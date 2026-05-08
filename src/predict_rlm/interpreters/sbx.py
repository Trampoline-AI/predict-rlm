"""Docker Sandboxes interpreter backend."""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput

from predict_rlm.files import get_synced_file_params
from predict_rlm.interpreter import SandboxFatalError

from .base import (
    STALE_RESPONSE_DISCARD_LIMIT,
    InterpreterExecutionGate,
    PredictRLMInterpreter,
    SbxConfig,
)

RUNNER_PATH = Path(__file__).parents[1] / "sandbox" / "python_runner.py"
DEFAULT_PACKAGE_DOMAINS = ["pypi.org", "files.pythonhosted.org"]
SBX_PYTHON_EXECUTABLE = "python3"


class SbxInterpreter(PredictRLMInterpreter):
    """Interpreter backend powered by Docker Sandboxes.

    The backend starts a Python JSON-RPC runner inside a Docker Sandbox and
    maps predict-rlm virtual paths under a per-run workspace staging root.
    """

    def __init__(
        self,
        *,
        config: SbxConfig | None = None,
        allowed_domains: list[str] | None = None,
        tools: dict[str, Callable[..., Any]] | None = None,
        output_fields: list[dict] | None = None,
        preinstall_packages: bool = True,
        skill_packages: list[str] | None = None,
        debug: bool = False,
        extra_read_paths: list[str] | None = None,
        extra_write_paths: list[str] | None = None,
        _runner_command: list[str] | None = None,
        _staging_root: str | Path | None = None,
    ) -> None:
        self.config = config or SbxConfig()
        self.allowed_domains = allowed_domains
        self.tools = tools or {}
        self.output_fields = output_fields or []
        self.preinstall_packages = preinstall_packages
        self.skill_packages = skill_packages or []
        self.debug = debug
        self.extra_read_paths = extra_read_paths or []
        self.extra_write_paths = extra_write_paths or []
        self._runner_command = _runner_command
        self._host_workspace = Path.cwd()
        self._owns_staging_root = _staging_root is None
        self._staging_root = Path(_staging_root) if _staging_root else (
            self._host_workspace / ".predict_rlm_sbx" / uuid.uuid4().hex
        )
        self._staging_root.mkdir(parents=True, exist_ok=True)
        self._proc: subprocess.Popen[str] | None = None
        self._stdout_lines: queue.Queue[str] = queue.Queue()
        self._stdout_reader: threading.Thread | None = None
        self._tool_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(4, len(self.tools) or 1)
        )
        self._pending_tool_calls: dict[concurrent.futures.Future[dict[str, Any]], int] = {}
        self._execution_gate = InterpreterExecutionGate("SBX interpreter")
        self._sandbox_name: str | None = None
        self._request_id = 0
        self._shutdown = False

    def execute(
        self, code: str, variables: dict[str, Any] | None = None
    ) -> Any:
        with self._execution_gate.top_level():
            return self._execute_top_level(code, variables)

    def _execute_top_level(
        self, code: str, variables: dict[str, Any] | None = None
    ) -> Any:
        if variables:
            mapped_variables = {
                name: self._map_variable_value(value) for name, value in variables.items()
            }
            assignments = "\n".join(
                f"{name} = {value!r}" for name, value in mapped_variables.items()
            )
            code = f"{assignments}\n{code}"
        response = self._send_request("execute", {"code": code})
        return self._unwrap_execute_response(response)

    async def aexecute(
        self, code: str, variables: dict[str, Any] | None = None
    ) -> Any:
        return await asyncio.to_thread(self.execute, code, variables)

    def mount_file_at(self, host_path: str, virtual_path: str) -> None:
        source = Path(host_path)
        target = self._host_path_for_virtual_path(virtual_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    def mkdir_p(self, virtual_path: str) -> None:
        self._host_path_for_virtual_path(virtual_path).mkdir(parents=True, exist_ok=True)

    def list_dir(self, virtual_path: str) -> list[str]:
        root = self._host_path_for_virtual_path(virtual_path)
        if not root.exists():
            return []
        return [
            self._virtual_path_for_host_path(path)
            for path in sorted(root.rglob("*"))
            if path.is_file()
        ]

    def sync_file_to(self, virtual_path: str, host_path: str) -> None:
        source = self._host_path_for_virtual_path(virtual_path)
        target = Path(host_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    def _host_path_for_virtual_path(self, virtual_path: str) -> Path:
        if virtual_path != "/sandbox" and not virtual_path.startswith("/sandbox/"):
            raise ValueError(f"Sbx virtual path must be under /sandbox: {virtual_path}")
        sandbox_root = (self._staging_root / "sandbox").resolve()
        rel = virtual_path.removeprefix("/sandbox").lstrip("/")
        host_path = (sandbox_root / rel).resolve()
        try:
            host_path.relative_to(sandbox_root)
        except ValueError as exc:
            raise ValueError(f"Sbx virtual path escapes /sandbox: {virtual_path}") from exc
        return host_path

    def _map_variable_value(self, value: Any) -> Any:
        if isinstance(value, str) and (value == "/sandbox" or value.startswith("/sandbox/")):
            return str(self._host_path_for_virtual_path(value))
        if isinstance(value, list):
            return [self._map_variable_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._map_variable_value(item) for item in value)
        if isinstance(value, dict):
            return {key: self._map_variable_value(item) for key, item in value.items()}
        return value

    def _virtual_path_for_host_path(self, host_path: Path) -> str:
        sandbox_root = (self._staging_root / "sandbox").resolve()
        rel = host_path.resolve().relative_to(sandbox_root)
        return "/sandbox/" + rel.as_posix()

    def configure_runtime(
        self,
        *,
        tools: dict[str, Callable[..., Any]] | None = None,
        output_fields: list[dict] | None = None,
    ) -> None:
        if tools is not None and tools is not self.tools:
            self.tools = tools
            self._tool_executor.shutdown(wait=False, cancel_futures=True)
            self._tool_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max(4, len(self.tools) or 1)
            )
        if output_fields is not None:
            self.output_fields = output_fields
        if self._proc and self._proc.poll() is None:
            if self.output_fields:
                self._send_request("register_output_fields", {"fields": self.output_fields})
            if self.tools:
                self._send_request("register_tools", {"tools": list(self.tools)})

    def prewarm(self) -> None:
        self._ensure_process()

    def reset(self) -> None:
        self._send_request("reset", {})
        sandbox_root = self._staging_root / "sandbox"
        shutil.rmtree(sandbox_root, ignore_errors=True)
        sandbox_root.mkdir(parents=True, exist_ok=True)

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        if self._proc and self._proc.poll() is None:
            try:
                self._send_request("shutdown", {})
            except Exception:
                pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)
        self._proc = None

        if self._runner_command is None and self._sandbox_name and self.config.remove_on_shutdown:
            if not self.config.persist:
                subprocess.run(
                    ["sbx", "rm", self._sandbox_name],
                    check=False,
                    capture_output=True,
                    text=True,
                )
        self._tool_executor.shutdown(wait=False, cancel_futures=True)
        self._cleanup_staging_root()

    def _cleanup_staging_root(self) -> None:
        if not self._owns_staging_root or self.config.persist:
            return
        shutil.rmtree(self._staging_root, ignore_errors=True)
        try:
            self._staging_root.parent.rmdir()
        except OSError:
            pass

    def _ensure_process(self) -> None:
        if self._proc and self._proc.poll() is None:
            return
        if self._proc and self._proc.poll() is not None:
            raise SandboxFatalError("Sbx runner process exited unexpectedly")

        if self._runner_command is not None:
            command = self._runner_command
        else:
            command = self._start_sbx_and_build_runner_command()

        env = os.environ.copy()
        env["PREDICT_RLM_SBX_ROOT"] = str(self._staging_root)
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=1,
        )
        self._start_stdout_reader()
        if self.output_fields:
            self._send_request("register_output_fields", {"fields": self.output_fields})
        if self.tools:
            self._send_request("register_tools", {"tools": list(self.tools)})

    def _start_stdout_reader(self) -> None:
        assert self._proc is not None
        assert self._proc.stdout is not None
        stdout = self._proc.stdout
        self._stdout_lines = queue.Queue()

        def read_stdout() -> None:
            for line in stdout:
                self._stdout_lines.put(line)

        self._stdout_reader = threading.Thread(
            target=read_stdout,
            name="predict-rlm-sbx-stdout",
            daemon=True,
        )
        self._stdout_reader.start()

    def _start_sbx_and_build_runner_command(self) -> list[str]:
        if shutil.which("sbx") is None:
            raise SandboxFatalError(
                "Docker Sandboxes backend requires the `sbx` CLI. "
                "Install it with `brew install docker/tap/sbx` and run `sbx login`."
            )

        runner_path = self._prepare_runner_script()

        primary_workspace = str(self._staging_root)
        if self.config.workspace_read_only:
            primary_workspace = f"{primary_workspace}:ro"
        create_cmd = [
            "sbx",
            "create",
            "shell",
            primary_workspace,
            *self.config.extra_workspaces,
        ]
        if self.config.name:
            create_cmd.extend(["--name", self.config.name])
        for flag, value in (
            ("--cpus", self.config.cpus),
            ("--memory", self.config.memory),
            ("--template", self.config.template),
            ("--kit", self.config.kit),
            ("--branch", self.config.branch),
        ):
            if value is not None:
                create_cmd.extend([flag, str(value)])
        try:
            created = subprocess.run(
                create_cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.config.create_timeout,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise SandboxFatalError(f"Failed to create sbx sandbox: {exc}") from exc

        self._sandbox_name = self.config.name or self._parse_sandbox_name(created.stdout)
        self._apply_network_policy()
        self._bootstrap_packages()

        runner_root = self._staging_root
        runner_root.mkdir(parents=True, exist_ok=True)
        return [
            "sbx",
            "exec",
            "-i",
            "-w",
            str(self._staging_root),
            self._sandbox_name,
            "env",
            f"PREDICT_RLM_SBX_ROOT={runner_root}",
            SBX_PYTHON_EXECUTABLE,
            "-u",
            str(runner_path),
        ]

    def _prepare_runner_script(self) -> Path:
        runner_dir = self._staging_root / ".predict_rlm_runner"
        runner_dir.mkdir(parents=True, exist_ok=True)
        runner_path = runner_dir / "python_runner.py"
        shutil.copy2(RUNNER_PATH, runner_path)
        return runner_path

    def _parse_sandbox_name(self, stdout: str) -> str:
        for token in reversed(stdout.replace("\n", " ").split()):
            if token.strip():
                return token.strip()
        raise SandboxFatalError("Could not determine created sbx sandbox name")

    def _apply_network_policy(self) -> None:
        domains = list(DEFAULT_PACKAGE_DOMAINS) if self.preinstall_packages else []
        domains.extend(self.allowed_domains or [])
        for domain in domains:
            subprocess.run(
                ["sbx", "policy", "allow", "network", domain],
                check=False,
                capture_output=True,
                text=True,
            )

    def _bootstrap_packages(self) -> None:
        packages = []
        if self.preinstall_packages:
            packages.extend(["pydantic", "pandas"])
        packages.extend(self.skill_packages)
        if not packages:
            return
        assert self._sandbox_name is not None
        command = [
            "sbx",
            "exec",
            "-w",
            str(self._staging_root),
            self._sandbox_name,
            SBX_PYTHON_EXECUTABLE,
            "-m",
            "pip",
            "install",
            "--break-system-packages",
            *packages,
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.config.exec_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise SandboxFatalError(
                f"Failed to bootstrap sbx packages {packages}: timed out after "
                f"{self.config.exec_timeout}s"
            ) from exc
        if result.returncode != 0:
            raise SandboxFatalError(
                "Failed to bootstrap sbx packages "
                f"{packages}: exit code {result.returncode}; "
                f"stdout: {result.stdout.strip()}; stderr: {result.stderr.strip()}"
            )

    def _read_stdout_line(self, deadline: float) -> str | None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        try:
            return self._stdout_lines.get(timeout=min(remaining, 0.05))
        except queue.Empty:
            return None

    def _fail_timed_out_request(self) -> None:
        assert self._proc is not None
        self._proc.kill()
        raise SandboxFatalError(
            f"Sbx runner request timed out after {self.config.exec_timeout}s"
        )

    def _submit_tool_call(self, request: dict[str, Any]) -> None:
        request_id = request.get("id")
        future = self._tool_executor.submit(self._build_tool_response, request)
        self._pending_tool_calls[future] = request_id

    def _drain_completed_tool_calls(self) -> None:
        completed = [
            future for future in self._pending_tool_calls
            if future.done()
        ]
        for future in completed:
            self._pending_tool_calls.pop(future)
            self._write_tool_response(future.result())

    def _build_tool_response(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = request.get("id")
        params = request.get("params", {})
        name = params.get("name")
        temp_dir: str | None = None
        try:
            if name not in self.tools:
                raise CodeInterpreterError(f"Unknown tool: {name}")
            tool = self.tools[name]
            args = list(params.get("args", []))
            kwargs = dict(params.get("kwargs", {}))
            args, kwargs, synced_entries, temp_dir = self._prepare_synced_file_tool_args(
                tool,
                args,
                kwargs,
            )
            with self._execution_gate.tool_callback():
                result = tool(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = asyncio.run(result)
            for sandbox_path, host_path, writeback in synced_entries:
                if writeback and os.path.isfile(host_path):
                    self.mount_file_at(host_path, sandbox_path)
            is_json = result is None or isinstance(result, (dict, list, int, float, bool))
            return {
                "jsonrpc": "2.0",
                "result": {
                    "value": json.dumps(result) if is_json else str(result or ""),
                    "type": "json" if is_json else "string",
                },
                "id": request_id,
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32000, "message": str(exc)},
                "id": request_id,
            }
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _prepare_synced_file_tool_args(
        self,
        tool: Callable[..., Any],
        args: list[Any],
        kwargs: dict[str, Any],
    ) -> tuple[list[Any], dict[str, Any], list[tuple[str, str, bool]], str | None]:
        synced_params = get_synced_file_params(tool)
        temp_dir: str | None = None
        synced_entries: list[tuple[str, str, bool]] = []
        if not synced_params:
            return args, kwargs, synced_entries, temp_dir

        sig = inspect.signature(tool)
        param_names = list(sig.parameters.keys())
        for param_name, synced_file in synced_params.items():
            sandbox_path = kwargs.get(param_name)
            if sandbox_path is None and param_name in param_names:
                idx = param_names.index(param_name)
                if idx < len(args):
                    sandbox_path = args[idx]
            if not sandbox_path or not isinstance(sandbox_path, str):
                continue

            if synced_file.host_dir is not None:
                host_dir = synced_file.host_dir
                os.makedirs(host_dir, exist_ok=True)
            else:
                if temp_dir is None:
                    temp_dir = tempfile.mkdtemp(prefix="tool-file-sync-")
                host_dir = temp_dir

            host_path = os.path.join(host_dir, os.path.basename(sandbox_path))
            self.sync_file_to(sandbox_path, host_path)
            synced_entries.append((sandbox_path, host_path, synced_file.writeback))

            if param_name in kwargs:
                kwargs[param_name] = host_path
            elif param_name in param_names:
                idx = param_names.index(param_name)
                if idx < len(args):
                    args[idx] = host_path

        return args, kwargs, synced_entries, temp_dir

    def _write_tool_response(self, response: dict[str, Any]) -> None:
        assert self._proc is not None
        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(response) + "\n")
        self._proc.stdin.flush()

    def _send_request(self, method: str, params: dict[str, Any] | None = None) -> dict:
        self._ensure_process_for_method(method)
        assert self._proc is not None
        if self._proc.stdin is None or self._proc.stdout is None:
            raise SandboxFatalError("Sbx runner stdio is unavailable")

        self._request_id += 1
        request_id = self._request_id
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": request_id,
        }
        try:
            self._proc.stdin.write(json.dumps(payload) + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError as exc:
            raise SandboxFatalError("Sbx runner pipe broke while sending request") from exc

        deadline = time.monotonic() + self.config.exec_timeout
        stale_discards = 0
        while True:
            self._drain_completed_tool_calls()
            if self._proc.poll() is not None:
                stderr = self._proc.stderr.read() if self._proc.stderr else ""
                raise SandboxFatalError(f"Sbx runner exited unexpectedly: {stderr}")
            if time.monotonic() > deadline:
                self._fail_timed_out_request()
            line = self._read_stdout_line(deadline)
            if not line:
                continue
            if not line.startswith("{"):
                continue
            response = json.loads(line)
            if response.get("method") == "tool_call":
                self._submit_tool_call(response)
                continue
            if response.get("id") == request_id:
                return response
            stale_discards += 1
            if stale_discards > STALE_RESPONSE_DISCARD_LIMIT:
                raise CodeInterpreterError(
                    "Too many stale top-level responses while resyncing "
                    f"SBX request id={request_id}"
                )

    def _ensure_process_for_method(self, method: str) -> None:
        if method == "shutdown" and self._proc is not None:
            return
        self._ensure_process()

    def _unwrap_execute_response(self, response: dict) -> Any:
        if "error" in response:
            error = response["error"]
            error_data = error.get("data", {})
            error_type = error_data.get("type", "Sandbox Error")
            if error_type == "SyntaxError":
                raise SyntaxError(error.get("message", "Invalid Python syntax"))
            raise CodeInterpreterError(
                f"{error_type}: {error_data.get('args') or error.get('message', '')}"
            )

        result = response.get("result", {})
        if "final" in result:
            return FinalOutput(result["final"])
        return result.get("output")


class SbxPool:
    """Thread-safe pool of prewarmed Docker Sandboxes interpreters."""

    def __init__(
        self,
        *,
        size: int,
        config: SbxConfig | None = None,
        allowed_domains: list[str] | None = None,
        tools: dict[str, Callable[..., Any]] | None = None,
        output_fields: list[dict] | None = None,
        preinstall_packages: bool = True,
        skill_packages: list[str] | None = None,
        debug: bool = False,
        extra_read_paths: list[str] | None = None,
        extra_write_paths: list[str] | None = None,
        _runner_command: list[str] | None = None,
        _staging_root: str | Path | None = None,
    ) -> None:
        if size < 1:
            raise ValueError("SbxPool size must be at least 1")
        self.size = size
        self.config = config or SbxConfig()
        self._pool_name_prefix = self.config.name or f"predict-rlm-sbx-pool-{uuid.uuid4().hex[:12]}"
        self._interpreter_kwargs = {
            "config": self.config,
            "allowed_domains": allowed_domains,
            "tools": tools,
            "output_fields": output_fields,
            "preinstall_packages": preinstall_packages,
            "skill_packages": skill_packages,
            "debug": debug,
            "extra_read_paths": extra_read_paths,
            "extra_write_paths": extra_write_paths,
            "_runner_command": _runner_command,
        }
        self._staging_root = Path(_staging_root) if _staging_root is not None else None
        self._available: queue.Queue[SbxInterpreter] = queue.Queue(maxsize=size)
        self._all_interpreters: list[SbxInterpreter] = []
        self._lock = threading.Lock()
        self._state_changed = threading.Condition(self._lock)
        self._started = False
        self._starting = False
        self._shutdown = False
        self._shutdown_requested = False
        self._shutting_down = False

    def __enter__(self) -> SbxPool:
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.shutdown()

    def start(self) -> None:
        if self._begin_start(allow_restart=True):
            self._finish_start()

    def _begin_start(self, *, allow_restart: bool) -> bool:
        with self._state_changed:
            if allow_restart:
                while self._shutting_down:
                    self._state_changed.wait()
            elif self._is_stopping_locked():
                raise RuntimeError("SbxPool is shut down")
            if self._started:
                return False
            while self._starting:
                self._state_changed.wait()
                if not allow_restart and self._is_stopping_locked():
                    raise RuntimeError("SbxPool is shut down")
                if self._started:
                    return False
            if not allow_restart and self._is_stopping_locked():
                raise RuntimeError("SbxPool is shut down")
            self._starting = True
            self._shutdown = False
            self._shutdown_requested = False
            return True

    def _finish_start(self) -> None:
        interpreters: list[SbxInterpreter] = []
        try:
            for index in range(self.size):
                interpreters.append(self._create_interpreter(index))
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.size) as executor:
                futures = [executor.submit(interpreter.prewarm) for interpreter in interpreters]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
        except Exception:
            self._shutdown_interpreters(interpreters, suppress_errors=True)
            with self._state_changed:
                self._drain_available_locked()
                self._all_interpreters.clear()
                self._started = False
                self._starting = False
                self._state_changed.notify_all()
            raise

        with self._state_changed:
            self._drain_available_locked()
            self._all_interpreters = interpreters
            for interpreter in interpreters:
                self._available.put(interpreter)
            self._started = True
            self._starting = False
            self._shutdown = False
            self._state_changed.notify_all()

    @contextmanager
    def lease(
        self,
        *,
        tools: dict[str, Callable[..., Any]] | None = None,
        output_fields: list[dict] | None = None,
    ):
        self._ensure_started_for_lease()
        interpreter = self._acquire_interpreter()
        try:
            interpreter.configure_runtime(tools=tools, output_fields=output_fields)
            yield interpreter
        finally:
            with self._state_changed:
                stopping = self._is_stopping_locked() or interpreter not in self._all_interpreters
            if stopping:
                return
            try:
                interpreter.reset()
            except Exception:
                interpreter.shutdown()
                with self._state_changed:
                    if self._is_stopping_locked() or interpreter not in self._all_interpreters:
                        self._state_changed.notify_all()
                        return
                    index = self._all_interpreters.index(interpreter)
                    replacement = self._create_interpreter(index)
                    replacement.prewarm()
                    self._all_interpreters[index] = replacement
                    interpreter = replacement
            with self._state_changed:
                if self._is_stopping_locked() or interpreter not in self._all_interpreters:
                    self._state_changed.notify_all()
                    return
                self._available.put_nowait(interpreter)
                self._state_changed.notify()

    def shutdown(self) -> None:
        with self._state_changed:
            while self._starting:
                self._shutdown_requested = True
                self._state_changed.notify_all()
                self._state_changed.wait()
            if self._shutdown:
                return
            self._shutdown = True
            self._shutdown_requested = False
            self._shutting_down = True
            interpreters = list(self._all_interpreters)
            self._drain_available_locked()
            self._all_interpreters.clear()
            self._started = False
            self._state_changed.notify_all()

        try:
            self._shutdown_interpreters(interpreters)
        finally:
            with self._state_changed:
                self._shutting_down = False
                self._state_changed.notify_all()

    def _ensure_started_for_lease(self) -> None:
        if self._begin_start(allow_restart=False):
            self._finish_start()
        with self._state_changed:
            if self._is_stopping_locked() or not self._started:
                raise RuntimeError("SbxPool is shut down")

    def _acquire_interpreter(self) -> SbxInterpreter:
        with self._state_changed:
            while True:
                if self._is_stopping_locked() or not self._started:
                    raise RuntimeError("SbxPool is shut down")
                try:
                    return self._available.get_nowait()
                except queue.Empty:
                    self._state_changed.wait()

    def _is_stopping_locked(self) -> bool:
        return self._shutdown or self._shutdown_requested or self._shutting_down

    def _create_interpreter(self, index: int) -> SbxInterpreter:
        kwargs = dict(self._interpreter_kwargs)
        if self.size > 1:
            kwargs["config"] = self.config.model_copy(
                update={"name": f"{self._pool_name_prefix}-{index}"}
            )
        if self._staging_root is not None:
            kwargs["_staging_root"] = self._staging_root / f"runner-{index}"
        return SbxInterpreter(**kwargs)

    def _drain_available_locked(self) -> None:
        while True:
            try:
                self._available.get_nowait()
            except queue.Empty:
                break

    def _shutdown_interpreters(
        self,
        interpreters: list[SbxInterpreter],
        *,
        suppress_errors: bool = False,
    ) -> None:
        first_error: BaseException | None = None
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, len(interpreters))
        ) as executor:
            futures = [executor.submit(interpreter.shutdown) for interpreter in interpreters]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
        if first_error is not None and not suppress_errors:
            raise first_error
