"""Docker Sandboxes interpreter backend."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import inspect
import json
import logging
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput

from predict_rlm._logging import (
    configure_predict_rlm_logging,
    emit_trace_error,
    emit_trace_result,
    emit_trace_tool_call,
    format_log_fields,
    interpreter_result_logging_enabled,
    live_tool_call_logging_enabled,
)
from predict_rlm.files import get_synced_file_params
from predict_rlm.interpreter import SandboxFatalError
from predict_rlm.trace import ToolCall, ms_since, record_tool_call

from .base import (
    STALE_RESPONSE_DISCARD_LIMIT,
    InterpreterExecutionGate,
    PredictRLMInterpreter,
    SandboxExecutionError,
    SbxConfig,
)
from .sbx_pool import SbxPool as SbxPool

RUNNER_PATH = Path(__file__).parents[1] / "sandbox" / "python_runner.py"
DEFAULT_PACKAGE_DOMAINS = ["pypi.org", "files.pythonhosted.org"]
SBX_PYTHON_EXECUTABLE = "python3"
logger = logging.getLogger(__name__)


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
        verbose: bool = False,
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
        self.verbose = verbose
        configure_predict_rlm_logging(debug=debug, verbose=verbose)
        self.extra_read_paths = extra_read_paths or []
        self.extra_write_paths = extra_write_paths or []
        self._runner_command = _runner_command
        self._host_workspace = Path.cwd()
        self._owns_staging_root = _staging_root is None
        self._staging_root = (
            Path(_staging_root)
            if _staging_root
            else (self._host_workspace / ".predict_rlm_sbx" / uuid.uuid4().hex)
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

    def configure_debug(self, enabled: bool) -> None:
        self.debug = enabled
        configure_predict_rlm_logging(debug=enabled)

    def configure_verbose(self, enabled: bool) -> None:
        self.verbose = enabled
        configure_predict_rlm_logging(verbose=enabled)

    def _log_lifecycle(self, event: str, **fields: Any) -> None:
        if not getattr(self, "debug", False):
            return
        process_pid = getattr(self._proc, "pid", None) if self._proc else None
        logger.debug(
            "%s%s",
            event,
            format_log_fields(
                {
                    "backend": "sbx",
                    "sandbox_name": getattr(self, "_sandbox_name", None),
                    "process_pid": process_pid,
                    "staging_root": str(getattr(self, "_staging_root", "")) or None,
                    **fields,
                }
            ),
        )

    def _log_partial_output(self, output: str, **fields: Any) -> None:
        if not getattr(self, "debug", False) or not output:
            return
        process_pid = getattr(self._proc, "pid", None) if self._proc else None
        logger.debug(
            "sandbox.partial_output%s\n%s",
            format_log_fields(
                {
                    "backend": "sbx",
                    "sandbox_name": getattr(self, "_sandbox_name", None),
                    "process_pid": process_pid,
                    "staging_root": str(getattr(self, "_staging_root", "")) or None,
                    "chars": len(output),
                    **fields,
                }
            ),
            output.rstrip(),
        )

    def execute(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        with self._execution_gate.top_level():
            return self._execute_top_level(code, variables)

    def _execute_top_level(self, code: str, variables: dict[str, Any] | None = None) -> Any:
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

    async def aexecute(self, code: str, variables: dict[str, Any] | None = None) -> Any:
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
        debug: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        if debug is not None:
            self.configure_debug(debug)
        if verbose is not None:
            self.configure_verbose(verbose)
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
        self._log_lifecycle(
            "sbx.runtime.configured",
            tools=len(self.tools),
            output_fields=len(self.output_fields),
            process_running=bool(self._proc and self._proc.poll() is None),
        )

    def prewarm(self) -> None:
        self._log_lifecycle("sbx.prewarm.start")
        self._ensure_process()
        self._log_lifecycle("sbx.prewarm.ok")

    def reset(self) -> None:
        self._log_lifecycle("sbx.reset.start")
        self._send_request("reset", {})
        sandbox_root = self._staging_root / "sandbox"
        shutil.rmtree(sandbox_root, ignore_errors=True)
        sandbox_root.mkdir(parents=True, exist_ok=True)
        self._log_lifecycle("sbx.reset.ok")

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self._log_lifecycle("sbx.shutdown.start")
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
                self._log_lifecycle("sbx.shutdown.kill", kill_result="sent")
        self._proc = None

        if (
            self._runner_command is None
            and self._sandbox_name
            and self.config.remove_on_shutdown
        ):
            if not self.config.persist:
                subprocess.run(
                    ["sbx", "rm", self._sandbox_name],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self._log_lifecycle("sbx.shutdown.rm")
        self._tool_executor.shutdown(wait=False, cancel_futures=True)
        self._cleanup_staging_root()
        self._log_lifecycle("sbx.shutdown.complete")

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
        self._log_lifecycle(
            "sbx.runner.start",
            command=command[0] if command else None,
        )
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=1,
        )
        self._log_lifecycle("sbx.runner.started")
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
            self._log_lifecycle("sbx.create.missing_cli", status="error")
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
        create_start = time.perf_counter()
        self._log_lifecycle(
            "sbx.create.start",
            create_timeout=self.config.create_timeout,
            workspace_read_only=self.config.workspace_read_only,
            extra_workspaces=len(self.config.extra_workspaces),
        )
        try:
            created = subprocess.run(
                create_cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.config.create_timeout,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            self._log_lifecycle(
                "sbx.create.error",
                duration_ms=ms_since(create_start),
                error_type=type(exc).__name__,
                status="error",
            )
            raise SandboxFatalError(f"Failed to create sbx sandbox: {exc}") from exc

        self._sandbox_name = self.config.name or self._parse_sandbox_name(created.stdout)
        self._log_lifecycle(
            "sbx.create.ok",
            duration_ms=ms_since(create_start),
            stdout_chars=len(created.stdout or ""),
            stderr_chars=len(created.stderr or ""),
        )
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
        self._log_lifecycle("sbx.network_policy.start", domains=len(domains))
        for domain in domains:
            result = subprocess.run(
                ["sbx", "policy", "allow", "network", domain],
                check=False,
                capture_output=True,
                text=True,
            )
            self._log_lifecycle(
                "sbx.network_policy.domain",
                domain=domain,
                returncode=result.returncode,
                status="ok" if result.returncode == 0 else "error",
            )
        self._log_lifecycle("sbx.network_policy.complete", domains=len(domains))

    def _bootstrap_packages(self) -> None:
        packages = []
        if self.preinstall_packages:
            packages.extend(["pydantic", "pandas"])
        packages.extend(self.skill_packages)
        if not packages:
            self._log_lifecycle("sbx.bootstrap.skip", packages=0)
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
        bootstrap_start = time.perf_counter()
        self._log_lifecycle(
            "sbx.bootstrap.start",
            packages=",".join(packages),
            timeout_seconds=self.config.exec_timeout,
        )
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.config.exec_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            self._log_lifecycle(
                "sbx.bootstrap.timeout",
                packages=",".join(packages),
                duration_ms=ms_since(bootstrap_start),
                status="error",
            )
            raise SandboxFatalError(
                f"Failed to bootstrap sbx packages {packages}: timed out after "
                f"{self.config.exec_timeout}s"
            ) from exc
        if result.returncode != 0:
            self._log_lifecycle(
                "sbx.bootstrap.error",
                packages=",".join(packages),
                duration_ms=ms_since(bootstrap_start),
                returncode=result.returncode,
                stdout_chars=len(result.stdout or ""),
                stderr_chars=len(result.stderr or ""),
                status="error",
            )
            raise SandboxFatalError(
                "Failed to bootstrap sbx packages "
                f"{packages}: exit code {result.returncode}; "
                f"stdout: {result.stdout.strip()}; stderr: {result.stderr.strip()}"
            )
        self._log_lifecycle(
            "sbx.bootstrap.ok",
            packages=",".join(packages),
            duration_ms=ms_since(bootstrap_start),
            stdout_chars=len(result.stdout or ""),
            stderr_chars=len(result.stderr or ""),
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
        self._log_lifecycle(
            "sbx.request.timeout",
            timeout_seconds=self.config.exec_timeout,
            status="error",
        )
        self._proc.kill()
        raise SandboxFatalError(
            f"Sbx runner request timed out after {self.config.exec_timeout}s"
        )

    def _submit_tool_call(self, request: dict[str, Any]) -> None:
        request_id = request.get("id")
        if self.verbose or live_tool_call_logging_enabled():
            params = request.get("params", {})
            emit_trace_tool_call(
                params.get("name"),
                args=params.get("args", []),
                kwargs=params.get("kwargs", {}),
            )
        params = request.get("params", {})
        self._log_lifecycle(
            "sbx.tool_call.start",
            tool=params.get("name"),
            request_id=request_id,
        )
        ctx = contextvars.copy_context()
        future = self._tool_executor.submit(ctx.run, self._build_tool_response, request)
        self._pending_tool_calls[future] = request_id

    def _drain_completed_tool_calls(self) -> None:
        completed = [future for future in self._pending_tool_calls if future.done()]
        for future in completed:
            self._pending_tool_calls.pop(future)
            self._write_tool_response(future.result())

    def _build_tool_response(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = request.get("id")
        params = request.get("params", {})
        name = params.get("name")
        temp_dir: str | None = None
        call_start = time.perf_counter()
        args: list[Any] = []
        kwargs: dict[str, Any] = {}
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
            response = {
                "jsonrpc": "2.0",
                "result": {
                    "value": json.dumps(result) if is_json else str(result or ""),
                    "type": "json" if is_json else "string",
                },
                "id": request_id,
            }
            if name != "predict":
                record_tool_call(
                    ToolCall(
                        name=name,
                        args=args,
                        kwargs={k: v for k, v in kwargs.items() if k != "pydantic_schemas"},
                        result=result,
                        duration_ms=ms_since(call_start),
                    )
                )
            self._log_lifecycle(
                "sbx.tool_call.ok",
                tool=name,
                request_id=request_id,
                duration_ms=ms_since(call_start),
            )
            return response
        except Exception as exc:
            if name != "predict":
                record_tool_call(
                    ToolCall(
                        name=name or "",
                        args=args,
                        kwargs={k: v for k, v in kwargs.items() if k != "pydantic_schemas"},
                        result=None,
                        error=str(exc),
                        duration_ms=ms_since(call_start),
                    )
                )
            self._log_lifecycle(
                "sbx.tool_call.error",
                tool=name,
                request_id=request_id,
                duration_ms=ms_since(call_start),
                error_type=type(exc).__name__,
                status="error",
            )
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
        request_start = time.perf_counter()
        self._log_lifecycle(
            "sbx.request.start",
            method=method,
            request_id=request_id,
            timeout_seconds=self.config.exec_timeout,
        )
        try:
            self._proc.stdin.write(json.dumps(payload) + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError as exc:
            self._log_lifecycle(
                "sbx.request.broken_pipe",
                method=method,
                request_id=request_id,
                status="error",
            )
            raise SandboxFatalError("Sbx runner pipe broke while sending request") from exc

        deadline = time.monotonic() + self.config.exec_timeout
        stale_discards = 0
        while True:
            self._drain_completed_tool_calls()
            if self._proc.poll() is not None:
                stderr = self._proc.stderr.read() if self._proc.stderr else ""
                self._log_lifecycle(
                    "sbx.runner.exited",
                    method=method,
                    request_id=request_id,
                    stderr_chars=len(stderr or ""),
                    status="error",
                )
                raise SandboxFatalError(f"Sbx runner exited unexpectedly: {stderr}")
            if time.monotonic() > deadline:
                self._fail_timed_out_request()
            line = self._read_stdout_line(deadline)
            if not line:
                continue
            if not line.startswith("{"):
                self._log_lifecycle(
                    "sbx.protocol.non_json_stdout",
                    method=method,
                    request_id=request_id,
                    preview=line[:200],
                )
                continue
            response = json.loads(line)
            if response.get("method") == "tool_call":
                self._submit_tool_call(response)
                continue
            if response.get("id") == request_id:
                self._log_lifecycle(
                    "sbx.request.ok",
                    method=method,
                    request_id=request_id,
                    duration_ms=ms_since(request_start),
                )
                return response
            stale_discards += 1
            self._log_lifecycle(
                "sbx.protocol.stale_response",
                method=method,
                request_id=request_id,
                response_id=response.get("id"),
                stale_discards=stale_discards,
            )
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
            partial_output = error_data.get("output") or ""
            if partial_output:
                self._log_partial_output(partial_output, error_type=error_type)
            if interpreter_result_logging_enabled(self.verbose):
                if partial_output:
                    emit_trace_result({"output": partial_output})
                emit_trace_error(
                    error_type,
                    error.get("message") or error_data.get("args", []),
                )
            if error_type == "SyntaxError":
                raise SyntaxError(error.get("message", "Invalid Python syntax"))
            raise SandboxExecutionError(
                f"{error_type}: {error_data.get('args') or error.get('message', '')}",
                partial_output=partial_output,
            )

        result = response.get("result", {})
        if interpreter_result_logging_enabled(self.verbose):
            emit_trace_result(result)
        if "final" in result:
            return FinalOutput(result["final"])
        return result.get("output")
