"""Docker Sandboxes interpreter backend."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import hashlib
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
from pathlib import Path
from typing import Any, Callable

from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput

from predict_rlm._logging import (
    configure_predict_rlm_logging,
    emit_trace_error,
    emit_trace_result,
    emit_trace_tool_call,
    interpreter_result_logging_enabled,
    live_tool_call_logging_enabled,
)
from predict_rlm._shared import strip_code_fences
from predict_rlm.execution_timeout import (
    ITERATION_TIMEOUT_FAILURE_CLASS,
    format_recoverable_timeout_result,
    recoverable_timeout_host_deadline_seconds,
    resolve_execution_timeout,
)
from predict_rlm.files import get_synced_file_params
from predict_rlm.interpreter import SandboxFatalError
from predict_rlm.runtime_hooks import RuntimeHook, RuntimeHookEvent
from predict_rlm.serialization import to_plain_data
from predict_rlm.trace import ToolCall, ms_since, record_tool_call
from predict_rlm.workspace import DirectWorkspaceMount, WorkspaceFileInfo

from .base import (
    InterpreterExecutionGate,
    PredictRLMInterpreter,
    SandboxExecutionError,
    SbxConfig,
)
from .persistent_runner import PersistentJsonRpcRunnerClient, PersistentSupervisorProcess
from .sbx_logging import log_interpreter_lifecycle, log_partial_output
from .sbx_pool import SbxPool as SbxPool

RUNNER_PATH = Path(__file__).parents[1] / "sandbox" / "python_runner.py"
DEFAULT_PACKAGE_DOMAINS = ["pypi.org", "files.pythonhosted.org"]
SBX_PYTHON_EXECUTABLE = "python3"


class SbxInterpreter(PersistentJsonRpcRunnerClient, PredictRLMInterpreter):
    """Interpreter backend powered by Docker Sandboxes.

    The backend starts a Python JSON-RPC supervisor inside a Docker Sandbox and
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
        _supervisor_command: list[str] | None = None,
        direct_workspace_mounts: list[DirectWorkspaceMount] | None = None,
        runtime_hooks: list[RuntimeHook] | None = None,
        on_runtime_hook_event: Callable[[RuntimeHookEvent], Any] | None = None,
        _runner_command: list[str] | None = None,
        _staging_root: str | Path | None = None,
    ) -> None:
        PersistentJsonRpcRunnerClient.__init__(self, supervisor_name="Sbx supervisor")
        self.config = config or SbxConfig()
        self.allowed_domains = allowed_domains
        self.tools = tools or {}
        self.output_fields = output_fields or []
        self.preinstall_packages = preinstall_packages
        self.skill_packages = skill_packages or []
        self.debug = debug
        self.verbose = verbose
        configure_predict_rlm_logging(
            debug=True if debug else None,
            verbose=True if verbose else None,
        )
        self.extra_read_paths = extra_read_paths or []
        self.extra_write_paths = extra_write_paths or []
        self._supervisor_command = _supervisor_command or _runner_command
        self._direct_workspace_mounts = list(direct_workspace_mounts or [])
        self.runtime_hooks = list(runtime_hooks or [])
        self.on_runtime_hook_event = on_runtime_hook_event
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
        self._prepared_runner_path: Path | None = None
        self._shutdown = False
        self._post_execute_hooks: list[Callable[[Any], Any]] = []
        self._owned_direct_aliases: list[Path] = []
        self._relocate_owned_staging_root_if_nested_in_direct_workspace()

    def configure_debug(self, enabled: bool) -> None:
        self.debug = enabled
        configure_predict_rlm_logging(debug=enabled)

    def configure_verbose(self, enabled: bool) -> None:
        self.verbose = enabled
        configure_predict_rlm_logging(verbose=enabled)

    def _log_lifecycle(self, event: str, **fields: Any) -> None:
        process_pid = getattr(self._proc, "pid", None) if self._proc else None
        log_interpreter_lifecycle(
            enabled=getattr(self, "debug", False),
            event=event,
            sandbox_name=getattr(self, "_sandbox_name", None),
            process_pid=process_pid,
            staging_root=getattr(self, "_staging_root", None),
            **fields,
        )

    def _log_partial_output(self, output: str, **fields: Any) -> None:
        process_pid = getattr(self._proc, "pid", None) if self._proc else None
        log_partial_output(
            enabled=getattr(self, "debug", False),
            output=output,
            sandbox_name=getattr(self, "_sandbox_name", None),
            process_pid=process_pid,
            staging_root=getattr(self, "_staging_root", None),
            **fields,
        )

    def execute(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        with self._execution_gate.top_level():
            try:
                return self._execute_top_level(code, variables, timeout=timeout)
            finally:
                self._run_post_execute_hooks()

    def _execute_top_level(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        code = strip_code_fences(code)
        if variables:
            mapped_variables = {
                name: self._map_variable_value(value) for name, value in variables.items()
            }
            assignments = "\n".join(
                f"{name} = {value!r}" for name, value in mapped_variables.items()
            )
            code = f"{assignments}\n{code}"
        params: dict[str, Any] = {"code": code}
        if timeout is not None:
            execution_timeout, _ = self._resolve_execution_timeout(timeout)
            params["execution_timeout_seconds"] = execution_timeout
        response = self._send_request("execute", params, timeout=timeout)
        return self._unwrap_execute_response(response)

    async def aexecute(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        return await asyncio.to_thread(self.execute, code, variables, timeout=timeout)

    def mount_file_at(self, host_path: str, virtual_path: str) -> None:
        source = Path(host_path)
        target = self._host_path_for_sandbox_path(virtual_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    def mkdir_p(self, virtual_path: str) -> None:
        self._host_path_for_sandbox_path(virtual_path).mkdir(parents=True, exist_ok=True)

    def list_dir(self, virtual_path: str) -> list[str]:
        root = self._host_path_for_sandbox_path(virtual_path)
        if not root.exists():
            return []
        return [
            self._sandbox_path_for_host_path(path)
            for path in sorted(root.rglob("*"))
            if path.is_file()
        ]

    def workspace_manifest(self, virtual_path: str) -> dict[str, WorkspaceFileInfo]:
        root = self._host_path_for_sandbox_path(virtual_path)
        if not root.exists():
            raise FileNotFoundError(f"Workspace mount does not exist: {virtual_path}")
        if not root.is_dir():
            raise NotADirectoryError(f"Workspace mount is not a directory: {virtual_path}")
        files: dict[str, WorkspaceFileInfo] = {}
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel_path = path.relative_to(root).as_posix()
            files[rel_path] = WorkspaceFileInfo(
                type="file",
                sha256=self._sha256_file(path),
                size=path.stat().st_size,
            )
        return files

    def add_post_execute_hook(self, hook: Callable[[Any], Any]) -> None:
        self._post_execute_hooks.append(hook)

    def _run_post_execute_hooks(self) -> None:
        for hook in self._post_execute_hooks:
            hook(self)

    def sync_file_to(self, virtual_path: str, host_path: str) -> None:
        source = self._host_path_for_sandbox_path(virtual_path)
        target = Path(host_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    def _sha256_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _host_path_for_virtual_path(self, virtual_path: str) -> Path:
        sandbox_root = (self._staging_root / "sandbox").resolve()
        rel = virtual_path.removeprefix("/sandbox").lstrip("/")
        host_path = (sandbox_root / rel).resolve()
        try:
            host_path.relative_to(sandbox_root)
        except ValueError as exc:
            raise ValueError(f"Sbx virtual path escapes /sandbox: {virtual_path}") from exc
        return host_path

    def _host_path_for_sandbox_path(self, sandbox_path: str) -> Path:
        for mount in self._direct_workspace_mounts:
            rel_path = self._relative_to_prefix(sandbox_path, mount.sandbox_path)
            if rel_path is not None:
                return Path(mount.host_path, *rel_path.parts)
        if sandbox_path == "/sandbox" or sandbox_path.startswith("/sandbox/"):
            return self._host_path_for_virtual_path(sandbox_path)
        raise ValueError(
            "Sbx path must be under /sandbox or a direct workspace mount: "
            f"{sandbox_path}"
        )

    def _map_variable_value(self, value: Any) -> Any:
        value = to_plain_data(value)
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

    def _sandbox_path_for_host_path(self, host_path: Path) -> str:
        for mount in self._direct_workspace_mounts:
            try:
                rel = host_path.resolve().relative_to(Path(mount.host_path).resolve())
            except ValueError:
                continue
            if rel.as_posix() == ".":
                return mount.sandbox_path
            return f"{mount.sandbox_path.rstrip('/')}/{rel.as_posix()}"
        return self._virtual_path_for_host_path(host_path)

    def _relative_to_prefix(self, path: str, prefix: str) -> Path | None:
        try:
            rel = Path(path).relative_to(Path(prefix))
        except ValueError:
            return None
        return Path() if rel.as_posix() == "." else rel

    def configure_direct_workspace_mounts(
        self, mounts: list[DirectWorkspaceMount]
    ) -> None:
        mounts = list(mounts)
        if self._same_direct_workspace_mounts(mounts):
            return
        if self._proc and self._proc.poll() is None:
            raise RuntimeError(
                "Direct workspace mounts must be configured before the SBX runner starts"
            )
        self._direct_workspace_mounts = mounts
        self._relocate_owned_staging_root_if_nested_in_direct_workspace()

    def _same_direct_workspace_mounts(self, mounts: list[DirectWorkspaceMount]) -> bool:
        return self._direct_workspace_mount_keys(mounts) == self._direct_workspace_mount_keys(
            self._direct_workspace_mounts
        )

    def _direct_workspace_mount_keys(
        self, mounts: list[DirectWorkspaceMount]
    ) -> list[tuple[str, str]]:
        return [
            (
                os.path.abspath(mount.host_path),
                os.path.normpath(mount.sandbox_path),
            )
            for mount in mounts
        ]

    def configure_runtime(
        self,
        *,
        tools: dict[str, Callable[..., Any]] | None = None,
        output_fields: list[dict] | None = None,
        debug: bool | None = None,
        verbose: bool | None = None,
        runtime_hooks: list[RuntimeHook] | None = None,
        on_runtime_hook_event: Callable[[RuntimeHookEvent], Any] | None = None,
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
        if runtime_hooks is not None:
            self.runtime_hooks = list(runtime_hooks)
            self.on_runtime_hook_event = on_runtime_hook_event
        if self._proc and self._proc.poll() is None:
            if self.output_fields:
                self._send_request("register_output_fields", {"fields": self.output_fields})
            if self.tools:
                self._send_request("register_tools", {"tools": list(self.tools)})
            self._register_runtime_hooks()
        self._log_lifecycle(
            "sbx.runtime.configured",
            tools=len(self.tools),
            output_fields=len(self.output_fields),
            process_running=bool(self._proc and self._proc.poll() is None),
        )

    def _register_runtime_hooks(self) -> None:
        self._send_request(
            "register_runtime_hooks",
            {"hooks": [hook.model_dump(mode="json") for hook in self.runtime_hooks]},
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

        if self._supervisor_command is None and self._sandbox_name and self.config.remove_on_shutdown:
            if not self.config.persist:
                subprocess.run(
                    ["sbx", "rm", "--force", self._sandbox_name],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self._log_lifecycle("sbx.shutdown.rm")
        self._tool_executor.shutdown(wait=False, cancel_futures=True)
        self._cleanup_direct_workspace_aliases_host_side()
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

    def _relocate_owned_staging_root_if_nested_in_direct_workspace(self) -> None:
        if not self._owns_staging_root:
            return
        staging_root = self._staging_root.resolve()
        for mount in self._direct_workspace_mounts:
            direct_root = Path(mount.host_path).resolve()
            try:
                staging_root.relative_to(direct_root)
            except ValueError:
                continue
            old_staging_root = self._staging_root
            self._staging_root = Path(tempfile.mkdtemp(prefix="predict-rlm-sbx-"))
            shutil.rmtree(old_staging_root, ignore_errors=True)
            try:
                old_staging_root.parent.rmdir()
            except OSError:
                pass
            return

    def _ensure_process(self) -> None:
        if self._proc and self._proc.poll() is None:
            return
        if self._proc and self._proc.poll() is not None:
            raise SandboxFatalError("Sbx supervisor process exited unexpectedly")

        try:
            if self._supervisor_command is not None:
                self._setup_direct_workspace_aliases_host_side()
                command = self._supervisor_command
            else:
                command = self._start_sbx_and_build_supervisor_command()
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
        except BaseException as exc:
            self._log_lifecycle(
                "sbx.runner.error",
                error_type=type(exc).__name__,
                status="error",
            )
            raise
        self._log_lifecycle("sbx.runner.started")
        self._start_stdout_reader()
        if self.output_fields:
            self._send_request("register_output_fields", {"fields": self.output_fields})
        if self.tools:
            self._send_request("register_tools", {"tools": list(self.tools)})
        self._register_runtime_hooks()

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

    def _start_sbx_and_build_supervisor_command(self) -> list[str]:
        if shutil.which("sbx") is None:
            self._log_lifecycle("sbx.create.missing_cli", status="error")
            raise SandboxFatalError(
                "Docker Sandboxes backend requires the `sbx` CLI. "
                "Install it with `brew install docker/tap/sbx` and run `sbx login`."
            )

        if self._sandbox_name is None:
            runner_path = self._prepare_runner_script()

            primary_workspace = str(self._staging_root)
            if self.config.workspace_read_only:
                primary_workspace = f"{primary_workspace}:ro"
            direct_workspaces = self._direct_workspace_args()
            create_cmd = [
                "sbx",
                "create",
                "shell",
                primary_workspace,
                *self.config.extra_workspaces,
                *direct_workspaces,
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
            except subprocess.CalledProcessError as exc:
                self._log_lifecycle(
                    "sbx.create.error",
                    duration_ms=ms_since(create_start),
                    error_type=type(exc).__name__,
                    status="error",
                )
                raise SandboxFatalError(
                    f"Failed to create sbx sandbox: {exc}. "
                    f"stdout: {exc.stdout or ''} stderr: {exc.stderr or ''}"
                ) from exc
            except subprocess.TimeoutExpired as exc:
                self._log_lifecycle(
                    "sbx.create.error",
                    duration_ms=ms_since(create_start),
                    error_type=type(exc).__name__,
                    status="error",
                )
                raise SandboxFatalError(
                    f"Failed to create sbx sandbox: timed out after "
                    f"{self.config.create_timeout}s. "
                    f"stdout: {exc.stdout or ''} stderr: {exc.stderr or ''}"
                ) from exc

            self._sandbox_name = self.config.name or self._parse_sandbox_name(created.stdout)
            self._log_lifecycle(
                "sbx.create.ok",
                duration_ms=ms_since(create_start),
                stdout_chars=len(created.stdout or ""),
                stderr_chars=len(created.stderr or ""),
            )
            self._apply_network_policy()
            self._bootstrap_packages()
            self._setup_direct_workspace_aliases_in_sandbox()
        else:
            runner_path = self._prepared_runner_path or self._prepare_runner_script()

        assert self._sandbox_name is not None
        runner_root = self._staging_root
        runner_root.mkdir(parents=True, exist_ok=True)
        command = [
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
        return command

    def _direct_workspace_args(self) -> list[str]:
        seen = {str(self._staging_root)}
        args: list[str] = []
        for mount in self._direct_workspace_mounts:
            if mount.host_path in seen:
                continue
            seen.add(mount.host_path)
            args.append(mount.host_path)
        return args

    def _direct_workspace_aliases(self) -> list[tuple[str, str]]:
        return [
            (mount.host_path, mount.sandbox_path)
            for mount in self._direct_workspace_mounts
            if mount.host_path != mount.sandbox_path
        ]

    def _setup_direct_workspace_aliases_in_sandbox(self) -> None:
        aliases = self._direct_workspace_aliases()
        if not aliases:
            return
        assert self._sandbox_name is not None
        script = (
            "import json, os, pathlib, sys\n"
            "for source, target in json.loads(sys.argv[1]):\n"
            "    source_path = pathlib.Path(source)\n"
            "    target_path = pathlib.Path(target)\n"
            "    if target_path.exists() or target_path.is_symlink():\n"
            "        if target_path.is_symlink() and os.readlink(target_path) == str(source_path):\n"
            "            continue\n"
            "        raise FileExistsError(f'Direct workspace alias already exists: {target}')\n"
            "    target_path.parent.mkdir(parents=True, exist_ok=True)\n"
            "    target_path.symlink_to(source_path, target_is_directory=True)\n"
        )
        result = subprocess.run(
            [
                "sbx",
                "exec",
                "-w",
                str(self._staging_root),
                "-u",
                "root",
                self._sandbox_name,
                SBX_PYTHON_EXECUTABLE,
                "-c",
                script,
                json.dumps(aliases),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=self.config.exec_timeout,
        )
        if result.returncode != 0:
            raise SandboxFatalError(
                "Failed to configure direct workspace aliases: "
                f"stdout: {result.stdout.strip()}; stderr: {result.stderr.strip()}"
            )

    def _setup_direct_workspace_aliases_host_side(self) -> None:
        for source, target in self._direct_workspace_aliases():
            source_path = Path(source)
            target_path = Path(target)
            if target_path.exists() or target_path.is_symlink():
                if target_path.is_symlink() and os.readlink(target_path) == str(source_path):
                    continue
                raise FileExistsError(f"Direct workspace alias already exists: {target}")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.symlink_to(source_path, target_is_directory=True)
            self._owned_direct_aliases.append(target_path)

    def _cleanup_direct_workspace_aliases_host_side(self) -> None:
        for path in reversed(self._owned_direct_aliases):
            try:
                if path.is_symlink():
                    path.unlink()
            except OSError:
                pass
        self._owned_direct_aliases.clear()

    def _start_sbx_and_build_runner_command(self) -> list[str]:
        return self._start_sbx_and_build_supervisor_command()

    def _prepare_runner_script(self) -> Path:
        runner_dir = self._staging_root / ".predict_rlm_runner"
        runner_dir.mkdir(parents=True, exist_ok=True)
        runner_path = runner_dir / "python_runner.py"
        shutil.copy2(RUNNER_PATH, runner_path)
        self._prepared_runner_path = runner_path
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

    def _read_supervisor_stdout_line(
        self,
        process: PersistentSupervisorProcess,
        *,
        deadline: float,
        timeout: float,
    ) -> str | None:
        return self._read_stdout_line(deadline) or ""

    def _resolve_execution_timeout(self, timeout: float | None) -> tuple[float, str]:
        return resolve_execution_timeout(timeout, default_timeout=self.config.exec_timeout)

    def _host_watchdog_timeout(
        self,
        timeout_seconds: float,
        timeout_failure_class: str,
    ) -> float:
        return recoverable_timeout_host_deadline_seconds(
            timeout_seconds,
            timeout_failure_class,
        )

    def _fail_timed_out_request(
        self,
        timeout_seconds: float,
        host_timeout_seconds: float,
        timeout_failure_class: str,
    ) -> None:
        assert self._proc is not None
        self._log_lifecycle(
            "sbx.request.timeout",
            timeout_seconds=self.config.exec_timeout,
            status="error",
        )
        self._proc.kill()
        if timeout_failure_class == ITERATION_TIMEOUT_FAILURE_CLASS:
            raise SandboxFatalError(
                "Sbx supervisor failed to recover from iteration timeout after "
                f"{timeout_seconds:g}s; waited {host_timeout_seconds:g}s before "
                "force-killing supervisor"
            )
        raise SandboxFatalError(
            f"Sbx supervisor request timed out after {host_timeout_seconds:g}s"
        )

    def _get_supervisor_process(self) -> subprocess.Popen[str] | None:
        return self._proc

    def _request_timeout_seconds(
        self,
        method: str,
        params: dict[str, Any],
        timeout: float | None,
    ) -> float:
        timeout_seconds, timeout_failure_class = self._resolve_execution_timeout(timeout)
        return self._host_watchdog_timeout(timeout_seconds, timeout_failure_class)

    def _execution_timeout_metadata_from_params(
        self,
        params: dict[str, Any],
    ) -> tuple[float, str]:
        if "execution_timeout_seconds" in params:
            return float(params["execution_timeout_seconds"]), ITERATION_TIMEOUT_FAILURE_CLASS
        return self._resolve_execution_timeout(None)

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
        context = contextvars.copy_context()
        future = self._tool_executor.submit(
            context.run,
            self._build_tool_response,
            request,
        )
        self._pending_tool_calls[future] = request_id

    def _drain_completed_tool_calls(self) -> None:
        completed = [future for future in self._pending_tool_calls if future.done()]
        for future in completed:
            self._pending_tool_calls.pop(future)
            self._write_tool_response(future.result())

    def _drain_completed_supervisor_work(self) -> None:
        self._drain_completed_tool_calls()

    def _handle_supervisor_control_message(
        self,
        message: dict[str, Any],
        *,
        deadline: float,
    ) -> bool:
        if message.get("method") == "tool_call":
            self._submit_tool_call(message)
            return True
        if message.get("method") == "runtime_hook_event":
            self._handle_runtime_hook_event(message)
            return True
        return False

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
            result = to_plain_data(result)
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

    def _send_request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict:
        return self._send_json_rpc_request(method, params, timeout=timeout)

    def _on_supervisor_request_start(
        self,
        method: str,
        params: dict[str, Any],
        *,
        request_id: int,
        request_timeout: float,
    ) -> None:
        timeout_seconds, _ = self._execution_timeout_metadata_from_params(params)
        self._log_lifecycle(
            "sbx.request.start",
            method=method,
            request_id=request_id,
            timeout_seconds=timeout_seconds,
            request_timeout=request_timeout,
            pending_tool_count=len(self._pending_tool_calls),
            code_chars=len(str((params or {}).get("code", ""))) if method == "execute" else None,
        )

    def _on_supervisor_request_response(
        self,
        method: str,
        *,
        request_id: int,
        request_start: float,
        response: dict[str, Any],
    ) -> None:
        self._log_lifecycle(
            "sbx.request.ok",
            method=method,
            request_id=request_id,
            duration_ms=ms_since(request_start),
            status="error_response" if "error" in response else "ok",
            pending_tool_count=len(self._pending_tool_calls),
        )

    def _handle_supervisor_send_error(
        self,
        method: str,
        request_id: int,
        exc: BrokenPipeError,
    ) -> None:
        self._log_lifecycle(
            "sbx.request.broken_pipe",
            method=method,
            request_id=request_id,
            error_type=type(exc).__name__,
            status="error",
        )
        raise SandboxFatalError("Sbx supervisor pipe broke while sending request") from exc

    def _handle_supervisor_exit_during_request(
        self,
        method: str,
        *,
        request_id: int,
        request_start: float,
        process: PersistentSupervisorProcess,
    ) -> None:
        stderr = self._read_stderr_for_process(process)
        self._log_lifecycle(
            "sbx.runner.exited",
            method=method,
            request_id=request_id,
            duration_ms=ms_since(request_start),
            stderr_chars=len(stderr or ""),
            status="error",
        )
        raise SandboxFatalError(f"Sbx supervisor exited unexpectedly: {stderr}")

    def _handle_supervisor_request_timeout(
        self,
        method: str,
        params: dict[str, Any],
        process: PersistentSupervisorProcess,
        *,
        request_id: int,
        request_timeout: float,
        request_start: float,
        stdout_tail: str,
    ) -> dict[str, Any]:
        timeout_seconds, timeout_failure_class = self._execution_timeout_metadata_from_params(
            params
        )
        self._log_lifecycle(
            "sbx.request.timeout",
            method=method,
            request_id=request_id,
            duration_ms=ms_since(request_start),
            timeout_seconds=timeout_seconds,
            request_timeout=request_timeout,
            pending_tool_count=len(self._pending_tool_calls),
            failure_class=timeout_failure_class,
            status="error",
        )
        self._fail_timed_out_request(
            timeout_seconds,
            request_timeout,
            timeout_failure_class,
        )

    def _handle_stale_response_limit(
        self,
        method: str,
        *,
        request_id: int,
        request_start: float,
    ) -> None:
        self._log_lifecycle(
            "sbx.protocol.stale_response_limit",
            method=method,
            request_id=request_id,
            duration_ms=ms_since(request_start),
            status="error",
        )
        raise CodeInterpreterError(
            "Too many stale top-level responses while resyncing "
            f"SBX request id={request_id}"
        )

    def _handle_runtime_hook_event(self, request: dict[str, Any]) -> None:
        if self.on_runtime_hook_event is None:
            return
        try:
            event = RuntimeHookEvent.model_validate(request.get("params") or {})
            self.on_runtime_hook_event(event)
        except Exception:
            return

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
        if isinstance(result, dict) and "timeout" in result:
            return format_recoverable_timeout_result(result)
        if interpreter_result_logging_enabled(self.verbose):
            emit_trace_result(result)
        if "final" in result:
            return FinalOutput(result["final"])
        return result.get("output")

    def _ensure_process_for_request(self, method: str) -> None:
        self._ensure_process_for_method(method)

    def _discard_supervisor_process(self) -> None:
        self._proc = None
        self._stdout_lines = queue.Queue()
        self._stdout_reader = None
        self._pending_tool_calls.clear()

    def _read_stderr_for_process(self, process: PersistentSupervisorProcess) -> str:
        stderr = process.stderr
        if stderr is None:
            return ""
        try:
            return stderr.read() or ""
        except Exception:
            return ""

    def _format_supervisor_restart_diagnostic(
        self,
        returncode: int | None,
        context: dict[str, Any],
        *,
        stderr: str,
    ) -> str:
        diagnostic = (
            "Sbx supervisor exited after the previous execute response. "
            "The supervisor process was restarted; Python globals from the "
            "prior supervisor were lost, while sandbox filesystem state is "
            "preserved. Re-run setup code before relying on in-memory variables."
            "\n"
            f"[supervisor lifecycle] {self._format_supervisor_exit_evidence(returncode, context)}"
        )
        if stderr:
            diagnostic = f"{diagnostic}\n[supervisor stderr before restart]\n{stderr.rstrip()}"
        return diagnostic

    def _raise_execute_error(self, response: dict[str, Any]) -> None:
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
