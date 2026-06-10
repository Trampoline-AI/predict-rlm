from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import inspect
import json
import math
import os
import queue
import re
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Protocol

from dspy.primitives.code_interpreter import CodeInterpreterError

from predict_rlm._shared import strip_code_fences
from predict_rlm.debug import debug_event
from predict_rlm.debug import is_enabled as predict_rlm_debug_enabled
from predict_rlm.execution_timeout import (
    DEFAULT_RECOVERABLE_EXECUTION_TIMEOUT_GRACE_SECONDS,
    ITERATION_TIMEOUT_FAILURE_CLASS,
    recoverable_timeout_host_deadline_seconds,
    validate_execution_timeout,
)
from predict_rlm.interpreter import SandboxFatalError
from predict_rlm.interpreters.base import ClientAdapterExecutionGate, PredictRLMClientAdapter
from predict_rlm.interpreters.persistent_runner import (
    PersistentJsonRpcRunnerClient,
    PersistentSupervisorProcess,
)

PYTHON_RUNNER_RECOVERABLE_TIMEOUT_GRACE_SECONDS = (
    DEFAULT_RECOVERABLE_EXECUTION_TIMEOUT_GRACE_SECONDS
)
HOST_TOOL_TIMEOUT_RESPONSE_MARGIN_SECONDS = 0.05
_CODE_PREVIEW_CHARS = 500
_SECRETISH_CODE_RE = re.compile(
    r"(?i)\b(api[_-]?key|authorization|bearer|credential|password|secret|token)\b"
)

__all__ = [
    "DirectProcessRunnerClientAdapter",
    "PYTHON_RUNNER_RECOVERABLE_TIMEOUT_GRACE_SECONDS",
    "PythonRunnerClientAdapter",
]


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _code_preview(code: str) -> str:
    preview = code[:_CODE_PREVIEW_CHARS]
    lines = []
    for line in preview.splitlines():
        lines.append(
            "[REDACTED secret-like code line]" if _SECRETISH_CODE_RE.search(line) else line
        )
    return "\n".join(lines)


def _process_debug_metadata(process: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    pid = getattr(process, "pid", None)
    if pid is not None:
        metadata["runner_pid"] = pid
    returncode = getattr(process, "returncode", None)
    if returncode is not None:
        metadata["runner_returncode"] = returncode
    for name in ("session_id", "command_id"):
        value = getattr(process, name, None)
        if value is not None:
            metadata[f"runner_{name}"] = value
    return metadata


def _runner_source() -> str:
    spec = importlib.util.find_spec("predict_rlm.sandbox.python_runner")
    if spec is None or spec.origin is None:
        raise RuntimeError("Could not locate predict_rlm.sandbox.python_runner")
    return Path(spec.origin).read_text(encoding="utf-8")


class _RunnerError:
    def __init__(self, *, type: str, message: str, args: list[Any] | None = None) -> None:
        self.type = type
        self.message = message
        self.args = args

    @classmethod
    def from_exception(cls, exc: BaseException) -> "_RunnerError":
        return cls(
            type=type(exc).__name__,
            message=str(exc),
            args=list(getattr(exc, "args", ())),
        )

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "_RunnerError":
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        return cls(
            type=str(data.get("type") or payload.get("type") or "RuntimeError"),
            message=str(payload.get("message") or ""),
            args=data.get("args") or payload.get("args"),
        )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": self.type, "message": self.message}
        if self.args is not None:
            payload["args"] = self.args
        return payload


class RunnerProcess(Protocol):
    stdin: Any
    stdout: Any
    stderr: Any

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def kill(self) -> None: ...


class RunnerBackend(Protocol):
    def copy_to(self, host_path: str, container_path: str) -> None: ...

    def copy_from(self, container_path: str, host_path: str) -> None: ...

    def exec(self, command: list[str], *, timeout: float | None = None) -> Any: ...

    def start_exec(
        self,
        command: list[str],
        *,
        workdir: str | None = None,
        timeout: float | None = None,
    ) -> RunnerProcess: ...


class _HostToolTimeoutError(TimeoutError):
    pass


class _TimeoutLineReader:
    def __init__(self, process: RunnerProcess) -> None:
        self.process = process
        self.pipe = process.stdout
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._closed = False

    def close(self) -> None:
        self._closed = True
        close_pipe = getattr(self.pipe, "close", None)
        if close_pipe is not None:
            close_pipe()
        if self._thread is not None:
            self._thread.join(timeout=1)

    def readline(self, timeout: float) -> str | None:
        timeout = max(0.0, timeout)
        read_with_timeout = getattr(self.pipe, "readline_timeout", None)
        if read_with_timeout is not None:
            return read_with_timeout(timeout)
        fd = self._fileno()
        if fd is not None:
            try:
                ready, _, _ = select.select([fd], [], [], timeout)
            except (OSError, ValueError):
                fd = None
            else:
                if ready:
                    return self.pipe.readline()
                return None
        self._ensure_thread()
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _fileno(self) -> int | None:
        fileno = getattr(self.pipe, "fileno", None)
        if fileno is None:
            return None
        try:
            fd = fileno()
        except (OSError, ValueError, AttributeError):
            return None
        return fd if isinstance(fd, int) and fd >= 0 else None

    def _ensure_thread(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._read_loop,
            name="python-runner-supervisor-stdout",
            daemon=True,
        )
        self._thread.start()

    def _read_loop(self) -> None:
        while not self._closed:
            line = self.pipe.readline()
            if line:
                self._queue.put(line)
                continue
            if self._closed or self.process.poll() is not None:
                self._queue.put("")
                return
            time.sleep(0.01)


class _DirectProcessRunnerAdapter:
    """Launches a runner process directly on the current machine."""

    supports_file_sync = True

    def __init__(self, *, workdir: str | None = None) -> None:
        self.workdir = workdir
        root = Path(workdir or os.getcwd()) / ".predict_rlm_runner_env"
        self.runner_root = root.resolve()
        self.sandbox_root = self.runner_root / "sandbox"

    def _cwd(self, workdir: str | None = None) -> str | None:
        return workdir or self.workdir

    def _path_for_runtime_path(self, path: str) -> Path:
        if path == "/sandbox" or path.startswith("/sandbox/"):
            rel = path.removeprefix("/sandbox").lstrip("/")
            return self.sandbox_root / rel
        return Path(path)

    def _map_command_arg(self, arg: str) -> str:
        if arg == "/sandbox" or arg.startswith("/sandbox/"):
            return str(self._path_for_runtime_path(arg))
        return arg

    def virtual_path_for_host_path(self, path: str) -> str:
        candidate = Path(path)
        try:
            rel = candidate.resolve().relative_to(self.sandbox_root)
        except ValueError:
            return path
        return "/sandbox/" + rel.as_posix()

    def copy_to(self, host_path: str, container_path: str) -> None:
        source = Path(host_path)
        destination = self._path_for_runtime_path(container_path)
        if source.resolve() == destination.resolve():
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    def copy_from(self, container_path: str, host_path: str) -> None:
        source = self._path_for_runtime_path(container_path)
        destination = Path(host_path)
        if source.resolve() == destination.resolve():
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    def exec(self, command: list[str], *, timeout: float | None = None) -> Any:
        return subprocess.run(
            [self._map_command_arg(arg) for arg in command],
            cwd=self._cwd(),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )

    def install_runner_script(
        self,
        source: str,
        runner_path: str,
        *,
        timeout: float | None = None,
    ) -> None:
        del timeout
        path = Path(runner_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")

    def start_exec(
        self,
        command: list[str],
        *,
        workdir: str | None = None,
        timeout: float | None = None,
    ) -> RunnerProcess:
        del timeout
        env = os.environ.copy()
        env["PREDICT_RLM_RUNNER_ROOT"] = str(self.runner_root)
        env["PREDICT_RLM_SBX_ROOT"] = str(self.runner_root)
        return subprocess.Popen(
            command,
            cwd=self._cwd(workdir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )


class PythonRunnerClientAdapter(PersistentJsonRpcRunnerClient, PredictRLMClientAdapter):
    _LIST_DIR_SCRIPT = (
        "import json, pathlib, sys; "
        "root = pathlib.Path(sys.argv[1]); "
        "print(json.dumps([str(p) for p in sorted(root.rglob('*')) if p.is_file()]))"
    )

    def __init__(
        self,
        backend: RunnerBackend,
        *,
        tools: dict[str, Callable[..., Any]] | None = None,
        output_fields: list[dict[str, Any]] | None = None,
        runner_path: str = "/tmp/predict_rlm_runner.py",
        python_executable: str = "python3",
        workdir: str | None = None,
        exec_timeout: float = 900.0,
        recoverable_timeout_grace: float = PYTHON_RUNNER_RECOVERABLE_TIMEOUT_GRACE_SECONDS,
        supervisor_name: str = "Python runner supervisor",
        debug_interpreter: str = "python_runner",
        debug_event_prefix: str = "python_runner.runner",
        restart_process_description: str = "supervisor process",
        filesystem_state_description: str = "runner filesystem state",
    ) -> None:
        PersistentJsonRpcRunnerClient.__init__(
            self,
            supervisor_name=supervisor_name,
        )
        self.adapter = backend
        self.tools = tools or {}
        self.output_fields = output_fields or []
        self.runner_path = runner_path
        self.python_executable = python_executable
        self.workdir = workdir
        self.exec_timeout = exec_timeout
        self._supervisor_name = supervisor_name
        self._debug_interpreter = debug_interpreter
        self._debug_event_prefix = debug_event_prefix
        self._restart_process_description = restart_process_description
        self._filesystem_state_description = filesystem_state_description
        self.recoverable_timeout_grace = self._resolve_recoverable_timeout_grace(
            recoverable_timeout_grace
        )
        self._process: RunnerProcess | None = None
        self._shutdown = False
        self._tools_registered = False
        self._output_fields_registered = False
        self._stdout_reader: _TimeoutLineReader | None = None
        self._execution_gate = ClientAdapterExecutionGate(supervisor_name)
        self._debug_request_context: dict[int, dict[str, Any]] = {}
        self._defer_next_submit_finalization = False

    def defer_next_submit_finalization(self) -> None:
        self._defer_next_submit_finalization = True

    def execute(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        with self._execution_gate.top_level():
            return self._execute_top_level(code, variables, timeout=timeout)

    async def aexecute(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        try:
            return await asyncio.to_thread(self.execute, code, variables, timeout=timeout)
        except asyncio.CancelledError:
            self._abort_supervisor_after_cancellation()
            raise

    def mount_file_at(self, host_path: str, virtual_path: str) -> None:
        self.adapter.copy_to(host_path, virtual_path)

    def mkdir_p(self, virtual_path: str) -> None:
        result = self.adapter.exec(["mkdir", "-p", virtual_path], timeout=self.exec_timeout)
        self._raise_for_exec_failure(result, f"creating {virtual_path}")

    def list_dir(self, virtual_path: str) -> list[str]:
        result = self.adapter.exec(
            [self.python_executable, "-c", self._LIST_DIR_SCRIPT, virtual_path],
            timeout=self.exec_timeout,
        )
        self._raise_for_exec_failure(result, f"listing {virtual_path}")
        stdout = getattr(result, "stdout", result if isinstance(result, str) else "")
        paths = list(json.loads(stdout or "[]"))
        virtualize = getattr(self.adapter, "virtual_path_for_host_path", None)
        if virtualize is not None:
            return [virtualize(path) for path in paths]
        return paths

    def sync_file_to(self, virtual_path: str, host_path: str) -> None:
        if getattr(self.adapter, "supports_file_sync", True) is False:
            self.adapter.copy_from(virtual_path, host_path)
            return
        Path(host_path).parent.mkdir(parents=True, exist_ok=True)
        self.adapter.copy_from(virtual_path, host_path)

    def reset(self) -> None:
        self._send_request("reset", {})

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        process = self._process
        if process is not None and process.poll() is None:
            try:
                self._send_request("shutdown", {"preserve_kernel_process": True})
            except Exception:
                pass
            try:
                process.wait(timeout=5)
            except Exception:
                process.kill()
        if self._stdout_reader is not None:
            self._stdout_reader.close()
        self._process = None
        self._stdout_reader = None

    def _abort_supervisor_after_cancellation(self) -> None:
        process = self._process
        if process is not None:
            self._debug_event(
                "cancel_abort",
                **_process_debug_metadata(process),
            )
            try:
                process.kill()
            except Exception:
                pass
            try:
                process.wait(timeout=1)
            except Exception:
                pass
        if self._stdout_reader is not None:
            self._stdout_reader.close()
        self._process = None
        self._stdout_reader = None

    def configure_runtime(
        self,
        *,
        tools: dict[str, Callable[..., Any]] | None = None,
        output_fields: list[dict[str, Any]] | None = None,
    ) -> None:
        if tools is not None and tools is not self.tools:
            self.tools = tools
            self._tools_registered = False
        if output_fields is not None:
            self.output_fields = output_fields
            self._output_fields_registered = False
        if self._process is not None:
            self._register_runtime()

    def _execute_top_level(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        code = strip_code_fences(code)
        if variables:
            assignments = "\n".join(f"{name} = {value!r}" for name, value in variables.items())
            code = f"{assignments}\n{code}"
        params: dict[str, Any] = {"code": code}
        if self._defer_next_submit_finalization:
            params["defer_final_output"] = True
            self._defer_next_submit_finalization = False
        host_timeout = self.exec_timeout
        if timeout is not None:
            execution_timeout = self._resolve_execution_timeout(timeout)
            params["execution_timeout_seconds"] = execution_timeout
            host_timeout = recoverable_timeout_host_deadline_seconds(
                execution_timeout,
                ITERATION_TIMEOUT_FAILURE_CLASS,
                grace_seconds=self.recoverable_timeout_grace,
            )
        response = self._send_request("execute", params, timeout=host_timeout)
        return self._unwrap_execute_response(response)

    def _resolve_recoverable_timeout_grace(self, grace: float) -> float:
        if (
            isinstance(grace, bool)
            or not isinstance(grace, (int, float))
            or not math.isfinite(float(grace))
            or float(grace) < 0
        ):
            raise ValueError("recoverable timeout grace must be a non-negative number")
        return float(grace)

    def _resolve_execution_timeout(self, timeout: float) -> float:
        execution_timeout = validate_execution_timeout(timeout)
        assert execution_timeout is not None
        return execution_timeout

    def _ensure_process(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._register_runtime()
            return
        if self._process is not None and self._process.poll() is not None:
            stderr = self._read_stderr()
            self._debug_event(
                "dead_before_request",
                stderr_len=len(stderr),
                **_process_debug_metadata(self._process),
            )
            raise SandboxFatalError(f"{self._supervisor_name} exited unexpectedly: {stderr}")
        self._copy_runner_script()
        self._process = self.adapter.start_exec(
            [self.python_executable, "-u", self.runner_path],
            workdir=self.workdir,
            timeout=self.exec_timeout,
        )
        self._stdout_reader = None
        self._debug_event(
            "start",
            runner_path=self.runner_path,
            python_executable=self.python_executable,
            workdir=self.workdir,
            **_process_debug_metadata(self._process),
        )
        self._register_runtime()

    def _copy_runner_script(self) -> None:
        install_runner_script = getattr(self.adapter, "install_runner_script", None)
        if install_runner_script is not None:
            install_runner_script(
                _runner_source(),
                self.runner_path,
                timeout=self.exec_timeout,
            )
            return
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".py", delete=False
        ) as tmp:
            tmp.write(_runner_source())
            tmp_path = tmp.name
        try:
            self.adapter.copy_to(tmp_path, self.runner_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _register_runtime(self) -> None:
        if self.output_fields and not self._output_fields_registered:
            self._send_request_without_ensure(
                "register_output_fields",
                {"fields": self.output_fields},
            )
            self._output_fields_registered = True
        if self.tools and not self._tools_registered:
            self._send_request_without_ensure("register_tools", {"tools": list(self.tools)})
            self._tools_registered = True

    def _send_request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        return self._send_json_rpc_request(method, params, timeout=timeout)

    def _send_request_without_ensure(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        return self._send_json_rpc_request_without_ensure(
            method,
            params,
            timeout=timeout,
        )

    def _get_supervisor_process(self) -> RunnerProcess | None:
        return self._process

    def _ensure_process_for_request(self, method: str) -> None:
        if method == "shutdown" and self._process is not None:
            return
        self._ensure_process()

    def _request_timeout_seconds(
        self,
        method: str,
        params: dict[str, Any],
        timeout: float | None,
    ) -> float:
        return self.exec_timeout if timeout is None else timeout

    def _debug_event(self, event: str, **metadata: Any) -> None:
        debug_event(
            f"{self._debug_event_prefix}.{event}",
            interpreter=self._debug_interpreter,
            **metadata,
        )

    def _request_debug_context(
        self,
        method: str,
        params: dict[str, Any],
        *,
        request_id: int,
        request_timeout: float,
    ) -> dict[str, Any]:
        context: dict[str, Any] = {
            "request_id": request_id,
            "method": method,
            "host_timeout_seconds": request_timeout,
            "recoverable_timeout_grace_seconds": self.recoverable_timeout_grace,
        }
        process = self._get_supervisor_process()
        if process is not None:
            context.update(_process_debug_metadata(process))
        code = params.get("code")
        if isinstance(code, str):
            context.update(
                {
                    "code_len": len(code),
                    "code_hash": _hash_text(code),
                    "code_nonempty": bool(code.strip()),
                }
            )
            if "execution_timeout_seconds" in params:
                context["execution_timeout_seconds"] = params.get("execution_timeout_seconds")
        return context

    def _on_supervisor_request_start(
        self,
        method: str,
        params: dict[str, Any],
        *,
        request_id: int,
        request_timeout: float,
    ) -> None:
        if not predict_rlm_debug_enabled():
            return
        context = self._request_debug_context(
            method,
            params,
            request_id=request_id,
            request_timeout=request_timeout,
        )
        self._debug_request_context[request_id] = context
        self._debug_event("request", **context)

    def _on_supervisor_request_response(
        self,
        method: str,
        *,
        request_id: int,
        request_start: float,
        response: dict[str, Any],
    ) -> None:
        if not predict_rlm_debug_enabled():
            return
        context = self._debug_request_context.pop(request_id, {})
        context.update(
            {
                "request_id": request_id,
                "method": method,
                "elapsed_seconds": time.perf_counter() - request_start,
                "response_id": response.get("id"),
                "response_kind": self._response_debug_kind(response),
            }
        )
        context.update(self._response_debug_lengths(response))
        self._debug_event("response", **context)

    def _record_supervisor_response(
        self,
        method: str,
        params: dict[str, Any],
        *,
        request_id: int,
        request_timeout: float,
        response: dict[str, Any],
    ) -> None:
        super()._record_supervisor_response(
            method,
            params,
            request_id=request_id,
            request_timeout=request_timeout,
            response=response,
        )
        if not predict_rlm_debug_enabled() or method != "execute":
            return
        result = response.get("result") if isinstance(response, dict) else None
        output = result.get("output") if isinstance(result, dict) else None
        code = params.get("code")
        if isinstance(code, str) and code.strip() and output == "":
            self._debug_event(
                "empty_execute_output",
                request_id=request_id,
                code_len=len(code),
                code_hash=_hash_text(code),
                code_preview=_code_preview(code),
                response_id=response.get("id"),
                response_kind=self._response_debug_kind(response),
                **self._response_debug_lengths(response),
            )

    def _response_debug_kind(self, response: dict[str, Any]) -> str:
        if "error" in response:
            error = response.get("error") or {}
            data = error.get("data") if isinstance(error, dict) else {}
            if isinstance(data, dict) and data.get("type"):
                return f"error:{data.get('type')}"
            return "error"
        result = response.get("result") or {}
        if isinstance(result, dict) and "timeout" in result:
            return "timeout"
        if isinstance(result, dict) and "final" in result:
            return "final"
        if isinstance(result, dict) and "submitted" in result:
            return "submitted"
        return "output"

    def _response_debug_lengths(self, response: dict[str, Any]) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        if "error" in response:
            error = response.get("error") or {}
            if isinstance(error, dict):
                metadata["error_type"] = (
                    (error.get("data") or {}).get("type")
                    if isinstance(error.get("data"), dict)
                    else None
                )
                metadata["error_message"] = error.get("message")
            return metadata
        result = response.get("result") or {}
        if not isinstance(result, dict):
            return metadata
        for key in ("output", "stdout", "stderr"):
            value = result.get(key)
            if isinstance(value, str):
                metadata[f"{key}_len"] = len(value)
        return metadata

    def _handle_supervisor_send_error(
        self,
        method: str,
        request_id: int,
        exc: BrokenPipeError,
    ) -> None:
        self._debug_event(
            "send_error",
            method=method,
            request_id=request_id,
            error_type=type(exc).__name__,
        )
        raise SandboxFatalError(f"{self._supervisor_name} pipe broke") from exc

    def _handle_supervisor_exit_during_request(
        self,
        method: str,
        *,
        request_id: int,
        request_start: float,
        process: PersistentSupervisorProcess,
    ) -> None:
        stderr = self._read_stderr_for_process(process)
        self._debug_request_context.pop(request_id, None)
        self._debug_event(
            "exit_during_request",
            method=method,
            request_id=request_id,
            elapsed_seconds=time.perf_counter() - request_start,
            stderr_len=len(stderr),
            **_process_debug_metadata(process),
        )
        raise SandboxFatalError(f"{self._supervisor_name} exited unexpectedly: {stderr}")

    def _on_supervisor_stale_response(
        self,
        method: str,
        *,
        expected_request_id: int,
        stale_response: dict[str, Any],
        stale_discards: int,
    ) -> None:
        self._debug_event(
            "stale_response",
            method=method,
            expected_request_id=expected_request_id,
            stale_response_id=stale_response.get("id"),
            stale_discards=stale_discards,
            response_kind=self._response_debug_kind(stale_response),
            **self._response_debug_lengths(stale_response),
        )

    def _recoverable_execution_timeout_seconds(
        self,
        method: str,
        params: dict[str, Any],
    ) -> float | None:
        if method != "execute" or "execution_timeout_seconds" not in params:
            return None
        return self._resolve_execution_timeout(params["execution_timeout_seconds"])

    def _handle_supervisor_request_timeout(
        self,
        method: str,
        params: dict[str, Any],
        process: PersistentSupervisorProcess,
        *,
        request_id: int,
        request_timeout: float,
        request_start: float,
        stdout_tail: str = "",
    ) -> dict[str, Any]:
        recoverable_timeout_seconds = self._recoverable_execution_timeout_seconds(
            method,
            params,
        )
        self._kill_process_after_timeout(process)
        stderr = self._read_stderr_for_process(process)
        context = self._debug_request_context.pop(request_id, {})
        context_metadata = {
            key: value
            for key, value in context.items()
            if key
            not in {
                "method",
                "request_id",
                "host_timeout_seconds",
                "runner_pid",
                "runner_returncode",
                "runner_session_id",
                "runner_command_id",
            }
        }
        self._debug_event(
            "request_timeout",
            method=method,
            request_id=request_id,
            elapsed_seconds=time.perf_counter() - request_start,
            host_timeout_seconds=request_timeout,
            recoverable_execution_timeout_seconds=recoverable_timeout_seconds,
            stdout_tail_len=len(stdout_tail),
            stderr_len=len(stderr),
            **context_metadata,
            **_process_debug_metadata(process),
        )
        if recoverable_timeout_seconds is None:
            raise SandboxFatalError(
                f"{self._supervisor_name} request timed out after {request_timeout:g}s"
            )

        self._discard_supervisor_process()
        restart_error: BaseException | None = None
        try:
            self._ensure_process()
        except BaseException as exc:
            restart_error = exc
        if restart_error is not None:
            raise SandboxFatalError(
                f"{self._supervisor_name} request timed out after "
                f"{request_timeout:g}s and the {self._restart_process_description} "
                "could not be restarted: "
                f"{restart_error}"
            ) from restart_error

        diagnostic = (
            f"{self._supervisor_name} request timed out after "
            f"{request_timeout:g}s before it returned a structured timeout. "
            f"The {self._restart_process_description} was killed and restarted; "
            "Python globals from the timed-out supervisor were lost, while "
            f"{self._filesystem_state_description} is preserved. "
            "Re-run setup code before relying on in-memory "
            "variables."
        )
        if stderr:
            diagnostic = f"{diagnostic}\n[supervisor stderr before restart]\n{stderr.rstrip()}"
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "timeout": {"seconds": recoverable_timeout_seconds},
                "stdout": stdout_tail,
                "stderr": diagnostic,
            },
        }

    def _read_supervisor_stdout_line(
        self,
        process: PersistentSupervisorProcess,
        *,
        deadline: float,
        timeout: float,
    ) -> str | None:
        if self._stdout_reader is None or self._stdout_reader.process is not process:
            self._stdout_reader = _TimeoutLineReader(process)
        return self._stdout_reader.readline(timeout)

    def _handle_supervisor_control_message(
        self,
        message: dict[str, Any],
        *,
        deadline: float,
    ) -> bool:
        if message.get("method") != "tool_call":
            return False
        self._write_tool_response(self._build_tool_response(message, deadline=deadline))
        return True

    def _kill_process_after_timeout(self, process: RunnerProcess) -> None:
        self._debug_event(
            "kill_after_timeout",
            **_process_debug_metadata(process),
        )
        process.kill()
        try:
            process.wait(timeout=1)
        except Exception:
            pass

    def _discard_supervisor_process(self) -> None:
        process = self._process
        if process is not None:
            self._debug_event(
                "discard",
                **_process_debug_metadata(process),
            )
        if self._stdout_reader is not None:
            self._stdout_reader.close()
        self._process = None
        self._stdout_reader = None
        self._tools_registered = False
        self._output_fields_registered = False

    def _format_supervisor_restart_diagnostic(
        self,
        returncode: int | None,
        context: dict[str, Any],
        *,
        stderr: str,
    ) -> str:
        diagnostic = (
            f"{self._supervisor_name} exited after the previous execute response. "
            f"The {self._restart_process_description} was restarted; Python globals "
            "from the prior supervisor were lost, while "
            f"{self._filesystem_state_description} is preserved. "
            "Re-run setup code before relying on in-memory variables."
            "\n"
            f"[supervisor lifecycle] {self._format_supervisor_exit_evidence(returncode, context)}"
        )
        if stderr:
            diagnostic = f"{diagnostic}\n[supervisor stderr before restart]\n{stderr.rstrip()}"
        return diagnostic

    def _build_tool_response(
        self,
        message: dict[str, Any],
        *,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        request_id = message.get("id")
        params = message.get("params") or {}
        name = params.get("name")
        try:
            if name not in self.tools:
                raise CodeInterpreterError(f"Unknown tool: {name}")
            tool = self.tools[name]
            args = list(params.get("args") or [])
            kwargs = dict(params.get("kwargs") or {})
            with self._execution_gate.tool_callback():
                result = self._run_host_tool_with_deadline(
                    tool,
                    args,
                    kwargs,
                    deadline=deadline,
                )
            is_json = result is None or isinstance(result, (dict, list, int, float, bool))
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "type": "json" if is_json else "string",
                    "value": json.dumps(result) if is_json else str(result or ""),
                },
            }
        except BaseException as exc:
            error = _RunnerError.from_exception(exc)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32000,
                    "message": error.message,
                    "data": error.to_payload(),
                },
            }

    def _run_host_tool_with_deadline(
        self,
        tool: Callable[..., Any],
        args: list[Any],
        kwargs: dict[str, Any],
        *,
        deadline: float | None,
    ) -> Any:
        if deadline is None:
            result = tool(*args, **kwargs)
            if inspect.isawaitable(result):
                result = asyncio.run(result)
            return result

        remaining = deadline - time.monotonic()
        callback_budget = remaining - HOST_TOOL_TIMEOUT_RESPONSE_MARGIN_SECONDS
        if callback_budget <= 0:
            raise _HostToolTimeoutError(
                "Host tool call timed out before dispatch because the "
                f"{self._supervisor_name} request deadline was exhausted"
            )

        result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

        def run() -> None:
            try:
                result = tool(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = asyncio.run(result)
            except BaseException as exc:
                result_queue.put(("error", exc))
            else:
                result_queue.put(("result", result))

        thread = threading.Thread(
            target=run,
            name="python-runner-host-tool",
            daemon=True,
        )
        thread.start()
        try:
            kind, value = result_queue.get(timeout=max(0.0, callback_budget))
        except queue.Empty as exc:
            raise _HostToolTimeoutError(
                "Host tool call timed out after exhausting the remaining "
                f"{self._supervisor_name} request budget "
                f"({callback_budget:g}s plus a "
                f"{HOST_TOOL_TIMEOUT_RESPONSE_MARGIN_SECONDS:g}s response margin). "
                "The tool runs in a single-use daemon thread, so Python cannot "
                "forcibly stop underlying provider or library work that ignores "
                "cancellation."
            ) from exc
        if kind == "error":
            raise value
        return value

    def _write_tool_response(self, response: dict[str, Any]) -> None:
        process = self._require_process()
        process.stdin.write(json.dumps(response, default=str, separators=(",", ":")) + "\n")
        process.stdin.flush()

    def _raise_execute_error(self, response: dict[str, Any]) -> None:
        error = _RunnerError.from_payload(response.get("error") or {})
        if error.type == "SyntaxError":
            raise SyntaxError(error.message)
        raise CodeInterpreterError(f"{error.type}: {error.message}")

    def _require_process(self) -> RunnerProcess:
        return self._require_supervisor_process()

    def _read_stderr(self) -> str:
        if self._process is None or self._process.stderr is None:
            return ""
        return self._read_stderr_for_process(self._process)

    def _read_stderr_for_process(self, process: RunnerProcess) -> str:
        if process.stderr is None:
            return ""
        try:
            stderr = process.stderr.read()
        except Exception:
            return ""
        if stderr:
            self._debug_event(
                "stderr_read",
                stderr_len=len(stderr),
                **_process_debug_metadata(process),
            )
        return stderr

    def _raise_for_exec_failure(self, result: Any, operation: str) -> None:
        returncode = getattr(result, "returncode", getattr(result, "return_code", 0))
        if returncode not in (0, None):
            stderr = getattr(result, "stderr", "")
            stdout = getattr(result, "stdout", "")
            raise SandboxFatalError(
                f"{self._supervisor_name} backend failed while {operation}: "
                f"exit code {returncode}; stdout: {stdout}; stderr: {stderr}"
            )


class DirectProcessRunnerClientAdapter(PythonRunnerClientAdapter):
    """PredictRLM interpreter that runs the persistent Python supervisor locally."""

    def __init__(
        self,
        *,
        tools: dict[str, Callable[..., Any]] | None = None,
        output_fields: list[dict[str, Any]] | None = None,
        runner_path: str = "/tmp/predict_rlm_runner.py",
        python_executable: str | None = None,
        workdir: str | None = None,
        exec_timeout: float = 900.0,
        recoverable_timeout_grace: float = PYTHON_RUNNER_RECOVERABLE_TIMEOUT_GRACE_SECONDS,
    ) -> None:
        super().__init__(
            _DirectProcessRunnerAdapter(workdir=workdir),
            tools=tools,
            output_fields=output_fields,
            runner_path=runner_path,
            python_executable=python_executable or sys.executable,
            workdir=workdir,
            exec_timeout=exec_timeout,
            recoverable_timeout_grace=recoverable_timeout_grace,
        )
