from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import os
import queue
import re
import select
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Protocol
from uuid import uuid4

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
from predict_rlm.interpreters.base import InterpreterExecutionGate, PredictRLMInterpreter
from predict_rlm.interpreters.persistent_runner import (
    PersistentJsonRpcRunnerClient,
    PersistentSupervisorProcess,
)

from .protocol import RunnerError
from .runner import runner_source

TERMINAL_BENCH_RECOVERABLE_TIMEOUT_GRACE_SECONDS = (
    DEFAULT_RECOVERABLE_EXECUTION_TIMEOUT_GRACE_SECONDS
)
HOST_TOOL_TIMEOUT_RESPONSE_MARGIN_SECONDS = 0.05
_CODE_PREVIEW_CHARS = 500
_SECRETISH_CODE_RE = re.compile(
    r"(?i)\b(api[_-]?key|authorization|bearer|credential|password|secret|token)\b"
)


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


def _shell_python_command(python_executable: str, args: list[str]) -> str:
    quoted_args = " ".join(shlex.quote(arg) for arg in args)
    if python_executable != "python3":
        return " ".join(part for part in (shlex.quote(python_executable), quoted_args) if part)
    resolver = (
        "if command -v python3 >/dev/null 2>&1; then _predict_rlm_python=python3; "
        "elif command -v python >/dev/null 2>&1; then _predict_rlm_python=python; "
        "else echo 'PredictRLM supervisor requires python3 or python on PATH' >&2; "
        "exit 127; fi; "
    )
    return f'{resolver}"$_predict_rlm_python" {quoted_args}'.rstrip()


def _python_bootstrap_command() -> str:
    return " ".join(
        [
            "if command -v python3 >/dev/null 2>&1 || command -v python >/dev/null 2>&1; then exit 0; fi;",
            "export DEBIAN_FRONTEND=noninteractive;",
            "if command -v apt-get >/dev/null 2>&1; then apt-get update && apt-get install -y python3;",
            "elif command -v apk >/dev/null 2>&1; then apk add --no-cache python3;",
            "elif command -v microdnf >/dev/null 2>&1; then microdnf install -y python3;",
            "elif command -v dnf >/dev/null 2>&1; then dnf install -y python3;",
            "elif command -v yum >/dev/null 2>&1; then yum install -y python3;",
            "else echo 'PredictRLM supervisor requires python3 or python on PATH and no supported package manager was found' >&2; exit 127; fi;",
            "command -v python3 >/dev/null 2>&1 || command -v python >/dev/null 2>&1",
        ]
    )


def _docker_compose_project_name(name: str) -> str:
    name = name.lower()
    if not re.match(r"^[a-z0-9]", name):
        name = "0" + name
    return re.sub(r"[^a-z0-9_-]", "-", name)


def _run_coroutine_on_loop(coro: Any, loop: asyncio.AbstractEventLoop) -> Any:
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None
    if running_loop is loop:
        close = getattr(coro, "close", None)
        if close is not None:
            close()
        raise SandboxFatalError(
            "HarborEnvironmentInterpreter sync methods must run outside the Harbor event loop"
        )
    return asyncio.run_coroutine_threadsafe(coro, loop).result()


def _resolve_maybe_awaitable(value: Any, loop: asyncio.AbstractEventLoop) -> Any:
    if inspect.isawaitable(value):
        return _run_coroutine_on_loop(value, loop)
    return value


class ContainerProcess(Protocol):
    stdin: Any
    stdout: Any
    stderr: Any

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def kill(self) -> None: ...


class ContainerAdapter(Protocol):
    def copy_to(self, host_path: str, container_path: str) -> None: ...

    def copy_from(self, container_path: str, host_path: str) -> None: ...

    def exec(self, command: list[str], *, timeout: float | None = None) -> Any: ...

    def start_exec(
        self,
        command: list[str],
        *,
        workdir: str | None = None,
        timeout: float | None = None,
    ) -> ContainerProcess: ...


class _HostToolTimeoutError(TimeoutError):
    pass


class _TimeoutLineReader:
    def __init__(self, process: ContainerProcess) -> None:
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
            name="terminal-bench-supervisor-stdout",
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


class HarborContainerAdapter:
    """Small adapter around the container object supplied by Terminal-Bench/Harbor."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.container = getattr(runtime, "container", runtime)
        self.user = self._coerce_optional_string(
            getattr(runtime, "_user", None) or getattr(runtime, "user", None)
        )
        self.workdir = self._coerce_optional_string(
            getattr(runtime, "workdir", None) or getattr(runtime, "_workdir", None)
        )

    @staticmethod
    def _coerce_optional_string(value: Any) -> str | None:
        if value is None:
            return None
        value = str(value)
        return value or None

    @property
    def _docker_container_id(self) -> str | None:
        container_id = getattr(self.container, "id", None)
        if container_id:
            return str(container_id)
        attrs = getattr(self.container, "attrs", None)
        if isinstance(attrs, dict) and attrs.get("Id"):
            return str(attrs["Id"])
        return None

    @property
    def _is_docker_sdk_runtime(self) -> bool:
        return self._docker_container_id is not None

    @property
    def supports_file_sync(self) -> bool:
        return not self._is_docker_sdk_runtime

    def _unsupported_minimal(self, operation: str) -> NotImplementedError:
        return NotImplementedError(
            "Terminal-Bench minimal smoke adapter does not support "
            f"{operation} yet; this smoke path only installs and starts the "
            "persistent supervisor."
        )

    def copy_to(self, host_path: str, container_path: str) -> None:
        if self._is_docker_sdk_runtime:
            raise self._unsupported_minimal("file sync")
        if hasattr(self.container, "copy_to"):
            self.container.copy_to(host_path, container_path)
            return
        if hasattr(self.container, "put_file"):
            self.container.put_file(host_path, container_path)
            return
        raise TypeError("Terminal-Bench container does not expose copy_to/put_file")

    def copy_from(self, container_path: str, host_path: str) -> None:
        if self._is_docker_sdk_runtime:
            raise self._unsupported_minimal("file sync")
        if hasattr(self.container, "copy_from"):
            self.container.copy_from(container_path, host_path)
            return
        if hasattr(self.container, "get_file"):
            self.container.get_file(container_path, host_path)
            return
        raise TypeError("Terminal-Bench container does not expose copy_from/get_file")

    def exec(self, command: list[str], *, timeout: float | None = None) -> Any:
        if self._is_docker_sdk_runtime:
            raise self._unsupported_minimal("list_dir/mkdir_p one-shot exec operations")
        if hasattr(self.container, "exec"):
            return self.container.exec(command, timeout=timeout)
        if hasattr(self.container, "run"):
            return self.container.run(command, timeout=timeout)
        raise TypeError("Terminal-Bench container does not expose exec/run")

    def install_runner_script(
        self,
        source: str,
        runner_path: str,
        *,
        timeout: float | None = None,
    ) -> None:
        container_id = self._docker_container_id
        if container_id is None:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                suffix=".py",
                delete=False,
            ) as tmp:
                tmp.write(source)
                tmp_path = tmp.name
            try:
                self.copy_to(tmp_path, runner_path)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            return
        result = subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                container_id,
                "sh",
                "-c",
                f"cat > {shlex.quote(runner_path)}",
            ],
            input=source,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise SandboxFatalError(
                "Terminal-Bench minimal smoke adapter failed to install runner "
                f"script: exit code {result.returncode}; stdout: {result.stdout}; "
                f"stderr: {result.stderr}"
            )

    def start_exec(
        self,
        command: list[str],
        *,
        workdir: str | None = None,
        timeout: float | None = None,
    ) -> ContainerProcess:
        container_id = self._docker_container_id
        if container_id is not None:
            docker_command = ["docker", "exec", "-i"]
            effective_workdir = workdir or self.workdir
            if effective_workdir:
                docker_command.extend(["-w", effective_workdir])
            if self.user:
                docker_command.extend(["-u", self.user])
            docker_command.append(container_id)
            docker_command.extend(command)
            return subprocess.Popen(
                docker_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        for name in ("start_exec", "exec_stream", "popen"):
            method = getattr(self.container, name, None)
            if method is not None:
                return method(command, workdir=workdir, timeout=timeout)
        raise TypeError(
            "Terminal-Bench container does not expose an interactive exec method; "
            "provide container_adapter=... with start_exec/copy/exec primitives"
        )


def _daytona_session_execute_request(command: str, *, run_async: bool) -> Any:
    try:
        from daytona.common.process import SessionExecuteRequest
    except ImportError as exc:
        raise SandboxFatalError("Daytona session execution requires the daytona SDK") from exc
    return SessionExecuteRequest(command=command, run_async=run_async)


class _DaytonaSessionInput:
    def __init__(
        self,
        process: Any,
        loop: asyncio.AbstractEventLoop,
        session_id: str,
        command_id: str,
        echo_filter: "_DaytonaSessionEchoFilter",
    ) -> None:
        self.process = process
        self.loop = loop
        self.session_id = session_id
        self.command_id = command_id
        self.echo_filter = echo_filter
        self._buffer = ""

    def write(self, data: str) -> int:
        self._buffer += data
        return len(data)

    def flush(self) -> None:
        if not self._buffer:
            return
        data = self._buffer
        self._buffer = ""
        self.echo_filter.expect(data)
        _run_coroutine_on_loop(
            self.process.send_session_command_input(self.session_id, self.command_id, data),
            self.loop,
        )


class _DaytonaSessionEchoFilter:
    def __init__(self) -> None:
        self._pending = ""

    def expect(self, data: str) -> None:
        self._pending += data

    def strip(self, chunk: str) -> str:
        while self._pending and chunk:
            if chunk.startswith(self._pending):
                chunk = chunk[len(self._pending) :]
                self._pending = ""
                break
            if self._pending.startswith(chunk):
                self._pending = self._pending[len(chunk) :]
                return ""
            break
        return chunk


class _DaytonaSessionOutput:
    def __init__(
        self,
        process: Any,
        loop: asyncio.AbstractEventLoop,
        session_id: str,
        command_id: str,
        stream: str,
        echo_filter: "_DaytonaSessionEchoFilter | None" = None,
    ) -> None:
        self.process = process
        self.loop = loop
        self.session_id = session_id
        self.command_id = command_id
        self.stream = stream
        self.echo_filter = echo_filter
        self._offset = 0
        self._pending: list[str] = []
        self._buffer = ""
        self._closed = False

    def close(self) -> None:
        self._closed = True

    def readline(self) -> str:
        while not self._closed:
            line = self._readline_once()
            if line is not None:
                return line
            time.sleep(0.01)
        return ""

    def readline_timeout(self, timeout: float) -> str | None:
        deadline = time.monotonic() + max(0.0, timeout)
        while not self._closed:
            line = self._readline_once()
            if line is not None:
                return line
            if time.monotonic() >= deadline:
                return None
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        return ""

    def _readline_once(self) -> str | None:
        if self._pending:
            return self._pending.pop(0)
        logs = _run_coroutine_on_loop(
            self.process.get_session_command_logs(self.session_id, self.command_id),
            self.loop,
        )
        text = str(getattr(logs, self.stream, "") or "")
        if len(text) > self._offset:
            chunk = text[self._offset :]
            self._offset = len(text)
            self._buffer += chunk
            self._queue_complete_lines()
            if self._pending:
                return self._pending.pop(0)
        command = _run_coroutine_on_loop(
            self.process.get_session_command(self.session_id, self.command_id),
            self.loop,
        )
        if getattr(command, "exit_code", None) is not None:
            if self._buffer:
                self._queue_complete_lines(flush=True)
            if self._pending:
                return self._pending.pop(0)
            return ""
        return None

    def _queue_complete_lines(self, *, flush: bool = False) -> None:
        while "\n" in self._buffer:
            index = self._buffer.find("\n") + 1
            self._queue_line(self._buffer[:index])
            self._buffer = self._buffer[index:]
        if flush and self._buffer:
            self._queue_line(self._buffer)
            self._buffer = ""

    def _queue_line(self, line: str) -> None:
        if self.echo_filter is not None:
            line = self.echo_filter.strip(line)
        if line:
            self._pending.append(line)


class DaytonaSessionProcess:
    def __init__(
        self,
        *,
        process: Any,
        loop: asyncio.AbstractEventLoop,
        session_id: str,
        command_id: str,
    ) -> None:
        self.process = process
        self.loop = loop
        self.session_id = session_id
        self.command_id = command_id
        stdout_echo_filter = _DaytonaSessionEchoFilter()
        self.stdin = _DaytonaSessionInput(
            process, loop, session_id, command_id, stdout_echo_filter
        )
        self.stdout = _DaytonaSessionOutput(
            process,
            loop,
            session_id,
            command_id,
            "stdout",
            stdout_echo_filter,
        )
        self.stderr = _DaytonaSessionOutput(process, loop, session_id, command_id, "stderr")

    def poll(self) -> int | None:
        command = _run_coroutine_on_loop(
            self.process.get_session_command(self.session_id, self.command_id),
            self.loop,
        )
        exit_code = getattr(command, "exit_code", None)
        return None if exit_code is None else int(exit_code)

    def wait(self, timeout: float | None = None) -> int:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            return_code = self.poll()
            if return_code is not None:
                return return_code
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(self.command_id, timeout)
            time.sleep(0.05)

    def kill(self) -> None:
        try:
            _run_coroutine_on_loop(
                self.process.send_session_command_input(
                    self.session_id, self.command_id, "\x03"
                ),
                self.loop,
            )
        except Exception:
            pass


class HarborEnvironmentAdapter(HarborContainerAdapter):
    """Adapter that bridges Harbor's async environment APIs into the runner protocol."""

    def __init__(
        self,
        environment: Any,
        *,
        loop: asyncio.AbstractEventLoop,
        exec_timeout: float,
    ) -> None:
        super().__init__(environment)
        self.environment = environment
        self.loop = loop
        self.exec_timeout = exec_timeout

    def _resolve(self, value: Any) -> Any:
        return _resolve_maybe_awaitable(value, self.loop)

    def copy_to(self, host_path: str, container_path: str) -> None:
        if not self._is_docker_sdk_runtime and hasattr(self.environment, "upload_file"):
            self._resolve(self.environment.upload_file(host_path, container_path))
            return
        super().copy_to(host_path, container_path)

    def copy_from(self, container_path: str, host_path: str) -> None:
        if not self._is_docker_sdk_runtime and hasattr(self.environment, "download_file"):
            self._resolve(self.environment.download_file(container_path, host_path))
            return
        super().copy_from(container_path, host_path)

    def exec(self, command: list[str], *, timeout: float | None = None) -> Any:
        if self._is_docker_sdk_runtime:
            return super().exec(command, timeout=timeout)
        method = getattr(self.environment, "exec", None)
        if method is None:
            return super().exec(command, timeout=timeout)
        return self._resolve(
            method(
                command=" ".join(shlex.quote(str(part)) for part in command),
                timeout_sec=int(timeout or self.exec_timeout),
            )
        )

    def install_runner_script(
        self,
        source: str,
        runner_path: str,
        *,
        timeout: float | None = None,
    ) -> None:
        if self._is_docker_sdk_runtime or not hasattr(self.environment, "upload_file"):
            super().install_runner_script(source, runner_path, timeout=timeout)
            return
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".py", delete=False
        ) as tmp:
            tmp.write(source)
            tmp_path = tmp.name
        try:
            self.copy_to(tmp_path, runner_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def start_exec(
        self,
        command: list[str],
        *,
        workdir: str | None = None,
        timeout: float | None = None,
    ) -> ContainerProcess:
        if self._is_docker_sdk_runtime:
            return super().start_exec(command, workdir=workdir, timeout=timeout)
        daytona_process = self._start_daytona_session_exec(
            command, workdir=workdir, timeout=timeout
        )
        if daytona_process is not None:
            return daytona_process
        for name in ("start_exec", "exec_stream", "popen"):
            method = getattr(self.environment, name, None)
            if method is None:
                continue
            try:
                return self._resolve(method(command, workdir=workdir, timeout=timeout))
            except TypeError as positional_exc:
                try:
                    return self._resolve(
                        method(
                            command=" ".join(shlex.quote(str(part)) for part in command),
                            cwd=workdir,
                            timeout_sec=int(timeout or self.exec_timeout),
                        )
                    )
                except TypeError:
                    raise positional_exc
        process = self._start_docker_compose_exec(command, workdir=workdir)
        if process is not None:
            return process
        raise TypeError(
            "Harbor environment does not expose an interactive exec method and "
            "no Docker SDK container id is available; persistent PredictRLM "
            "supervisor execution requires start_exec/exec_stream/popen or docker exec -i"
        )

    def _start_daytona_session_exec(
        self,
        command: list[str],
        *,
        workdir: str | None = None,
        timeout: float | None = None,
    ) -> ContainerProcess | None:
        sandbox = getattr(self.environment, "_sandbox", None)
        process = getattr(sandbox, "process", None)
        if not all(
            callable(getattr(process, name, None))
            for name in (
                "create_session",
                "execute_session_command",
                "get_session_command",
                "get_session_command_logs",
                "send_session_command_input",
            )
        ):
            return None
        session_id = f"predict-rlm-{uuid4().hex}"
        self._resolve(process.create_session(session_id))
        shell_command = " ".join(shlex.quote(str(part)) for part in command)
        effective_workdir = workdir or self.workdir
        if effective_workdir:
            shell_command = f"cd {shlex.quote(effective_workdir)} && {shell_command}"
        request = _daytona_session_execute_request(shell_command, run_async=True)
        response = self._resolve(
            process.execute_session_command(
                session_id,
                request,
                timeout=int(timeout or self.exec_timeout),
            )
        )
        command_id = getattr(response, "cmd_id", None)
        if not command_id:
            raise SandboxFatalError("Daytona session command did not return a command id")
        return DaytonaSessionProcess(
            process=process,
            loop=self.loop,
            session_id=session_id,
            command_id=str(command_id),
        )

    def _start_docker_compose_exec(
        self,
        command: list[str],
        *,
        workdir: str | None = None,
    ) -> ContainerProcess | None:
        compose_paths = getattr(self.environment, "_docker_compose_paths", None)
        environment_dir = getattr(self.environment, "environment_dir", None)
        session_id = getattr(self.environment, "session_id", None)
        if not compose_paths or environment_dir is None or session_id is None:
            return None
        docker_command = [
            "docker",
            "compose",
            "--project-name",
            _docker_compose_project_name(str(session_id)),
            "--project-directory",
            str(Path(environment_dir).resolve().absolute()),
        ]
        for path in compose_paths:
            docker_command.extend(["-f", str(Path(path).resolve().absolute())])
        docker_command.extend(["exec", "-T"])
        effective_workdir = workdir or self.workdir
        if effective_workdir:
            docker_command.extend(["-w", effective_workdir])
        user = self._coerce_optional_string(getattr(self.environment, "default_user", None))
        if user:
            docker_command.extend(["-u", user])
        docker_command.append("main")
        docker_command.extend(command)
        compose_env_vars = getattr(self.environment, "_compose_env_vars", None)
        env = compose_env_vars(include_os_env=True) if compose_env_vars is not None else None
        return subprocess.Popen(
            docker_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )


class LocalProcessRunnerAdapter:
    """Runner adapter for an agent already executing inside the task machine."""

    supports_file_sync = True

    def __init__(self, *, workdir: str | None = None) -> None:
        self.workdir = workdir

    def _cwd(self, workdir: str | None = None) -> str | None:
        return workdir or self.workdir

    def copy_to(self, host_path: str, container_path: str) -> None:
        source = Path(host_path)
        destination = Path(container_path)
        if source.resolve() == destination.resolve():
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    def copy_from(self, container_path: str, host_path: str) -> None:
        source = Path(container_path)
        destination = Path(host_path)
        if source.resolve() == destination.resolve():
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    def exec(self, command: list[str], *, timeout: float | None = None) -> Any:
        return subprocess.run(
            command,
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
    ) -> ContainerProcess:
        del timeout
        return subprocess.Popen(
            command,
            cwd=self._cwd(workdir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )


class TerminalBenchRunnerInterpreter(PersistentJsonRpcRunnerClient, PredictRLMInterpreter):
    _LIST_DIR_SCRIPT = (
        "import json, pathlib, sys; "
        "root = pathlib.Path(sys.argv[1]); "
        "print(json.dumps([str(p) for p in sorted(root.rglob('*')) if p.is_file()]))"
    )

    def __init__(
        self,
        container: Any,
        *,
        container_adapter: ContainerAdapter | None = None,
        tools: dict[str, Callable[..., Any]] | None = None,
        output_fields: list[dict[str, Any]] | None = None,
        runner_path: str = "/tmp/predict_rlm_runner.py",
        python_executable: str = "python3",
        workdir: str | None = None,
        exec_timeout: float = 900.0,
        recoverable_timeout_grace: float = TERMINAL_BENCH_RECOVERABLE_TIMEOUT_GRACE_SECONDS,
    ) -> None:
        PersistentJsonRpcRunnerClient.__init__(
            self,
            supervisor_name="Terminal-Bench supervisor",
        )
        self.container = container
        self.adapter = container_adapter or HarborContainerAdapter(container)
        self.tools = tools or {}
        self.output_fields = output_fields or []
        self.runner_path = runner_path
        self.python_executable = python_executable
        self.workdir = workdir
        self.exec_timeout = exec_timeout
        self.recoverable_timeout_grace = self._resolve_recoverable_timeout_grace(
            recoverable_timeout_grace
        )
        self._process: ContainerProcess | None = None
        self._shutdown = False
        self._tools_registered = False
        self._output_fields_registered = False
        self._stdout_reader: _TimeoutLineReader | None = None
        self._execution_gate = InterpreterExecutionGate("Terminal-Bench supervisor")
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
        return list(json.loads(stdout or "[]"))

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
                "terminal_bench.runner.cancel_abort",
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
                "terminal_bench.runner.dead_before_request",
                stderr_len=len(stderr),
                **_process_debug_metadata(self._process),
            )
            raise SandboxFatalError(f"Terminal-Bench supervisor exited unexpectedly: {stderr}")
        self._copy_runner_script()
        self._process = self.adapter.start_exec(
            [self.python_executable, "-u", self.runner_path],
            workdir=self.workdir,
            timeout=self.exec_timeout,
        )
        self._stdout_reader = None
        self._debug_event(
            "terminal_bench.runner.start",
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
                runner_source(),
                self.runner_path,
                timeout=self.exec_timeout,
            )
            return
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".py", delete=False
        ) as tmp:
            tmp.write(runner_source())
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

    def _get_supervisor_process(self) -> ContainerProcess | None:
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
        debug_event(event, interpreter="terminal_bench", **metadata)

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
        self._debug_event("terminal_bench.runner.request", **context)

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
        self._debug_event("terminal_bench.runner.response", **context)

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
                "terminal_bench.runner.empty_execute_output",
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
            "terminal_bench.runner.send_error",
            method=method,
            request_id=request_id,
            error_type=type(exc).__name__,
        )
        raise SandboxFatalError("Terminal-Bench supervisor pipe broke") from exc

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
            "terminal_bench.runner.exit_during_request",
            method=method,
            request_id=request_id,
            elapsed_seconds=time.perf_counter() - request_start,
            stderr_len=len(stderr),
            **_process_debug_metadata(process),
        )
        raise SandboxFatalError(f"Terminal-Bench supervisor exited unexpectedly: {stderr}")

    def _on_supervisor_stale_response(
        self,
        method: str,
        *,
        expected_request_id: int,
        stale_response: dict[str, Any],
        stale_discards: int,
    ) -> None:
        self._debug_event(
            "terminal_bench.runner.stale_response",
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
            "terminal_bench.runner.request_timeout",
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
                f"Terminal-Bench supervisor request timed out after {request_timeout:g}s"
            )

        self._discard_supervisor_process()
        restart_error: BaseException | None = None
        try:
            self._ensure_process()
        except BaseException as exc:
            restart_error = exc
        if restart_error is not None:
            raise SandboxFatalError(
                "Terminal-Bench supervisor request timed out after "
                f"{request_timeout:g}s and the copied supervisor could not be restarted: "
                f"{restart_error}"
            ) from restart_error

        diagnostic = (
            "Terminal-Bench supervisor request timed out after "
            f"{request_timeout:g}s before it returned a structured timeout. "
            "The copied supervisor process was killed and restarted; Python globals "
            "from the timed-out supervisor were lost, while task container filesystem "
            "state is preserved. Re-run setup code before relying on in-memory "
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

    def _kill_process_after_timeout(self, process: ContainerProcess) -> None:
        self._debug_event(
            "terminal_bench.runner.kill_after_timeout",
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
                "terminal_bench.runner.discard",
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
            "Terminal-Bench supervisor exited after the previous execute response. "
            "The copied supervisor process was restarted; Python globals from the "
            "prior supervisor were lost, while task container filesystem state is "
            "preserved. Re-run setup code before relying on in-memory variables."
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
            error = RunnerError.from_exception(exc)
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
                "Terminal-Bench request deadline was exhausted"
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
            name="terminal-bench-host-tool",
            daemon=True,
        )
        thread.start()
        try:
            kind, value = result_queue.get(timeout=max(0.0, callback_budget))
        except queue.Empty as exc:
            raise _HostToolTimeoutError(
                "Host tool call timed out after exhausting the remaining "
                "Terminal-Bench request budget "
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
        error = RunnerError.from_payload(response.get("error") or {})
        if error.type == "SyntaxError":
            raise SyntaxError(error.message)
        raise CodeInterpreterError(f"{error.type}: {error.message}")

    def _require_process(self) -> ContainerProcess:
        return self._require_supervisor_process()

    def _read_stderr(self) -> str:
        if self._process is None or self._process.stderr is None:
            return ""
        return self._read_stderr_for_process(self._process)

    def _read_stderr_for_process(self, process: ContainerProcess) -> str:
        if process.stderr is None:
            return ""
        try:
            stderr = process.stderr.read()
        except Exception:
            return ""
        if stderr:
            self._debug_event(
                "terminal_bench.runner.stderr_read",
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
                f"Terminal-Bench container failed while {operation}: "
                f"exit code {returncode}; stdout: {stdout}; stderr: {stderr}"
            )


class LocalProcessRunnerInterpreter(TerminalBenchRunnerInterpreter):
    """PredictRLM interpreter that runs the persistent supervisor locally."""

    def __init__(
        self,
        *,
        tools: dict[str, Callable[..., Any]] | None = None,
        output_fields: list[dict[str, Any]] | None = None,
        runner_path: str = "/tmp/predict_rlm_runner.py",
        python_executable: str | None = None,
        workdir: str | None = None,
        exec_timeout: float = 900.0,
        recoverable_timeout_grace: float = TERMINAL_BENCH_RECOVERABLE_TIMEOUT_GRACE_SECONDS,
    ) -> None:
        super().__init__(
            Path(workdir or os.getcwd()),
            container_adapter=LocalProcessRunnerAdapter(workdir=workdir),
            tools=tools,
            output_fields=output_fields,
            runner_path=runner_path,
            python_executable=python_executable or sys.executable,
            workdir=workdir,
            exec_timeout=exec_timeout,
            recoverable_timeout_grace=recoverable_timeout_grace,
        )


class HarborEnvironmentInterpreter(TerminalBenchRunnerInterpreter):
    """PredictRLM interpreter that executes through Harbor's BaseEnvironment API."""

    def __init__(
        self,
        environment: Any,
        *,
        loop: asyncio.AbstractEventLoop,
        tools: dict[str, Callable[..., Any]] | None = None,
        output_fields: list[dict[str, Any]] | None = None,
        runner_path: str = "/tmp/predict_rlm_runner.py",
        python_executable: str = "python3",
        workdir: str | None = None,
        exec_timeout: float = 900.0,
        recoverable_timeout_grace: float = TERMINAL_BENCH_RECOVERABLE_TIMEOUT_GRACE_SECONDS,
    ) -> None:
        adapter = HarborEnvironmentAdapter(
            environment,
            loop=loop,
            exec_timeout=exec_timeout,
        )
        super().__init__(
            environment,
            container_adapter=adapter,
            tools=tools,
            output_fields=output_fields,
            runner_path=runner_path,
            python_executable=python_executable,
            workdir=workdir,
            exec_timeout=exec_timeout,
            recoverable_timeout_grace=recoverable_timeout_grace,
        )
        self.environment = environment
        self.loop = loop
        self._python_available = False

    def mount_file_at(self, host_path: str, virtual_path: str) -> None:
        if hasattr(self.environment, "upload_file"):
            self._run_coro(self.environment.upload_file(host_path, virtual_path))
            return
        super().mount_file_at(host_path, virtual_path)

    def mkdir_p(self, virtual_path: str) -> None:
        if not hasattr(self.environment, "exec"):
            super().mkdir_p(virtual_path)
            return
        result = self._run_coro(
            self.environment.exec(
                command=f"mkdir -p {shlex.quote(virtual_path)}",
                timeout_sec=int(self.exec_timeout),
            )
        )
        self._raise_for_harbor_exec_failure(result, f"creating {virtual_path}")

    def list_dir(self, virtual_path: str) -> list[str]:
        if not hasattr(self.environment, "exec"):
            return super().list_dir(virtual_path)
        self._ensure_python_available()
        result = self._run_coro(
            self.environment.exec(
                command=_shell_python_command(
                    self.python_executable,
                    ["-c", self._LIST_DIR_SCRIPT, virtual_path],
                ),
                timeout_sec=int(self.exec_timeout),
            )
        )
        self._raise_for_harbor_exec_failure(result, f"listing {virtual_path}")
        return list(json.loads(getattr(result, "stdout", "") or "[]"))

    def sync_file_to(self, virtual_path: str, host_path: str) -> None:
        if not hasattr(self.environment, "download_file"):
            super().sync_file_to(virtual_path, host_path)
            return
        Path(host_path).parent.mkdir(parents=True, exist_ok=True)
        self._run_coro(self.environment.download_file(virtual_path, host_path))

    def _ensure_process(self) -> None:
        self._ensure_python_available()
        super()._ensure_process()

    def shutdown(self) -> None:
        super().shutdown()

    def _ensure_python_available(self) -> None:
        if self._python_available or self.python_executable != "python3":
            return
        if getattr(self.adapter, "_is_docker_sdk_runtime", False):
            self._python_available = True
            return
        if not hasattr(self.environment, "exec"):
            self._python_available = True
            return
        result = self._run_coro(
            self.environment.exec(
                command=_python_bootstrap_command(),
                timeout_sec=int(self.exec_timeout),
            )
        )
        self._raise_for_harbor_exec_failure(result, "installing Python")
        self._python_available = True

    def _run_coro(self, coro: Any) -> Any:
        return _resolve_maybe_awaitable(coro, self.loop)

    def _raise_for_harbor_exec_failure(self, result: Any, operation: str) -> None:
        return_code = getattr(result, "return_code", getattr(result, "returncode", 0))
        if return_code not in (0, None):
            stderr = getattr(result, "stderr", "")
            stdout = getattr(result, "stdout", "")
            raise SandboxFatalError(
                f"Harbor environment failed while {operation}: "
                f"exit code {return_code}; stdout: {stdout}; stderr: {stderr}"
            )
