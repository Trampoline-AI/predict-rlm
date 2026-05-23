from __future__ import annotations

import asyncio
import inspect
import json
import math
import os
import queue
import re
import select
import shlex
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Protocol

from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput

from predict_rlm.execution_timeout import (
    DEFAULT_RECOVERABLE_EXECUTION_TIMEOUT_GRACE_SECONDS,
    ITERATION_TIMEOUT_FAILURE_CLASS,
    format_recoverable_timeout_result,
    recoverable_timeout_host_deadline_seconds,
    validate_execution_timeout,
)
from predict_rlm.interpreter import SandboxFatalError
from predict_rlm.interpreters.base import InterpreterExecutionGate, PredictRLMInterpreter

from .protocol import RunnerError, dumps, request
from .runner import runner_source

TERMINAL_BENCH_RECOVERABLE_TIMEOUT_GRACE_SECONDS = (
    DEFAULT_RECOVERABLE_EXECUTION_TIMEOUT_GRACE_SECONDS
)


def _shell_python_command(python_executable: str, args: list[str]) -> str:
    quoted_args = " ".join(shlex.quote(arg) for arg in args)
    if python_executable != "python3":
        return " ".join(part for part in (shlex.quote(python_executable), quoted_args) if part)
    resolver = (
        "if command -v python3 >/dev/null 2>&1; then _predict_rlm_python=python3; "
        "elif command -v python >/dev/null 2>&1; then _predict_rlm_python=python; "
        "else echo 'PredictRLM runner requires python3 or python on PATH' >&2; "
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
            "else echo 'PredictRLM runner requires python3 or python on PATH and no supported package manager was found' >&2; exit 127; fi;",
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


class _TimeoutLineReader:
    def __init__(self, process: ContainerProcess) -> None:
        self.process = process
        self.pipe = process.stdout
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread: threading.Thread | None = None

    def readline(self, timeout: float) -> str | None:
        timeout = max(0.0, timeout)
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
            name="terminal-bench-runner-stdout",
            daemon=True,
        )
        self._thread.start()

    def _read_loop(self) -> None:
        while True:
            line = self.pipe.readline()
            if line:
                self._queue.put(line)
                continue
            if self.process.poll() is not None:
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
            "persistent runner."
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
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".py", delete=False) as tmp:
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
            "runner execution requires start_exec/exec_stream/popen or docker exec -i"
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


class TerminalBenchRunnerInterpreter(PredictRLMInterpreter):
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
        self._request_id = 0
        self._shutdown = False
        self._tools_registered = False
        self._output_fields_registered = False
        self._stdout_reader: _TimeoutLineReader | None = None
        self._execution_gate = InterpreterExecutionGate("Terminal-Bench runner")

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
        return await asyncio.to_thread(self.execute, code, variables, timeout=timeout)

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
        if self._process is not None and self._process.poll() is None:
            try:
                self._send_request("shutdown", {})
            except Exception:
                pass
            try:
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
        self._process = None

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
        if variables:
            assignments = "\n".join(f"{name} = {value!r}" for name, value in variables.items())
            code = f"{assignments}\n{code}"
        params: dict[str, Any] = {"code": code}
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
            raise SandboxFatalError(f"Terminal-Bench runner exited unexpectedly: {stderr}")
        self._copy_runner_script()
        self._process = self.adapter.start_exec(
            [self.python_executable, "-u", self.runner_path],
            workdir=self.workdir,
            timeout=self.exec_timeout,
        )
        self._stdout_reader = None
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
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".py", delete=False) as tmp:
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
        if method == "shutdown" and self._process is not None:
            return self._send_request_without_ensure(method, params)
        self._ensure_process()
        return self._send_request_without_ensure(method, params, timeout=timeout)

    def _send_request_without_ensure(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        process = self._require_process()
        self._request_id += 1
        request_id = self._request_id
        payload = request(request_id, method, params or {})
        try:
            process.stdin.write(dumps(payload) + "\n")
            process.stdin.flush()
        except BrokenPipeError as exc:
            raise SandboxFatalError("Terminal-Bench runner pipe broke") from exc

        request_timeout = self.exec_timeout if timeout is None else timeout
        deadline = time.monotonic() + request_timeout
        while True:
            if process.poll() is not None:
                raise SandboxFatalError(
                    f"Terminal-Bench runner exited unexpectedly: {self._read_stderr()}"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._kill_process_after_timeout(process)
                raise SandboxFatalError(
                    f"Terminal-Bench runner request timed out after {request_timeout:g}s"
                )
            line = self._read_stdout_line(process, timeout=remaining)
            if line is None:
                self._kill_process_after_timeout(process)
                raise SandboxFatalError(
                    f"Terminal-Bench runner request timed out after {request_timeout:g}s"
                )
            if not line:
                time.sleep(0.01)
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if message.get("method") == "tool_call":
                self._write_tool_response(self._build_tool_response(message))
                continue
            if message.get("id") == request_id:
                return message

    def _read_stdout_line(
        self,
        process: ContainerProcess,
        *,
        timeout: float,
    ) -> str | None:
        if self._stdout_reader is None or self._stdout_reader.process is not process:
            self._stdout_reader = _TimeoutLineReader(process)
        return self._stdout_reader.readline(timeout)

    def _kill_process_after_timeout(self, process: ContainerProcess) -> None:
        process.kill()
        try:
            process.wait(timeout=1)
        except Exception:
            pass

    def _build_tool_response(self, message: dict[str, Any]) -> dict[str, Any]:
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
                result = tool(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = asyncio.run(result)
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

    def _write_tool_response(self, response: dict[str, Any]) -> None:
        process = self._require_process()
        process.stdin.write(dumps(response) + "\n")
        process.stdin.flush()

    def _unwrap_execute_response(self, response: dict[str, Any]) -> Any:
        if "error" in response:
            error = RunnerError.from_payload(response.get("error") or {})
            if error.type == "SyntaxError":
                raise SyntaxError(error.message)
            raise CodeInterpreterError(f"{error.type}: {error.message}")
        result = response.get("result") or {}
        if isinstance(result, dict) and "timeout" in result:
            return format_recoverable_timeout_result(result)
        if "final" in result:
            return FinalOutput(result["final"])
        return result.get("output")

    def _require_process(self) -> ContainerProcess:
        if self._process is None:
            raise SandboxFatalError("Terminal-Bench runner is not started")
        if self._process.stdin is None or self._process.stdout is None:
            raise SandboxFatalError("Terminal-Bench runner stdio is unavailable")
        return self._process

    def _read_stderr(self) -> str:
        if self._process is None or self._process.stderr is None:
            return ""
        try:
            return self._process.stderr.read()
        except Exception:
            return ""

    def _raise_for_exec_failure(self, result: Any, operation: str) -> None:
        returncode = getattr(result, "returncode", getattr(result, "return_code", 0))
        if returncode not in (0, None):
            stderr = getattr(result, "stderr", "")
            stdout = getattr(result, "stdout", "")
            raise SandboxFatalError(
                f"Terminal-Bench container failed while {operation}: "
                f"exit code {returncode}; stdout: {stdout}; stderr: {stderr}"
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
