"""Python JSON-RPC runner used by the Docker Sandboxes backend."""

from __future__ import annotations

import ast
import asyncio
import builtins
import contextlib
import hashlib
import inspect
import io
import json
import math
import multiprocessing
import os
import pathlib
import pickle
import queue
import select
import shutil
import signal
import sys
import tempfile
import time
from typing import Any

PROTOCOL_STDIN = sys.stdin
PROTOCOL_STDOUT = sys.stdout
REAL_OPEN = builtins.open
REAL_PATH = type(pathlib.Path())
SANDBOX_ROOT = REAL_PATH(
    os.environ.get("PREDICT_RLM_SBX_ROOT")
    or tempfile.mkdtemp(prefix="predict-rlm-sbx-runner-")
).resolve()
SANDBOX_DIR = SANDBOX_ROOT / "sandbox"
SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
TOOL_REQUEST_ID = 0
TOOL_RESPONSE_LOCK = asyncio.Lock()
PENDING_TOOL_RESPONSES: dict[int, dict[str, Any]] = {}
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _debug_enabled() -> bool:
    return os.environ.get("PREDICT_RLM_DEBUG", "").strip().lower() in _TRUE_VALUES


def _debug_event(event: str, **metadata: Any) -> None:
    if not _debug_enabled():
        return
    log_path = os.environ.get("PREDICT_RLM_DEBUG_LOG")
    if not log_path:
        return
    payload = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "event": event,
        **metadata,
    }
    try:
        pathlib.Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with REAL_OPEN(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
    except Exception:
        return


def _code_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8", errors="replace")).hexdigest()[:16]


class _FinalOutputError(Exception):
    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__("SUBMIT")
        self.payload = payload


class _RunnerExecutionError(Exception):
    def __init__(self, payload: dict[str, Any]) -> None:
        message = payload.get("message") or payload.get("type") or "execution failed"
        super().__init__(str(message))
        self.payload = payload


class _ExecutionCapture:
    def __init__(
        self,
        stdout: io.StringIO | None = None,
        stderr: io.StringIO | None = None,
    ) -> None:
        self.stdout = stdout or io.StringIO()
        self.stderr = stderr or io.StringIO()


class _FdTextStream(io.TextIOBase):
    def __init__(self, fd: int) -> None:
        self.fd = fd
        self._buffer = io.StringIO()

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        if not isinstance(text, str):
            text = str(text)
        self._buffer.write(text)
        if text:
            os.write(self.fd, text.encode("utf-8", errors="replace"))
        return len(text)

    def flush(self) -> None:
        return None

    def getvalue(self) -> str:
        return self._buffer.getvalue()


class _VirtualPath(str):
    def __new__(cls, virtual_path: str, real_path: str):
        obj = str.__new__(cls, real_path)
        obj.virtual_path = virtual_path
        return obj


class _PredictResult:
    __slots__ = ("_store",)

    def __init__(self, value: dict[str, Any] | None) -> None:
        object.__setattr__(self, "_store", dict(value or {}))

    def __getattribute__(self, key: str) -> Any:
        if not key.startswith("_"):
            store = object.__getattribute__(self, "_store")
            if key in store:
                return store[key]
        return object.__getattribute__(self, key)

    def __getitem__(self, key: str) -> Any:
        return object.__getattribute__(self, "_store")[key]

    def __setitem__(self, key: str, value: Any) -> None:
        object.__getattribute__(self, "_store")[key] = value

    def __contains__(self, key: object) -> bool:
        return key in object.__getattribute__(self, "_store")

    def __iter__(self):
        return iter(object.__getattribute__(self, "_store"))

    def __len__(self) -> int:
        return len(object.__getattribute__(self, "_store"))

    def __repr__(self) -> str:
        return f"PredictResult({object.__getattribute__(self, '_store')!r})"

    def keys(self) -> list[str]:
        return list(object.__getattribute__(self, "_store").keys())

    def values(self) -> list[Any]:
        return list(object.__getattribute__(self, "_store").values())

    def items(self) -> list[tuple[str, Any]]:
        return list(object.__getattribute__(self, "_store").items())

    def get(self, key: str, default: Any = None) -> Any:
        return object.__getattribute__(self, "_store").get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return dict(object.__getattribute__(self, "_store"))


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, _PredictResult):
        return _to_jsonable(value.to_dict())
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(val) for val in value]
    if isinstance(value, _VirtualPath):
        return value.virtual_path
    return value


def _map_virtual_path(path: Any) -> Any:
    raw = os.fspath(path) if isinstance(path, os.PathLike) else path
    if isinstance(raw, str) and (raw == "/sandbox" or raw.startswith("/sandbox/")):
        rel = raw.removeprefix("/sandbox").lstrip("/")
        return _VirtualPath(raw, str(SANDBOX_DIR / rel))
    return path


def _virtual_from_real(path: pathlib.Path) -> str:
    rel = path.resolve().relative_to(SANDBOX_DIR)
    return "/sandbox/" + rel.as_posix()


def _open(path: Any, *args: Any, **kwargs: Any):
    return REAL_OPEN(_map_virtual_path(path), *args, **kwargs)


def _submit(**kwargs: Any) -> None:
    raise _FinalOutputError(_to_jsonable(kwargs))


def _send_protocol(message: dict[str, Any]) -> None:
    PROTOCOL_STDOUT.write(json.dumps(message, default=str) + "\n")
    PROTOCOL_STDOUT.flush()


def _read_protocol_response_line() -> dict[str, Any]:
    while True:
        line = PROTOCOL_STDIN.readline()
        if not line:
            raise RuntimeError("Host closed stdin while waiting for tool response")
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue


async def _read_protocol_response(request_id: int) -> dict[str, Any]:
    async with TOOL_RESPONSE_LOCK:
        if request_id in PENDING_TOOL_RESPONSES:
            return PENDING_TOOL_RESPONSES.pop(request_id)
        while True:
            response = await asyncio.to_thread(_read_protocol_response_line)
            response_id = response.get("id")
            if response_id == request_id:
                return response
            if isinstance(response_id, int):
                PENDING_TOOL_RESPONSES[response_id] = response


async def _call_host_tool(name: str, *args: Any, **kwargs: Any) -> Any:
    global TOOL_REQUEST_ID
    TOOL_REQUEST_ID += 1
    request_id = TOOL_REQUEST_ID
    _send_protocol({
        "jsonrpc": "2.0",
        "method": "tool_call",
        "params": {"name": name, "args": list(args), "kwargs": kwargs},
        "id": request_id,
    })
    response = await _read_protocol_response(request_id)
    if "error" in response:
        raise RuntimeError(response["error"].get("message", response["error"]))
    result = response.get("result", {})
    value = result.get("value")
    if result.get("type") == "json":
        value = json.loads(value)
    if name == "predict" and isinstance(value, dict):
        return _PredictResult(value)
    return value


def _register_tools(params: dict[str, Any], globals_dict: dict[str, Any]) -> dict[str, Any]:
    for name in params.get("tools", []):
        async def _tool(*args: Any, __tool_name: str = name, **kwargs: Any) -> Any:
            return await _call_host_tool(__tool_name, *args, **kwargs)

        globals_dict[name] = _tool
    return {}


def _install_virtual_filesystem(globals_dict: dict[str, Any]) -> None:
    builtins.open = _open
    globals_dict.setdefault("SUBMIT", _submit)


def _new_globals() -> dict[str, Any]:
    globals_dict: dict[str, Any] = {"__name__": "__main__"}
    _install_virtual_filesystem(globals_dict)
    return globals_dict


def _response(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "result": result, "id": request_id}


def _exception_payload(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, _RunnerExecutionError):
        return exc.payload
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "args": list(getattr(exc, "args", ())),
    }


def _error(request_id: Any, exc: BaseException) -> dict[str, Any]:
    data = _exception_payload(exc)
    partial_output = getattr(exc, "_predict_rlm_output", "")
    if partial_output:
        data["output"] = partial_output
    return {
        "jsonrpc": "2.0",
        "error": {
            "code": -32000,
            "message": str(exc),
            "data": data,
        },
        "id": request_id,
    }


async def _execute_code(
    code: str,
    globals_dict: dict[str, Any],
    capture: _ExecutionCapture | None = None,
) -> dict[str, Any]:
    capture = capture or _ExecutionCapture()
    try:
        compiled = compile(
            code,
            "<sbx-runner>",
            "exec",
            flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
        )
        with (
            contextlib.redirect_stdout(capture.stdout),
            contextlib.redirect_stderr(capture.stderr),
        ):
            result = eval(compiled, globals_dict, globals_dict)
            if inspect.isawaitable(result):
                await result
    except _FinalOutputError as final:
        return {"final": final.payload}
    except BaseException as exc:
        setattr(
            exc,
            "_predict_rlm_output",
            capture.stdout.getvalue() + capture.stderr.getvalue(),
        )
        raise
    return {"output": capture.stdout.getvalue() + capture.stderr.getvalue()}


def _execution_timeout_seconds(params: dict[str, Any]) -> float | None:
    timeout = params.get("execution_timeout_seconds")
    if timeout is None:
        return None
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or float(timeout) <= 0
    ):
        raise ValueError("execution_timeout_seconds must be a positive number")
    return float(timeout)


def _pickleable_globals(globals_dict: dict[str, Any]) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    for name, value in globals_dict.items():
        if name.startswith("__") and name.endswith("__"):
            continue
        if name == "SUBMIT":
            continue
        try:
            pickle.dumps(value)
        except Exception:
            continue
        updates[name] = value
    return updates


def _raise_runner_error(payload: dict[str, Any]) -> None:
    raise _RunnerExecutionError(payload)


def _runner_context() -> multiprocessing.context.BaseContext:
    try:
        return multiprocessing.get_context("fork")
    except ValueError as exc:
        raise RuntimeError("isolated execution requires a fork-capable Python runtime") from exc


def _execute_code_runner(
    code: str,
    globals_dict: dict[str, Any],
    result_queue: multiprocessing.Queue,
    stdin_fd: int,
    stdout_fd: int,
    stderr_fd: int,
) -> None:
    with contextlib.suppress(OSError):
        os.setsid()
    global PROTOCOL_STDIN, PROTOCOL_STDOUT
    PROTOCOL_STDIN = os.fdopen(stdin_fd, "r", encoding="utf-8", buffering=1)
    PROTOCOL_STDOUT = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    devnull_stdin = open(os.devnull, "r", encoding="utf-8")
    os.dup2(devnull_stdin.fileno(), 0)
    sys.stdin = devnull_stdin
    global PENDING_TOOL_RESPONSES, TOOL_RESPONSE_LOCK
    PENDING_TOOL_RESPONSES = {}
    TOOL_RESPONSE_LOCK = asyncio.Lock()
    os.dup2(stdout_fd, 1)
    os.dup2(stderr_fd, 2)
    capture = _ExecutionCapture(
        stdout=_FdTextStream(stdout_fd),
        stderr=_FdTextStream(stderr_fd),
    )
    try:
        result = asyncio.run(_execute_code(code, globals_dict, capture))
        result_queue.put({
            "ok": True,
            "result": result,
            "globals": _pickleable_globals(globals_dict),
        })
    except BaseException as exc:
        result_queue.put({"ok": False, "error": _exception_payload(exc)})
    finally:
        with contextlib.suppress(OSError):
            os.close(stdout_fd)
        with contextlib.suppress(OSError):
            os.close(stderr_fd)


def _drain_fd(fd: int, parts: list[str]) -> bool:
    while True:
        try:
            chunk = os.read(fd, 65536)
        except BlockingIOError:
            return True
        except OSError:
            return False
        if not chunk:
            return False
        parts.append(chunk.decode("utf-8", errors="replace"))


def _runner_process_group_id(process: multiprocessing.Process) -> int | None:
    pid = process.pid
    if pid is None:
        return None
    try:
        pgid = os.getpgid(pid)
    except OSError:
        return None
    return pgid if pgid == pid else None


def _signal_runner_process_group(pgid: int | None, sig: int) -> bool:
    if pgid is None:
        return False
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return False
    except OSError:
        return False
    return True


def _terminate_runner(process: multiprocessing.Process) -> None:
    pgid = _runner_process_group_id(process)
    if not process.is_alive() and pgid is None:
        return

    if not _signal_runner_process_group(pgid, signal.SIGINT) and process.is_alive():
        process.terminate()
    process.join(timeout=0.2)

    if not _signal_runner_process_group(pgid, signal.SIGTERM) and process.is_alive():
        process.terminate()
    process.join(timeout=0.3)

    if not _signal_runner_process_group(pgid, signal.SIGKILL) and process.is_alive():
        if hasattr(process, "kill"):
            process.kill()
        else:
            process.terminate()
    process.join(timeout=0.5)


async def _execute_code_in_runner_with_timeout(
    code: str,
    globals_dict: dict[str, Any],
    timeout_seconds: float | None,
) -> dict[str, Any]:
    code_digest = _code_hash(code)
    ctx = _runner_context()
    stdout_read_fd, stdout_write_fd = os.pipe()
    stderr_read_fd, stderr_write_fd = os.pipe()
    protocol_stdin_fd = os.dup(0)
    os.set_blocking(stdout_read_fd, False)
    os.set_blocking(stderr_read_fd, False)
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=_execute_code_runner,
        args=(
            code,
            globals_dict,
            result_queue,
            protocol_stdin_fd,
            stdout_write_fd,
            stderr_write_fd,
        ),
    )
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    active_fds = {stdout_read_fd: stdout_parts, stderr_read_fd: stderr_parts}
    process.start()
    os.close(protocol_stdin_fd)
    os.close(stdout_write_fd)
    os.close(stderr_write_fd)
    deadline = (
        time.monotonic() + timeout_seconds if timeout_seconds is not None else None
    )
    runner_message: dict[str, Any] | None = None

    try:
        while True:
            for fd, parts in list(active_fds.items()):
                if not _drain_fd(fd, parts):
                    active_fds.pop(fd, None)
            try:
                runner_message = result_queue.get_nowait()
                break
            except queue.Empty:
                pass
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _terminate_runner(process)
                    for fd, parts in list(active_fds.items()):
                        _drain_fd(fd, parts)
                    stdout = "".join(stdout_parts)
                    stderr = "".join(stderr_parts)
                    _debug_event(
                        "sbx.python_runner.execute",
                        code_hash=code_digest,
                        code_len=len(code),
                        timeout=True,
                        timeout_seconds=timeout_seconds,
                        stdout_len=len(stdout),
                        stderr_len=len(stderr),
                        child_pid=process.pid,
                        child_exitcode=process.exitcode,
                        active_fd_count=len(active_fds),
                    )
                    return {
                        "timeout": {"seconds": timeout_seconds},
                        "stdout": stdout,
                        "stderr": stderr,
                    }
                select_timeout = min(0.01, remaining)
            else:
                if not process.is_alive():
                    process.join(timeout=0.5)
                    break
                select_timeout = 0.01
            if active_fds:
                ready, _, _ = await asyncio.to_thread(
                    select.select,
                    list(active_fds),
                    [],
                    [],
                    select_timeout,
                )
                for fd in ready:
                    parts = active_fds.get(fd)
                    if parts is not None and not _drain_fd(fd, parts):
                        active_fds.pop(fd, None)
            else:
                await asyncio.sleep(select_timeout)

        process.join(timeout=0.5)
        if process.is_alive():
            _terminate_runner(process)
        for fd, parts in list(active_fds.items()):
            if not _drain_fd(fd, parts):
                active_fds.pop(fd, None)
    finally:
        for fd in (stdout_read_fd, stderr_read_fd):
            with contextlib.suppress(OSError):
                os.close(fd)
        result_queue.close()

    if runner_message is None:
        exitcode = process.exitcode
        _debug_event(
            "sbx.python_runner.execute",
            code_hash=code_digest,
            code_len=len(code),
            timeout=False,
            error=True,
            error_type="RuntimeError",
            stdout_len=len("".join(stdout_parts)),
            stderr_len=len("".join(stderr_parts)),
            child_pid=process.pid,
            child_exitcode=exitcode,
            active_fd_count=len(active_fds),
        )
        if exitcode is None:
            raise RuntimeError("execution runner exited without a result")
        raise RuntimeError(
            f"execution runner exited without a result (exitcode={exitcode})"
        )
    if not runner_message.get("ok"):
        _debug_event(
            "sbx.python_runner.execute",
            code_hash=code_digest,
            code_len=len(code),
            timeout=False,
            error=True,
            error_type=(runner_message.get("error") or {}).get("type"),
            error_message=(runner_message.get("error") or {}).get("message"),
            stdout_len=len("".join(stdout_parts)),
            stderr_len=len("".join(stderr_parts)),
            child_pid=process.pid,
            child_exitcode=process.exitcode,
            active_fd_count=len(active_fds),
        )
        _raise_runner_error(runner_message.get("error") or {})
    globals_dict.update(runner_message.get("globals") or {})
    result = runner_message.get("result") or {}
    if isinstance(result, dict) and "output" in result:
        result = dict(result)
        result["output"] = "".join(stdout_parts) + "".join(stderr_parts)
    _debug_event(
        "sbx.python_runner.execute",
        code_hash=code_digest,
        code_len=len(code),
        timeout=False,
        error=False,
        result_kind=(
            "timeout"
            if isinstance(result, dict) and "timeout" in result
            else "final"
            if isinstance(result, dict) and "final" in result
            else "output"
        ),
        output_len=len(result.get("output", "")) if isinstance(result, dict) else None,
        stdout_len=len("".join(stdout_parts)),
        stderr_len=len("".join(stderr_parts)),
        child_pid=process.pid,
        child_exitcode=process.exitcode,
        active_fd_count=len(active_fds),
    )
    return result


async def _execute_code_with_timeout(
    code: str,
    globals_dict: dict[str, Any],
    timeout_seconds: float | None,
) -> dict[str, Any]:
    return await _execute_code_in_runner_with_timeout(code, globals_dict, timeout_seconds)


def _mount_file(params: dict[str, Any]) -> dict[str, Any]:
    source = REAL_PATH(params["host_path"])
    target = REAL_PATH(_map_virtual_path(params["virtual_path"]))
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return {}


def _mkdir_p(params: dict[str, Any]) -> dict[str, Any]:
    REAL_PATH(_map_virtual_path(params["path"])).mkdir(parents=True, exist_ok=True)
    return {}


def _list_dir(params: dict[str, Any]) -> dict[str, Any]:
    root = REAL_PATH(_map_virtual_path(params["path"]))
    if not root.exists():
        return {"files": []}
    files = [
        _virtual_from_real(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    return {"files": files}


def _sync_file(params: dict[str, Any]) -> dict[str, Any]:
    source = REAL_PATH(_map_virtual_path(params["virtual_path"]))
    target = REAL_PATH(params["host_path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return {}


async def _handle_request(
    request: dict[str, Any], globals_dict: dict[str, Any]
) -> dict[str, Any] | None:
    request_id = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}

    try:
        if method == "execute":
            return _response(
                request_id,
                await _execute_code_with_timeout(
                    params.get("code", ""),
                    globals_dict,
                    _execution_timeout_seconds(params),
                ),
            )
        if method == "register_output_fields":
            return _response(request_id, {})
        if method == "register_tools":
            return _response(request_id, _register_tools(params, globals_dict))
        if method == "mount_file":
            return _response(request_id, _mount_file(params))
        if method == "mkdir_p":
            return _response(request_id, _mkdir_p(params))
        if method == "list_dir":
            return _response(request_id, _list_dir(params))
        if method == "sync_file":
            return _response(request_id, _sync_file(params))
        if method == "shutdown":
            return _response(request_id, {"shutdown": True})
        raise ValueError(f"Unknown method: {method}")
    except SyntaxError as exc:
        return _error(request_id, exc)
    except BaseException as exc:
        return _error(request_id, exc)


async def _main() -> None:
    globals_dict = _new_globals()

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        if request.get("method") == "reset":
            PENDING_TOOL_RESPONSES.clear()
            globals_dict = _new_globals()
            _send_protocol(_response(request.get("id"), {}))
            continue

        response = await _handle_request(request, globals_dict)
        if response is not None:
            _send_protocol(response)
        if request.get("method") == "shutdown":
            break


if __name__ == "__main__":
    asyncio.run(_main())
