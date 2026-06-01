"""Python JSON-RPC runner used by the Docker Sandboxes backend."""

from __future__ import annotations

import ast
import asyncio
import builtins
import contextlib
import importlib
import inspect
import io
import json
import math
import multiprocessing
import os
import pathlib
import pickle
import queue
import re
import select
import shutil
import signal
import sys
import tempfile
import time
from typing import Any, Callable

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
RUNTIME_HOOK_ORIGINALS: dict[str, tuple[Any, str, Callable[..., Any]]] = {}
RUNTIME_HOOK_SPECS: dict[str, set[str]] = {}
RUNTIME_HOOKS_ENABLED = False
PREDICT_SCHEMA_BUILTINS = {
    "Any",
    "BaseModel",
    "Dict",
    "Image",
    "List",
    "Literal",
    "Optional",
    "Set",
    "Tuple",
    "Union",
}


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
        return _to_jsonable(value.model_dump())
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
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


class _SandboxPath(REAL_PATH):
    def __new__(cls, *args: Any, **kwargs: Any):
        if args:
            args = (_map_virtual_path(args[0]), *args[1:])
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args:
            args = (_map_virtual_path(args[0]), *args[1:])
        super().__init__(*args, **kwargs)


def _submit(**kwargs: Any) -> None:
    raise _FinalOutputError(_to_jsonable(kwargs))


def _send_protocol(message: dict[str, Any]) -> None:
    PROTOCOL_STDOUT.write(json.dumps(message, default=str) + "\n")
    PROTOCOL_STDOUT.flush()


def _summarize_hook_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 2:
        return {"type": type(value).__name__, "repr": _short_repr(value)}
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, str) and len(value) > 500:
            return value[:500] + f"... ({len(value)} chars)"
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if type(value).__name__ == "CompletedProcess":
        return {
            "type": "CompletedProcess",
            "args": _summarize_hook_value(getattr(value, "args", None), depth=depth + 1),
            "returncode": getattr(value, "returncode", None),
            "stdout_chars": len(getattr(value, "stdout", "") or ""),
            "stderr_chars": len(getattr(value, "stderr", "") or ""),
        }
    if isinstance(value, bytes):
        if len(value) > 128:
            return {"type": "bytes", "len": len(value), "preview": value[:128].hex()}
        return {"type": "bytes", "len": len(value), "preview": value.hex()}
    if isinstance(value, (list, tuple)):
        return [_summarize_hook_value(item, depth=depth + 1) for item in list(value)[:20]]
    if isinstance(value, dict):
        return {
            str(key): _summarize_hook_value(val, depth=depth + 1)
            for key, val in list(value.items())[:20]
        }
    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": str(value)}
    return {"type": type(value).__name__, "repr": _short_repr(value)}


def _short_repr(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 500 else text[:500] + f"... ({len(text)} chars)"


def _emit_runtime_hook_event(
    target: str,
    phase: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    result: Any = None,
    error: BaseException | None = None,
    duration_ms: int | None = None,
) -> None:
    if not RUNTIME_HOOKS_ENABLED:
        return
    if phase not in RUNTIME_HOOK_SPECS.get(target, set()):
        return
    params = {
        "target": target,
        "phase": phase,
        "args": [_summarize_hook_value(arg) for arg in args],
        "kwargs": _summarize_hook_value(kwargs),
        "result": _summarize_hook_value(result),
        "error": str(error) if error is not None else None,
        "duration_ms": duration_ms,
        "timestamp": time.time(),
    }
    _send_protocol({"jsonrpc": "2.0", "method": "runtime_hook_event", "params": params})


def _runtime_hook_owner(target: str) -> tuple[Any, str]:
    if target == "builtins.open":
        return builtins, "open"
    if target.startswith("pathlib.Path."):
        return REAL_PATH, target.rsplit(".", 1)[1]
    return _resolve_dotted_runtime_hook_owner(target)


def _resolve_dotted_runtime_hook_owner(target: str) -> tuple[Any, str]:
    parts = target.split(".")
    if len(parts) < 2 or any(not part for part in parts):
        raise ValueError(f"Runtime hook target must be a dotted path: {target}")
    for module_end in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:module_end])
        try:
            owner = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name or module_name.startswith(f"{exc.name}."):
                continue
            raise ValueError(
                f"Runtime hook target module import failed: {module_name}"
            ) from exc
        except Exception as exc:
            raise ValueError(
                f"Runtime hook target module import failed: {module_name}"
            ) from exc
        for attr in parts[module_end:-1]:
            try:
                owner = getattr(owner, attr)
            except AttributeError as exc:
                raise ValueError(f"Runtime hook target does not exist: {target}") from exc
        return owner, parts[-1]
    raise ValueError(f"Unsupported runtime hook target: {target}")


def _runtime_hook_callable(target: str) -> tuple[Any, str, Callable[..., Any]]:
    owner, attr = _runtime_hook_owner(target)
    try:
        original = getattr(owner, attr)
    except AttributeError as exc:
        raise ValueError(f"Runtime hook target does not exist: {target}") from exc
    if not callable(original):
        raise ValueError(f"Runtime hook target is not callable: {target}")
    return owner, attr, original


def _restore_runtime_hooks() -> None:
    for target, (owner, attr, original) in reversed(list(RUNTIME_HOOK_ORIGINALS.items())):
        setattr(owner, attr, original)
    RUNTIME_HOOK_ORIGINALS.clear()
    RUNTIME_HOOK_SPECS.clear()


def _make_runtime_hook_wrapper(target: str, original: Callable[..., Any]) -> Callable[..., Any]:
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        _emit_runtime_hook_event(target, "before", args, kwargs)
        started = time.perf_counter()
        try:
            result = original(*args, **kwargs)
        except BaseException as exc:
            _emit_runtime_hook_event(
                target,
                "error",
                args,
                kwargs,
                error=exc,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
            raise
        _emit_runtime_hook_event(
            target,
            "after",
            args,
            kwargs,
            result=result,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        return result

    return _wrapped


def _runtime_hook_specs_by_target(params: dict[str, Any]) -> dict[str, set[str]]:
    specs_by_target: dict[str, set[str]] = {}
    for hook in params.get("hooks", []):
        target = hook.get("target")
        if not isinstance(target, str):
            continue
        phases = hook.get("phases") or ["before"]
        specs_by_target.setdefault(target, set()).update(str(phase) for phase in phases)
    return specs_by_target


def _register_runtime_hooks(params: dict[str, Any]) -> dict[str, Any]:
    _restore_runtime_hooks()
    planned_hooks = []
    for target, phases in _runtime_hook_specs_by_target(params).items():
        owner, attr, original = _runtime_hook_callable(target)
        planned_hooks.append((target, phases, owner, attr, original))
    for target, phases, owner, attr, original in planned_hooks:
        RUNTIME_HOOK_SPECS[target] = phases
        RUNTIME_HOOK_ORIGINALS[target] = (owner, attr, original)
        setattr(owner, attr, _make_runtime_hook_wrapper(target, original))
    return {}


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


def _lookup_type(name: str, globals_dict: dict[str, Any]) -> Any:
    if name in globals_dict:
        return globals_dict[name]

    main_module = sys.modules.get("__main__")
    if main_module is not None and name in main_module.__dict__:
        return main_module.__dict__[name]

    frame = inspect.currentframe()
    while frame is not None:
        if name in frame.f_globals:
            return frame.f_globals[name]
        if name in frame.f_locals:
            return frame.f_locals[name]
        frame = frame.f_back

    return None


def _get_pydantic_schemas(signature: str, globals_dict: dict[str, Any]) -> dict[str, dict]:
    schemas: dict[str, dict] = {}
    for match in re.finditer(r"(?<![.\w])([A-Z][A-Za-z0-9_]*)", signature):
        name = match.group(1)
        if name in PREDICT_SCHEMA_BUILTINS:
            continue

        model_type = _lookup_type(name, globals_dict)
        if model_type is None or not hasattr(model_type, "model_json_schema"):
            continue

        schema = model_type.model_json_schema()
        json.dumps(schema)
        schemas[name] = schema

    return schemas


def _predict_signature(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    signature = args[0] if args else kwargs["signature"]
    if not isinstance(signature, str):
        raise TypeError(f"predict signature must be str, got {type(signature).__name__}")
    return signature


def _reconstruct_output_types(
    signature: str,
    result: Any,
    globals_dict: dict[str, Any],
) -> Any:
    if isinstance(result, _PredictResult):
        result = result.to_dict()
    if not isinstance(result, dict):
        return result

    outputs_part = signature.split("->", 1)[1] if "->" in signature else ""
    pattern = r"(\w+)\s*:\s*((?:Optional\[|list\[|List\[)*)\s*([A-Z][A-Za-z0-9_]*)"
    for match in re.finditer(pattern, outputs_part):
        field_name = match.group(1)
        wrapper = match.group(2) or ""
        type_name = match.group(3)
        if type_name in PREDICT_SCHEMA_BUILTINS:
            continue

        model_type = _lookup_type(type_name, globals_dict)
        if (
            model_type is None
            or not hasattr(model_type, "model_validate")
            or not hasattr(model_type, "model_fields")
        ):
            continue

        from pydantic import ConfigDict

        output_type = type(
            model_type.__name__,
            (model_type,),
            {"model_config": ConfigDict(extra="allow")},
        )
        value = result.get(field_name)
        if value is None:
            continue
        if "list[" in wrapper.lower() and isinstance(value, list):
            if all(isinstance(item, dict) for item in value):
                result[field_name] = [output_type.model_validate(item) for item in value]
        elif isinstance(value, dict):
            result[field_name] = output_type.model_validate(value)

    return _PredictResult(result)


async def _call_predict_tool(
    *args: Any,
    globals_dict: dict[str, Any],
    **kwargs: Any,
) -> Any:
    signature = _predict_signature(args, kwargs)
    safe_args = _to_jsonable(list(args))
    safe_kwargs = _to_jsonable(kwargs)

    pydantic_schemas = _get_pydantic_schemas(signature, globals_dict)
    if pydantic_schemas:
        safe_kwargs["pydantic_schemas"] = pydantic_schemas

    result = await _call_host_tool("predict", *safe_args, **safe_kwargs)
    return _reconstruct_output_types(signature, result, globals_dict)


async def _drain_execution_tasks(baseline_tasks: set[asyncio.Task]) -> None:
    current_task = asyncio.current_task()
    pending_tasks = [
        task
        for task in asyncio.all_tasks()
        if task is not current_task and task not in baseline_tasks and not task.done()
    ]
    if pending_tasks:
        await asyncio.gather(*pending_tasks, return_exceptions=True)


def _register_tools(params: dict[str, Any], globals_dict: dict[str, Any]) -> dict[str, Any]:
    for name in params.get("tools", []):
        if name == "predict":
            async def _tool(*args: Any, **kwargs: Any) -> Any:
                return await _call_predict_tool(
                    *args,
                    globals_dict=globals_dict,
                    **kwargs,
                )
        else:
            async def _tool(*args: Any, __tool_name: str = name, **kwargs: Any) -> Any:
                return await _call_host_tool(__tool_name, *args, **kwargs)

        globals_dict[name] = _tool
    return {}


def _install_virtual_filesystem(globals_dict: dict[str, Any]) -> None:
    builtins.open = _open
    pathlib.Path = _SandboxPath  # type: ignore[assignment]
    globals_dict.setdefault("SUBMIT", _submit)


def _new_globals() -> dict[str, Any]:
    _restore_runtime_hooks()
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
    return {
        "jsonrpc": "2.0",
        "error": {
            "code": -32000,
            "message": str(exc),
            "data": _exception_payload(exc),
        },
        "id": request_id,
    }


async def _execute_code(
    code: str,
    globals_dict: dict[str, Any],
    capture: _ExecutionCapture | None = None,
) -> dict[str, Any]:
    capture = capture or _ExecutionCapture()
    baseline_tasks = set(asyncio.all_tasks())
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
    except BaseException:
        await _drain_execution_tasks(baseline_tasks)
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
    global RUNTIME_HOOKS_ENABLED
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
        RUNTIME_HOOKS_ENABLED = True
        result = asyncio.run(_execute_code(code, globals_dict, capture))
        RUNTIME_HOOKS_ENABLED = False
        result_queue.put({
            "ok": True,
            "result": result,
            "globals": _pickleable_globals(globals_dict),
            "sys_path": list(sys.path),
        })
    except BaseException as exc:
        RUNTIME_HOOKS_ENABLED = False
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
                    return {
                        "timeout": {"seconds": timeout_seconds},
                        "stdout": "".join(stdout_parts),
                        "stderr": "".join(stderr_parts),
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
        if exitcode is None:
            raise RuntimeError("execution runner exited without a result")
        raise RuntimeError(
            f"execution runner exited without a result (exitcode={exitcode})"
        )
    if not runner_message.get("ok"):
        _raise_runner_error(runner_message.get("error") or {})
    globals_dict.update(runner_message.get("globals") or {})
    sys.path[:] = runner_message["sys_path"]
    result = runner_message.get("result") or {}
    if isinstance(result, dict) and "output" in result:
        result = dict(result)
        result["output"] = "".join(stdout_parts) + "".join(stderr_parts)
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
        if method == "register_runtime_hooks":
            return _response(request_id, _register_runtime_hooks(params))
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
