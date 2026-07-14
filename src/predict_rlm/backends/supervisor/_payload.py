"""Executable Python JSON-RPC supervisor payload for native backends."""

from __future__ import annotations

import ast
import asyncio
import builtins
import collections.abc
import contextlib
import dataclasses
import hashlib
import importlib
import inspect
import io
import json
import math
import multiprocessing
import os
import pathlib
import queue
import re
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from argparse import ArgumentParser
from typing import Any

PROTOCOL_STDIN = sys.stdin
PROTOCOL_STDOUT = sys.stdout
REAL_OPEN = builtins.open
REAL_PATH = type(pathlib.Path())
ORIGINAL_PATH = pathlib.Path
SANDBOX_ROOT = REAL_PATH(
    os.environ.get("PREDICT_RLM_RUNNER_ROOT")
    or os.environ.get("PREDICT_RLM_SBX_ROOT")
    or tempfile.mkdtemp(prefix="predict-rlm-sbx-runner-")
).resolve()
SANDBOX_DIR = SANDBOX_ROOT / "sandbox"
SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
TOOL_REQUEST_ID = 0
PENDING_TOOL_RESPONSES: dict[int, dict[str, Any]] = {}
TOOL_RESPONSE_CONDITION = threading.Condition()
TOOL_RESPONSE_READER_THREAD: threading.Thread | None = None
TOOL_RESPONSE_READER_ERROR: BaseException | None = None
TOOL_RESPONSE_READ_BUFFER = ""
WAITING_TOOL_RESPONSE_IDS: set[int] = set()
HOST_TOOL_REQUEST_QUEUE: multiprocessing.Queue | None = None
HOST_TOOL_RESPONSE_QUEUE: multiprocessing.Queue | None = None
_TRUE_VALUES = {"1", "true", "yes", "on"}
_KERNEL_PROCESS: multiprocessing.Process | None = None
_KERNEL_REQUEST_QUEUE: multiprocessing.Queue | None = None
_KERNEL_RESULT_QUEUE: multiprocessing.Queue | None = None
# Interrupt requests are handled out of band from the serial request queue so
# the websocket receiver can abort a cell while the run loop waits on execute.
_INTERRUPT_REQUESTED = False
_EXECUTION_ACTIVE = False
_DEFAULT_TIMEOUT_INTERRUPT_GRACE_SECONDS = 0.5
_INTERNAL_GLOBAL_NAMES = {
    "SUBMIT",
    "__predict_rlm_tool_names__",
}
RUNTIME_HOOK_ORIGINALS: dict[str, tuple[Any, str, Any]] = {}
RUNTIME_HOOK_SPECS: dict[str, set[str]] = {}
RUNTIME_HOOKS_ENABLED = False


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
    rel_text = rel.as_posix()
    if rel_text == ".":
        return "/sandbox"
    return "/sandbox/" + rel_text


def _snapshot_path_value(value: pathlib.PurePath) -> pathlib.PurePath | str:
    try:
        return _virtual_from_real(REAL_PATH(os.fspath(value)))
    except ValueError:
        return value


def _open(path: Any, *args: Any, **kwargs: Any):
    return REAL_OPEN(_map_virtual_path(path), *args, **kwargs)


class _SandboxPath(REAL_PATH):
    def __new__(cls, *args: Any, **kwargs: Any):
        if args:
            args = (_map_virtual_path(args[0]), *args[1:])
        current_path = pathlib.Path
        pathlib.Path = ORIGINAL_PATH  # type: ignore[assignment]
        try:
            return super().__new__(cls, *args, **kwargs)
        finally:
            pathlib.Path = current_path  # type: ignore[assignment]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if sys.version_info < (3, 12):
            return
        if args:
            args = (_map_virtual_path(args[0]), *args[1:])
        current_path = pathlib.Path
        pathlib.Path = ORIGINAL_PATH  # type: ignore[assignment]
        try:
            super().__init__(*args, **kwargs)
        finally:
            pathlib.Path = current_path  # type: ignore[assignment]


def _submit(**kwargs: Any) -> None:
    raise _FinalOutputError(_to_jsonable(kwargs))


def _short_repr(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 500 else text[:500] + f"... ({len(text)} chars)"


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


def _runtime_hook_callable(target: str) -> tuple[Any, str, Any]:
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


def _make_runtime_hook_wrapper(target: str, original: Any) -> Any:
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


def _send_protocol(message: dict[str, Any]) -> None:
    if HOST_TOOL_REQUEST_QUEUE is not None:
        HOST_TOOL_REQUEST_QUEUE.put(message)
        return
    PROTOCOL_STDOUT.write(json.dumps(message, default=str) + "\n")
    PROTOCOL_STDOUT.flush()


def _read_protocol_response_line(timeout: float | None = None) -> dict[str, Any] | None:
    if HOST_TOOL_RESPONSE_QUEUE is not None:
        try:
            return HOST_TOOL_RESPONSE_QUEUE.get(timeout=timeout)
        except queue.Empty:
            return None
    global TOOL_RESPONSE_READ_BUFFER
    fd = PROTOCOL_STDIN.fileno()
    while True:
        while "\n" not in TOOL_RESPONSE_READ_BUFFER:
            if timeout is not None:
                ready, _, _ = select.select([fd], [], [], timeout)
                if not ready:
                    return None
            chunk = os.read(fd, 65536)
            if not chunk:
                raise RuntimeError("Host closed stdin while waiting for tool response")
            TOOL_RESPONSE_READ_BUFFER += chunk.decode("utf-8", errors="replace")
        line, TOOL_RESPONSE_READ_BUFFER = TOOL_RESPONSE_READ_BUFFER.split("\n", 1)
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue


async def _read_protocol_response(request_id: int) -> dict[str, Any]:
    with TOOL_RESPONSE_CONDITION:
        WAITING_TOOL_RESPONSE_IDS.add(request_id)
    _ensure_tool_response_reader()
    try:
        while True:
            with TOOL_RESPONSE_CONDITION:
                if request_id in PENDING_TOOL_RESPONSES:
                    return PENDING_TOOL_RESPONSES.pop(request_id)
                if TOOL_RESPONSE_READER_ERROR is not None:
                    raise TOOL_RESPONSE_READER_ERROR
            await asyncio.sleep(0.01)
    finally:
        with TOOL_RESPONSE_CONDITION:
            WAITING_TOOL_RESPONSE_IDS.discard(request_id)
            TOOL_RESPONSE_CONDITION.notify_all()


def _ensure_tool_response_reader() -> None:
    global TOOL_RESPONSE_READER_THREAD
    if (
        TOOL_RESPONSE_READER_THREAD is not None
        and TOOL_RESPONSE_READER_THREAD.is_alive()
    ):
        return
    TOOL_RESPONSE_READER_THREAD = threading.Thread(
        target=_tool_response_reader_loop,
        name="predict-rlm-tool-response-reader",
        daemon=True,
    )
    TOOL_RESPONSE_READER_THREAD.start()
    _debug_event("sbx.python_runner.tool_response_reader.start")


def _tool_response_reader_loop() -> None:
    global TOOL_RESPONSE_READER_ERROR, TOOL_RESPONSE_READER_THREAD
    try:
        while True:
            with TOOL_RESPONSE_CONDITION:
                if not WAITING_TOOL_RESPONSE_IDS:
                    TOOL_RESPONSE_READER_THREAD = None
                    TOOL_RESPONSE_CONDITION.notify_all()
                    return
            response = _read_protocol_response_line(timeout=0.05)
            if response is None:
                continue
            response_id = response.get("id")
            _debug_event(
                "sbx.python_runner.tool_response_reader.response",
                response_id=response_id,
            )
            if isinstance(response_id, int):
                with TOOL_RESPONSE_CONDITION:
                    PENDING_TOOL_RESPONSES[response_id] = response
                    TOOL_RESPONSE_CONDITION.notify_all()
    except BaseException as exc:
        with TOOL_RESPONSE_CONDITION:
            TOOL_RESPONSE_READER_ERROR = exc
            TOOL_RESPONSE_READER_THREAD = None
            TOOL_RESPONSE_CONDITION.notify_all()


def _wait_for_idle_tool_response_reader() -> None:
    with TOOL_RESPONSE_CONDITION:
        while (
            TOOL_RESPONSE_READER_THREAD is not None
            and not WAITING_TOOL_RESPONSE_IDS
        ):
            TOOL_RESPONSE_CONDITION.wait()


def _publish_kernel_result(result_queue: multiprocessing.Queue, result: dict[str, Any]) -> None:
    _wait_for_idle_tool_response_reader()
    result_queue.put(result)


def _reset_tool_protocol_state() -> None:
    global PENDING_TOOL_RESPONSES, TOOL_REQUEST_ID, TOOL_RESPONSE_CONDITION
    global TOOL_RESPONSE_READER_ERROR, TOOL_RESPONSE_READER_THREAD
    global TOOL_RESPONSE_READ_BUFFER
    global WAITING_TOOL_RESPONSE_IDS
    PENDING_TOOL_RESPONSES = {}
    TOOL_RESPONSE_CONDITION = threading.Condition()
    TOOL_RESPONSE_READER_THREAD = None
    TOOL_RESPONSE_READER_ERROR = None
    TOOL_RESPONSE_READ_BUFFER = ""
    WAITING_TOOL_RESPONSE_IDS = set()
    TOOL_REQUEST_ID = os.getpid() << 32


def _has_waiting_tool_responses() -> bool:
    with TOOL_RESPONSE_CONDITION:
        return bool(WAITING_TOOL_RESPONSE_IDS)


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


_PREDICT_SIGNATURE_BUILTIN_TYPES = {
    "Image", "List", "Optional", "Dict", "Any", "Union", "Literal", "Tuple", "Set", "BaseModel",
}


def _predict_pydantic_schemas(
    signature: Any, user_globals: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Extract JSON schemas for custom Pydantic types named in a predict() signature.

    The host builds the structured-output signature, so it needs each custom type's
    schema; without it the host can't resolve the name and falls back to a plain
    string signature. Resolves capitalized type names from the kernel's execution
    globals first -- that's where REPL top-level classes live, and it works inside
    asyncio.gather where call-stack introspection can't reach the defining frame --
    then falls back to the call stack. Mirrors the JSPI backend so SBX predict()
    handles custom output types on par.
    """
    if not isinstance(signature, str):
        return {}
    schemas: dict[str, Any] = {}
    for match in re.finditer(r"(?<![.\w])([A-Z][A-Za-z0-9_]*)", signature):
        name = match.group(1)
        if name in _PREDICT_SIGNATURE_BUILTIN_TYPES or name in schemas:
            continue
        cls = None
        if user_globals is not None and name in user_globals:
            cls = user_globals[name]
        if cls is None:
            frame = inspect.currentframe()
            while frame is not None:
                if name in frame.f_globals:
                    cls = frame.f_globals[name]
                    break
                if name in frame.f_locals:
                    cls = frame.f_locals[name]
                    break
                frame = frame.f_back
        if cls is not None and hasattr(cls, "model_json_schema"):
            try:
                schemas[name] = _to_jsonable(_model_json_schema(cls, user_globals))
            except Exception:
                continue
    return schemas


def _model_json_schema(cls: Any, user_globals: dict[str, Any] | None) -> Any:
    """``cls.model_json_schema()``, rebuilding against the user's globals if needed.

    User code execs in a dedicated globals_dict tagged ``__name__='__main__'``,
    but ``sys.modules['__main__']`` is the payload's own module — so a model that
    references sibling models (``list[PageItem]``) can't resolve them by module
    name and schema generation raises "not fully defined". Rebuilding with the
    execution globals as the types namespace makes those siblings resolvable,
    mirroring how Pyodide/JSPI runs user code directly in ``__main__``.
    """
    try:
        return cls.model_json_schema()
    except Exception:
        rebuild = getattr(cls, "model_rebuild", None)
        if rebuild is None:
            raise
        rebuild(force=True, _types_namespace=dict(user_globals or {}))
        return cls.model_json_schema()


def _reconstruct_output_types(
    signature: Any, result: Any, user_globals: dict[str, Any]
) -> Any:
    """Revive custom-model output fields of a predict() result into real instances.

    The host serializes Pydantic outputs to dicts for JSON transport, so a field
    declared ``-> finding: PageFinding`` arrives as a plain dict and ``finding.page``
    would raise ``'dict' object has no attribute 'page'``. For each model-typed output
    field this validates the dict back into the user's class (an ``extra='allow'``
    subclass, matching the JSPI/Deno backend so fields the LM adds beyond the schema
    survive). Validation errors propagate -- a malformed output is a real problem the
    caller should see, not something to silently paper over.
    """
    if not isinstance(result, _PredictResult):
        return result
    outputs_part = signature.split("->", 1)[1] if isinstance(signature, str) and "->" in signature else ""
    for match in re.finditer(
        r"(\w+)\s*:\s*((?:Optional\[|list\[|List\[)*)\s*([A-Z][A-Za-z0-9_]*)", outputs_part
    ):
        field_name, wrapper, type_name = match.group(1), match.group(2) or "", match.group(3)
        if type_name in _PREDICT_SIGNATURE_BUILTIN_TYPES or field_name not in result:
            continue
        value = result[field_name]
        if value is None:
            continue
        cls = user_globals.get(type_name)
        if cls is None or not hasattr(cls, "model_validate") or not hasattr(cls, "model_fields"):
            continue
        from pydantic import ConfigDict as _ConfigDict

        cls = type(cls.__name__, (cls,), {"model_config": _ConfigDict(extra="allow")})
        if "list[" in wrapper.lower() and isinstance(value, list):
            result[field_name] = [cls.model_validate(item) for item in value]
        elif isinstance(value, dict):
            result[field_name] = cls.model_validate(value)
    return result


def _register_tools(params: dict[str, Any], globals_dict: dict[str, Any]) -> dict[str, Any]:
    tool_names = globals_dict.setdefault("__predict_rlm_tool_names__", set())
    for name in params.get("tools", []):
        if name == "predict":
            async def _predict_tool(*args: Any, _globals: dict[str, Any] = globals_dict, **kwargs: Any) -> Any:
                signature = args[0] if args else kwargs.get("signature", "")
                if "pydantic_schemas" not in kwargs:
                    schemas = _predict_pydantic_schemas(signature, _globals)
                    if schemas:
                        kwargs = {**kwargs, "pydantic_schemas": schemas}
                result = await _call_host_tool("predict", *args, **kwargs)
                return _reconstruct_output_types(signature, result, _globals)

            globals_dict[name] = _predict_tool
        else:
            async def _tool(*args: Any, __tool_name: str = name, **kwargs: Any) -> Any:
                return await _call_host_tool(__tool_name, *args, **kwargs)

            globals_dict[name] = _tool
        tool_names.add(name)
    _discard_kernel()
    return {}


def _install_virtual_filesystem(globals_dict: dict[str, Any]) -> None:
    builtins.open = _open
    pathlib.Path = _SandboxPath  # type: ignore[assignment]
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
    *,
    defer_final_output: bool = False,
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
        if defer_final_output:
            return {"submitted": final.payload}
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


def _timeout_interrupt_grace_seconds(params: dict[str, Any]) -> float:
    grace = params.get(
        "execution_timeout_interrupt_grace_seconds",
        _DEFAULT_TIMEOUT_INTERRUPT_GRACE_SECONDS,
    )
    if (
        isinstance(grace, bool)
        or not isinstance(grace, (int, float))
        or not math.isfinite(float(grace))
        or float(grace) < 0
    ):
        raise ValueError(
            "execution_timeout_interrupt_grace_seconds must be a non-negative number"
        )
    return float(grace)


def _live_kernel_state() -> dict[str, Any]:
    return {
        "preserved": True,
        "source": "live_kernel",
        "scope": "full_live",
    }


def _pickle_snapshot_state(
    snapshot: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "preserved": False,
        "source": "pickle_snapshot",
        "scope": "pickleable_globals",
        "reason": reason,
        "restored_globals": list(snapshot.get("restored_globals") or []),
        "lost_globals": list(snapshot.get("lost_globals") or []),
    }


def _build_interrupt_result(
    timeout_seconds: float | None,
    oob_interrupted: bool,
    stdout: str,
    stderr: str,
    snapshot: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    """Build the recoverable timeout-style result for an interrupted cell."""
    timeout_info: dict[str, Any] = {"seconds": timeout_seconds}
    if oob_interrupted:
        timeout_info["interrupted"] = True
    return {
        "timeout": timeout_info,
        "stdout": stdout,
        "stderr": stderr,
        "state": _pickle_snapshot_state(snapshot, reason),
    }


def _is_user_global(name: str, globals_dict: dict[str, Any]) -> bool:
    if name.startswith("__") and name.endswith("__"):
        return False
    if name in _INTERNAL_GLOBAL_NAMES:
        return False
    tool_names = globals_dict.get("__predict_rlm_tool_names__", set())
    return name not in tool_names


_SAFE_SNAPSHOT_SCALAR_TYPES = (type(None), bool, int, float, str, bytes)


def _safe_snapshot_value(value: Any, seen: set[int] | None = None) -> tuple[bool, Any]:
    if isinstance(value, _VirtualPath):
        return True, value.virtual_path
    if isinstance(value, _SAFE_SNAPSHOT_SCALAR_TYPES):
        return True, value
    if isinstance(value, pathlib.PurePath):
        return True, _snapshot_path_value(value)
    if inspect.ismodule(value) or inspect.isfunction(value) or inspect.isclass(value):
        return False, None

    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return False, None

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        seen.add(value_id)
        fields: dict[str, Any] = {}
        try:
            dataclass_fields = dataclasses.fields(value)
        except TypeError:
            return False, None
        for field in dataclass_fields:
            try:
                field_value = getattr(value, field.name)
            except Exception:
                return False, None
            ok, snapshot = _safe_snapshot_value(field_value, seen)
            if not ok:
                return False, None
            fields[field.name] = snapshot
        seen.discard(value_id)
        return True, fields

    if isinstance(value, list):
        seen.add(value_id)
        items = []
        for item in value:
            ok, snapshot = _safe_snapshot_value(item, seen)
            if not ok:
                return False, None
            items.append(snapshot)
        seen.discard(value_id)
        return True, items

    if isinstance(value, tuple):
        seen.add(value_id)
        items = []
        for item in value:
            ok, snapshot = _safe_snapshot_value(item, seen)
            if not ok:
                return False, None
            items.append(snapshot)
        seen.discard(value_id)
        return True, tuple(items)

    if isinstance(value, (set, frozenset)):
        seen.add(value_id)
        items = []
        for item in value:
            ok, snapshot = _safe_snapshot_value(item, seen)
            if not ok:
                return False, None
            try:
                hash(snapshot)
            except TypeError:
                return False, None
            items.append(snapshot)
        seen.discard(value_id)
        return True, type(value)(items)

    if not isinstance(value, collections.abc.Mapping):
        return False, None
    seen.add(value_id)
    mapping: dict[Any, Any] = {}
    for key, item in value.items():
        key_ok, key_snapshot = _safe_snapshot_value(key, seen)
        item_ok, item_snapshot = _safe_snapshot_value(item, seen)
        if not key_ok or not item_ok:
            return False, None
        try:
            hash(key_snapshot)
        except TypeError:
            return False, None
        mapping[key_snapshot] = item_snapshot
    seen.discard(value_id)
    return True, mapping


def _pickleable_globals_snapshot(globals_dict: dict[str, Any]) -> dict[str, Any]:
    restored: dict[str, Any] = {}
    lost: list[str] = []
    for name, value in sorted(globals_dict.items()):
        if not _is_user_global(name, globals_dict):
            continue
        ok, snapshot = _safe_snapshot_value(value)
        if not ok:
            lost.append(name)
            continue
        restored[name] = snapshot
    return {
        "globals": restored,
        "restored_globals": sorted(restored),
        "lost_globals": sorted(lost),
    }


def _reset_globals_from_pickle_snapshot(
    globals_dict: dict[str, Any],
    snapshot: dict[str, Any],
) -> None:
    tool_names = set(globals_dict.get("__predict_rlm_tool_names__", set()))
    tools = {
        name: globals_dict[name]
        for name in tool_names
        if name in globals_dict
    }
    globals_dict.clear()
    globals_dict.update(_new_globals())
    if tool_names:
        globals_dict["__predict_rlm_tool_names__"] = tool_names
        globals_dict.update(tools)
    globals_dict.update(snapshot.get("globals") or {})


def _raise_runner_error(payload: dict[str, Any]) -> None:
    raise _RunnerExecutionError(payload)


def _runner_context() -> multiprocessing.context.BaseContext:
    try:
        return multiprocessing.get_context("fork")
    except ValueError as exc:
        raise RuntimeError("isolated execution requires a fork-capable Python runtime") from exc


def _capture_file_path(kind: str) -> pathlib.Path:
    handle = tempfile.NamedTemporaryFile(
        prefix=f"predict-rlm-{kind}-",
        suffix=".txt",
        dir=SANDBOX_ROOT,
        delete=False,
    )
    path = pathlib.Path(handle.name)
    handle.close()
    return path


def _read_capture_file(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _unlink_capture_files(*paths: pathlib.Path) -> None:
    for path in paths:
        with contextlib.suppress(OSError):
            path.unlink()


@contextlib.contextmanager
def _redirect_process_stdio_to_files(
    stdout_path: str,
    stderr_path: str,
):
    saved_stdin_fd = os.dup(0)
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    stdout_fd = os.open(stdout_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    stderr_fd = os.open(stderr_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    devnull_stdin = REAL_OPEN(os.devnull, "r", encoding="utf-8")
    old_stdin, old_stdout, old_stderr = sys.stdin, sys.stdout, sys.stderr
    capture = _ExecutionCapture(
        stdout=_FdTextStream(stdout_fd),
        stderr=_FdTextStream(stderr_fd),
    )
    try:
        os.dup2(devnull_stdin.fileno(), 0)
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        sys.stdin = devnull_stdin
        yield capture
    finally:
        for stream in (sys.stdout, sys.stderr):
            with contextlib.suppress(Exception):
                stream.flush()
        os.dup2(saved_stdin_fd, 0)
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        sys.stdin, sys.stdout, sys.stderr = old_stdin, old_stdout, old_stderr
        for fd in (
            stdout_fd,
            stderr_fd,
            saved_stdin_fd,
            saved_stdout_fd,
            saved_stderr_fd,
        ):
            with contextlib.suppress(OSError):
                os.close(fd)
        devnull_stdin.close()


async def _execute_code_to_capture_files(
    code: str,
    globals_dict: dict[str, Any],
    stdout_path: str,
    stderr_path: str,
    *,
    defer_final_output: bool = False,
) -> dict[str, Any]:
    global RUNTIME_HOOKS_ENABLED
    with _redirect_process_stdio_to_files(stdout_path, stderr_path) as capture:
        RUNTIME_HOOKS_ENABLED = True
        try:
            result = await _execute_code(
                code,
                globals_dict,
                capture,
                defer_final_output=defer_final_output,
            )
        except BaseException as exc:
            setattr(
                exc,
                "_predict_rlm_output",
                _read_capture_file(pathlib.Path(stdout_path))
                + _read_capture_file(pathlib.Path(stderr_path)),
            )
            raise
        finally:
            RUNTIME_HOOKS_ENABLED = False
    if isinstance(result, dict) and "output" in result:
        result = dict(result)
        result["output"] = _read_capture_file(pathlib.Path(stdout_path)) + _read_capture_file(
            pathlib.Path(stderr_path)
        )
    return result


def _run_kernel_coroutine(coro: Any) -> Any:
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        # Cancel any tasks the code block left pending (e.g. predict() calls
        # orphaned when an asyncio.gather() raised early on a sibling). Without
        # this they leak across executes — their in-flight tool calls desync the
        # host<->kernel protocol and the next predict() hangs. Mirrors what
        # asyncio.run() does via _cancel_all_tasks before closing the loop.
        try:
            _cancel_all_kernel_tasks(loop)
        finally:
            asyncio.set_event_loop(None)
            loop.close()


def _cancel_all_kernel_tasks(loop: Any) -> None:
    to_cancel = [t for t in asyncio.all_tasks(loop) if not t.done()]
    if not to_cancel:
        return
    for task in to_cancel:
        task.cancel()
    loop.run_until_complete(asyncio.gather(*to_cancel, return_exceptions=True))


def _descendant_process_ids(parent_pid: int) -> list[int]:
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,ppid="],
            check=False,
            capture_output=True,
            text=True,
            timeout=1,
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []

    children_by_parent: dict[int, list[int]] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            pid, ppid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        children_by_parent.setdefault(ppid, []).append(pid)

    descendants: list[int] = []
    pending = list(children_by_parent.get(parent_pid, []))
    while pending:
        pid = pending.pop()
        descendants.append(pid)
        pending.extend(children_by_parent.get(pid, []))
    return descendants


def _reap_direct_children(child_pids: set[int]) -> None:
    deadline = time.monotonic() + 0.5
    while child_pids and time.monotonic() < deadline:
        for pid in tuple(child_pids):
            try:
                reaped, _ = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                child_pids.discard(pid)
                continue
            except OSError:
                child_pids.discard(pid)
                continue
            if reaped:
                child_pids.discard(pid)
        if child_pids:
            time.sleep(0.02)


def _terminate_descendant_processes(parent_pid: int | None = None) -> None:
    child_pids = set(_descendant_process_ids(parent_pid or os.getpid()))
    if not child_pids:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in tuple(child_pids):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                child_pids.discard(pid)
            except OSError:
                child_pids.discard(pid)
        _reap_direct_children(child_pids)


def _start_parent_death_watchdog(poll_seconds: float = 0.25) -> None:
    """Force-exit this kernel child if its supervisor parent dies.

    The kernel child ``setsid()``'s into its own session so it can manage its
    own subprocess descendants; that also means a force-kill of the supervisor
    (for example host-side iteration-timeout recovery) does not reach it.
    Without this watchdog an orphaned kernel child blocked awaiting tool
    responses that will never arrive would live forever, leaking a process per
    recovery. Polling ``getppid()`` lets the orphan self-terminate promptly.
    """
    initial_ppid = os.getppid()

    def _watch() -> None:
        while True:
            time.sleep(poll_seconds)
            try:
                current_ppid = os.getppid()
            except OSError:
                continue
            if current_ppid != initial_ppid:
                with contextlib.suppress(Exception):
                    _terminate_descendant_processes()
                os._exit(1)

    thread = threading.Thread(
        target=_watch,
        name="predict-rlm-kernel-parent-watchdog",
        daemon=True,
    )
    thread.start()


def _persistent_kernel_runner(
    globals_dict: dict[str, Any],
    request_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    stdin_fd: int,
    host_tool_request_queue: multiprocessing.Queue | None = None,
    host_tool_response_queue: multiprocessing.Queue | None = None,
) -> None:
    with contextlib.suppress(OSError):
        os.setsid()
    _start_parent_death_watchdog()
    global HOST_TOOL_REQUEST_QUEUE, HOST_TOOL_RESPONSE_QUEUE, PROTOCOL_STDIN, PROTOCOL_STDOUT
    PROTOCOL_STDIN = os.fdopen(stdin_fd, "r", encoding="utf-8", buffering=1)
    PROTOCOL_STDOUT = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    HOST_TOOL_REQUEST_QUEUE = host_tool_request_queue
    HOST_TOOL_RESPONSE_QUEUE = host_tool_response_queue
    _reset_tool_protocol_state()

    while True:
        request = request_queue.get()
        if request is None:
            return
        try:
            if request.get("op") == "snapshot":
                _publish_kernel_result(
                    result_queue,
                    {
                        "ok": True,
                        "snapshot": _pickleable_globals_snapshot(globals_dict),
                    },
                )
                continue
            if request.get("op") == "register_runtime_hooks":
                _publish_kernel_result(
                    result_queue,
                    {"ok": True, "result": _register_runtime_hooks(request["params"])}
                )
                continue
            if request.get("timeout_seconds") is not None:
                signal.signal(signal.SIGINT, signal.default_int_handler)
            result = _run_kernel_coroutine(
                _execute_code_to_capture_files(
                    request["code"],
                    globals_dict,
                    request["stdout_path"],
                    request["stderr_path"],
                    defer_final_output=bool(request.get("defer_final_output")),
                )
            )
            _publish_kernel_result(result_queue, {"ok": True, "result": result})
            if _has_waiting_tool_responses():
                return
        except KeyboardInterrupt:
            timeout_seconds = request.get("timeout_seconds")
            if timeout_seconds is None:
                _publish_kernel_result(
                    result_queue,
                    {"ok": False, "error": _exception_payload(KeyboardInterrupt())},
                )
                if _has_waiting_tool_responses():
                    return
                continue
            _terminate_descendant_processes()
            _publish_kernel_result(
                result_queue,
                {
                    "ok": True,
                    "result": {
                        "timeout": {"seconds": timeout_seconds},
                        "stdout": _read_capture_file(pathlib.Path(request["stdout_path"])),
                        "stderr": _read_capture_file(pathlib.Path(request["stderr_path"])),
                        "state": _live_kernel_state(),
                    },
                },
            )
            if _has_waiting_tool_responses():
                return
        except BaseException as exc:
            error = _exception_payload(exc)
            output = _read_capture_file(pathlib.Path(request["stdout_path"])) + _read_capture_file(
                pathlib.Path(request["stderr_path"])
            )
            if output:
                error["output"] = output
            _publish_kernel_result(result_queue, {"ok": False, "error": error})
            if _has_waiting_tool_responses():
                return


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


def _signal_runner(process: multiprocessing.Process, sig: int) -> bool:
    pgid = _runner_process_group_id(process)
    if _signal_runner_process_group(pgid, sig):
        return True
    pid = process.pid
    if pid is None or not process.is_alive():
        return False
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return False
    except OSError:
        return False
    return True


def _terminate_runner(process: multiprocessing.Process) -> None:
    pgid = _runner_process_group_id(process)
    if not process.is_alive() and pgid is None:
        return

    if not _signal_runner(process, signal.SIGINT) and process.is_alive():
        process.terminate()
    process.join(timeout=0.2)

    _signal_runner_process_group(pgid, signal.SIGTERM)
    if process.is_alive():
        process.terminate()
    process.join(timeout=0.3)

    _signal_runner_process_group(pgid, signal.SIGKILL)
    if process.is_alive():
        if hasattr(process, "kill"):
            process.kill()
        else:
            process.terminate()
    process.join(timeout=0.5)


def _request_interrupt() -> bool:
    """Latch an interrupt request and return whether a cell is running."""
    global _INTERRUPT_REQUESTED
    _INTERRUPT_REQUESTED = True
    return _EXECUTION_ACTIVE


def _consume_interrupt_request() -> bool:
    """Atomically read-and-clear the latched interrupt flag."""
    global _INTERRUPT_REQUESTED
    requested = _INTERRUPT_REQUESTED
    _INTERRUPT_REQUESTED = False
    return requested


async def _handle_interrupt_request(request: dict[str, Any]) -> dict[str, Any]:
    """Handle an ``interrupt`` JSON-RPC request outside the serial queue."""
    running = _request_interrupt()
    return _response(request.get("id"), {"running": running})


def _discard_kernel() -> None:
    global _KERNEL_PROCESS, _KERNEL_REQUEST_QUEUE, _KERNEL_RESULT_QUEUE
    process = _KERNEL_PROCESS
    if process is not None:
        _terminate_runner(process)
    for message_queue in (_KERNEL_REQUEST_QUEUE, _KERNEL_RESULT_QUEUE):
        if message_queue is not None:
            with contextlib.suppress(Exception):
                message_queue.close()
    _KERNEL_PROCESS = None
    _KERNEL_REQUEST_QUEUE = None
    _KERNEL_RESULT_QUEUE = None


class _HostToolBridge:
    def __init__(
        self,
        connection: Any,
        request_queue: multiprocessing.Queue,
        response_queue: multiprocessing.Queue,
    ) -> None:
        self.connection = connection
        self.request_queue = request_queue
        self.response_queue = response_queue

    async def drain_requests(self) -> None:
        while True:
            try:
                request = self.request_queue.get_nowait()
            except queue.Empty:
                return
            await self.connection.send(json.dumps(request, default=str))

    def deliver_response(self, response: dict[str, Any]) -> None:
        self.response_queue.put(response)

    def clear(self) -> None:
        for message_queue in (self.request_queue, self.response_queue):
            while True:
                try:
                    message_queue.get_nowait()
                except queue.Empty:
                    break


def _ensure_kernel(
    globals_dict: dict[str, Any],
    host_tool_bridge: _HostToolBridge | None = None,
) -> multiprocessing.Process:
    global _KERNEL_PROCESS, _KERNEL_REQUEST_QUEUE, _KERNEL_RESULT_QUEUE
    if _KERNEL_PROCESS is not None and _KERNEL_PROCESS.is_alive():
        return _KERNEL_PROCESS
    _discard_kernel()
    ctx = _runner_context()
    _KERNEL_REQUEST_QUEUE = ctx.Queue()
    _KERNEL_RESULT_QUEUE = ctx.Queue()
    protocol_stdin_fd = os.dup(0)
    _KERNEL_PROCESS = ctx.Process(
        target=_persistent_kernel_runner,
        args=(
            globals_dict,
            _KERNEL_REQUEST_QUEUE,
            _KERNEL_RESULT_QUEUE,
            protocol_stdin_fd,
            host_tool_bridge.request_queue if host_tool_bridge is not None else None,
            host_tool_bridge.response_queue if host_tool_bridge is not None else None,
        ),
    )
    _KERNEL_PROCESS.start()
    os.close(protocol_stdin_fd)
    return _KERNEL_PROCESS


async def _kernel_pickle_snapshot(
    process: multiprocessing.Process,
) -> dict[str, Any]:
    assert _KERNEL_REQUEST_QUEUE is not None
    assert _KERNEL_RESULT_QUEUE is not None
    _KERNEL_REQUEST_QUEUE.put({"op": "snapshot"})
    while process.is_alive():
        try:
            message = _KERNEL_RESULT_QUEUE.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.01)
            continue
        if message.get("ok"):
            return message.get("snapshot") or {
                "globals": {},
                "restored_globals": [],
                "lost_globals": [],
            }
        _raise_runner_error(message.get("error") or {})
    return {"globals": {}, "restored_globals": [], "lost_globals": []}


async def _register_runtime_hooks_in_runner(
    params: dict[str, Any],
    globals_dict: dict[str, Any],
    host_tool_bridge: _HostToolBridge | None = None,
) -> dict[str, Any]:
    process = _ensure_kernel(globals_dict, host_tool_bridge)
    assert _KERNEL_REQUEST_QUEUE is not None
    assert _KERNEL_RESULT_QUEUE is not None
    _KERNEL_REQUEST_QUEUE.put({"op": "register_runtime_hooks", "params": params})
    while process.is_alive():
        try:
            message = _KERNEL_RESULT_QUEUE.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.01)
            continue
        if message.get("ok"):
            return message.get("result") or {}
        _raise_runner_error(message.get("error") or {})
    _discard_kernel()
    raise RuntimeError("execution runner exited while registering runtime hooks")


async def _execute_code_in_runner_with_timeout(
    code: str,
    globals_dict: dict[str, Any],
    timeout_seconds: float | None,
    timeout_interrupt_grace_seconds: float,
    *,
    defer_final_output: bool = False,
    host_tool_bridge: _HostToolBridge | None = None,
) -> dict[str, Any]:
    global _EXECUTION_ACTIVE
    process = _ensure_kernel(globals_dict, host_tool_bridge)
    assert _KERNEL_REQUEST_QUEUE is not None
    assert _KERNEL_RESULT_QUEUE is not None
    pre_timeout_snapshot: dict[str, Any] | None = await _kernel_pickle_snapshot(process)
    _consume_interrupt_request()
    _EXECUTION_ACTIVE = True
    stdout_path = _capture_file_path("stdout")
    stderr_path = _capture_file_path("stderr")
    _KERNEL_REQUEST_QUEUE.put({
        "code": code,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "timeout_seconds": timeout_seconds,
        "defer_final_output": defer_final_output,
    })
    deadline = (
        time.monotonic() + timeout_seconds if timeout_seconds is not None else None
    )
    interrupt_deadline: float | None = None
    interrupt_sent = False
    oob_interrupted = False
    runner_message: dict[str, Any] | None = None
    try:
        while True:
            if host_tool_bridge is not None:
                await host_tool_bridge.drain_requests()
            now = time.monotonic()
            try:
                runner_message = _KERNEL_RESULT_QUEUE.get_nowait()
                break
            except queue.Empty:
                pass
            if not process.is_alive():
                process.join(timeout=0.5)
                break
            if not interrupt_sent and _consume_interrupt_request():
                interrupt_sent = True
                oob_interrupted = True
                interrupt_deadline = now + timeout_interrupt_grace_seconds
                interrupted = _signal_runner(process, signal.SIGINT)
                _debug_event(
                    "sbx.python_runner.execute.interrupt",
                    code_hash=_code_hash(code),
                    code_len=len(code),
                    reason="out_of_band_interrupt",
                    interrupt_sent=interrupted,
                    interrupt_grace_seconds=timeout_interrupt_grace_seconds,
                    child_pid=process.pid,
                    child_exitcode=process.exitcode,
                )
                if interrupted and timeout_interrupt_grace_seconds > 0:
                    await asyncio.sleep(0.01)
                    continue
            if deadline is not None and not interrupt_sent and now >= deadline:
                interrupt_sent = True
                interrupt_deadline = now + timeout_interrupt_grace_seconds
                interrupted = _signal_runner(process, signal.SIGINT)
                _debug_event(
                    "sbx.python_runner.execute.interrupt",
                    code_hash=_code_hash(code),
                    code_len=len(code),
                    timeout_seconds=timeout_seconds,
                    interrupt_sent=interrupted,
                    interrupt_grace_seconds=timeout_interrupt_grace_seconds,
                    child_pid=process.pid,
                    child_exitcode=process.exitcode,
                )
                if interrupted and timeout_interrupt_grace_seconds > 0:
                    await asyncio.sleep(0.01)
                    continue
            if interrupt_sent and interrupt_deadline is not None and now >= interrupt_deadline:
                _terminate_runner(process)
                stdout = _read_capture_file(stdout_path)
                stderr = _read_capture_file(stderr_path)
                _discard_kernel()
                reason = "kernel did not respond to SIGINT before hard kill"
                snapshot = pre_timeout_snapshot or {
                    "globals": {},
                    "restored_globals": [],
                    "lost_globals": [],
                }
                _reset_globals_from_pickle_snapshot(globals_dict, snapshot)
                _debug_event(
                    "sbx.python_runner.execute",
                    code_hash=_code_hash(code),
                    code_len=len(code),
                    timeout=True,
                    timeout_seconds=timeout_seconds,
                    state_preserved=False,
                    state_source="pickle_snapshot",
                    state_loss_reason=reason,
                    restored_globals=snapshot.get("restored_globals", []),
                    lost_globals=snapshot.get("lost_globals", []),
                    stdout_len=len(stdout),
                    stderr_len=len(stderr),
                    child_pid=process.pid,
                    child_exitcode=process.exitcode,
                )
                return _build_interrupt_result(
                    timeout_seconds, oob_interrupted, stdout, stderr, snapshot, reason
                )
            await asyncio.sleep(0.01)
        if host_tool_bridge is not None:
            await host_tool_bridge.drain_requests()

        if runner_message is None:
            exitcode = process.exitcode
            _discard_kernel()
            if interrupt_sent:
                stdout = _read_capture_file(stdout_path)
                stderr = _read_capture_file(stderr_path)
                reason = (
                    "kernel exited after SIGINT before returning structured timeout"
                )
                snapshot = pre_timeout_snapshot or {
                    "globals": {},
                    "restored_globals": [],
                    "lost_globals": [],
                }
                _reset_globals_from_pickle_snapshot(globals_dict, snapshot)
                _debug_event(
                    "sbx.python_runner.execute",
                    code_hash=_code_hash(code),
                    code_len=len(code),
                    timeout=True,
                    timeout_seconds=timeout_seconds,
                    state_preserved=False,
                    state_source="pickle_snapshot",
                    state_loss_reason=reason,
                    restored_globals=snapshot.get("restored_globals", []),
                    lost_globals=snapshot.get("lost_globals", []),
                    stdout_len=len(stdout),
                    stderr_len=len(stderr),
                    child_pid=process.pid,
                    child_exitcode=exitcode,
                )
                return _build_interrupt_result(
                    timeout_seconds, oob_interrupted, stdout, stderr, snapshot, reason
                )
            _debug_event(
                "sbx.python_runner.execute",
                code_hash=_code_hash(code),
                code_len=len(code),
                timeout=False,
                error=True,
                error_type="RuntimeError",
                stdout_len=len(_read_capture_file(stdout_path)),
                stderr_len=len(_read_capture_file(stderr_path)),
                child_pid=process.pid,
                child_exitcode=exitcode,
            )
            if exitcode is None:
                raise RuntimeError("execution runner exited without a result")
            raise RuntimeError(
                f"execution runner exited without a result (exitcode={exitcode})"
            )
        if interrupt_sent and not runner_message.get("ok"):
            stdout = _read_capture_file(stdout_path)
            stderr = _read_capture_file(stderr_path)
            reason = "kernel interrupted by SIGINT"
            snapshot = pre_timeout_snapshot or {
                "globals": {},
                "restored_globals": [],
                "lost_globals": [],
            }
            _reset_globals_from_pickle_snapshot(globals_dict, snapshot)
            _debug_event(
                "sbx.python_runner.execute",
                code_hash=_code_hash(code),
                code_len=len(code),
                timeout=True,
                interrupted=oob_interrupted,
                state_preserved=False,
                state_source="pickle_snapshot",
                state_loss_reason=reason,
                restored_globals=snapshot.get("restored_globals", []),
                lost_globals=snapshot.get("lost_globals", []),
                stdout_len=len(stdout),
                stderr_len=len(stderr),
                child_pid=process.pid,
                child_exitcode=process.exitcode,
            )
            return _build_interrupt_result(
                timeout_seconds, oob_interrupted, stdout, stderr, snapshot, reason
            )
        if not runner_message.get("ok"):
            _debug_event(
                "sbx.python_runner.execute",
                code_hash=_code_hash(code),
                code_len=len(code),
                timeout=False,
                error=True,
                error_type=(runner_message.get("error") or {}).get("type"),
                error_message=(runner_message.get("error") or {}).get("message"),
                stdout_len=len(_read_capture_file(stdout_path)),
                stderr_len=len(_read_capture_file(stderr_path)),
                child_pid=process.pid,
                child_exitcode=process.exitcode,
            )
            _raise_runner_error(runner_message.get("error") or {})
        result = runner_message.get("result") or {}
        if isinstance(result, dict) and "timeout" in result:
            _terminate_descendant_processes(process.pid)
        _debug_event(
            "sbx.python_runner.execute",
            code_hash=_code_hash(code),
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
            stdout_len=len(_read_capture_file(stdout_path)),
            stderr_len=len(_read_capture_file(stderr_path)),
            state_preserved=(
                (result.get("state") or {}).get("preserved")
                if isinstance(result, dict)
                else None
            ),
            child_pid=process.pid,
            child_exitcode=process.exitcode,
        )
        return result
    finally:
        _EXECUTION_ACTIVE = False
        _unlink_capture_files(stdout_path, stderr_path)


async def _execute_code_with_timeout(
    code: str,
    globals_dict: dict[str, Any],
    timeout_seconds: float | None,
    timeout_interrupt_grace_seconds: float,
    *,
    defer_final_output: bool = False,
    host_tool_bridge: _HostToolBridge | None = None,
) -> dict[str, Any]:
    return await _execute_code_in_runner_with_timeout(
        code,
        globals_dict,
        timeout_seconds,
        timeout_interrupt_grace_seconds,
        defer_final_output=defer_final_output,
        host_tool_bridge=host_tool_bridge,
    )


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
    request: dict[str, Any],
    globals_dict: dict[str, Any],
    host_tool_bridge: _HostToolBridge | None = None,
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
                    _timeout_interrupt_grace_seconds(params),
                    defer_final_output=bool(params.get("defer_final_output")),
                    host_tool_bridge=host_tool_bridge,
                ),
            )
        if method == "register_output_fields":
            return _response(request_id, {})
        if method == "register_tools":
            return _response(request_id, _register_tools(params, globals_dict))
        if method == "register_runtime_hooks":
            return _response(
                request_id,
                await _register_runtime_hooks_in_runner(
                    params, globals_dict, host_tool_bridge
                ),
            )
        if method == "mount_file":
            return _response(request_id, _mount_file(params))
        if method == "mkdir_p":
            return _response(request_id, _mkdir_p(params))
        if method == "list_dir":
            return _response(request_id, _list_dir(params))
        if method == "sync_file":
            return _response(request_id, _sync_file(params))
        if method == "interrupt":
            return await _handle_interrupt_request(request)
        if method == "shutdown":
            return _response(request_id, {"shutdown": True})
        raise ValueError(f"Unknown method: {method}")
    except SyntaxError as exc:
        return _error(request_id, exc)
    except BaseException as exc:
        return _error(request_id, exc)


async def _stdio_main() -> None:
    globals_dict = _new_globals()

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        if request.get("method") == "reset":
            _discard_kernel()
            PENDING_TOOL_RESPONSES.clear()
            globals_dict = _new_globals()
            _send_protocol(_response(request.get("id"), {}))
            continue

        response = await _handle_request(request, globals_dict)
        if response is not None:
            _send_protocol(response)
        if request.get("method") == "shutdown":
            if (request.get("params") or {}).get("preserve_kernel_process"):
                os._exit(0)
            _discard_kernel()
            break


class _WebSocketSupervisorSession:
    def __init__(self, connection: Any, stop_event: asyncio.Event) -> None:
        self.connection = connection
        self.stop_event = stop_event
        ctx = multiprocessing.get_context("fork")
        self.host_tool_bridge = _HostToolBridge(
            connection,
            ctx.Queue(),
            ctx.Queue(),
        )
        self.requests: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def run(self) -> None:
        globals_dict = _new_globals()
        receiver = asyncio.create_task(self._receive_messages())
        try:
            while True:
                request = await self.requests.get()
                if request is None:
                    break
                if request.get("method") == "reset":
                    _discard_kernel()
                    self.host_tool_bridge.clear()
                    globals_dict = _new_globals()
                    await self.connection.send(json.dumps(_response(request.get("id"), {})))
                    continue

                response = await _handle_request(
                    request,
                    globals_dict,
                    self.host_tool_bridge,
                )
                await self.host_tool_bridge.drain_requests()
                if response is not None:
                    await self.connection.send(json.dumps(response, default=str))
                if request.get("method") == "shutdown":
                    if (request.get("params") or {}).get("preserve_kernel_process"):
                        os._exit(0)
                    _discard_kernel()
                    self.stop_event.set()
                    break
        finally:
            receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await receiver

    async def _receive_messages(self) -> None:
        try:
            async for raw in self.connection:
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(message, dict):
                    continue
                if message.get("method") == "interrupt":
                    response = await _handle_interrupt_request(message)
                    await self.connection.send(json.dumps(response, default=str))
                elif message.get("method"):
                    await self.requests.put(message)
                elif "id" in message:
                    self.host_tool_bridge.deliver_response(message)
        finally:
            await self.requests.put(None)


async def _websocket_main(
    *,
    host: str,
    port: int,
    path: str,
    max_message_bytes: int,
) -> None:
    from websockets.asyncio.server import serve
    from websockets.datastructures import Headers
    from websockets.http11 import Request, Response

    def authorize(connection: Any, request: Request) -> Response | None:
        if request.path == path:
            return None
        return Response(404, "Not Found", Headers(), b"Not Found")

    stop_event = asyncio.Event()

    async def handler(connection: Any) -> None:
        await _WebSocketSupervisorSession(connection, stop_event).run()

    async with serve(
        handler,
        host,
        port,
        process_request=authorize,
        max_size=max_message_bytes,
        max_queue=32,
    ) as server:
        await stop_event.wait()
        server.close()
        await server.wait_closed()


def _parse_args() -> Any:
    parser = ArgumentParser()
    parser.add_argument("--websocket-host")
    parser.add_argument("--websocket-port", type=int)
    parser.add_argument("--websocket-path")
    parser.add_argument("--websocket-max-message-bytes", type=int, default=32 * 1024 * 1024)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.websocket_host is not None or args.websocket_port is not None:
        if args.websocket_host is None or args.websocket_port is None or not args.websocket_path:
            raise SystemExit("websocket host, port, and path are required together")
        asyncio.run(
            _websocket_main(
                host=args.websocket_host,
                port=args.websocket_port,
                path=args.websocket_path,
                max_message_bytes=args.websocket_max_message_bytes,
            )
        )
    else:
        asyncio.run(_stdio_main())
