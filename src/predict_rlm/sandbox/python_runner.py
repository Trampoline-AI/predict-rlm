"""Python JSON-RPC runner used by the Docker Sandboxes backend."""

from __future__ import annotations

import ast
import asyncio
import builtins
import contextlib
import inspect
import io
import json
import os
import pathlib
import shutil
import sys
import tempfile
from typing import Any

REAL_STDOUT = sys.stdout
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


class _FinalOutputError(Exception):
    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__("SUBMIT")
        self.payload = payload


class _VirtualPath(str):
    def __new__(cls, virtual_path: str, real_path: str):
        obj = str.__new__(cls, real_path)
        obj.virtual_path = virtual_path
        return obj


def _to_jsonable(value: Any) -> Any:
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


def _path_factory(*args: Any, **kwargs: Any) -> pathlib.Path:
    if args:
        args = (_map_virtual_path(args[0]), *args[1:])
    return REAL_PATH(*args, **kwargs)


def _submit(**kwargs: Any) -> None:
    raise _FinalOutputError(_to_jsonable(kwargs))


def _send_protocol(message: dict[str, Any]) -> None:
    REAL_STDOUT.write(json.dumps(message, default=str) + "\n")
    REAL_STDOUT.flush()


def _read_protocol_response_line() -> dict[str, Any]:
    while True:
        line = sys.stdin.readline()
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
        return json.loads(value)
    return value


def _register_tools(params: dict[str, Any], globals_dict: dict[str, Any]) -> dict[str, Any]:
    for name in params.get("tools", []):
        async def _tool(*args: Any, __tool_name: str = name, **kwargs: Any) -> Any:
            return await _call_host_tool(__tool_name, *args, **kwargs)

        globals_dict[name] = _tool
    return {}


def _install_virtual_filesystem(globals_dict: dict[str, Any]) -> None:
    builtins.open = _open
    pathlib.Path = _path_factory  # type: ignore[assignment]
    globals_dict.setdefault("SUBMIT", _submit)


def _new_globals() -> dict[str, Any]:
    globals_dict: dict[str, Any] = {"__name__": "__main__"}
    _install_virtual_filesystem(globals_dict)
    return globals_dict


def _response(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "result": result, "id": request_id}


def _error(request_id: Any, exc: BaseException) -> dict[str, Any]:
    data = {
        "type": type(exc).__name__,
        "args": list(getattr(exc, "args", ())),
    }
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


async def _execute_code(code: str, globals_dict: dict[str, Any]) -> dict[str, Any]:
    output = io.StringIO()
    try:
        compiled = compile(
            code,
            "<sbx-runner>",
            "exec",
            flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
        )
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            result = eval(compiled, globals_dict, globals_dict)
            if inspect.isawaitable(result):
                await result
    except _FinalOutputError as final:
        return {"final": final.payload}
    except BaseException as exc:
        setattr(exc, "_predict_rlm_output", output.getvalue())
        raise
    return {"output": output.getvalue()}


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
            return _response(request_id, await _execute_code(params.get("code", ""), globals_dict))
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
