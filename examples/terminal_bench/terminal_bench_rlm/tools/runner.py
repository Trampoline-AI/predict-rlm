"""JSONL runner copied into a Terminal-Bench task container."""

from __future__ import annotations

import ast
import asyncio
import builtins
import contextlib
import inspect
import io
import json
import sys
from pathlib import Path
from typing import Any

REAL_STDOUT = sys.stdout
TOOL_REQUEST_ID = 0
PENDING_TOOL_RESPONSES: dict[Any, dict[str, Any]] = {}


class _FinalOutputError(Exception):
    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__("SUBMIT")
        self.payload = payload


def runner_script_path() -> Path:
    return Path(__file__).resolve()


def runner_source() -> str:
    return runner_script_path().read_text(encoding="utf-8")


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(val) for val in value]
    return value


def _submit(**kwargs: Any) -> None:
    raise _FinalOutputError(_to_jsonable(kwargs))


def _response_ok(request_id: Any, result: Any | None = None) -> dict[str, Any]:
    return {"id": request_id, "ok": True, "result": {} if result is None else result}


def _response_error(request_id: Any, exc: BaseException) -> dict[str, Any]:
    return {
        "id": request_id,
        "ok": False,
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "args": list(getattr(exc, "args", ())),
        },
    }


def _send(message: dict[str, Any]) -> None:
    REAL_STDOUT.write(json.dumps(message, default=str) + "\n")
    REAL_STDOUT.flush()


def _read_message() -> dict[str, Any]:
    while True:
        line = sys.stdin.readline()
        if not line:
            raise RuntimeError("Host closed stdin while waiting for a response")
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(message, dict):
            return message


async def _read_tool_response(request_id: int) -> dict[str, Any]:
    if request_id in PENDING_TOOL_RESPONSES:
        return PENDING_TOOL_RESPONSES.pop(request_id)
    while True:
        response = await asyncio.to_thread(_read_message)
        response_id = response.get("id")
        if response_id == request_id:
            return response
        PENDING_TOOL_RESPONSES[response_id] = response


async def _call_host_tool(name: str, *args: Any, **kwargs: Any) -> Any:
    global TOOL_REQUEST_ID
    TOOL_REQUEST_ID += 1
    request_id = TOOL_REQUEST_ID
    _send(
        {
            "id": request_id,
            "method": "tool_call",
            "params": {"name": name, "args": list(args), "kwargs": kwargs},
        }
    )
    response = await _read_tool_response(request_id)
    if not response.get("ok"):
        error = response.get("error") or {}
        raise RuntimeError(error.get("message") or error)
    result = response.get("result") or {}
    value = result.get("value")
    if result.get("type") == "json" and isinstance(value, str):
        return json.loads(value)
    return value


def _register_tools(params: dict[str, Any], namespace: dict[str, Any]) -> dict[str, Any]:
    for name in params.get("tools", []):

        async def _tool(*args: Any, __tool_name: str = name, **kwargs: Any) -> Any:
            return await _call_host_tool(__tool_name, *args, **kwargs)

        namespace[name] = _tool
    return {}


def _new_namespace() -> dict[str, Any]:
    namespace: dict[str, Any] = {"__name__": "__main__", "SUBMIT": _submit}
    namespace["open"] = builtins.open
    return namespace


async def _execute_code(code: str, namespace: dict[str, Any]) -> dict[str, Any]:
    output = io.StringIO()
    try:
        compiled = compile(
            code,
            "<terminal-bench-runner>",
            "exec",
            flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
        )
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            result = eval(compiled, namespace, namespace)
            if inspect.isawaitable(result):
                await result
    except _FinalOutputError as final:
        return {"final": final.payload}
    return {"output": output.getvalue()}


async def _handle_request(
    request: dict[str, Any], namespace: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    request_id = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}

    if method == "reset":
        PENDING_TOOL_RESPONSES.clear()
        return _response_ok(request_id), _new_namespace()

    try:
        if method == "execute":
            return _response_ok(
                request_id,
                await _execute_code(str(params.get("code") or ""), namespace),
            ), namespace
        if method == "register_tools":
            return _response_ok(request_id, _register_tools(params, namespace)), namespace
        if method == "register_output_fields":
            return _response_ok(request_id), namespace
        if method == "shutdown":
            return _response_ok(request_id, {"shutdown": True}), namespace
        raise ValueError(f"Unknown method: {method}")
    except BaseException as exc:
        return _response_error(request_id, exc), namespace


async def _main() -> None:
    namespace = _new_namespace()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(message, dict):
            continue
        response, namespace = await _handle_request(message, namespace)
        _send(response)
        if message.get("method") == "shutdown":
            break


if __name__ == "__main__":
    asyncio.run(_main())
