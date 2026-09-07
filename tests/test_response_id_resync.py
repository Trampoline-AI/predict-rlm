from __future__ import annotations

import asyncio
import json
import types

import pytest
from dspy.primitives.code_interpreter import CodeInterpreterError  # noqa: E402

from predict_rlm.backends import JspiBackend
from predict_rlm.backends.base import STALE_RESPONSE_DISCARD_LIMIT  # noqa: E402


def _build_interp(stdout_lines: list[str]):
    """Build a JspiBackend whose ``_read_with_timeout`` is mocked
    to pop from a scripted queue. Avoids real fd/select plumbing.
    """
    interp = JspiBackend.__new__(JspiBackend)
    interp._stdout_fd = -1
    interp._read_buf = ""
    interp._use_jspi = False
    interp._stdin_fd = -1
    interp._loop = None
    interp._request_id = 6  # next request will be id=7

    class _QuietStdin:
        def __init__(self):
            self.writes = []

        def write(self, data):
            self.writes.append(data)

        def flush(self):
            pass

        def close(self):
            pass

    interp.deno_process = types.SimpleNamespace(
        stdin=_QuietStdin(),
        stdout=None,  # unused because we override _read_with_timeout below
        stderr=None,
        poll=lambda: None,
    )

    queue = list(stdout_lines)

    def _mock_read(timeout):
        if not queue:
            return None
        return queue.pop(0).rstrip("\n")

    interp._read_with_timeout = _mock_read  # type: ignore[assignment]
    return interp


def test_multiple_stale_responses_are_discarded():
    """If the buffer holds several stale frames (e.g. the process
    recovered from multiple timeouts in a row), the resync loop must
    keep reading until it finds the matching id.
    """
    stale1 = json.dumps({"jsonrpc": "2.0", "id": 3, "result": {"output": "stale3"}}) + "\n"
    stale2 = json.dumps({"jsonrpc": "2.0", "id": 4, "result": {"output": "stale4"}}) + "\n"
    stale3 = json.dumps({"jsonrpc": "2.0", "id": 5, "result": {"output": "stale5"}}) + "\n"
    fresh = json.dumps({"jsonrpc": "2.0", "id": 7, "result": {"output": "real"}}) + "\n"

    interp = _build_interp([stale1, stale2, stale3, fresh])
    result = interp._send_request("m", {}, context="t")
    assert result.get("result", {}).get("output") == "real"


def test_exhausted_resync_raises_cleanly():
    """The resync must have a safety cap so a runaway stdout (every
    response has the wrong id, e.g. deno bug) doesn't hang forever.
    Raising CodeInterpreterError is the cleanly-propagated signal.
    """
    bogus = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}) + "\n"
    # Feed the same stale id forever-ish (100 copies); should bail
    # well before 100.
    interp = _build_interp([bogus] * 100)
    with pytest.raises(CodeInterpreterError, match="stale|resync"):
        interp._send_request("m", {}, context="t")


def _build_execute_loop_interp(stdout_lines: list[str]):
    interp = JspiBackend.__new__(JspiBackend)
    interp._pending_file_ops = {}
    interp._debug = False
    interp.deno_process = types.SimpleNamespace(
        stderr=types.SimpleNamespace(read=lambda: ""),
    )
    lines = list(stdout_lines)

    async def _mock_read(timeout):
        if not lines:
            return None
        return lines.pop(0).rstrip("\n")

    async def _noop_responses(pending):
        return None

    async def _noop_sync_files():
        return None

    interp._async_files = _noop_sync_files  # type: ignore[assignment]
    interp._read_with_timeout_async = _mock_read  # type: ignore[assignment]
    interp._send_completed_responses = _noop_responses  # type: ignore[assignment]
    interp._wait_and_send_all_responses = _noop_responses  # type: ignore[assignment]
    return interp


@pytest.mark.asyncio
async def test_jspi_execute_loop_discards_stale_top_level_response():
    stale = json.dumps({"jsonrpc": "2.0", "id": 5, "result": {"output": "stale"}})
    fresh = json.dumps({"jsonrpc": "2.0", "id": 7, "result": {"output": "fresh"}})
    interp = _build_execute_loop_interp([stale, fresh])

    result = await interp._execute_async(7)

    assert result == "fresh"


@pytest.mark.asyncio
async def test_jspi_execute_loop_exhausted_resync_raises_cleanly():
    stale = json.dumps({"jsonrpc": "2.0", "id": 5, "result": {"output": "stale"}})
    interp = _build_execute_loop_interp([stale] * (STALE_RESPONSE_DISCARD_LIMIT + 1))

    with pytest.raises(CodeInterpreterError, match="stale|resync"):
        await interp._execute_async(7)


@pytest.mark.asyncio
async def test_jspi_execute_loop_routes_file_operation_response_before_resync():
    file_op = json.dumps({"jsonrpc": "2.0", "id": 5, "result": {}})
    fresh = json.dumps({"jsonrpc": "2.0", "id": 7, "result": {"output": "fresh"}})
    interp = _build_execute_loop_interp([file_op, fresh])
    future = asyncio.get_running_loop().create_future()
    interp._pending_file_ops = {5: future}

    result = await interp._execute_async(7)

    assert result == "fresh"
    assert future.result()["id"] == 5


@pytest.mark.asyncio
async def test_jspi_execute_loop_routes_tool_calls_without_counting_them_stale():
    tool_calls = [
        json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "tool_call",
                "params": {"name": "tool", "args": [], "kwargs": {}},
                "id": f"tool-{idx}",
            }
        )
        for idx in range(STALE_RESPONSE_DISCARD_LIMIT + 1)
    ]
    fresh = json.dumps({"jsonrpc": "2.0", "id": 7, "result": {"output": "fresh"}})
    interp = _build_execute_loop_interp([*tool_calls, fresh])
    called: list[str] = []

    async def _execute_tool(name, params, request_id=None):
        called.append(request_id if request_id is not None else name)
        return {"value": "ok", "type": "string"}

    async def _wait_all(pending):
        await asyncio.gather(*pending.values())
        pending.clear()

    interp._execute_tool_async = _execute_tool  # type: ignore[assignment]
    interp._wait_and_send_all_responses = _wait_all  # type: ignore[assignment]

    result = await interp._execute_async(7)

    assert result == "fresh"
    assert len(called) == STALE_RESPONSE_DISCARD_LIMIT + 1
