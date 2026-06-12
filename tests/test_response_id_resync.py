"""RED-GREEN repro for the Response-ID desync bomb.

Background:
    ``JspiBackend._send_request`` increments ``self._request_id``,
    writes a JSON-RPC message to deno's stdin, then reads one line from
    stdout and asserts the response's ``id`` matches the request. If it
    doesn't, the helper raises ``Response ID mismatch``.

    That assumption breaks when a previous ``_send_request`` hit its
    ``DENO_REQUEST_TIMEOUT_SEC`` budget and returned ``None``: the deno
    process may still deliver its response later, leaving a stale
    JSON-RPC frame sitting in the stdout buffer. The NEXT
    ``_send_request`` writes a fresh request, reads the stale response
    first, sees an older id, and raises.

    The raised error is fed back to the RLM as ``[Error] Response ID
    mismatch …``, which the model interprets as a code-format problem.
    Since the code isn't broken, the model resubmits identical code
    and gets the same mismatch error on the next iteration, burning
    through the 50-iteration budget until the task hits
    ``task_timeout=600s``. A 2026-04-18 eval run turned 15 tasks into
    infinite retry bombs this way — each recorded as a score-0 timeout.

Fix (option B): treat a non-matching id as a STALE response, discard
    it, and read again. Only raise if we exhaust a safety cap on the
    number of stale frames or if reading fails. Natural resync; no
    fragile buffer-purge logic.

RED: stale frame preceding the right one → ``Response ID mismatch`` raised
GREEN: stale frame discarded, real response returned cleanly
"""

from __future__ import annotations

import asyncio
import json
import queue
import types

import pytest
from dspy.primitives.code_interpreter import CodeInterpreterError

import predict_rlm.backends.jspi.backend as rlm_interpreter
from predict_rlm.backends import JspiBackend, SbxBackend, SbxConfig
from predict_rlm.backends.base import STALE_RESPONSE_DISCARD_LIMIT


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


def test_stale_response_is_discarded_then_fresh_response_is_returned(monkeypatch):
    """When a stale id=5 frame is sitting in the stdout buffer ahead of
    the fresh id=7 response, ``_send_request`` must skip the stale and
    return the fresh — NOT raise Response ID mismatch.

    Without the fix: first readline returns the stale id=5 frame, the
    id check fires, CodeInterpreterError is raised. The model then
    sees the error as a code-format bug and burns its iteration budget
    retrying identical code.
    """
    # Keep test fast — don't let the real DENO_REQUEST_TIMEOUT_SEC=30s
    # stretch into a test failure if something unexpected blocks.
    monkeypatch.setattr(rlm_interpreter, "DENO_REQUEST_TIMEOUT_SEC", 0.5)

    stale_line = json.dumps({
        "jsonrpc": "2.0",
        "id": 5,
        "result": {"output": "[Error] timed out on iter 5"},
    }) + "\n"
    fresh_line = json.dumps({
        "jsonrpc": "2.0",
        "id": 7,
        "result": {"output": "ok"},
    }) + "\n"

    interp = _build_interp([stale_line, fresh_line])

    # Must NOT raise Response ID mismatch. The resync loop discards
    # id=5 and returns the id=7 payload.
    result = interp._send_request("test_method", {}, context="unit-test")

    assert result is not None, "expected a result dict; got None"
    assert result.get("result", {}).get("output") == "ok", (
        f"expected fresh id=7 response (output=ok), got {result!r}. "
        "If this is the stale id=5 response the resync is broken."
    )


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


def test_matching_response_passes_through_unchanged():
    """Guardrail: when the first readline IS the matching response
    (normal case), the resync loop returns it without discarding.
    """
    good = json.dumps({"jsonrpc": "2.0", "id": 7, "result": {"output": "hi"}}) + "\n"
    interp = _build_interp([good])
    result = interp._send_request("m", {}, context="t")
    assert result.get("result", {}).get("output") == "hi"


def _build_execute_loop_interp(stdout_lines: list[str]):
    interp = JspiBackend.__new__(JspiBackend)
    interp._pending_file_ops = {}
    interp._debug = False
    interp._sync_files = lambda: None
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
        json.dumps({
            "jsonrpc": "2.0",
            "method": "tool_call",
            "params": {"name": "tool", "args": [], "kwargs": {}},
            "id": f"tool-{idx}",
        })
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


class _BufferingStdin:
    def __init__(self) -> None:
        self.data: list[str] = []

    def write(self, data: str) -> None:
        self.data.append(data)

    def flush(self) -> None:
        return None


def _build_sbx_request_interp(tmp_path, stdout_lines: list[str]) -> SbxBackend:
    interp = SbxBackend(
        config=SbxConfig(name="resync-test", exec_timeout=1),
        preinstall_packages=False,
        _runner_command=["unused"],
        _staging_root=tmp_path / "staging",
    )
    interp._ensure_process_for_method = lambda method: None  # type: ignore[method-assign]
    interp._proc = types.SimpleNamespace(
        stdin=_BufferingStdin(),
        stdout=types.SimpleNamespace(),
        stderr=None,
        poll=lambda: None,
    )
    interp._stdout_lines = queue.Queue()
    for line in stdout_lines:
        interp._stdout_lines.put(line)
    return interp


def _close_sbx_request_interp(interp: SbxBackend) -> None:
    interp._proc = None


def test_sbx_send_request_discards_stale_top_level_response(tmp_path):
    stale = json.dumps({"jsonrpc": "2.0", "id": 5, "result": {"output": "stale"}})
    fresh = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"output": "fresh"}})
    interp = _build_sbx_request_interp(tmp_path, [stale, fresh])

    try:
        result = interp._send_request("execute", {"code": "print('fresh')"})
    finally:
        _close_sbx_request_interp(interp)

    assert result["result"]["output"] == "fresh"


def test_sbx_send_request_exhausted_resync_raises_cleanly(tmp_path):
    stale = json.dumps({"jsonrpc": "2.0", "id": 5, "result": {"output": "stale"}})
    interp = _build_sbx_request_interp(
        tmp_path,
        [stale] * (STALE_RESPONSE_DISCARD_LIMIT + 1),
    )

    try:
        with pytest.raises(CodeInterpreterError, match="stale|resync"):
            interp._send_request("execute", {"code": "print('fresh')"})
    finally:
        _close_sbx_request_interp(interp)


def test_sbx_send_request_routes_tool_calls_without_counting_them_stale(
    tmp_path, monkeypatch
):
    tool_calls = [
        json.dumps({
            "jsonrpc": "2.0",
            "method": "tool_call",
            "params": {"name": "tool", "args": [], "kwargs": {}},
            "id": f"tool-{idx}",
        })
        for idx in range(STALE_RESPONSE_DISCARD_LIMIT + 1)
    ]
    fresh = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"output": "fresh"}})
    interp = _build_sbx_request_interp(tmp_path, [*tool_calls, fresh])
    submitted: list[dict] = []
    monkeypatch.setattr(interp, "_submit_tool_call", submitted.append)

    try:
        result = interp._send_request("execute", {"code": "print('fresh')"})
    finally:
        _close_sbx_request_interp(interp)

    assert result["result"]["output"] == "fresh"
    assert len(submitted) == STALE_RESPONSE_DISCARD_LIMIT + 1
