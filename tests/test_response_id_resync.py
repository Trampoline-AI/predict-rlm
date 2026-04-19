"""RED-GREEN repro for the Response-ID desync bomb.

Background:
    ``JspiInterpreter._send_request`` increments ``self._request_id``,
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

import json
import types

import pytest

import predict_rlm.interpreter as rlm_interpreter
from predict_rlm.interpreter import CodeInterpreterError, JspiInterpreter


def _build_interp(stdout_lines: list[str]):
    """Build a JspiInterpreter whose ``_read_with_timeout`` is mocked
    to pop from a scripted queue. Avoids real fd/select plumbing.
    """
    interp = JspiInterpreter.__new__(JspiInterpreter)
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
