"""When Codex fails mid-stream (rate limit, backend error, incomplete,
dropped connection) we raise :class:`CodexStreamError` with the upstream
details instead of silently handing DSPy an empty response — which would
manifest as a confusing parser failure downstream.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
from conftest import make_completed, make_text_delta
from dspy_codex_lm import CodexStreamError


def _failed_event(code: str, message: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="response.failed",
        response=SimpleNamespace(
            error=SimpleNamespace(code=code, message=message),
        ),
    )


def _incomplete_event(reason: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="response.incomplete",
        response=SimpleNamespace(
            incomplete_details=SimpleNamespace(reason=reason),
        ),
    )


def _error_event(code: str, message: str) -> SimpleNamespace:
    return SimpleNamespace(type="error", code=code, message=message)


def _patch_responses(events):
    return mock.patch(
        "dspy_codex_lm.lm.litellm.responses",
        side_effect=lambda **_: iter(events),
    )


# ---- unit: _handle_event captures failure into state ----


def test_handle_event_captures_failed(lm):
    state = lm._fresh_state()
    lm._handle_event(_failed_event("rate_limit_exceeded", "Too many requests"), state)
    assert state["failure"] is not None
    assert state["failure"]["kind"] == "failed"
    assert state["failure"]["code"] == "rate_limit_exceeded"
    assert state["failure"]["message"] == "Too many requests"
    assert state["completed"] is False


def test_handle_event_captures_incomplete(lm):
    state = lm._fresh_state()
    lm._handle_event(_incomplete_event("max_output_tokens"), state)
    assert state["failure"]["kind"] == "incomplete"
    assert state["failure"]["code"] == "max_output_tokens"


def test_handle_event_captures_error(lm):
    state = lm._fresh_state()
    lm._handle_event(_error_event("server_error", "internal"), state)
    assert state["failure"]["kind"] == "error"
    assert state["failure"]["code"] == "server_error"


def test_handle_event_sets_completed_flag(lm):
    state = lm._fresh_state()
    lm._handle_event(make_completed(input_tokens=1, output_tokens=1), state)
    assert state["completed"] is True
    assert state["failure"] is None


# ---- forward() should raise, not return empty ----


def test_forward_raises_on_failed_event(lm):
    events = [
        make_text_delta("partial"),
        _failed_event("rate_limit_exceeded", "Too many requests"),
    ]
    with _patch_responses(events):
        with pytest.raises(CodexStreamError) as excinfo:
            lm.forward(prompt="hi")
    msg = str(excinfo.value)
    assert "failed" in msg
    assert "rate_limit_exceeded" in msg
    assert "Too many requests" in msg


def test_forward_raises_on_incomplete_event(lm):
    events = [make_text_delta("partial"), _incomplete_event("content_filter")]
    with _patch_responses(events):
        with pytest.raises(CodexStreamError, match="content_filter"):
            lm.forward(prompt="hi")


def test_forward_raises_on_error_event(lm):
    events = [_error_event("server_error", "upstream down")]
    with _patch_responses(events):
        with pytest.raises(CodexStreamError, match="server_error"):
            lm.forward(prompt="hi")


def test_forward_raises_on_truncated_stream(lm):
    # Stream produces text deltas but never a completed / failed event.
    events = [make_text_delta("hello"), make_text_delta(" world")]
    with _patch_responses(events):
        with pytest.raises(CodexStreamError, match="without.*completed"):
            lm.forward(prompt="hi")


def test_forward_raises_on_empty_stream(lm):
    with _patch_responses([]):
        with pytest.raises(CodexStreamError):
            lm.forward(prompt="hi")


# ---- logger emits a warning before raising ----


def test_failure_logs_warning(lm, caplog):
    import logging as _logging

    caplog.set_level(_logging.WARNING, logger="dspy_codex_lm.lm")
    events = [_failed_event("rate_limit_exceeded", "slow down")]
    with _patch_responses(events), pytest.raises(CodexStreamError):
        lm.forward(prompt="hi")
    records = [r for r in caplog.records if r.name == "dspy_codex_lm.lm"]
    assert records
    assert records[0].levelname == "WARNING"
    assert "rate_limit_exceeded" in records[0].getMessage()


# ---- async parity ----


def _async_iter(items):
    async def _gen():
        for item in items:
            yield item

    return _gen()


def _patch_aresponses(events):
    async def fake(**_):
        return _async_iter(events)

    return mock.patch("dspy_codex_lm.lm.litellm.aresponses", side_effect=fake)


async def test_aforward_raises_on_failed_event(lm):
    events = [_failed_event("rate_limit_exceeded", "Too many requests")]
    with _patch_aresponses(events):
        with pytest.raises(CodexStreamError, match="rate_limit_exceeded"):
            await lm.aforward(prompt="hi")


async def test_aforward_raises_on_truncated_stream(lm):
    events = [make_text_delta("partial")]
    with _patch_aresponses(events):
        with pytest.raises(CodexStreamError, match="without.*completed"):
            await lm.aforward(prompt="hi")
