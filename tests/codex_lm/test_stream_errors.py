"""When Codex fails mid-stream (rate limit, backend error, incomplete,
dropped connection) we raise :class:`CodexStreamError` with the upstream
details instead of silently handing DSPy an empty response — which would
manifest as a confusing parser failure downstream.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
from conftest import make_text_delta
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
