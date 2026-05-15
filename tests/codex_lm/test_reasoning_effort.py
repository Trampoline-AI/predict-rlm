"""reasoning_effort is DSPy/LiteLLM's standard knob for GPT-5 / o-series
reasoning control. DSPy's request converter rewrites it to
``reasoning: {effort, summary}`` for the Responses API. These tests verify
that users can set it either on the CodexLM constructor or per-call, and
that it reaches the actual litellm.responses / aresponses call.
"""

from unittest import mock

import pytest
from conftest import build_stream_events
from dspy_codex_lm import CodexLM

# ---- unit: _build_request rewrites effort into reasoning field ----


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_ctor_effort_flows_into_request(effort: str):
    lm = CodexLM(
        model="gpt-5.3-codex",
        reasoning_effort=effort,
        access_token="fake",
        account_id="fake",
    )
    request, _ = lm._build_request(prompt="x", messages=None, kwargs={})
    assert "reasoning_effort" not in request
    assert request["reasoning"] == {"effort": effort, "summary": "auto"}


def test_per_call_effort_flows_into_request(lm):
    request, _ = lm._build_request(
        prompt="x", messages=None, kwargs={"reasoning_effort": "high"}
    )
    assert "reasoning_effort" not in request
    assert request["reasoning"] == {"effort": "high", "summary": "auto"}


def test_per_call_effort_overrides_ctor_effort():
    lm = CodexLM(
        model="gpt-5.3-codex",
        reasoning_effort="low",
        access_token="fake",
        account_id="fake",
    )
    request, _ = lm._build_request(
        prompt="x", messages=None, kwargs={"reasoning_effort": "high"}
    )
    assert request["reasoning"] == {"effort": "high", "summary": "auto"}


def test_no_effort_no_reasoning_field(lm):
    request, _ = lm._build_request(prompt="x", messages=None, kwargs={})
    assert "reasoning" not in request
    assert "reasoning_effort" not in request


def test_ctor_effort_none_does_not_emit_reasoning_field():
    """Explicit None on construction must not produce ``effort: null``."""
    lm = CodexLM(
        model="gpt-5.3-codex",
        reasoning_effort=None,
        access_token="fake",
        account_id="fake",
    )
    request, _ = lm._build_request(prompt="x", messages=None, kwargs={})
    assert "reasoning_effort" not in request
    assert "reasoning" not in request


def test_per_call_effort_none_does_not_emit_reasoning_field(lm):
    request, _ = lm._build_request(prompt="x", messages=None, kwargs={"reasoning_effort": None})
    assert "reasoning_effort" not in request
    assert "reasoning" not in request


def test_per_call_effort_none_clears_ctor_effort():
    """Passing None at call time should override a ctor effort (clear it)."""
    lm = CodexLM(
        model="gpt-5.3-codex",
        reasoning_effort="high",
        access_token="fake",
        account_id="fake",
    )
    request, _ = lm._build_request(prompt="x", messages=None, kwargs={"reasoning_effort": None})
    assert "reasoning" not in request


# ---- integration: effort reaches the mocked litellm call ----


def _patch_responses_capture(events, captured):
    def fake(**kwargs):
        captured.update(kwargs)
        return iter(events)

    return mock.patch("dspy_codex_lm.lm.litellm.responses", side_effect=fake)


def _patch_aresponses_capture(events, captured):
    async def _aiter(items):
        for i in items:
            yield i

    async def fake(**kwargs):
        captured.update(kwargs)
        return _aiter(events)

    return mock.patch("dspy_codex_lm.lm.litellm.aresponses", side_effect=fake)


def test_forward_sends_reasoning_to_litellm():
    lm = CodexLM(
        model="gpt-5.3-codex",
        reasoning_effort="high",
        access_token="fake",
        account_id="fake",
    )
    events = build_stream_events("ok", input_tokens=5, output_tokens=1)
    captured: dict = {}
    with _patch_responses_capture(events, captured):
        lm.forward(prompt="hi")
    assert captured["reasoning"] == {"effort": "high", "summary": "auto"}
    assert "reasoning_effort" not in captured


async def test_aforward_sends_reasoning_to_litellm():
    lm = CodexLM(
        model="gpt-5.3-codex",
        reasoning_effort="medium",
        access_token="fake",
        account_id="fake",
    )
    events = build_stream_events("ok", input_tokens=5, output_tokens=1)
    captured: dict = {}
    with _patch_aresponses_capture(events, captured):
        await lm.aforward(prompt="hi")
    assert captured["reasoning"] == {"effort": "medium", "summary": "auto"}
    assert "reasoning_effort" not in captured


def test_per_call_effort_reaches_litellm():
    lm = CodexLM(
        model="gpt-5.3-codex",
        access_token="fake",
        account_id="fake",
    )
    events = build_stream_events("ok", input_tokens=5, output_tokens=1)
    captured: dict = {}
    with _patch_responses_capture(events, captured):
        lm.forward(prompt="hi", reasoning_effort="low")
    assert captured["reasoning"] == {"effort": "low", "summary": "auto"}
