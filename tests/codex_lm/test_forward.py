import copy
from types import SimpleNamespace
from unittest import mock

import dspy
import pytest
from conftest import build_stream_events, make_completed, make_text_delta
from dspy.utils.usage_tracker import UsageTracker


def test_stream_assembly_accounts_for_cached_input_and_output(lm):
    events = [
        make_text_delta("hel"),
        SimpleNamespace(type="response.in_progress"),
        make_text_delta("lo"),
        make_completed(input_tokens=100, cached_tokens=80, output_tokens=10),
    ]
    with mock.patch("dspy_codex_lm.lm.litellm.responses", return_value=iter(events)):
        response = lm.forward(prompt="hi", cache=False)

    assert response.output[0].content[0].text == "hello"
    assert response.usage.input_tokens == 100
    assert response.usage.output_tokens == 10
    assert response.usage.cost == pytest.approx(20 * 1.75e-6 + 80 * 1.75e-7 + 10 * 1.4e-5)


def test_cached_calls_do_not_double_count_usage_or_mutate_fresh_history(lm):
    events = build_stream_events("hi", input_tokens=1000, output_tokens=50)
    tracker = UsageTracker()
    with dspy.settings.context(usage_tracker=tracker):
        with mock.patch(
            "dspy_codex_lm.lm.litellm.responses",
            side_effect=lambda **_: iter(copy.deepcopy(events)),
        ) as transport:
            assert lm(prompt="same")[0]["text"] == "hi"
            assert lm(prompt="same")[0]["text"] == "hi"

    assert transport.call_count == 1
    totals = tracker.get_total_tokens()[lm.model]
    assert totals["prompt_tokens"] == 1000
    assert totals["completion_tokens"] == 50
    fresh, cached = lm.history
    assert fresh["usage"]["prompt_tokens"] == 1000
    assert fresh["usage"]["completion_tokens"] == 50
    assert fresh["cost"] == pytest.approx(1000 * 1.75e-6 + 50 * 1.4e-5)
    assert dict(cached["usage"]) == {}


async def test_async_prediction_exposes_parsed_answer_and_billable_usage(lm):
    events = build_stream_events(
        "[[ ## answer ## ]]\n42\n[[ ## completed ## ]]",
        input_tokens=500,
        output_tokens=20,
    )

    async def fake_aresponses(**_):
        async def stream():
            for event in copy.deepcopy(events):
                yield event

        return stream()

    with dspy.settings.context(lm=lm, track_usage=True):
        with mock.patch("dspy_codex_lm.lm.litellm.aresponses", side_effect=fake_aresponses):
            prediction = await dspy.Predict("question -> answer").acall(
                question="six times seven?"
            )

    assert prediction.answer.strip() == "42"
    usage = prediction.get_lm_usage()[lm.model]
    assert usage["prompt_tokens"] == 500
    assert usage["completion_tokens"] == 20
