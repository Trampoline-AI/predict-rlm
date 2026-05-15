from unittest import mock

import dspy
from conftest import build_stream_events


def _patch_responses(events, call_count=None):
    """side_effect that returns a fresh iterator on every call."""

    def fake(**_):
        if call_count is not None:
            call_count["n"] += 1
        return iter(events)

    return mock.patch("dspy_codex_lm.lm.litellm.responses", side_effect=fake)


def test_forward_single_call(lm):
    events = build_stream_events("4", input_tokens=100, output_tokens=1)
    with _patch_responses(events):
        resp = lm.forward(prompt="What is 2+2?")
    assert resp.output[0].content[0].text == "4"
    assert resp.usage.input_tokens == 100
    assert resp.usage.cost > 0


def test_forward_via_dspy_predict(lm):
    text = "[[ ## answer ## ]]\n4\n[[ ## completed ## ]]"
    events = build_stream_events(text, input_tokens=50, output_tokens=10)
    with _patch_responses(events):
        dspy.configure(lm=lm)
        result = dspy.Predict("question -> answer")(question="What is 2+2?")
    assert result.answer.strip() == "4"


def test_forward_cost_in_history(lm):
    text = "[[ ## answer ## ]]\n4\n[[ ## completed ## ]]"
    events = build_stream_events(text, input_tokens=100, output_tokens=10)
    with _patch_responses(events):
        dspy.configure(lm=lm)
        dspy.Predict("question -> answer")(question="What is 2+2?")
    history_cost = dspy.settings.lm.history[-1]["cost"]
    # 100 * 1.75e-6 + 10 * 1.4e-5 = 0.000315
    assert abs(history_cost - 0.000315) < 1e-9


def test_forward_second_call_hits_cache(lm):
    events = build_stream_events("first", input_tokens=10, output_tokens=3)
    count = {"n": 0}
    with _patch_responses(events, call_count=count):
        r1 = lm.forward(prompt="same question")
        r2 = lm.forward(prompt="same question")
    assert count["n"] == 1  # second call hit cache
    assert r1.output[0].content[0].text == r2.output[0].content[0].text == "first"
