from unittest import mock

import dspy
from conftest import build_stream_events


async def _async_iter(items):
    for item in items:
        yield item


def _patch_aresponses(events, call_count=None):
    async def fake(**_):
        if call_count is not None:
            call_count["n"] += 1
        return _async_iter(events)

    return mock.patch("dspy_codex_lm.lm.litellm.aresponses", side_effect=fake)


async def test_aforward_single_call(lm):
    events = build_stream_events("4", input_tokens=50, output_tokens=1)
    with _patch_aresponses(events):
        resp = await lm.aforward(prompt="What is 2+2?")
    assert resp.output[0].content[0].text == "4"
    assert resp.usage.cost > 0


async def test_aforward_via_dspy_predict(lm):
    text = "[[ ## answer ## ]]\n4\n[[ ## completed ## ]]"
    events = build_stream_events(text, input_tokens=50, output_tokens=10)
    with _patch_aresponses(events):
        dspy.configure(lm=lm)
        result = await dspy.Predict("question -> answer").acall(question="What is 2+2?")
    assert result.answer.strip() == "4"


async def test_aforward_second_call_hits_cache(lm):
    events = build_stream_events("first", input_tokens=10, output_tokens=3)
    count = {"n": 0}
    with _patch_aresponses(events, call_count=count):
        r1 = await lm.aforward(prompt="same")
        r2 = await lm.aforward(prompt="same")
    assert count["n"] == 1
    assert r1.output[0].content[0].text == r2.output[0].content[0].text == "first"
