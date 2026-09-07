import asyncio
from unittest import mock

import pytest
from conftest import build_stream_events
from dspy_codex_lm import CodexStreamError


async def test_silent_stream_raises_instead_of_hanging(lm, monkeypatch):
    monkeypatch.setattr("dspy_codex_lm.lm.CODEX_STREAM_HEARTBEAT_SEC", 0.05)

    async def fake_aresponses(**_):
        async def stream():
            await asyncio.Future()
            yield

        return stream()

    with mock.patch("dspy_codex_lm.lm.litellm.aresponses", side_effect=fake_aresponses):
        with pytest.raises(CodexStreamError, match="stalled"):
            await asyncio.wait_for(lm.aforward(prompt="hi"), timeout=3)


async def test_completion_does_not_wait_for_stream_disconnect(lm, monkeypatch):
    monkeypatch.setattr("dspy_codex_lm.lm.CODEX_STREAM_HEARTBEAT_SEC", 0.05)

    async def fake_aresponses(**_):
        async def stream():
            for event in build_stream_events("ok", input_tokens=5, output_tokens=1):
                yield event
            await asyncio.Future()

        return stream()

    with mock.patch("dspy_codex_lm.lm.litellm.aresponses", side_effect=fake_aresponses):
        response = await asyncio.wait_for(lm.aforward(prompt="hi"), timeout=3)
    assert response.output[0].content[0].text == "ok"


def test_sync_completion_stops_before_reading_another_http_event(lm):
    def stream():
        yield from build_stream_events("ok", input_tokens=5, output_tokens=1)
        raise AssertionError("read past response.completed")

    with mock.patch("dspy_codex_lm.lm.litellm.responses", return_value=stream()):
        response = lm.forward(prompt="hi")
    assert response.output[0].content[0].text == "ok"
