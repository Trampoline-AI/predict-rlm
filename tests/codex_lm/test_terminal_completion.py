from __future__ import annotations

import asyncio
import threading
from unittest import mock

from conftest import build_stream_events


class _CompletedThenPendingSync:
    def __init__(self, events):
        self._events = iter(events)
        self.pending_reads = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._events)
        except StopIteration:
            self.pending_reads += 1
            threading.Event().wait(0.25)
            raise StopIteration


class _CompletedThenPendingAsync:
    def __init__(self, events):
        self._events = iter(events)
        self.pending_reads = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._events)
        except StopIteration:
            self.pending_reads += 1
            await asyncio.Future()
            raise AssertionError("unreachable")  # pragma: no cover


def test_forward_stops_at_response_completed_before_pending_stream(lm, monkeypatch):
    monkeypatch.setattr("dspy_codex_lm.lm.CODEX_STREAM_HEARTBEAT_SEC", 0.05)
    stream = _CompletedThenPendingSync(build_stream_events("ok", input_tokens=5, output_tokens=1))

    with mock.patch("dspy_codex_lm.lm.litellm.responses", return_value=stream):
        response = lm.forward(prompt="hi")

    assert response.output[0].content[0].text == "ok"
    assert stream.pending_reads == 0


async def test_aforward_stops_at_response_completed_before_pending_stream(lm, monkeypatch):
    monkeypatch.setattr("dspy_codex_lm.lm.CODEX_STREAM_HEARTBEAT_SEC", 0.05)
    stream = _CompletedThenPendingAsync(
        build_stream_events("ok", input_tokens=5, output_tokens=1)
    )

    with mock.patch("dspy_codex_lm.lm.litellm.aresponses", return_value=stream):
        response = await asyncio.wait_for(lm.aforward(prompt="hi"), timeout=1.0)

    assert response.output[0].content[0].text == "ok"
    assert stream.pending_reads == 0
