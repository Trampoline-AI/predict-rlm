"""RED-GREEN repro for the SSE-stream deadlock.

Background:
    ``CodexLM.aforward`` iterates the response stream via ``async for
    event in stream:``. If the underlying HTTP transport stalls —
    headers received, no body delivered, no FIN, no error — that loop
    blocks indefinitely below Python's asyncio cancellation layer. An
    outer ``asyncio.wait_for(task_timeout=...)`` cannot cancel a hung
    socket read that never hits an ``await`` point, and the kernel's
    TCP keepalive is 2 hours by default on macOS.

    This caused a real production stall on 2026-04-18: a SpreadsheetBench
    eval seized at 10:37 with 30 concurrent workers all parked inside
    ``async for event in stream:``. ``wait_for(300s)`` fired ``cancel()``
    at 10:42 but the tasks never received the CancelledError. The whole
    process became a CPU-0% zombie until manually killed.

RED (pre-fix): ``test_async_stream_that_never_yields_raises_within_heartbeat``
    runs to the outer safety-net timeout (3s) instead of raising
    CodexStreamError, failing the pytest.raises assertion with a
    ``asyncio.TimeoutError`` — proof the heartbeat guard is absent.

GREEN (post-fix): each ``__anext__`` is bounded by
    ``CODEX_STREAM_HEARTBEAT_SEC``. A silent stream raises
    ``CodexStreamError`` within that window.

NOTE on sync path: the sync ``forward()`` suffers the same bug but a
test for it requires subprocess isolation (a hung ``for event in stream:``
in a pytest-owned thread kills the whole test runner). Covering the
sync path is tracked separately — the async test in this file exercises
the surface the production eval actually uses.
"""

from __future__ import annotations

import asyncio
from unittest import mock

import pytest
from dspy_codex_lm import CodexStreamError
from dspy_codex_lm import lm as codex_lm

# Safety net: without the fix, the aforward call hangs forever. Wrapping
# in ``asyncio.wait_for`` gives the test a clean, bounded failure path
# (TimeoutError != CodexStreamError → the pytest.raises assertion fails
# loudly) instead of hanging the whole pytest process.
_TEST_SAFETY_TIMEOUT = 3.0


class _NeverYieldsAsync:
    """Async iterator that simulates a fully-hung SSE stream: ``__anext__``
    parks forever in ``asyncio.sleep``, replicating the production
    deadlock where the socket is established, headers are delivered, but
    no event body ever arrives and the server never closes the connection.
    """

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(3600)  # one hour — simulates "never"
        raise AssertionError("unreachable")  # pragma: no cover


async def test_default_stream_heartbeat_matches_upstream_idle_timeout():
    assert codex_lm.CODEX_STREAM_HEARTBEAT_SEC == 300.0


async def test_async_stream_that_never_yields_raises_within_heartbeat(lm, monkeypatch):
    """The async stream consumer must raise CodexStreamError within the
    configured heartbeat window when the stream goes silent.

    RED state (no fix): heartbeat knob doesn't exist, the monkeypatch is
    skipped, the inner ``async for event in stream:`` hangs forever,
    the outer ``asyncio.wait_for(3s)`` safety net fires a TimeoutError
    → the ``pytest.raises(CodexStreamError)`` assertion fails.

    GREEN state (with fix): heartbeat=0.1s fires, tenacity retries
    (disabled to 1 attempt in tests), CodexStreamError propagates
    cleanly well under the safety net.
    """
    # Shorten the heartbeat if the fix is present. Test stays meaningful
    # in RED state (falls through to the safety net).
    if hasattr(codex_lm, "CODEX_STREAM_HEARTBEAT_SEC"):
        monkeypatch.setattr(codex_lm, "CODEX_STREAM_HEARTBEAT_SEC", 0.1)

    async def _fake_aresponses(**_):
        return _NeverYieldsAsync()

    with mock.patch("dspy_codex_lm.lm.litellm.aresponses", side_effect=_fake_aresponses):
        with pytest.raises(CodexStreamError, match="stalled"):
            await asyncio.wait_for(
                lm.aforward(prompt="hello"),
                timeout=_TEST_SAFETY_TIMEOUT,
            )


async def test_heartbeat_does_not_interrupt_healthy_stream(lm, monkeypatch):
    """Streams that emit events within the heartbeat window pass through
    unchanged — the heartbeat is a ceiling, not a floor. This guards
    against the fix regressing to a too-aggressive timeout that would
    break normal-latency responses.

    Passes in both RED and GREEN states (no-heartbeat == permissive),
    so it anchors the "healthy streams still work" property.
    """
    if hasattr(codex_lm, "CODEX_STREAM_HEARTBEAT_SEC"):
        monkeypatch.setattr(codex_lm, "CODEX_STREAM_HEARTBEAT_SEC", 1.0)

    from conftest import build_stream_events

    events = build_stream_events("ok", input_tokens=10, output_tokens=1)

    async def _async_iter():
        for e in events:
            await asyncio.sleep(0.01)  # well under the 1.0s heartbeat
            yield e

    async def _fake_aresponses(**_):
        return _async_iter()

    with mock.patch("dspy_codex_lm.lm.litellm.aresponses", side_effect=_fake_aresponses):
        resp = await lm.aforward(prompt="hi")
        assert resp.output[0].content[0].text == "ok"
