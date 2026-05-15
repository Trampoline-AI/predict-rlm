import asyncio
import time
from unittest import mock

from conftest import build_stream_events


async def _delayed_async_iter(items, delay):
    await asyncio.sleep(delay)
    for item in items:
        yield item


def _patch_slow_aresponses(events, delay):
    async def fake(**_):
        # Each call gets its own async iterator so concurrent iteration works
        return _delayed_async_iter(events, delay)

    return mock.patch("dspy_codex_lm.lm.litellm.aresponses", side_effect=fake)


async def test_concurrent_cache_hits_fast(lm):
    events = build_stream_events("answer", input_tokens=10, output_tokens=3)
    with _patch_slow_aresponses(events, delay=0.1):
        # Warm the cache with one call
        await lm.aforward(prompt="same question")

    # Now patch with a fresh slow responder, but all these calls should hit cache
    with _patch_slow_aresponses(events, delay=10.0):  # if cache misses we'd hang
        t0 = time.monotonic()
        results = await asyncio.gather(*[lm.aforward(prompt="same question") for _ in range(5)])
        dt = time.monotonic() - t0

    assert len(results) == 5
    assert all(r.output[0].content[0].text == "answer" for r in results)
    # 5 cache hits should complete far under the 10s delay
    assert dt < 0.5, f"cache hits took {dt:.3f}s, expected << 10s"


async def test_concurrent_misses_run_in_parallel(lm):
    """5 distinct prompts = 5 misses. Should run concurrently, not serially."""
    events = build_stream_events("ok", input_tokens=10, output_tokens=1)
    delay = 0.1
    with _patch_slow_aresponses(events, delay=delay):
        t0 = time.monotonic()
        await asyncio.gather(*[lm.aforward(prompt=f"question {i}") for i in range(5)])
        dt = time.monotonic() - t0

    # Serialized would be ~5*delay = 0.5s; parallel should be ~delay = 0.1s.
    # Allow some headroom for scheduling overhead.
    assert dt < delay * 2.5, f"concurrent misses took {dt:.3f}s, expected ~{delay:.2f}s"
