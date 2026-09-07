import asyncio
from unittest import mock

from conftest import build_stream_events


async def test_concurrent_requests_keep_results_and_cache_entries_isolated(lm):
    entered = 0
    both_entered = asyncio.Event()
    streams = iter(
        [
            build_stream_events("alpha", input_tokens=10, output_tokens=1),
            build_stream_events("beta", input_tokens=20, output_tokens=2),
        ]
    )

    async def fake_aresponses(**_):
        nonlocal entered
        events = next(streams)
        entered += 1
        if entered == 2:
            both_entered.set()

        async def stream():
            await both_entered.wait()
            for event in events:
                yield event
                await asyncio.sleep(0)

        return stream()

    prompts = ["first", "second"]
    with mock.patch("dspy_codex_lm.lm.litellm.aresponses", side_effect=fake_aresponses):
        fresh = await asyncio.wait_for(
            asyncio.gather(*(lm.aforward(prompt=prompt) for prompt in prompts)),
            timeout=3,
        )
    assert {response.output[0].content[0].text for response in fresh} == {"alpha", "beta"}
    assert {response.usage.input_tokens for response in fresh} == {10, 20}
    expected = [response.output[0].content[0].text for response in fresh]

    with mock.patch(
        "dspy_codex_lm.lm.litellm.aresponses",
        side_effect=AssertionError("cache hit reached transport"),
    ):
        cached = await asyncio.gather(*(lm.aforward(prompt=prompt) for prompt in prompts))
    assert [response.output[0].content[0].text for response in cached] == expected
