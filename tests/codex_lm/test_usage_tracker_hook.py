"""CodexLM fires ``dspy.settings.usage_tracker.add_usage`` after each call.

DSPy's LM base class fires this at ``clients/lm.py:167`` (sync) and :205
(async) so that ``dspy.track_usage()`` can attribute tokens to the
prediction. CodexLM overrides ``forward``/``aforward`` entirely, which
before this fix silently bypassed the hook. Downstream consumers
(predict-rlm's ``pred.get_lm_usage()`` accumulation, cost_log, etc.)
then saw $0 for every CodexLM-routed call.

These tests pin the hook contract:
  - Within a ``track_usage()`` context, each CodexLM call populates the
    tracker under the LM's ``self.model`` slug.
  - Cache hits DO NOT fire the hook (matching DSPy's behavior).
  - Without a tracker in context, the call succeeds silently.

Real tests exercise the full CodexLM.aforward path with litellm mocked
at the responses layer — not the LM method itself — so the hook's
guard conditions run against real response objects, not MagicMocks
that would silently make ``dict(usage)`` empty.
"""

import copy
from unittest import mock

import dspy
from conftest import build_stream_events
from dspy.utils.usage_tracker import UsageTracker


def test_aforward_populates_usage_tracker(lm):
    events = build_stream_events("hi", input_tokens=1000, output_tokens=50)

    import asyncio

    async def _fake_aresponses(**_):
        async def _gen():
            for ev in copy.deepcopy(events):
                yield ev

        return _gen()

    tracker = UsageTracker()

    async def run():
        with dspy.settings.context(usage_tracker=tracker):
            with mock.patch(
                "dspy_codex_lm.lm.litellm.aresponses",
                side_effect=_fake_aresponses,
            ):
                await lm.aforward(prompt="hi")

    asyncio.run(run())
    totals = tracker.get_total_tokens()
    assert lm.model in totals, (
        f"expected tracker to have {lm.model!r}, got {list(totals.keys())}"
    )
    assert totals[lm.model]["prompt_tokens"] == 1000
    assert totals[lm.model]["completion_tokens"] == 50


def test_forward_populates_usage_tracker(lm):
    events = build_stream_events("hi", input_tokens=1000, output_tokens=50)

    def _fake_responses(**_):
        return iter(copy.deepcopy(events))

    tracker = UsageTracker()

    with dspy.settings.context(usage_tracker=tracker):
        with mock.patch(
            "dspy_codex_lm.lm.litellm.responses",
            side_effect=_fake_responses,
        ):
            lm.forward(prompt="hi")

    totals = tracker.get_total_tokens()
    assert totals[lm.model]["prompt_tokens"] == 1000
    assert totals[lm.model]["completion_tokens"] == 50


def test_no_tracker_in_context_call_still_succeeds(lm):
    """When no ``usage_tracker`` is set in context, the hook does nothing
    and the call proceeds normally — no crash, no exception.
    """
    events = build_stream_events("hi", input_tokens=1000, output_tokens=50)

    def _fake_responses(**_):
        return iter(copy.deepcopy(events))

    # No usage_tracker set
    with mock.patch(
        "dspy_codex_lm.lm.litellm.responses",
        side_effect=_fake_responses,
    ):
        resp = lm.forward(prompt="hi")
    assert resp.output[0].content[0].text == "hi"


def test_track_usage_via_context_populates_pred_lm_usage(lm):
    """The end-to-end integration: wrap ``dspy.Predict`` in
    ``track_usage=True``, predict once, confirm ``pred.get_lm_usage()``
    shows the CodexLM's tokens. This is the exact path predict-rlm's
    cost accounting depends on.
    """
    import asyncio

    class Sig(dspy.Signature):
        q: str = dspy.InputField()
        a: str = dspy.OutputField()

    dspy.settings.configure(lm=lm)
    predict = dspy.Predict(Sig)

    # Completion text in the content must be a valid JSON/Chat response
    # the adapter can parse — we use JSON-format output for simplicity.
    events = build_stream_events('{"a": "42"}', input_tokens=500, output_tokens=20)

    async def _fake_aresponses(**_):
        async def _gen():
            for ev in copy.deepcopy(events):
                yield ev

        return _gen()

    async def run():
        with dspy.settings.context(track_usage=True):
            with mock.patch(
                "dspy_codex_lm.lm.litellm.aresponses",
                side_effect=_fake_aresponses,
            ):
                pred = await predict.acall(q="hi")
        return pred.get_lm_usage()

    usage = asyncio.run(run())
    assert usage, f"pred.get_lm_usage() should be populated, got {usage!r}"
    assert lm.model in usage
    assert usage[lm.model]["prompt_tokens"] > 0


def test_cache_hit_skips_usage_tracker_hook(lm, monkeypatch):
    """A cached response MUST NOT populate the tracker — no new tokens
    were consumed. Mirrors DSPy's own ``not getattr(results, "cache_hit",
    False)`` guard.
    """
    events = build_stream_events("hi", input_tokens=1000, output_tokens=50)

    def _fake_responses(**_):
        return iter(copy.deepcopy(events))

    tracker = UsageTracker()

    with dspy.settings.context(usage_tracker=tracker):
        with mock.patch(
            "dspy_codex_lm.lm.litellm.responses",
            side_effect=_fake_responses,
        ):
            # Warm the cache
            lm.forward(prompt="same prompt")
            # Second call = cache hit
            lm.forward(prompt="same prompt")

    totals = tracker.get_total_tokens()
    # Only the first (non-cache-hit) call populated tokens — not 2×
    assert totals[lm.model]["prompt_tokens"] == 1000, (
        f"cache hit should not double-count: got {totals}"
    )
