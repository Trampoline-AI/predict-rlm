"""Pins DSPy's cache-hit behavior as observed through CodexLM history.

Background: ``dspy.clients.cache.Cache.get`` on a cache hit does
``response.usage = {}`` and ``response.cache_hit = True`` but does NOT clear
``response._hidden_params["response_cost"]``. As a result, ``BaseLM`` writes
an entry with empty usage and the original call's cost. Downstream cost
aggregators that sum ``entry["cost"]`` naively will double-count.

These tests pin that contract so if DSPy changes the behaviour upstream
(e.g. zeros cost too, or keeps usage populated), we notice it here first.
"""

from unittest import mock

import pytest
from conftest import build_stream_events


def _patch_responses(events):
    import copy

    def fake(**_):
        return iter(copy.deepcopy(events))

    return mock.patch("dspy_codex_lm.lm.litellm.responses", side_effect=fake)


def test_cache_miss_history_has_populated_usage(lm):
    """Sanity: a cold call populates both input_tokens/output_tokens and
    the Chat-Completions-shaped prompt_tokens/completion_tokens aliases.
    """
    events = build_stream_events("hi", input_tokens=1000, output_tokens=50)
    with _patch_responses(events):
        lm(prompt="fresh")

    entry = lm.history[0]
    usage = dict(entry["usage"])
    assert usage["input_tokens"] == 1000
    assert usage["output_tokens"] == 50
    # dspy-codex-lm aliases these so downstream Chat-Completions consumers work
    assert usage["prompt_tokens"] == 1000
    assert usage["completion_tokens"] == 50
    assert entry["cost"] > 0


def test_cache_hit_leaves_empty_usage_and_preserves_cost(lm):
    """Upstream DSPy contract: cache hit zeros response.usage, keeps cost.

    If this test fails, DSPy has changed its Cache.get() semantics — update
    predict-rlm's ``usage_since`` cache-hit detection accordingly.
    """
    events = build_stream_events("hi", input_tokens=1000, output_tokens=50)
    call_count = 0

    def counting_fake(**_):
        nonlocal call_count
        call_count += 1
        import copy

        return iter(copy.deepcopy(events))

    with mock.patch("dspy_codex_lm.lm.litellm.responses", side_effect=counting_fake):
        lm(prompt="same prompt twice")
        lm(prompt="same prompt twice")

    # Cache worked — only one underlying API call
    assert call_count == 1
    assert len(lm.history) == 2

    hit = lm.history[1]
    hit_usage = dict(hit["usage"])
    # Cache hit: usage zeroed, cost preserved, cache_hit flag on response
    assert hit_usage == {}, (
        f"DSPy cache-hit contract changed: usage={hit_usage!r} (expected empty)"
    )
    assert hit["cost"] == pytest.approx(lm.history[0]["cost"]), (
        "DSPy cache-hit contract changed: cost was reset on cache hit "
        "(previously preserved). predict-rlm's cache-hit cost discounting "
        "may now be double-zeroing."
    )
    assert getattr(hit["response"], "cache_hit", False) is True


def test_cache_hit_cost_would_double_count_without_filter():
    """Documents the naïve-sum failure mode: summing entry['cost'] across
    fresh+cached entries reports 2x the real API spend even though only one
    call was made. predict-rlm's ``usage_since`` filters this out.
    """
    history = [
        {"usage": {"prompt_tokens": 1000, "completion_tokens": 50}, "cost": 0.001},
        {"usage": {}, "cost": 0.001},  # cache hit shape
    ]
    naive_total = sum(e["cost"] for e in history)
    # Naïve sum reports 2x what was actually charged
    assert naive_total == pytest.approx(0.002)
    # Real API spend (only cache-MISS entries with populated usage)
    real_total = sum(
        e["cost"]
        for e in history
        if e["usage"].get("prompt_tokens", 0) or e["usage"].get("completion_tokens", 0)
    )
    assert real_total == pytest.approx(0.001)
