"""Jittered exponential backoff — prevents thundering-herd retries.

Context:
    With ``wait_exponential``, every retry wakes up at exactly the same
    offset after a failure (t+2s, t+6s, t+14s). When N concurrent
    callers all hit a rate-limit window simultaneously (e.g. 30 eval
    workers hammering Codex during peak hours), their retries land in
    perfect lockstep — recreating the same concentrated load on the
    very endpoint that was already struggling. Backoff is supposed to
    *spread* load, not re-concentrate it.

    Switching to ``wait_random_exponential`` uniformly randomizes each
    wait in [0, exponential_cap_at_attempt], so N concurrent retries
    fan out across the backoff window instead of arriving in lockstep.

RED (pre-change): ``wait_exponential(multiplier=2, max=8)`` called at
    ``attempt_number=2`` always returns 4.0s — deterministic, no
    variance. The "multiple calls produce distinct waits" assertion
    fails.

GREEN (post-change): ``wait_random_exponential(multiplier=2, max=8)``
    at ``attempt_number=2`` returns a fresh uniform sample from [0, 4]
    on each invocation — multiple calls yield a distribution of values.
"""

from __future__ import annotations

from dspy_codex_lm import lm as codex_lm
from tenacity import wait_random_exponential


def _sample_waits(wait_fn, attempt_number: int, n: int = 50) -> list[float]:
    """Invoke ``wait_fn`` ``n`` times at a fixed ``attempt_number`` and
    collect the returned wait durations. Tenacity wait callables accept
    a ``RetryCallState`` and return the wait in seconds; we fabricate a
    minimal state to avoid booting a real retry loop.
    """

    class _MinimalState:
        def __init__(self, attempt_no: int):
            self.attempt_number = attempt_no
            self.outcome = None
            self.outcome_timestamp = None
            self.start_time = 0.0
            self.retry_object = None
            self.args = ()
            self.kwargs = {}
            self.fn = None
            self.idle_for = 0.0
            self.next_action = None

    state = _MinimalState(attempt_number)
    return [wait_fn(state) for _ in range(n)]


def test_retry_wait_is_jittered(monkeypatch):
    """Pulling the actual wait function from the production retry kwargs
    and sampling it 50 times at attempt=2 must show variance > 0 —
    deterministic ``wait_exponential`` yields a single repeated value,
    jittered ``wait_random_exponential`` yields a uniform spread.

    The conftest auto-fixture zeroes ``CODEX_STREAM_WAIT_MAX`` to keep
    unrelated tests fast. Restore a non-zero cap here so the jitter
    has something to spread over.
    """
    monkeypatch.setattr(codex_lm, "CODEX_STREAM_WAIT_MULTIPLIER", 2.0)
    monkeypatch.setattr(codex_lm, "CODEX_STREAM_WAIT_MAX", 8.0)

    kwargs = codex_lm._codex_retry_kwargs()
    wait_fn = kwargs["wait"]

    samples = _sample_waits(wait_fn, attempt_number=2, n=50)

    # Jitter test: at least 10 distinct values out of 50 samples. With a
    # uniform [0, 4] distribution, the probability of fewer than 10
    # distinct values is effectively zero; with a deterministic function
    # there's only ever one distinct value.
    distinct = len(set(samples))
    assert distinct >= 10, (
        f"expected ≥10 distinct wait values across 50 samples (indicates "
        f"jittered backoff), got {distinct}: {sorted(set(samples))[:5]}..."
    )


def test_retry_wait_respects_exponential_cap(monkeypatch):
    """The jittered wait should still honor ``CODEX_STREAM_WAIT_MAX`` —
    a [0, cap] uniform sample never exceeds the cap. Guards against a
    hypothetical future swap to an unbounded jitter strategy.
    """
    monkeypatch.setattr(codex_lm, "CODEX_STREAM_WAIT_MULTIPLIER", 2.0)
    monkeypatch.setattr(codex_lm, "CODEX_STREAM_WAIT_MAX", 8.0)

    kwargs = codex_lm._codex_retry_kwargs()
    wait_fn = kwargs["wait"]

    # Sample at a high attempt_number where a deterministic
    # ``wait_exponential`` would have saturated at ``max`` (8.0) long
    # ago. For ``wait_random_exponential`` the cap is the upper bound
    # of the uniform distribution.
    samples = _sample_waits(wait_fn, attempt_number=10, n=100)

    assert max(samples) <= 8.0 + 1e-6, (
        f"wait exceeded configured max of 8.0s: max observed = {max(samples)}"
    )
    assert min(samples) >= 0, f"wait went negative: min = {min(samples)}"


def test_retry_wait_function_is_jitter_type():
    """Source-anchor: the production retry config must use a jittered
    wait strategy. If someone ever swaps it back to the deterministic
    ``wait_exponential``, this fails explicitly instead of regressing
    silently into a thundering-herd pattern.
    """
    kwargs = codex_lm._codex_retry_kwargs()
    wait_fn = kwargs["wait"]
    assert isinstance(wait_fn, wait_random_exponential), (
        "retry wait strategy regressed to deterministic wait_exponential — "
        "this re-enables the thundering-herd failure mode that the jitter "
        "swap was introduced to prevent"
    )
