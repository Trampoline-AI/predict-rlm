"""CodexLM auto-retries transient stream failures.

Codex streams drop or rate-limit frequently in long-running workloads
(24h+ optimize runs). Before this contract, a single dropped stream would
propagate up to GEPA and — depending on where the call lives — either
crash the proposer (losing an hour of inner-loop state) or zero out a
minibatch case. We now retry with exponential backoff so transient
errors are invisible above the LM layer.

Persistent failures (all retries exhausted) still raise CodexStreamError
— the caller deserves to see the real failure mode, not a silent succeed.
"""

import copy
from unittest import mock

import pytest
from conftest import build_stream_events
from dspy_codex_lm import CodexStreamError


def _failed_events(code: str = "rate_limit_exceeded", msg: str = "slow down"):
    from types import SimpleNamespace

    return [
        SimpleNamespace(
            type="response.failed",
            response=SimpleNamespace(
                error=SimpleNamespace(code=code, message=msg),
            ),
        )
    ]


def test_retries_on_transient_stream_failure_and_succeeds(lm, monkeypatch):
    """One transient failure, then success → the caller sees the success
    and the retry happens invisibly.
    """
    # Override the conftest's test-default (max_attempts=1) to actually
    # exercise retries. Wait knobs stay zero so the test is instant.
    monkeypatch.setattr("dspy_codex_lm.lm.CODEX_STREAM_MAX_ATTEMPTS", 4)

    good_events = build_stream_events("ok", input_tokens=5, output_tokens=1)
    call_sequence = [
        iter(copy.deepcopy(_failed_events())),
        iter(copy.deepcopy(good_events)),
    ]

    def fake(**_):
        return call_sequence.pop(0)

    with mock.patch("dspy_codex_lm.lm.litellm.responses", side_effect=fake):
        resp = lm.forward(prompt="hi")

    assert resp.output[0].content[0].text == "ok"
    # Both the failing and succeeding stream were consumed
    assert call_sequence == []


def test_retries_exhausted_raises_codex_stream_error(lm, monkeypatch):
    """After all configured retries fail, the last CodexStreamError
    propagates with the real upstream error message.
    """
    monkeypatch.setattr("dspy_codex_lm.lm.CODEX_STREAM_MAX_ATTEMPTS", 4)

    attempts = {"n": 0}

    def fake(**_):
        attempts["n"] += 1
        return iter(copy.deepcopy(_failed_events(code="503", msg="upstream down")))

    with mock.patch("dspy_codex_lm.lm.litellm.responses", side_effect=fake):
        with pytest.raises(CodexStreamError, match="upstream down"):
            lm.forward(prompt="hi")

    # max_attempts attempts were made in total
    from dspy_codex_lm.lm import CODEX_STREAM_MAX_ATTEMPTS

    assert attempts["n"] == CODEX_STREAM_MAX_ATTEMPTS


async def test_aforward_retries_on_transient_failure(lm, monkeypatch):
    """The async path mirrors the sync retry behaviour."""
    monkeypatch.setattr("dspy_codex_lm.lm.CODEX_STREAM_MAX_ATTEMPTS", 4)

    good_events = build_stream_events("ok", input_tokens=5, output_tokens=1)

    async def _make_fail_iter():
        for ev in _failed_events():
            yield ev

    async def _make_good_iter():
        for ev in copy.deepcopy(good_events):
            yield ev

    call_sequence = [_make_fail_iter(), _make_good_iter()]

    async def fake(**_):
        return call_sequence.pop(0)

    with mock.patch("dspy_codex_lm.lm.litellm.aresponses", side_effect=fake):
        resp = await lm.aforward(prompt="hi")

    assert resp.output[0].content[0].text == "ok"


def test_pydantic_unexpected_value_warning_is_silenced():
    """``PydanticSerializationUnexpectedValue(... ResponseAPIUsage ...)``
    fires whenever a downstream pydantic model (DSPy's Prediction, a
    user's logging wrapper, etc.) serializes a field declared as
    ``ResponseAPIUsage`` whose actual value carries our Chat-Completions
    aliases (``prompt_tokens`` / ``completion_tokens``). The aliases are
    load-bearing for cost trackers, so rather than dropping them we
    silence the resulting UserWarning at module load via
    ``_install_pydantic_warning_filter``.
    """
    import warnings

    from dspy_codex_lm.lm import _install_pydantic_warning_filter

    with warnings.catch_warnings(record=True) as caught:
        warnings.resetwarnings()
        _install_pydantic_warning_filter()
        # Emit the exact warning shape the user saw from pydantic.
        warnings.warn(
            "Pydantic serializer warnings:\n"
            "  PydanticSerializationUnexpectedValue(Expected "
            "`ResponseAPIUsage` - serialized value may not be as "
            "expected [field_name='usage', input_value={...}])",
            UserWarning,
            stacklevel=2,
        )

    assert not caught, (
        f"expected ResponseAPIUsage serializer warnings to be silenced, "
        f"got: {[str(w.message)[:80] for w in caught]}"
    )
