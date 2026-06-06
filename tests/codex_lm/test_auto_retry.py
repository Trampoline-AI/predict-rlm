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
import json
from pathlib import Path
from unittest import mock

import pytest
from conftest import build_stream_events
from dspy_codex_lm import CodexStreamError


def _write_auth(path: Path, *, access_token: str, account_id: str) -> Path:
    path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": access_token,
                    "account_id": account_id,
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def _failed_events(
    code: str = "rate_limit_exceeded",
    msg: str = "slow down",
    **error_fields,
):
    from types import SimpleNamespace

    return [
        SimpleNamespace(
            type="response.failed",
            response=SimpleNamespace(
                error=SimpleNamespace(code=code, message=msg, **error_fields),
            ),
        )
    ]


def test_default_stream_attempts_match_upstream_codex():
    import dspy_codex_lm.lm as codex_lm

    assert codex_lm.DEFAULT_CODEX_STREAM_MAX_ATTEMPTS == 5


def test_response_failed_retry_after_ms_is_attached_to_stream_error(lm):
    from dspy_codex_lm.lm import _codex_stream_error_from_state

    state = lm._fresh_state()
    lm._handle_event(_failed_events(retry_after_ms=1250)[0], state)

    error = _codex_stream_error_from_state(state)

    assert error is not None
    assert error.retry_after_seconds == 1.25


def test_retry_wait_prefers_server_requested_retry_after(monkeypatch):
    from dspy_codex_lm.lm import _codex_retry_kwargs

    monkeypatch.setattr("dspy_codex_lm.lm.CODEX_STREAM_WAIT_MULTIPLIER", 0.0)
    monkeypatch.setattr("dspy_codex_lm.lm.CODEX_STREAM_WAIT_MAX", 0.0)
    wait = _codex_retry_kwargs()["wait"]
    error = CodexStreamError("slow down", retry_after_seconds=3.5)

    class Outcome:
        def exception(self):
            return error

    class RetryState:
        outcome = Outcome()

    assert wait(RetryState()) == 3.5


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


def test_retry_uses_alternate_rotation_profile_after_stream_stall(tmp_path, monkeypatch):
    from dspy_codex_lm import CodexHTTPLM as CodexLM
    from dspy_codex_lm.auth import import_auth_profile
    from dspy_codex_lm.cli import main

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr("dspy_codex_lm.lm.CODEX_STREAM_MAX_ATTEMPTS", 2)
    import_auth_profile(
        "alpha",
        _write_auth(tmp_path / "alpha.json", access_token="alpha-token", account_id="acct-alpha"),
    )
    import_auth_profile(
        "beta",
        _write_auth(tmp_path / "beta.json", access_token="beta-token", account_id="acct-beta"),
    )
    assert main(["codex-lm", "rotation", "on"]) == 0

    selected_accounts = iter(["acct-beta", "acct-beta"])

    def choose_credentials(credentials):
        credentials = tuple(credentials)
        selected = next(selected_accounts)
        return next(
            credential for credential in credentials if credential.account_id == selected
        )

    seen_headers = []
    seen_api_keys = []
    good_events = build_stream_events("ok", input_tokens=5, output_tokens=1)

    def fake_responses(*, headers, api_key, **_):
        seen_headers.append(headers["ChatGPT-Account-Id"])
        seen_api_keys.append(api_key)
        if len(seen_headers) == 1:
            raise CodexStreamError("Codex stream stalled")
        return iter(copy.deepcopy(good_events))

    monkeypatch.setattr("dspy_codex_lm.lm.random.choice", choose_credentials)
    with mock.patch("dspy_codex_lm.lm.litellm.responses", side_effect=fake_responses):
        response = CodexLM(model="gpt-5.3-codex").forward(prompt="hi")

    assert response.output[0].content[0].text == "ok"
    assert seen_headers == ["acct-beta", "acct-alpha"]
    assert seen_api_keys == ["beta-token", "alpha-token"]


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


async def test_aforward_retry_uses_alternate_rotation_profile_after_stream_stall(
    tmp_path,
    monkeypatch,
):
    from dspy_codex_lm import CodexHTTPLM as CodexLM
    from dspy_codex_lm.auth import import_auth_profile
    from dspy_codex_lm.cli import main

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr("dspy_codex_lm.lm.CODEX_STREAM_MAX_ATTEMPTS", 2)
    import_auth_profile(
        "alpha",
        _write_auth(tmp_path / "alpha.json", access_token="alpha-token", account_id="acct-alpha"),
    )
    import_auth_profile(
        "beta",
        _write_auth(tmp_path / "beta.json", access_token="beta-token", account_id="acct-beta"),
    )
    assert main(["codex-lm", "rotation", "on"]) == 0

    selected_accounts = iter(["acct-beta", "acct-beta"])

    def choose_credentials(credentials):
        credentials = tuple(credentials)
        selected = next(selected_accounts)
        return next(
            credential for credential in credentials if credential.account_id == selected
        )

    seen_headers = []
    seen_api_keys = []
    good_events = build_stream_events("ok", input_tokens=5, output_tokens=1)

    async def good_stream():
        for event in copy.deepcopy(good_events):
            yield event

    async def fake_aresponses(*, headers, api_key, **_):
        seen_headers.append(headers["ChatGPT-Account-Id"])
        seen_api_keys.append(api_key)
        if len(seen_headers) == 1:
            raise CodexStreamError("Codex stream stalled")
        return good_stream()

    monkeypatch.setattr("dspy_codex_lm.lm.random.choice", choose_credentials)
    with mock.patch("dspy_codex_lm.lm.litellm.aresponses", side_effect=fake_aresponses):
        response = await CodexLM(model="gpt-5.3-codex").aforward(prompt="hi")

    assert response.output[0].content[0].text == "ok"
    assert seen_headers == ["acct-beta", "acct-alpha"]
    assert seen_api_keys == ["beta-token", "alpha-token"]


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
