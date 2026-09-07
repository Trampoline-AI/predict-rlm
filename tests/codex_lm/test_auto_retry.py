"""Retry recovery, exhaustion, and account failover."""

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


def test_retry_uses_alternate_rotation_profile_after_stream_stall(tmp_path, monkeypatch):
    from dspy_codex_lm import CodexHTTPLM as CodexLM
    from dspy_codex_lm.auth import import_auth_profile
    from dspy_codex_lm.cli import main

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr("dspy_codex_lm.lm.CODEX_STREAM_MAX_ATTEMPTS", 2)
    import_auth_profile(
        "alpha",
        _write_auth(
            tmp_path / "alpha.json", access_token="alpha-token", account_id="acct-alpha"
        ),
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
        with pytest.raises(CodexStreamError, match="upstream down") as caught:
            lm.forward(prompt="hi")

    assert caught.value.failure_kind == "failed"
    assert caught.value.failure_code == "503"
    assert attempts["n"] == 4
