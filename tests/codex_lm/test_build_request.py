import os
from unittest import mock

from conftest import build_stream_events
from dspy_codex_lm import CodexHTTPLM


def test_request_conversion_does_not_mutate_constructor_reasoning():
    reasoning = {"effort": "high"}
    lm = CodexHTTPLM(
        model="gpt-5.6-sol",
        access_token="fake",
        account_id="acct",
        reasoning=reasoning,
    )
    request, _ = lm._build_request(prompt="x", messages=None, kwargs={})
    assert request["reasoning"]["context"] == "all_turns"
    assert reasoning == {"effort": "high"}


def test_proxy_environment_is_restored_after_request(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://original:8080")
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    lm = CodexHTTPLM(
        model="gpt-5.5",
        proxy_url="http://127.0.0.1:8898",
        access_token="fake",
        account_id="fake",
    )

    def transport(**_):
        assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:8898"
        assert os.environ["HTTP_PROXY"] == "http://127.0.0.1:8898"
        return iter(build_stream_events("ok"))

    with mock.patch("dspy_codex_lm.lm.litellm.responses", side_effect=transport):
        response = lm.forward(prompt="hi", cache=False)
    assert response.output[0].content[0].text == "ok"
    assert os.environ["HTTPS_PROXY"] == "http://original:8080"
    assert "HTTP_PROXY" not in os.environ
