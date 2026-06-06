from __future__ import annotations

import os
from unittest import mock

import pytest
from conftest import build_stream_events
from dspy_codex_lm import CodexHTTPLM as CodexLM


def _patch_responses_capture(events, captured):
    def fake(**kwargs):
        captured["kwargs"] = kwargs
        captured["https_proxy"] = os.environ.get("HTTPS_PROXY")
        captured["http_proxy"] = os.environ.get("HTTP_PROXY")
        return iter(events)

    return mock.patch("dspy_codex_lm.lm.litellm.responses", side_effect=fake)


def test_priority_service_tier_flows_into_request():
    lm = CodexLM(
        model="gpt-5.5",
        service_tier="priority",
        access_token="fake",
        account_id="fake",
    )

    request, _ = lm._build_request(prompt="x", messages=None, kwargs={})

    assert request["service_tier"] == "priority"


def test_per_call_service_tier_overrides_constructor():
    lm = CodexLM(
        model="gpt-5.5",
        service_tier="default",
        access_token="fake",
        account_id="fake",
    )

    request, _ = lm._build_request(
        prompt="x", messages=None, kwargs={"service_tier": "priority"}
    )

    assert request["service_tier"] == "priority"


@pytest.mark.parametrize(
    ("env_name", "captured_name"),
    [("HTTPS_PROXY", "https_proxy"), ("HTTP_PROXY", "http_proxy")],
)
def test_proxy_url_is_scoped_to_litellm_call(
    monkeypatch, env_name: str, captured_name: str
):
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    lm = CodexLM(
        model="gpt-5.5",
        proxy_url="http://127.0.0.1:8898",
        access_token="fake",
        account_id="fake",
    )
    events = build_stream_events("ok", input_tokens=5, output_tokens=1)
    captured: dict = {}

    with _patch_responses_capture(events, captured):
        lm.forward(prompt="hi", cache=False)

    assert captured[captured_name] == "http://127.0.0.1:8898"
    assert os.environ.get(env_name) is None
    assert "proxy_url" not in captured["kwargs"]


async def test_async_proxy_url_is_scoped_to_litellm_call(monkeypatch):
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    lm = CodexLM(
        model="gpt-5.5",
        proxy_url="http://127.0.0.1:8898",
        access_token="fake",
        account_id="fake",
    )
    events = build_stream_events("ok", input_tokens=5, output_tokens=1)
    captured: dict = {}

    async def _aiter(items):
        for item in items:
            yield item

    async def fake(**kwargs):
        captured["kwargs"] = kwargs
        captured["https_proxy"] = os.environ.get("HTTPS_PROXY")
        captured["http_proxy"] = os.environ.get("HTTP_PROXY")
        return _aiter(events)

    with mock.patch("dspy_codex_lm.lm.litellm.aresponses", side_effect=fake):
        await lm.aforward(prompt="hi", cache=False)

    assert captured["https_proxy"] == "http://127.0.0.1:8898"
    assert captured["http_proxy"] == "http://127.0.0.1:8898"
    assert os.environ.get("HTTPS_PROXY") is None
    assert os.environ.get("HTTP_PROXY") is None
    assert "proxy_url" not in captured["kwargs"]
