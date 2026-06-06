from __future__ import annotations

import copy
from dataclasses import dataclass
from types import SimpleNamespace
from unittest import mock

import pytest
from conftest import build_stream_events
from dspy_codex_lm import CodexHTTPLM, CodexStreamError, CodexWSLM


@dataclass
class RecordedTurn:
    request_id: str
    sticky_state: dict
    request: dict
    headers: dict


class FakeWSTransport:
    def __init__(self, streams):
        self.streams = list(streams)
        self.turns: list[RecordedTurn] = []

    def stream_turn(self, *, request, headers, request_id, sticky_state):
        self.turns.append(
            RecordedTurn(
                request_id=request_id,
                sticky_state=sticky_state,
                request=copy.deepcopy(request),
                headers=copy.deepcopy(headers),
            )
        )
        stream = self.streams.pop(0)
        if isinstance(stream, BaseException):
            raise stream
        return iter(copy.deepcopy(stream))

    async def astream_turn(self, *, request, headers, request_id, sticky_state):
        self.turns.append(
            RecordedTurn(
                request_id=request_id,
                sticky_state=sticky_state,
                request=copy.deepcopy(request),
                headers=copy.deepcopy(headers),
            )
        )
        stream = self.streams.pop(0)
        if isinstance(stream, BaseException):
            raise stream

        async def _events():
            for event in copy.deepcopy(stream):
                yield event

        return _events()


class FakeHTTPFallback:
    def __init__(self):
        self.forward_calls = []
        self.aforward_calls = []

    def forward(self, prompt=None, messages=None, **kwargs):
        self.forward_calls.append((prompt, messages, dict(kwargs)))
        return SimpleNamespace(source="http", prompt=prompt)

    async def aforward(self, prompt=None, messages=None, **kwargs):
        self.aforward_calls.append((prompt, messages, dict(kwargs)))
        return SimpleNamespace(source="http", prompt=prompt)


def _failed_events(code: str = "rate_limit_exceeded", msg: str = "slow down"):
    return [
        SimpleNamespace(
            type="response.failed",
            response=SimpleNamespace(
                error=SimpleNamespace(code=code, message=msg),
            ),
        )
    ]


def test_codex_ws_lm_is_exported_and_accepts_codex_lm_kwargs():
    lm = CodexWSLM(
        model="gpt-5.3-codex",
        instructions="You are Ada.",
        access_token="fake-access",
        account_id="fake-account",
        proxy_url="http://127.0.0.1:9999",
        reasoning_effort="low",
    )

    request, _ = lm._build_request(prompt="hi", messages=None, kwargs={})
    assert request["instructions"] == "You are Ada."
    assert request["reasoning"]["effort"] == "low"


def test_two_forward_calls_create_distinct_ws_turns():
    transport = FakeWSTransport(
        [
            build_stream_events("one", response_id="resp-one"),
            build_stream_events("two", response_id="resp-two"),
        ]
    )
    lm = CodexWSLM(
        model="gpt-5.3-codex",
        access_token="fake-access",
        account_id="fake-account",
        ws_transport=transport,
    )

    first = lm.forward(prompt="one", cache=False)
    second = lm.forward(prompt="two", cache=False)

    assert first.output[0].content[0].text == "one"
    assert second.output[0].content[0].text == "two"
    assert len(transport.turns) == 2
    assert transport.turns[0].request_id != transport.turns[1].request_id
    assert transport.turns[0].headers["session_id"] != transport.turns[1].headers["session_id"]
    assert transport.turns[0].sticky_state is not transport.turns[1].sticky_state


def test_retry_stays_inside_one_invocation_turn_boundary(monkeypatch):
    monkeypatch.setattr("dspy_codex_lm.lm.CODEX_STREAM_MAX_ATTEMPTS", 2)
    transport = FakeWSTransport(
        [
            _failed_events(),
            build_stream_events("ok", response_id="resp-ok"),
            build_stream_events("next", response_id="resp-next"),
        ]
    )
    lm = CodexWSLM(
        model="gpt-5.3-codex",
        access_token="fake-access",
        account_id="fake-account",
        ws_transport=transport,
    )

    first = lm.forward(prompt="retry", cache=False)
    second = lm.forward(prompt="next", cache=False)

    assert first.output[0].content[0].text == "ok"
    assert second.output[0].content[0].text == "next"
    assert len(transport.turns) == 3
    assert transport.turns[0].request_id == transport.turns[1].request_id
    assert transport.turns[0].sticky_state is transport.turns[1].sticky_state
    assert transport.turns[1].request_id != transport.turns[2].request_id
    assert transport.turns[1].sticky_state is not transport.turns[2].sticky_state


def test_ws_fallback_exhaustion_routes_later_invocations_to_http(monkeypatch):
    monkeypatch.setattr("dspy_codex_lm.lm.CODEX_STREAM_MAX_ATTEMPTS", 1)
    transport = FakeWSTransport([CodexStreamError("ws unavailable")])
    fallback = FakeHTTPFallback()
    lm = CodexWSLM(
        model="gpt-5.3-codex",
        access_token="fake-access",
        account_id="fake-account",
        ws_transport=transport,
        fallback_lm=fallback,
    )

    first = lm.forward(prompt="first", cache=False)
    second = lm.forward(prompt="second", cache=False)

    assert first.source == "http"
    assert second.source == "http"
    assert [call[0] for call in fallback.forward_calls] == ["first", "second"]
    assert len(transport.turns) == 1


async def test_two_aforward_calls_create_distinct_ws_turns():
    transport = FakeWSTransport(
        [
            build_stream_events("one", response_id="resp-one"),
            build_stream_events("two", response_id="resp-two"),
        ]
    )
    lm = CodexWSLM(
        model="gpt-5.3-codex",
        access_token="fake-access",
        account_id="fake-account",
        ws_transport=transport,
    )

    first = await lm.aforward(prompt="one", cache=False)
    second = await lm.aforward(prompt="two", cache=False)

    assert first.output[0].content[0].text == "one"
    assert second.output[0].content[0].text == "two"
    assert len(transport.turns) == 2
    assert transport.turns[0].request_id != transport.turns[1].request_id
    assert transport.turns[0].sticky_state is not transport.turns[1].sticky_state


@pytest.mark.parametrize("lm_kind", ["http", "ws"])
def test_protocol_forward_response_contract_is_shared(lm_kind):
    events = build_stream_events("contract", input_tokens=100, output_tokens=1)
    if lm_kind == "http":
        lm = CodexHTTPLM(
            model="gpt-5.3-codex",
            access_token="fake-access",
            account_id="fake-account",
        )
        with mock.patch("dspy_codex_lm.lm.litellm.responses", return_value=iter(events)):
            resp = lm.forward(prompt="shared", cache=False)
    else:
        lm = CodexWSLM(
            model="gpt-5.3-codex",
            access_token="fake-access",
            account_id="fake-account",
            ws_transport=FakeWSTransport([events]),
        )
        resp = lm.forward(prompt="shared", cache=False)

    assert resp.output[0].content[0].text == "contract"
    assert resp.usage.input_tokens == 100
    assert resp.usage.cost > 0
