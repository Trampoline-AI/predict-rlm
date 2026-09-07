import copy
from types import SimpleNamespace
from unittest import mock

from conftest import build_stream_events
from dspy_codex_lm import CodexHTTPLM, CodexStreamError, CodexWSLM


class FakeWSTransport:
    def __init__(self, streams):
        self.streams = iter(streams)
        self.turns = []

    def stream_turn(self, *, request, headers, request_id, sticky_state):
        self.turns.append((request_id, sticky_state))
        stream = next(self.streams)
        if isinstance(stream, BaseException):
            raise stream
        return iter(copy.deepcopy(stream))


def test_retry_shares_turn_state_without_leaking_it_to_next_request(monkeypatch):
    monkeypatch.setattr("dspy_codex_lm.lm.CODEX_STREAM_MAX_ATTEMPTS", 2)
    failed = SimpleNamespace(
        type="response.failed",
        response=SimpleNamespace(
            error=SimpleNamespace(code="rate_limit_exceeded", message="slow down")
        ),
    )
    transport = FakeWSTransport(
        [
            [failed],
            build_stream_events("recovered"),
            build_stream_events("next"),
        ]
    )
    lm = CodexWSLM(
        model="gpt-5.3-codex",
        access_token="fake",
        account_id="fake",
        ws_transport=transport,
        ws_fallback=False,
    )
    assert lm.forward(prompt="retry", cache=False).output[0].content[0].text == "recovered"
    assert lm.forward(prompt="next", cache=False).output[0].content[0].text == "next"
    first, retry, subsequent = transport.turns
    assert first[0] == retry[0]
    assert first[1] is retry[1]
    assert subsequent[0] != retry[0]
    assert subsequent[1] is not retry[1]


def test_exhausted_websocket_stays_on_http_for_later_invocations():
    transport = FakeWSTransport([CodexStreamError("ws unavailable")])
    fallback = CodexHTTPLM(model="gpt-5.3-codex", access_token="fake", account_id="fake")
    lm = CodexWSLM(
        model="gpt-5.3-codex",
        access_token="fake",
        account_id="fake",
        ws_transport=transport,
        fallback_lm=fallback,
    )
    with mock.patch(
        "dspy_codex_lm.lm.litellm.responses",
        side_effect=[
            iter(build_stream_events("first", input_tokens=5, output_tokens=1)),
            iter(build_stream_events("second", input_tokens=9, output_tokens=2)),
        ],
    ):
        first = lm.forward(prompt="first", cache=False)
        second = lm.forward(prompt="second", cache=False)
    assert first.output[0].content[0].text == "first"
    assert first.usage.input_tokens == 5
    assert second.output[0].content[0].text == "second"
    assert second.usage.input_tokens == 9
    assert len(transport.turns) == 1
