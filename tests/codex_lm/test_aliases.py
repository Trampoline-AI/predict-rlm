from dspy_codex_lm import CodexHTTPLM, CodexLM, CodexWSLM


def test_codex_lm_aliases_websocket_by_default():
    assert CodexLM is CodexWSLM


def test_http_transport_remains_available_as_codex_http_lm():
    lm = CodexHTTPLM(
        model="gpt-5.3-codex",
        access_token="fake-access",
        account_id="fake-account",
    )

    assert isinstance(lm, CodexHTTPLM)
    assert not isinstance(lm, CodexWSLM)
