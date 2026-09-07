import json
from types import SimpleNamespace
from unittest import mock

import pytest
from dspy_codex_lm import CodexStreamError

from predict_rlm.debug import reset_debug_logger_for_tests


def test_stream_failure_logs_do_not_expose_credentials(lm, monkeypatch, tmp_path):
    log_path = tmp_path / "codex-debug.jsonl"
    for name in ("RLM_DEBUG", "CODEX_LM_DEBUG_LOG"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PREDICT_RLM_DEBUG", "1")
    monkeypatch.setenv("PREDICT_RLM_DEBUG_JSON", "1")
    monkeypatch.setenv("PREDICT_RLM_DEBUG_LOG", str(log_path))
    event = SimpleNamespace(
        type="response.failed",
        response=SimpleNamespace(
            error=SimpleNamespace(code="rate_limit_exceeded", message="slow down"),
        ),
    )
    reset_debug_logger_for_tests()
    try:
        with mock.patch("dspy_codex_lm.lm.litellm.responses", return_value=iter([event])):
            with pytest.raises(CodexStreamError):
                lm.forward(prompt="hi", cache=False)
        text = log_path.read_text()
        records = [json.loads(line) for line in text.splitlines()]
        assert any(record["event"] == "codex_lm.stream.error" for record in records)
        assert "fake-access" not in text
        assert "fake-account" not in text
    finally:
        reset_debug_logger_for_tests()
