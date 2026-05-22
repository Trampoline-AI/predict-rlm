import json

import pytest

from predict_rlm.debug import debug_event, reset_debug_logger_for_tests, sanitize_metadata


@pytest.fixture(autouse=True)
def reset_debug_logging(monkeypatch):
    for name in (
        "PREDICT_RLM_DEBUG",
        "RLM_DEBUG",
        "PREDICT_RLM_DEBUG_LOG",
        "PREDICT_RLM_DEBUG_JSON",
    ):
        monkeypatch.delenv(name, raising=False)
    reset_debug_logger_for_tests()
    yield
    reset_debug_logger_for_tests()


def test_debug_logging_disabled_by_default(capsys):
    debug_event("predict_rlm.test", count=1)

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_predict_rlm_debug_enables_stderr_output(monkeypatch, capsys):
    monkeypatch.setenv("PREDICT_RLM_DEBUG", "1")

    debug_event("predict_rlm.test", count=1)

    captured = capsys.readouterr()
    assert "predict_rlm.test" in captured.err
    assert "count=1" in captured.err


def test_rlm_debug_enables_stderr_output(monkeypatch, capsys):
    monkeypatch.setenv("RLM_DEBUG", "1")

    debug_event("predict_rlm.shared", enabled=True)

    captured = capsys.readouterr()
    assert "predict_rlm.shared" in captured.err
    assert "enabled=True" in captured.err


def test_predict_rlm_debug_log_writes_file(monkeypatch, tmp_path, capsys):
    log_path = tmp_path / "predict-rlm-debug.log"
    monkeypatch.setenv("PREDICT_RLM_DEBUG", "1")
    monkeypatch.setenv("PREDICT_RLM_DEBUG_LOG", str(log_path))

    debug_event("predict_rlm.file", status="ok")

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "predict_rlm.file" in log_path.read_text()


def test_json_debug_logging_redacts_obvious_secrets(monkeypatch, tmp_path):
    log_path = tmp_path / "predict-rlm-debug.jsonl"
    monkeypatch.setenv("PREDICT_RLM_DEBUG", "1")
    monkeypatch.setenv("PREDICT_RLM_DEBUG_LOG", str(log_path))
    monkeypatch.setenv("PREDICT_RLM_DEBUG_JSON", "1")

    debug_event(
        "predict_rlm.redact",
        api_key="sk-testsecret123456",
        nested={"authorization": "Bearer abcdefghijk"},
        harmless="visible",
    )

    record = json.loads(log_path.read_text())
    assert record["api_key"] == "[REDACTED]"
    assert record["nested"]["authorization"] == "[REDACTED]"
    assert record["harmless"] == "visible"
    assert "sk-testsecret123456" not in log_path.read_text()
    assert "Bearer abcdefghijk" not in log_path.read_text()


def test_sanitize_metadata_redacts_secret_values_in_nonsecret_keys():
    assert sanitize_metadata({"value": "Bearer abcdefghijk"}) == {"value": "[REDACTED]"}
