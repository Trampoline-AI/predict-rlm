import json
from pathlib import Path

import pytest

from predict_rlm.debug import debug_event, reset_debug_logger_for_tests
from predict_rlm.telemetry import (
    JsonlTelemetrySink,
    TelemetryContext,
    candidate_hash,
)


def test_jsonl_sink_creates_parent_directory_and_writes_one_object_per_line(tmp_path: Path):
    path = tmp_path / "run" / "telemetry" / "events.jsonl"
    sink = JsonlTelemetrySink(path)
    context = TelemetryContext(sink=sink, trace_id="trace-1", parent_span_id="parent-1")

    context.write_span(
        "sandbox.execute",
        event_domain="sandbox",
        span_id="span-1",
        start_time_unix_nano=1_000_000_000,
        end_time_unix_nano=1_250_000_000,
        status={"code": "ERROR", "message": "OPENAI_API_KEY=secret-value failed"},
        attributes={
            "failure.class": "sandbox_exec_timeout",
            "db.password": "plain-password",
            "nested": {"message": "Bearer secret-token", "safe": "ok"},
            "list": ["GITHUB_TOKEN=ghp_realvalue", "safe"],
        },
    )
    context.write_span(
        "spreadbench.case",
        event_domain="spreadbench",
        span_id="span-2",
        duration_ms=5,
        status="OK",
    )

    serialized = path.read_text(encoding="utf-8")
    first, second = [json.loads(line) for line in serialized.splitlines()]
    assert [first["span_id"], second["span_id"]] == ["span-1", "span-2"]
    assert first["trace_id"] == second["trace_id"] == "trace-1"
    assert first["parent_span_id"] == "parent-1"
    assert first["duration_ms"] == 250
    assert second["duration_ms"] == 5
    assert first["status"]["message"] == "OPENAI_API_KEY=[REDACTED] failed"
    assert first["attributes"]["failure.class"] == "sandbox_exec_timeout"
    assert first["attributes"]["db.password"] == "[REDACTED]"
    assert first["attributes"]["nested"] == {"message": "[REDACTED]", "safe": "ok"}
    assert first["attributes"]["list"] == ["GITHUB_TOKEN=[REDACTED]", "safe"]
    for secret in ("secret-value", "plain-password", "secret-token", "ghp_realvalue"):
        assert secret not in serialized


def test_candidate_hash_is_deterministic_for_equivalent_dicts():
    left = {"b": [2, 3], "a": {"nested": True}}
    right = {"a": {"nested": True}, "b": [2, 3]}

    assert candidate_hash(left) == candidate_hash(right)
    assert candidate_hash(left) != candidate_hash({"a": {"nested": False}, "b": [2, 3]})


@pytest.fixture
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


def test_json_debug_logging_redacts_obvious_secrets(monkeypatch, tmp_path, reset_debug_logging):
    log_path = tmp_path / "predict-rlm-debug.jsonl"
    monkeypatch.setenv("PREDICT_RLM_DEBUG", "1")
    monkeypatch.setenv("PREDICT_RLM_DEBUG_LOG", str(log_path))
    monkeypatch.setenv("PREDICT_RLM_DEBUG_JSON", "1")

    debug_event(
        "predict_rlm.redact",
        api_key="sk-testsecret123456",
        nested={"authorization": "Bearer abcdefghijk"},
        harmless="visible",
        value="Bearer value-secret",
    )

    record = json.loads(log_path.read_text())
    assert record["api_key"] == "[REDACTED]"
    assert record["nested"]["authorization"] == "[REDACTED]"
    assert record["harmless"] == "visible"
    assert record["value"] == "[REDACTED]"
    assert "sk-testsecret123456" not in log_path.read_text()
    assert "Bearer abcdefghijk" not in log_path.read_text()
