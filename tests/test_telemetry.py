import json
from pathlib import Path

import pytest

from predict_rlm.telemetry import (
    JsonlTelemetrySink,
    NoopTelemetrySink,
    TelemetryContext,
    candidate_hash,
    make_span_record,
    redact_attributes,
)


def test_jsonl_sink_creates_parent_directory_and_writes_one_object_per_line(tmp_path: Path):
    path = tmp_path / "run" / "telemetry" / "events.jsonl"
    sink = JsonlTelemetrySink(path)
    context = TelemetryContext(
        sink=sink,
        trace_id="trace-1",
        parent_span_id="parent-1",
        run_id="run-1",
        eval_kind="valset",
        eval_idx=2,
        attempt_id="0002",
        example_id="581-46",
        case_idx=56,
        candidate_hash="cand_sha256_abc",
    )

    first = context.write_span(
        "sandbox.execute",
        event_domain="sandbox",
        span_id="span-1",
        start_time_unix_nano=1_000_000_000,
        end_time_unix_nano=1_250_000_000,
        status={"code": "ERROR", "message": "Sandbox exec timed out after 250ms"},
        attributes={"failure.class": "sandbox_exec_timeout", "process.pid": 123},
    )
    second = context.write_span(
        "spreadbench.case",
        event_domain="spreadbench",
        span_id="span-2",
        duration_ms=5,
        status="OK",
    )

    assert path.parent.is_dir()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["span_id"] for line in lines] == ["span-1", "span-2"]

    assert first["duration_ms"] == 250
    assert first["schema_version"] == 1
    assert first["record_type"] == "span"
    assert first["event_domain"] == "sandbox"
    assert first["trace_id"] == "trace-1"
    assert first["span_id"] == "span-1"
    assert first["parent_span_id"] == "parent-1"
    assert first["name"] == "sandbox.execute"
    assert first["span_kind"] == "internal"
    assert first["start_time_unix_nano"] == 1_000_000_000
    assert first["end_time_unix_nano"] == 1_250_000_000
    assert first["status"]["code"] == "ERROR"
    assert first["attributes"]["rlm.run_id"] == "run-1"
    assert first["attributes"]["rlm.eval_kind"] == "valset"
    assert first["attributes"]["rlm.eval_idx"] == 2
    assert first["attributes"]["rlm.attempt_id"] == "0002"
    assert first["attributes"]["spreadbench.example_id"] == "581-46"
    assert first["attributes"]["spreadbench.case_idx"] == 56
    assert first["attributes"]["rlm.candidate_hash"] == "cand_sha256_abc"
    assert first["attributes"]["failure.class"] == "sandbox_exec_timeout"

    assert second["duration_ms"] == 5


def test_make_span_record_can_override_parent_and_redacts_status_message():
    context = TelemetryContext(
        sink=NoopTelemetrySink(),
        trace_id="trace-1",
        parent_span_id="parent-1",
    )

    record = make_span_record(
        context,
        name="host_tool.recalculate",
        event_domain="host_tool",
        span_id="span-1",
        parent_span_id="override-parent",
        status={"code": "ERROR", "message": "OPENAI_API_KEY=secret-value failed"},
        attributes={"safe": "value"},
    )

    assert record["parent_span_id"] == "override-parent"
    assert record["status"]["message"] == "OPENAI_API_KEY=[REDACTED] failed"


def test_noop_sink_and_disabled_jsonl_sink_do_not_write(tmp_path: Path):
    path = tmp_path / "telemetry" / "events.jsonl"
    NoopTelemetrySink().write({"trace_id": "trace-1"})

    sink = JsonlTelemetrySink(path, enabled=False)
    sink.write({"trace_id": "trace-1"})

    assert not path.exists()


def test_jsonl_sink_write_errors_are_best_effort(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    path = tmp_path / "telemetry" / "events.jsonl"
    sink = JsonlTelemetrySink(path)

    def fail_open(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "open", fail_open)

    try:
        raise RuntimeError("caller failure")
    except RuntimeError as exc:
        sink.write({"trace_id": "trace-1", "bad": object()})
        assert str(exc) == "caller failure"


def test_candidate_hash_is_deterministic_for_equivalent_dicts():
    left = {"b": [2, 3], "a": {"nested": True}}
    right = {"a": {"nested": True}, "b": [2, 3]}

    assert candidate_hash(left) == candidate_hash(right)
    assert candidate_hash(left).startswith("cand_sha256_")
    assert candidate_hash(left) != candidate_hash({"a": {"nested": False}, "b": [2, 3]})


def test_redact_attributes_removes_obvious_secret_values():
    redacted = redact_attributes(
        {
            "OPENAI_API_KEY": "sk-real-value",
            "db.password": "plain-password",
            "message": "export GITHUB_TOKEN=ghp_realvalue and continue",
            "nested": {"auth_header": "Bearer secret-token", "safe": "ok"},
            "list": ["OPENAI_API_KEY=abc123", "safe"],
        }
    )

    assert redacted["OPENAI_API_KEY"] == "[REDACTED]"
    assert redacted["db.password"] == "[REDACTED]"
    assert redacted["message"] == "export GITHUB_TOKEN=[REDACTED] and continue"
    assert redacted["nested"]["auth_header"] == "[REDACTED]"
    assert redacted["nested"]["safe"] == "ok"
    assert redacted["list"] == ["OPENAI_API_KEY=[REDACTED]", "safe"]
