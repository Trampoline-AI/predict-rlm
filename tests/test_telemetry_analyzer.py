import pytest

from predict_rlm.telemetry import classify_failure
from rlm_gepa.runtime.telemetry_analyzer import analyze_run, analyze_trace_rows

pytestmark = pytest.mark.gepa


def test_precedence_prefers_sandbox_lifecycle_over_all_other_classes():
    events = [
        {"attributes": {"failure.class": "model_generated_bad_code"}},
        {"attributes": {"failure.class": "resource_saturation_unknown"}},
        {"attributes": {"failure.class": "evaluator_exception"}},
        {"attributes": {"failure.class": "outer_task_timeout"}},
        {"attributes": {"failure.class": "host_tool_timeout_or_leak"}},
        {"attributes": {"failure.class": "sandbox_exec_timeout"}},
        {"attributes": {"failure.class": "rlm_iteration_execution_timeout"}},
        {"attributes": {"failure.class": "sandbox_lifecycle_failure"}},
    ]

    assert classify_failure({"score": 0}, events) == "sandbox_lifecycle_failure"


def test_model_output_truncated_precedes_generic_no_code_failure():
    events = [
        {
            "name": "rlm.action_generation.parse_error",
            "status": {"code": "ERROR", "message": "parse failed"},
            "attributes": {
                "failure.class": "model_no_code_generated",
                "lm.truncated": True,
                "lm.truncation_reason": "max_tokens",
                "lm.finish_reason": "length",
                "lm.max_tokens": 50000,
                "lm.output_tokens": 50000,
            },
        }
    ]

    assert classify_failure({"score": 0}, events) == "model_output_truncated"


def test_row_failure_class_participates_in_precedence_with_events():
    row = {"score": 0, "failure_class": "outer_task_timeout"}
    events = [{"attributes": {"failure.class": "sandbox_exec_timeout"}}]

    assert classify_failure(row, events) == "sandbox_exec_timeout"


def test_analyzer_loads_task_traces_and_telemetry_events(tmp_path):
    trace_dir = tmp_path / "task_traces"
    telemetry_dir = tmp_path / "telemetry"
    trace_dir.mkdir()
    telemetry_dir.mkdir()
    trace_id = "run:cand:valset:0:example"
    (trace_dir / "eval.jsonl").write_text(
        "\n".join(
            [
                '{"example_id":"example","candidate_hash":"cand","score":0,'
                '"telemetry_ref":{"trace_id":"run:cand:valset:0:example"}}',
                '{"example_id":"ok","candidate_hash":"cand","score":1}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (telemetry_dir / "events.jsonl").write_text(
        "\n".join(
            [
                (
                    '{"trace_id":"%s","name":"sandbox.execute","event_domain":"sandbox",'
                    '"status":{"code":"ERROR","message":"execution timed out"},'
                    '"attributes":{"failure.class":"sandbox_exec_timeout"}}'
                )
                % trace_id,
                (
                    '{"trace_id":"health","name":"sandbox.health_check",'
                    '"event_domain":"sandbox","status":{"code":"ERROR",'
                    '"message":"No response during health check"},'
                    '"attributes":{"rlm.candidate_hash":"cand","rlm.attempt_id":"attempt_0000",'
                    '"rlm.eval_kind":"valset","spreadbench.example_id":"example"}}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyze_run(tmp_path)

    assert report.zero_rows_by_failure_class == {"sandbox_exec_timeout": 1}
    assert report.classification_table()[0]["failure_class"] == "sandbox_exec_timeout"
    assert report.candidate_scores == [
        {
            "candidate": "cand",
            "row_count": 2,
            "raw_score": 0.5,
            "infra_excluded_score": 1.0,
            "infra_excluded_rows": 1,
        }
    ]
    assert report.health_check_failures[0]["candidate_hash"] == "cand"
    assert report.sandbox_tool_timeout_counts["sandbox"] == 1


def test_analyzer_marks_missing_partial_evidence_unknown():
    report = analyze_trace_rows(
        [{"example_id": "missing", "candidate_hash": "cand", "score": 0}],
        [{"trace_id": "other", "name": "sandbox.execute", "status": {"code": "OK"}}],
    )

    assert report.zero_rows_by_failure_class == {"unknown": 1}
    assert report.unknown_rows[0]["example_id"] == "missing"
