import pytest

from predict_rlm.telemetry import classify_failure, classify_zero_score_failure


@pytest.mark.parametrize(
    ("failure_class", "expected"),
    [
        ("sandbox_lifecycle_failure", "sandbox_lifecycle_failure"),
        ("sandbox_exec_timeout", "sandbox_exec_timeout"),
        ("host_tool_timeout_or_leak", "host_tool_timeout_or_leak"),
        ("outer_task_timeout", "outer_task_timeout"),
        ("evaluator_limitation", "evaluator_limitation"),
        ("evaluator_exception", "evaluator_exception"),
        ("model_no_code_generated", "model_no_code_generated"),
        ("model_generated_bad_code", "model_generated_bad_code"),
        ("resource_saturation_unknown", "resource_saturation_unknown"),
    ],
)
def test_classifies_closed_failure_classes_from_failure_class_attribute(
    failure_class: str,
    expected: str,
):
    events = [
        {
            "name": "synthetic.event",
            "status": {"code": "ERROR"},
            "attributes": {"failure.class": failure_class},
        }
    ]

    assert classify_failure({"score": 0}, events) == expected


def test_missing_or_partial_evidence_returns_unknown():
    assert classify_failure({"score": 0}, []) == "unknown"
    assert classify_failure({"score": 0, "status": "failed"}, None) == "unknown"
    assert classify_failure(None, [{"name": "sandbox.execute", "status": {"code": "OK"}}]) == "unknown"


def test_precedence_prefers_sandbox_lifecycle_over_all_other_classes():
    events = [
        {"attributes": {"failure.class": "model_generated_bad_code"}},
        {"attributes": {"failure.class": "resource_saturation_unknown"}},
        {"attributes": {"failure.class": "evaluator_exception"}},
        {"attributes": {"failure.class": "outer_task_timeout"}},
        {"attributes": {"failure.class": "host_tool_timeout_or_leak"}},
        {"attributes": {"failure.class": "sandbox_exec_timeout"}},
        {"attributes": {"failure.class": "sandbox_lifecycle_failure"}},
    ]

    assert classify_failure({"score": 0}, events) == "sandbox_lifecycle_failure"


def test_precedence_prefers_sandbox_exec_timeout_over_host_tool_and_outer_timeout():
    events = [
        {"attributes": {"failure.class": "outer_task_timeout"}},
        {"attributes": {"failure.class": "host_tool_timeout_or_leak"}},
        {"attributes": {"failure.class": "sandbox_exec_timeout"}},
    ]

    assert classify_failure({"score": 0}, events) == "sandbox_exec_timeout"


def test_precedence_prefers_host_tool_timeout_over_outer_timeout():
    events = [
        {"attributes": {"failure.class": "outer_task_timeout"}},
        {"attributes": {"failure.class": "host_tool_timeout_or_leak"}},
    ]

    assert classify_failure({"score": 0}, events) == "host_tool_timeout_or_leak"


def test_outer_task_timeout_is_used_when_no_lower_level_timeout_evidence_exists():
    events = [
        {
            "name": "gepa.case.outer_timeout",
            "status": {"code": "ERROR", "message": "outer task timed out after 300s"},
            "attributes": {"failure.class": "outer_task_timeout"},
        }
    ]

    assert classify_failure({"score": 0}, events) == "outer_task_timeout"


def test_evaluator_limitation_precedes_evaluator_exception():
    events = [
        {"attributes": {"failure.class": "evaluator_exception"}},
        {"attributes": {"failure.class": "evaluator_limitation"}},
    ]

    assert classify_failure({"score": 0}, events) == "evaluator_limitation"


def test_model_no_code_precedes_model_generated_bad_code():
    events = [
        {"attributes": {"failure.class": "model_generated_bad_code"}},
        {"attributes": {"failure.class": "model_no_code_generated"}},
    ]

    assert classify_failure({"score": 0}, events) == "model_no_code_generated"


def test_resource_saturation_is_only_above_unknown():
    events = [{"attributes": {"failure.class": "resource_saturation_unknown"}}]

    assert classify_failure({"score": 0}, events) == "resource_saturation_unknown"


def test_row_failure_class_participates_in_precedence_with_events():
    row = {"score": 0, "failure_class": "outer_task_timeout"}
    events = [{"attributes": {"failure.class": "sandbox_exec_timeout"}}]

    assert classify_failure(row, events) == "sandbox_exec_timeout"


def test_infers_classes_from_otel_shaped_synthetic_events():
    events = [
        {
            "name": "sandbox.health_check",
            "event_domain": "sandbox",
            "status": {"code": "ERROR", "message": "No response during health check"},
            "attributes": {},
        },
        {
            "name": "host_tool.recalculate",
            "event_domain": "host_tool",
            "status": {"code": "ERROR", "message": "LibreOffice subprocess timed out"},
            "attributes": {},
        },
    ]

    assert classify_zero_score_failure({"score": 0}, events) == "sandbox_lifecycle_failure"
