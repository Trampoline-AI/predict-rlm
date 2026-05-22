from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_EXAMPLE_DIR = Path(__file__).resolve().parent.parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

from terminal_bench_rlm.scoring import (  # noqa: E402
    hard_score,
    score_details,
    soft_score,
    to_gepa_example_result,
)


def test_harbor_reward_one_scores_as_resolved() -> None:
    trial = {"verifier_result": {"rewards": {"reward": 1.0}}}

    assert soft_score(trial) == 1.0
    assert hard_score(trial) == 1.0
    assert score_details(trial) == {
        "soft_score": 1.0,
        "hard_score": 1.0,
        "passed": 1,
        "total": 1,
        "is_resolved": True,
    }


def test_harbor_partial_reward_scores_soft_without_hard_pass() -> None:
    trial = SimpleNamespace(verifier_result=SimpleNamespace(rewards={"reward": 0.25}))

    assert soft_score(trial) == 0.25
    assert hard_score(trial) == 0.0
    assert score_details(trial)["is_resolved"] is False


def test_harbor_ctrf_details_override_zero_reward_for_partial_credit() -> None:
    trial = {
        "verifier_result": {
            "rewards": {"reward": 0.0},
            "ctrf": {
                "results": {
                    "summary": {"tests": 5, "passed": 4, "failed": 1},
                    "tests": [
                        {"name": "test_a", "status": "passed"},
                        {"name": "test_b", "status": "passed"},
                        {"name": "test_c", "status": "passed"},
                        {"name": "test_d", "status": "passed"},
                        {"name": "test_e", "status": "failed"},
                    ],
                }
            },
        }
    }

    assert soft_score(trial) == 0.8
    assert hard_score(trial) == 0.0
    assert score_details(trial) == {
        "soft_score": 0.8,
        "hard_score": 0.0,
        "passed": 4,
        "total": 5,
        "is_resolved": False,
        "failures": {"test_e": "failed"},
    }


def test_harbor_exception_keeps_zero_score_even_with_ctrf_details() -> None:
    trial = {
        "exception_info": {"type": "RuntimeError", "message": "agent failed"},
        "verifier_result": {
            "rewards": {"reward": 0.0},
            "ctrf": {"results": {"summary": {"tests": 5, "passed": 4}}},
        },
    }

    assert score_details(trial) == {
        "soft_score": 0.0,
        "hard_score": 0.0,
        "passed": 0,
        "total": 0,
        "is_resolved": False,
        "failure_class": "evaluator_exception",
    }


def test_harbor_timeout_exception_class_is_preserved_in_feedback_and_objectives() -> None:
    long_stdout = "stdout-" + ("x" * 5000)
    long_stderr = "stderr-" + ("y" * 5000)
    trial = {
        "exception_info": {
            "exception_type": "AgentTimeoutError",
            "exception_message": "Agent execution timed out after 900.0 seconds",
            "phase": "agent",
            "timed_out": True,
            "timeout_seconds": 900.0,
            "stdout": long_stdout,
            "stderr": long_stderr,
        }
    }

    details = score_details(trial)
    result = to_gepa_example_result(trial, traces=[])

    assert details["failure_class"] == "outer_task_timeout"
    assert "failure_class=outer_task_timeout" in result.feedback
    assert result.objective_scores is not None
    assert result.objective_scores["failure_class"] == "outer_task_timeout"
    assert result.objective_scores["failure_phase"] == "agent"
    assert result.objective_scores["exception_type"] == "AgentTimeoutError"
    assert result.objective_scores["diagnostic_text"] == "Agent execution timed out after 900.0 seconds"
    assert result.objective_scores["timed_out"] is True
    assert result.objective_scores["timeout_seconds"] == 900.0
    assert result.objective_scores["stdout_tail"].startswith("x")
    assert result.objective_scores["stdout_tail"].endswith("x" * 20)
    assert len(result.objective_scores["stdout_tail"]) <= 2000
    assert result.objective_scores["stderr_tail"].startswith("y")
    assert result.objective_scores["stderr_tail"].endswith("y" * 20)
    assert len(result.objective_scores["stderr_tail"]) <= 2000


def test_outer_timeout_placeholder_yields_to_harbor_verifier_pass_evidence() -> None:
    trial = {
        "exception_info": {
            "exception_type": "HarnessTimeoutError",
            "exception_message": "Terminal-Bench CLI timed out after 900s",
            "phase": "harness",
            "timed_out": True,
        },
        "verifier_result": {
            "rewards": {"reward": 0.0},
            "ctrf": {"results": {"summary": {"tests": 4, "passed": 4}}},
        },
    }

    details = score_details(trial)
    result = to_gepa_example_result(trial, traces=[])

    assert details["soft_score"] == 1.0
    assert details["hard_score"] == 1.0
    assert details["passed"] == 4
    assert details["total"] == 4
    assert details["is_resolved"] is True
    assert result.objective_scores is not None
    assert result.objective_scores["failure_class"] == "outer_task_timeout"
    assert result.objective_scores["failure_phase"] == "harness"
    assert result.objective_scores["timed_out"] is True


def test_internal_timeout_yields_to_harbor_verifier_pass_evidence() -> None:
    trial = {
        "exception_info": {
            "exception_type": "SandboxExecutionTimeout",
            "exception_message": "command timed out after 30s",
            "phase": "sandbox_exec",
            "timed_out": True,
        },
        "verifier_result": {
            "rewards": {"reward": 0.0},
            "ctrf": {"results": {"summary": {"tests": 2, "passed": 2}}},
        },
    }

    details = score_details(trial)
    result = to_gepa_example_result(trial, traces=[])

    assert details["soft_score"] == 1.0
    assert details["hard_score"] == 1.0
    assert details["passed"] == 2
    assert details["total"] == 2
    assert details["is_resolved"] is True
    assert result.score == 1.0
    assert result.objective_scores is not None
    assert result.objective_scores["failure_class"] == "sandbox_exec_timeout"
    assert result.objective_scores["failure_phase"] == "sandbox_exec"
    assert result.objective_scores["timed_out"] is True


def test_gepa_objective_scores_are_numeric_when_harbor_trial_has_exception() -> None:
    result = to_gepa_example_result({}, traces=[])

    assert result.objective_scores == {
        "soft_score": 0.0,
        "hard_score": 0.0,
        "passed": 0,
        "total": 0,
        "is_resolved": 0.0,
    }
