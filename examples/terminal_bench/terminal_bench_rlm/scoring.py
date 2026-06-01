from __future__ import annotations

from collections.abc import Mapping
from typing import Any

PASSED = "passed"
DIAGNOSTIC_TAIL_LIMIT = 2000


def hard_score(result_or_parser_results: Any) -> float:
    if _exception_blocks_score(result_or_parser_results):
        return 0.0
    ctrf_score = _harbor_ctrf_score(result_or_parser_results)
    if ctrf_score is not None:
        passed, total, _ctrf_parser_results = ctrf_score
        return 1.0 if total and passed == total else 0.0
    reward = _harbor_reward(result_or_parser_results)
    if reward is not None:
        return 1.0 if reward >= 1.0 else 0.0
    parser_results = _parser_results(result_or_parser_results)
    if parser_results:
        return 1.0 if all(_is_passed(status) for status in parser_results.values()) else 0.0
    return 1.0 if _is_resolved(result_or_parser_results) is True else 0.0


def soft_score(result_or_parser_results: Any) -> float:
    if _exception_blocks_score(result_or_parser_results):
        return 0.0
    ctrf_score = _harbor_ctrf_score(result_or_parser_results)
    if ctrf_score is not None:
        passed, total, _ctrf_parser_results = ctrf_score
        return passed / total if total else 0.0
    reward = _harbor_reward(result_or_parser_results)
    if reward is not None:
        return reward
    parser_results = _parser_results(result_or_parser_results)
    if parser_results:
        passed, total = _pass_counts(parser_results)
        return passed / total if total else 0.0
    return 1.0 if _is_resolved(result_or_parser_results) is True else 0.0


def score_details(result_or_parser_results: Any) -> dict[str, Any]:
    diagnostics = _diagnostic_metadata(result_or_parser_results)
    if _exception_blocks_score(result_or_parser_results):
        return {
            "soft_score": 0.0,
            "hard_score": 0.0,
            "passed": 0,
            "total": 0,
            "is_resolved": False,
            "failure_class": diagnostics.get("failure_class")
            or _failure_class(result_or_parser_results),
        }
    ctrf_score = _harbor_ctrf_score(result_or_parser_results)
    if ctrf_score is not None:
        passed, total, parser_results = ctrf_score
        details = {
            "soft_score": passed / total if total else 0.0,
            "hard_score": 1.0 if total and passed == total else 0.0,
            "passed": passed,
            "total": total,
            "is_resolved": bool(total and passed == total),
        }
        failures = {
            str(name): _status_label(status)
            for name, status in parser_results.items()
            if not _is_passed(status)
        }
        if failures:
            details["failures"] = failures
        if diagnostics.get("failure_class"):
            details["failure_class"] = diagnostics["failure_class"]
        return details
    reward = _harbor_reward(result_or_parser_results)
    if reward is not None:
        details = {
            "soft_score": reward,
            "hard_score": 1.0 if reward >= 1.0 else 0.0,
            "passed": 1 if reward >= 1.0 else 0,
            "total": 1,
            "is_resolved": reward >= 1.0,
        }
        if diagnostics.get("failure_class"):
            details["failure_class"] = diagnostics["failure_class"]
        return details
    parser_results = _parser_results(result_or_parser_results)
    passed, total = _pass_counts(parser_results)
    details = {
        "soft_score": soft_score(result_or_parser_results),
        "hard_score": hard_score(result_or_parser_results),
        "passed": passed,
        "total": total,
        "is_resolved": _is_resolved(result_or_parser_results),
    }
    failures = {
        str(name): _status_label(status)
        for name, status in parser_results.items()
        if not _is_passed(status)
    }
    if failures:
        details["failures"] = failures
    if diagnostics.get("failure_class"):
        details["failure_class"] = diagnostics["failure_class"]
    return details


def feedback(result_or_parser_results: Any) -> str:
    details = score_details(result_or_parser_results)
    message = (
        "Terminal-Bench score: "
        f"soft={details['soft_score']:.3f} "
        f"hard={details['hard_score']:.3f} "
        f"passed={details['passed']}/{details['total']}"
    )
    failures = details.get("failures") or {}
    if failures:
        labels = ", ".join(f"{name}={status}" for name, status in sorted(failures.items()))
        message = f"{message}; failures: {labels}"
    failure_class = details.get("failure_class")
    if failure_class:
        message = f"{message}; failure_class={failure_class}"
    return message


def to_gepa_example_result(
    result_or_parser_results: Any,
    *,
    traces: list[Any],
    example_id: str | None = None,
    rlm_inputs: Mapping[str, Any] | None = None,
):
    from rlm_gepa import RLMGepaExampleResult

    details = score_details(result_or_parser_results)
    objective_scores = {
        "soft_score": details["soft_score"],
        "hard_score": details["hard_score"],
        "passed": details["passed"],
        "total": details["total"],
        "is_resolved": 1.0 if details["is_resolved"] is True else 0.0,
    }
    objective_scores.update(_numeric_diagnostic_metadata(_diagnostic_metadata(result_or_parser_results)))
    return RLMGepaExampleResult(
        score=details["soft_score"],
        feedback=feedback(result_or_parser_results),
        traces=traces,
        rlm_inputs=dict(rlm_inputs or {}),
        example_id=example_id,
        objective_scores=objective_scores,
    )


def _numeric_diagnostic_metadata(metadata: Mapping[str, Any]) -> dict[str, float]:
    numeric: dict[str, float] = {}
    if "timed_out" in metadata:
        numeric["timed_out"] = 1.0 if metadata["timed_out"] else 0.0
    timeout_seconds = metadata.get("timeout_seconds")
    if isinstance(timeout_seconds, (int, float)):
        numeric["timeout_seconds"] = float(timeout_seconds)
    return numeric


def _pass_counts(parser_results: Mapping[Any, Any]) -> tuple[int, int]:
    total = len(parser_results)
    passed = sum(1 for status in parser_results.values() if _is_passed(status))
    return passed, total


def _harbor_ctrf_score(result_or_parser_results: Any) -> tuple[int, int, Mapping[Any, Any]] | None:
    ctrf = _harbor_ctrf(result_or_parser_results)
    if ctrf is None:
        return None
    parser_results = _ctrf_parser_results(ctrf)
    if parser_results:
        passed, total = _pass_counts(parser_results)
        return passed, total, parser_results
    summary_counts = _ctrf_summary_counts(ctrf)
    if summary_counts is None:
        return None
    passed, total = summary_counts
    return passed, total, {}


def _harbor_ctrf(result_or_parser_results: Any) -> Mapping[Any, Any] | None:
    verifier_result = _field(result_or_parser_results, "verifier_result")
    for owner in (verifier_result, result_or_parser_results):
        for name in ("ctrf", "ctrf_json", "verifier_details"):
            value = _field(owner, name)
            if isinstance(value, Mapping):
                return value
    return None


def _ctrf_results(ctrf: Mapping[Any, Any]) -> Mapping[Any, Any]:
    results = ctrf.get("results")
    return results if isinstance(results, Mapping) else ctrf


def _ctrf_parser_results(ctrf: Mapping[Any, Any]) -> dict[str, Any]:
    results = _ctrf_results(ctrf)
    tests = results.get("tests")
    if not isinstance(tests, list):
        return {}
    parser_results: dict[str, Any] = {}
    for index, test in enumerate(tests):
        if not isinstance(test, Mapping):
            continue
        name = test.get("name") or test.get("fullName") or test.get("title") or str(index)
        status = test.get("status") or test.get("raw_status")
        if status is not None:
            parser_results[str(name)] = status
    return parser_results


def _ctrf_summary_counts(ctrf: Mapping[Any, Any]) -> tuple[int, int] | None:
    results = _ctrf_results(ctrf)
    summary = results.get("summary")
    if not isinstance(summary, Mapping):
        return None
    passed = summary.get("passed")
    total = summary.get("tests")
    if not isinstance(passed, int):
        return None
    if not isinstance(total, int):
        total = sum(
            value
            for key, value in summary.items()
            if key in {"passed", "failed", "skipped", "pending", "other"}
            and isinstance(value, int)
        )
    return passed, total


def _parser_results(result_or_parser_results: Any) -> Mapping[Any, Any]:
    if isinstance(result_or_parser_results, Mapping):
        if "parser_results" in result_or_parser_results:
            value = result_or_parser_results.get("parser_results")
            return value if isinstance(value, Mapping) else {}
        return result_or_parser_results
    value = getattr(result_or_parser_results, "parser_results", None)
    return value if isinstance(value, Mapping) else {}


def _harbor_reward(result_or_parser_results: Any) -> float | None:
    verifier_result = _field(result_or_parser_results, "verifier_result")
    rewards = _field(verifier_result, "rewards")
    if not isinstance(rewards, Mapping):
        return None
    reward = rewards.get("reward")
    if isinstance(reward, (int, float)):
        return float(reward)
    return None


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _has_exception(result_or_parser_results: Any) -> bool:
    exception_info = _field(result_or_parser_results, "exception_info")
    return exception_info not in (None, False, {})


def _exception_blocks_score(result_or_parser_results: Any) -> bool:
    if not _has_exception(result_or_parser_results):
        return False
    failure_class = _failure_class(result_or_parser_results)
    if failure_class not in {"outer_task_timeout", "sandbox_exec_timeout"}:
        return True
    return not _has_successful_harbor_verifier_evidence(result_or_parser_results)


def _has_successful_harbor_verifier_evidence(result_or_parser_results: Any) -> bool:
    ctrf_score = _harbor_ctrf_score(result_or_parser_results)
    if ctrf_score is not None:
        passed, total, _parser_results = ctrf_score
        return bool(total and passed == total)
    reward = _harbor_reward(result_or_parser_results)
    return reward is not None and reward >= 1.0


def _failure_class(result_or_parser_results: Any) -> str:
    explicit = _field(result_or_parser_results, "failure_class")
    if explicit:
        return str(explicit)
    exception_info = _field(result_or_parser_results, "exception_info")
    exception_type = str(
        _field(exception_info, "exception_type")
        or _field(exception_info, "type")
        or _field(exception_info, "class")
        or ""
    )
    exception_message = str(
        _field(exception_info, "exception_message") or _field(exception_info, "message") or ""
    )
    text = f"{exception_type} {exception_message}".lower()
    if (
        "agenttimeouterror" in text
        or "harnesstimeouterror" in text
        or "agent execution timed out" in text
        or "terminal-bench cli timed out" in text
    ):
        return "outer_task_timeout"
    if "sandboxfatalerror" in text or "exit code 137" in text or "exit code 143" in text:
        return "sandbox_lifecycle_failure"
    if "timed out" in text or "timeout" in text:
        return "sandbox_exec_timeout"
    return "evaluator_exception"


def _diagnostic_metadata(result_or_parser_results: Any) -> dict[str, Any]:
    if not _has_exception(result_or_parser_results) and not _field(
        result_or_parser_results, "failure_class"
    ):
        return {}
    exception_info = _field(result_or_parser_results, "exception_info")
    metadata: dict[str, Any] = {"failure_class": _failure_class(result_or_parser_results)}
    phase = _first_field(exception_info, result_or_parser_results, names=("phase", "failure_phase"))
    if phase:
        metadata["failure_phase"] = str(phase)
    exception_type = _first_field(
        exception_info,
        result_or_parser_results,
        names=("exception_type", "type", "class"),
    )
    if exception_type:
        metadata["exception_type"] = str(exception_type)
    diagnostic_text = _first_field(
        exception_info,
        result_or_parser_results,
        names=("exception_message", "message", "diagnostic_text", "error"),
    )
    if diagnostic_text:
        metadata["diagnostic_text"] = _compact_text(diagnostic_text)
    returncode = _first_field(
        exception_info,
        result_or_parser_results,
        names=("returncode", "return_code", "exit_code"),
    )
    if returncode is not None:
        metadata["returncode"] = returncode
    timed_out = _first_field(exception_info, result_or_parser_results, names=("timed_out",))
    if timed_out is not None:
        metadata["timed_out"] = bool(timed_out)
    elif metadata["failure_class"] in {"outer_task_timeout", "sandbox_exec_timeout"}:
        metadata["timed_out"] = True
    timeout_seconds = _first_field(
        exception_info,
        result_or_parser_results,
        names=("timeout_seconds", "timeout_sec", "timeout"),
    )
    if timeout_seconds is not None:
        metadata["timeout_seconds"] = timeout_seconds
    for stream in ("stdout", "stderr"):
        tail = _first_field(
            exception_info,
            result_or_parser_results,
            names=(f"{stream}_tail", stream),
        )
        if tail:
            metadata[f"{stream}_tail"] = _tail_text(tail)
    return metadata


def _first_field(*owners: Any, names: tuple[str, ...]) -> Any:
    for owner in owners:
        for name in names:
            value = _field(owner, name)
            if value is not None:
                return value
    return None


def _compact_text(value: Any) -> str:
    return _tail_text(value)


def _tail_text(value: Any, *, limit: int = DIAGNOSTIC_TAIL_LIMIT) -> str:
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    text = str(value)
    return text[-limit:] if len(text) > limit else text


def _is_resolved(result_or_parser_results: Any) -> bool | None:
    if isinstance(result_or_parser_results, Mapping):
        value = result_or_parser_results.get("is_resolved")
    else:
        value = getattr(result_or_parser_results, "is_resolved", None)
    return value if isinstance(value, bool) else None


def _is_passed(status: Any) -> bool:
    return _status_label(status) == PASSED


def _status_label(status: Any) -> str:
    if hasattr(status, "value"):
        return str(status.value).lower()
    if hasattr(status, "name"):
        return str(status.name).lower()
    return str(status).lower()
