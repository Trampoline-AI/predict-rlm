from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from predict_rlm.telemetry import FailureClass, classify_failure

INFRA_FAILURE_CLASSES = {
    "sandbox_lifecycle_failure",
    "sandbox_exec_timeout",
    "host_tool_timeout_or_leak",
    "outer_task_timeout",
    "evaluator_limitation",
    "evaluator_exception",
    "resource_saturation_unknown",
}


@dataclass(frozen=True)
class ClassifiedTraceRow:
    row: dict[str, Any]
    events: list[dict[str, Any]]
    failure_class: FailureClass


@dataclass(frozen=True)
class TelemetryReport:
    rows: list[ClassifiedTraceRow]
    zero_rows_by_failure_class: dict[str, int]
    candidate_scores: list[dict[str, Any]]
    health_check_failures: list[dict[str, Any]]
    sandbox_tool_timeout_counts: dict[str, int]
    evaluator_counts: dict[str, int]
    unknown_rows: list[dict[str, Any]]

    def classification_table(self) -> list[dict[str, Any]]:
        return [
            {
                "example_id": classified.row.get("example_id"),
                "candidate_hash": classified.row.get("candidate_hash"),
                "kind": classified.row.get("kind"),
                "score": classified.row.get("score"),
                "failure_class": classified.failure_class,
                "failure_reason": classified.row.get("failure_reason"),
                "trace_id": (classified.row.get("telemetry_ref") or {}).get("trace_id"),
            }
            for classified in self.rows
        ]


def analyze_run(run_dir: str | Path) -> TelemetryReport:
    run_path = Path(run_dir)
    return analyze_trace_rows(
        load_task_trace_rows(run_path),
        load_telemetry_events(run_path),
    )


def analyze_trace_rows(
    rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> TelemetryReport:
    events_by_trace = _events_by_trace(events)
    classified_rows: list[ClassifiedTraceRow] = []
    zero_counts: Counter[str] = Counter()
    unknown_rows: list[dict[str, Any]] = []

    for row in rows:
        trace_events = _events_for_row(row, events_by_trace)
        failure_class: FailureClass = "unknown"
        if _is_zero_row(row):
            failure_class = classify_failure(row, trace_events)
            zero_counts[failure_class] += 1
            if failure_class == "unknown":
                unknown_rows.append(row)
        classified_rows.append(
            ClassifiedTraceRow(
                row=row,
                events=trace_events,
                failure_class=failure_class,
            )
        )

    return TelemetryReport(
        rows=classified_rows,
        zero_rows_by_failure_class=dict(zero_counts),
        candidate_scores=_candidate_scores(classified_rows),
        health_check_failures=_health_check_failures(events),
        sandbox_tool_timeout_counts=_sandbox_tool_timeout_counts(events),
        evaluator_counts={
            "evaluator_limitation": zero_counts.get("evaluator_limitation", 0),
            "evaluator_exception": zero_counts.get("evaluator_exception", 0),
        },
        unknown_rows=unknown_rows,
    )


def load_task_trace_rows(run_dir: str | Path) -> list[dict[str, Any]]:
    trace_dir = Path(run_dir) / "task_traces"
    rows: list[dict[str, Any]] = []
    if not trace_dir.is_dir():
        return rows
    for path in sorted(trace_dir.glob("*.jsonl")):
        rows.extend(_load_jsonl(path))
    return rows


def load_telemetry_events(run_dir: str | Path) -> list[dict[str, Any]]:
    return _load_jsonl(Path(run_dir) / "telemetry" / "events.jsonl")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _events_by_trace(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        trace_id = event.get("trace_id")
        if trace_id is not None:
            grouped[str(trace_id)].append(event)
    return dict(grouped)


def _events_for_row(
    row: dict[str, Any],
    events_by_trace: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    telemetry_ref = row.get("telemetry_ref")
    if isinstance(telemetry_ref, dict) and telemetry_ref.get("trace_id") is not None:
        return events_by_trace.get(str(telemetry_ref["trace_id"]), [])
    trace_id = row.get("trace_id")
    if trace_id is not None:
        return events_by_trace.get(str(trace_id), [])
    return []


def _candidate_scores(rows: list[ClassifiedTraceRow]) -> list[dict[str, Any]]:
    grouped: dict[str, list[ClassifiedTraceRow]] = defaultdict(list)
    for row in rows:
        candidate = row.row.get("candidate_id") or row.row.get("candidate_hash") or "unknown"
        grouped[str(candidate)].append(row)

    result: list[dict[str, Any]] = []
    for candidate, candidate_rows in sorted(grouped.items()):
        raw_scores = [float(item.row.get("score") or 0.0) for item in candidate_rows]
        model_rows = [
            item
            for item in candidate_rows
            if not (_is_zero_row(item.row) and item.failure_class in INFRA_FAILURE_CLASSES)
        ]
        model_scores = [float(item.row.get("score") or 0.0) for item in model_rows]
        result.append(
            {
                "candidate": candidate,
                "row_count": len(candidate_rows),
                "raw_score": sum(raw_scores) / len(raw_scores) if raw_scores else 0.0,
                "infra_excluded_score": (
                    sum(model_scores) / len(model_scores) if model_scores else None
                ),
                "infra_excluded_rows": len(candidate_rows) - len(model_rows),
            }
        )
    return result


def _health_check_failures(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for event in events:
        text = _event_text(event)
        if "health" not in text or "error" not in text:
            continue
        attrs = event.get("attributes") if isinstance(event.get("attributes"), dict) else {}
        failures.append(
            {
                "trace_id": event.get("trace_id"),
                "candidate_hash": attrs.get("rlm.candidate_hash"),
                "attempt_id": attrs.get("rlm.attempt_id"),
                "phase": attrs.get("rlm.eval_kind") or attrs.get("rlm.phase"),
                "example_id": attrs.get("spreadbench.example_id"),
                "case_idx": attrs.get("spreadbench.case_idx"),
                "message": (event.get("status") or {}).get("message")
                if isinstance(event.get("status"), dict)
                else None,
            }
        )
    return failures


def _sandbox_tool_timeout_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for event in events:
        text = _event_text(event)
        if "timeout" not in text and "timed out" not in text:
            continue
        if "sandbox" in text:
            counts["sandbox"] += 1
        if "host_tool" in text or "libreoffice" in text or "formulas" in text:
            counts["host_tool"] += 1
    return dict(counts)


def _event_text(event: dict[str, Any]) -> str:
    attrs = event.get("attributes") if isinstance(event.get("attributes"), dict) else {}
    status = event.get("status") if isinstance(event.get("status"), dict) else {}
    return " ".join(
        str(part).lower()
        for part in [
            event.get("name"),
            event.get("event_domain"),
            status.get("code"),
            status.get("message"),
            *attrs.keys(),
            *attrs.values(),
        ]
        if part is not None
    )


def _is_zero_row(row: dict[str, Any]) -> bool:
    try:
        return float(row.get("score") or 0.0) == 0.0
    except (TypeError, ValueError):
        return False
