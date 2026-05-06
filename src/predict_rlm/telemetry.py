"""Dependency-free OTel-shaped telemetry helpers for run-local JSONL artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import time
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

SCHEMA_VERSION = 1

FailureClass = Literal[
    "model_generated_bad_code",
    "model_no_code_generated",
    "model_output_truncated",
    "sandbox_lifecycle_failure",
    "sandbox_exec_timeout",
    "host_tool_timeout_or_leak",
    "outer_task_timeout",
    "evaluator_limitation",
    "evaluator_exception",
    "resource_saturation_unknown",
    "unknown",
]

FAILURE_CLASSES: tuple[str, ...] = (
    "model_generated_bad_code",
    "model_no_code_generated",
    "model_output_truncated",
    "sandbox_lifecycle_failure",
    "sandbox_exec_timeout",
    "host_tool_timeout_or_leak",
    "outer_task_timeout",
    "evaluator_limitation",
    "evaluator_exception",
    "resource_saturation_unknown",
    "unknown",
)

_PRECEDENCE: tuple[FailureClass, ...] = (
    "sandbox_lifecycle_failure",
    "sandbox_exec_timeout",
    "host_tool_timeout_or_leak",
    "outer_task_timeout",
    "evaluator_limitation",
    "evaluator_exception",
    "model_output_truncated",
    "model_no_code_generated",
    "model_generated_bad_code",
    "resource_saturation_unknown",
    "unknown",
)

_SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|auth|bearer|credential|password|secret|token)", re.IGNORECASE
)
_ENV_SECRET_RE = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:API[_-]?KEY|AUTH|CREDENTIAL|PASSWORD|SECRET|TOKEN)[A-Z0-9_]*)\s*=\s*([^\s,;]+)"
)
_VALUE_SECRET_RE = re.compile(r"(?i)^(sk-|xox[baprs]-|gh[pousr]_|bearer\s+)")
_REDACTED = "[REDACTED]"
_current_context: ContextVar["TelemetryContext | None"] = ContextVar(
    "predict_rlm_telemetry_context",
    default=None,
)


class TelemetrySink(Protocol):
    """Destination for OTel-shaped telemetry records."""

    def write(self, record: dict[str, Any]) -> None:
        """Write a single telemetry record."""


class JsonlTelemetrySink:
    """Best-effort JSONL sink with one JSON object per line."""

    def __init__(self, path: str | Path, *, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled

    def write(self, record: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True, default=str))
                f.write("\n")
        except Exception:
            # Telemetry is diagnostic only. It must never change caller behavior.
            return


class NoopTelemetrySink:
    """Telemetry sink for disabled instrumentation."""

    def write(self, record: dict[str, Any]) -> None:
        return


@dataclass(frozen=True)
class TelemetryContext:
    sink: TelemetrySink
    trace_id: str
    parent_span_id: str | None = None
    run_id: str | None = None
    eval_kind: str | None = None
    eval_idx: int | None = None
    attempt_id: str | None = None
    example_id: str | None = None
    case_idx: int | None = None
    candidate_id: str | None = None
    candidate_hash: str | None = None
    telemetry_level: str = "minimal"

    def write_span(
        self,
        name: str,
        *,
        event_domain: str,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        span_kind: str = "internal",
        start_time_unix_nano: int | None = None,
        end_time_unix_nano: int | None = None,
        duration_ms: int | None = None,
        status: str | dict[str, Any] = "OK",
        attributes: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = make_span_record(
            self,
            name=name,
            event_domain=event_domain,
            span_id=span_id,
            parent_span_id=parent_span_id,
            span_kind=span_kind,
            start_time_unix_nano=start_time_unix_nano,
            end_time_unix_nano=end_time_unix_nano,
            duration_ms=duration_ms,
            status=status,
            attributes=attributes,
        )
        self.sink.write(record)
        return record


def current_telemetry_context() -> TelemetryContext | None:
    """Return telemetry context for host tools that cannot receive it directly."""

    return _current_context.get()


def set_current_telemetry_context(
    telemetry_context: TelemetryContext | None,
) -> Token[TelemetryContext | None]:
    """Set the current telemetry context and return a reset token."""

    return _current_context.set(telemetry_context)


def reset_current_telemetry_context(token: Token[TelemetryContext | None]) -> None:
    """Reset the current telemetry context from a token."""

    _current_context.reset(token)


def make_trace_id(*parts: Any) -> str:
    """Create a deterministic trace id from stable identity parts."""

    joined = ":".join(str(part) for part in parts if part is not None)
    if joined:
        return joined
    return f"trace_{secrets.token_hex(8)}"


def make_span_id(prefix: str = "span") -> str:
    """Create a compact span id suitable for local JSONL telemetry."""

    return f"{prefix}_{secrets.token_hex(8)}"


def candidate_hash(candidate: dict[str, Any]) -> str:
    """Return a deterministic hash for a candidate payload."""

    canonical = json.dumps(candidate, sort_keys=True, separators=(",", ":"), default=str)
    return "cand_sha256_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def redact_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    """Redact obvious secrets from telemetry attributes."""

    return {
        str(key): _redact_value(str(key), value)
        for key, value in attributes.items()
    }


def make_span_record(
    context: TelemetryContext,
    *,
    name: str,
    event_domain: str,
    span_id: str | None = None,
    parent_span_id: str | None = None,
    span_kind: str = "internal",
    start_time_unix_nano: int | None = None,
    end_time_unix_nano: int | None = None,
    duration_ms: int | None = None,
    status: str | dict[str, Any] = "OK",
    attributes: dict[str, Any] | None = None,
    record_type: str = "span",
) -> dict[str, Any]:
    """Build a minimal OTel-shaped span record."""

    end = end_time_unix_nano if end_time_unix_nano is not None else time.time_ns()
    start = start_time_unix_nano if start_time_unix_nano is not None else end
    if duration_ms is None:
        duration_ms = max(0, round((end - start) / 1_000_000))

    merged_attributes = _context_attributes(context)
    if attributes:
        merged_attributes.update(redact_attributes(attributes))

    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": record_type,
        "event_domain": event_domain,
        "trace_id": context.trace_id,
        "span_id": span_id or make_span_id(_safe_id_prefix(name)),
        "parent_span_id": (
            context.parent_span_id if parent_span_id is None else parent_span_id
        ),
        "name": name,
        "span_kind": span_kind,
        "start_time_unix_nano": start,
        "end_time_unix_nano": end,
        "duration_ms": duration_ms,
        "status": _normalize_status(status),
        "attributes": merged_attributes,
    }


def classify_failure(
    row: dict[str, Any] | None,
    events: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
) -> FailureClass:
    """Classify a zero-score row from synthetic trace row and telemetry evidence."""

    evidence = _collect_failure_evidence(row, events)
    for failure_class in _PRECEDENCE:
        if failure_class in evidence:
            return failure_class
    return "unknown"


def classify_zero_score_failure(
    row: dict[str, Any] | None,
    events: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
) -> FailureClass:
    """Alias with a more explicit analyzer name."""

    return classify_failure(row, events)


def _context_attributes(context: TelemetryContext) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "telemetry.level": context.telemetry_level,
    }
    mappings = (
        ("rlm.run_id", context.run_id),
        ("rlm.eval_kind", context.eval_kind),
        ("rlm.eval_idx", context.eval_idx),
        ("rlm.attempt_id", context.attempt_id),
        ("spreadbench.example_id", context.example_id),
        ("spreadbench.case_idx", context.case_idx),
        ("rlm.candidate_id", context.candidate_id),
        ("rlm.candidate_hash", context.candidate_hash),
    )
    for key, value in mappings:
        if value is not None:
            attrs[key] = value
    return attrs


def _normalize_status(status: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(status, dict):
        code = str(status.get("code", "UNSET")).upper()
        normalized = dict(status)
        normalized["code"] = code
        if "message" in normalized and normalized["message"] is not None:
            normalized["message"] = _redact_value("status.message", normalized["message"])
        return normalized
    return {"code": status.upper()}


def _safe_id_prefix(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_") or "span"


def _redact_value(key: str, value: Any) -> Any:
    if _SECRET_KEY_RE.search(key):
        return _REDACTED
    if isinstance(value, str):
        if _VALUE_SECRET_RE.search(value):
            return _REDACTED
        return _ENV_SECRET_RE.sub(lambda m: f"{m.group(1)}={_REDACTED}", value)
    if isinstance(value, dict):
        return {str(k): _redact_value(str(k), v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(key, item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(key, item) for item in value)
    return value


def _collect_failure_evidence(
    row: dict[str, Any] | None,
    events: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
) -> set[FailureClass]:
    evidence: set[FailureClass] = set()
    for item in [row or {}, *(events or [])]:
        failure_class = _failure_class_from_item(item)
        if failure_class is not None:
            evidence.add(failure_class)
        evidence.update(_infer_failure_classes(item))
    return evidence


def _failure_class_from_item(item: dict[str, Any]) -> FailureClass | None:
    candidates = [
        item.get("failure_class"),
        item.get("failure.class"),
        (item.get("attributes") or {}).get("failure.class")
        if isinstance(item.get("attributes"), dict)
        else None,
    ]
    for candidate in candidates:
        if candidate in FAILURE_CLASSES:
            return candidate
    return None


def _infer_failure_classes(item: dict[str, Any]) -> set[FailureClass]:
    if not item:
        return set()

    attrs = item.get("attributes") if isinstance(item.get("attributes"), dict) else {}
    status = item.get("status") if isinstance(item.get("status"), dict) else {}
    text = " ".join(
        str(part).lower()
        for part in [
            item.get("name"),
            item.get("event_domain"),
            item.get("record_type"),
            item.get("error"),
            item.get("failure_reason"),
            status.get("code"),
            status.get("message"),
            *attrs.keys(),
            *attrs.values(),
        ]
        if part is not None
    )

    inferred: set[FailureClass] = set()
    if (
        attrs.get("lm.truncated") is True
        or item.get("truncated") is True
        or "model_output_truncated" in text
        or (
            "finish_reason" in text
            and any(marker in text for marker in ("length", "max_tokens", "max output"))
        )
        or "lm response was truncated" in text
    ):
        inferred.add("model_output_truncated")

    is_error = "error" in text or "timeout" in text or "timed out" in text
    if not is_error:
        return inferred

    if "sandbox" in text and any(
        marker in text
        for marker in ("health", "lifecycle", "startup", "start", "stale response", "no response")
    ):
        inferred.add("sandbox_lifecycle_failure")
    if "sandbox" in text and ("timeout" in text or "timed out" in text):
        inferred.add("sandbox_exec_timeout")
    if any(marker in text for marker in ("host_tool", "libreoffice", "formula", "subprocess", "worker")) and any(
        marker in text for marker in ("timeout", "timed out", "leak", "future")
    ):
        inferred.add("host_tool_timeout_or_leak")
    if "outer" in text and ("timeout" in text or "timed out" in text):
        inferred.add("outer_task_timeout")
    if "evaluator" in text and any(
        marker in text for marker in ("limitation", "unsupported", "missing workbook", "missing sheet")
    ):
        inferred.add("evaluator_limitation")
    if "evaluator" in text and ("exception" in text or "traceback" in text):
        inferred.add("evaluator_exception")
    if "no code" in text or "no output workbook" in text or "parse failed" in text:
        inferred.add("model_no_code_generated")
    if "bad code" in text or "wrong workbook" in text or "wrong output" in text:
        inferred.add("model_generated_bad_code")
    if any(marker in text for marker in ("resource", "semaphore", "process", "memory", "cancelled")):
        inferred.add("resource_saturation_unknown")
    return inferred
