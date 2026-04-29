from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from ..schema import SCHEMA_VERSION, CostRow, LMCost


def append_cost_rows(log_path: str | Path, rows: Iterable[CostRow | dict[str, Any]]) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            payload = row.to_dict() if isinstance(row, CostRow) else dict(row)
            payload.setdefault("schema_version", SCHEMA_VERSION)
            payload.setdefault("ts", datetime.now().isoformat())
            f.write(json.dumps(payload, default=str) + "\n")


def aggregate_costs_from_log(
    log_path: str | Path,
    role_order: list[str] | None = None,
    *,
    logical: bool = False,
) -> list[LMCost]:
    path = Path(log_path)
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or row.get("event") == "startup":
                continue
            rows.append(row)

    if logical:
        deduped: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
        for index, row in enumerate(rows):
            role = str(row.get("role") or "unknown")
            model = str(row.get("model") or "unknown")
            operation_id = row.get("operation_id")
            attempt_id = row.get("attempt_id")
            event_id = row.get("event_id")
            if operation_id or attempt_id:
                key = ("operation", str(operation_id or ""), str(attempt_id or ""), role, model)
            elif event_id:
                key = ("event", str(event_id), "", role, model)
            else:
                key = ("row", str(index), "", role, model)
            deduped[key] = row
        rows = list(deduped.values())

    agg: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        role = str(row.get("role") or "unknown")
        model = str(row.get("model") or "unknown")
        bucket = agg.setdefault(
            (role, model),
            {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0},
        )
        row_calls = row.get("calls")
        bucket["calls"] += 1 if row_calls is None else int(row_calls)
        bucket["prompt_tokens"] += int(row.get("input_tokens") or 0)
        bucket["completion_tokens"] += int(row.get("output_tokens") or 0)
        bucket["cost_usd"] += float(row.get("cost_usd") or 0.0)

    def sort_key(item: tuple[tuple[str, str], dict[str, Any]]) -> tuple[int, str, str]:
        (role, model), _bucket = item
        if role_order and role in role_order:
            return role_order.index(role), role, model
        return (len(role_order) if role_order else 0), role, model

    return [
        LMCost(
            role=role,
            model=model,
            calls=bucket["calls"],
            prompt_tokens=bucket["prompt_tokens"],
            completion_tokens=bucket["completion_tokens"],
            cost_usd=bucket["cost_usd"],
        )
        for (role, model), bucket in sorted(agg.items(), key=sort_key)
    ]


def costs_to_dicts(costs: Iterable[LMCost]) -> list[dict[str, Any]]:
    return [cost.to_dict() for cost in costs]


def append_trace_cost_rows(
    log_path: str | Path | None,
    *,
    event: str,
    event_id: str,
    operation_id: str,
    attempt_id: str,
    main_role: str,
    sub_role: str,
    trace: Any,
    sum_traces: Any,
) -> None:
    if log_path is None or trace is None:
        return
    main_usage, sub_usage, main_model, sub_model, main_calls, sub_calls = sum_traces([trace])
    rows: list[CostRow] = []
    if main_usage is not None and (
        main_usage.input_tokens or main_usage.output_tokens or main_usage.cache_hits
    ):
        rows.append(
            CostRow(
                event_id=event_id,
                operation_id=operation_id,
                attempt_id=attempt_id,
                event=event,
                role=main_role,
                model=main_model,
                calls=main_calls,
                input_tokens=main_usage.input_tokens,
                output_tokens=main_usage.output_tokens,
                cost_usd=main_usage.cost,
                cache_hits=main_usage.cache_hits,
            )
        )
    if sub_usage is not None and sub_model and (
        sub_usage.input_tokens or sub_usage.output_tokens or sub_usage.cache_hits
    ):
        rows.append(
            CostRow(
                event_id=event_id,
                operation_id=operation_id,
                attempt_id=attempt_id,
                event=event,
                role=sub_role,
                model=sub_model,
                calls=sub_calls,
                input_tokens=sub_usage.input_tokens,
                output_tokens=sub_usage.output_tokens,
                cost_usd=sub_usage.cost,
                cache_hits=sub_usage.cache_hits,
            )
        )
    if rows:
        append_cost_rows(log_path, rows)
