from __future__ import annotations

import json
import math
import pickle
import re
from pathlib import Path
from statistics import mean
from typing import Any

from .cost import LMCost, aggregate_costs_from_log

HARD_THRESHOLD = 0.999
ANSI_ITALIC = "\033[3m"
ANSI_MUTED = "\033[38;5;248m"
ANSI_BOLD_GOLD = "\033[1;38;5;220m"
ANSI_RESET = "\033[0m"
CELL_MATCH_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s+cells?\s+match")
ALL_CELLS_MATCH_RE = re.compile(r"All\s+(\d+)\s+cells?\s+match")
DECIMAL_RE = re.compile(r"(?<![\w.])(?P<sign>[+-]?)(?P<whole>\d+)\.(?P<fraction>\d+)")
LEADING_ZERO_DECIMAL_RE = re.compile(r"(?<![\w.])(?P<sign>[+-]?)0\.(?P<fraction>\d+)")
TEXT_COLUMNS = {"model", "outcome", "parent", "scope", "task"}
COST_GROUPS = [
    ("executor", [("main", {"executor", "main"}), ("sub", {"sub_lm", "sub"})]),
    (
        "proposer",
        [
            ("main", {"proposer", "reflection"}),
            ("sub", {"proposer_sub_lm", "proposer_sub"}),
        ],
    ),
    (
        "merge",
        [
            ("trace main", {"merge_trace_executor", "merge_trace_main"}),
            ("trace sub", {"merge_trace_sub_lm", "merge_trace_sub"}),
            ("proposer main", {"merge_proposer"}),
            ("proposer sub", {"merge_proposer_sub_lm", "merge_proposer_sub"}),
        ],
    ),
]


def load_run_state(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "gepa_state.bin"
    with path.open("rb") as f:
        state = pickle.load(f)
    if isinstance(state, dict):
        return state
    return dict(getattr(state, "__dict__", {}))


def load_summary(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "optimization_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_eval_report(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "eval.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_run_metadata(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "run_metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def header_summary(run_dir: str | Path) -> str:
    state = load_run_state(run_dir)
    subscores = state.get("prog_candidate_val_subscores") or []
    means = [_mean_scores(scores) for scores in subscores]
    best = max(means) if means else 0.0
    fire = " 🔥" if best >= 0.80 else ""
    return (
        f"iter={state.get('i', 0)}, candidates={len(state.get('program_candidates') or [])}, "
        f"evals={state.get('total_num_evals', 0)}, mb-score agg={best:.4f}{fire}"
    )


def iteration_rows(run_dir: str | Path) -> list[dict[str, Any]]:
    state = load_run_state(run_dir)
    best_idx = _best_candidate_idx(run_dir, state.get("prog_candidate_val_subscores") or [])
    rows: list[dict[str, Any]] = []
    for entry in state.get("full_program_trace") or []:
        row_scores = _iteration_scores(entry)
        if row_scores is None:
            continue
        parent_scores, new_scores, parent_label = row_scores
        gains, losses = _hard_flips(parent_scores, new_scores)
        soft_change, soft_secondary = _format_soft_change(parent_scores, new_scores)
        hard_change, hard_secondary = _format_hard_change(parent_scores, new_scores)
        flips, flips_secondary = _format_flips(gains, losses)
        iteration = entry.get("i", len(rows))
        rows.append(
            {
                "iter": _format_iter_parents(iteration, parent_label),
                "soft: par → child": soft_change,
                "hard: par → child": hard_change,
                "flips": flips,
                "p": f"{_mcnemar_exact_p(gains, losses):.2f}",
                "outcome": f"→ cand {entry['new_program_idx']}" if "new_program_idx" in entry else "REJECTED",
                "_highlight": entry.get("new_program_idx") == best_idx,
                "_muted_prefix": {
                    "soft: par → child": soft_change.removesuffix(soft_secondary),
                    "hard: par → child": hard_change.removesuffix(hard_secondary),
                    "flips": flips.removesuffix(flips_secondary),
                },
            }
        )
    return rows


def candidate_rows(run_dir: str | Path) -> list[dict[str, Any]]:
    state = load_run_state(run_dir)
    subscores = state.get("prog_candidate_val_subscores") or []
    parents = state.get("parent_program_for_candidate") or []
    if not subscores:
        return _candidate_rows_from_artifact(run_dir)

    task_ids = list(subscores[0].keys()) if subscores and subscores[0] else []
    seed_mean = _mean_scores(subscores[0]) if subscores else 0.0
    best_idx = _best_candidate_idx(run_dir, subscores)
    rows: list[dict[str, Any]] = []
    for index, scores in enumerate(subscores):
        values = _score_values(scores)
        pareto_count = 0
        exclusive_scores: list[float] = []
        for task_id in task_ids:
            best = max((candidate_scores.get(task_id, 0.0) for candidate_scores in subscores), default=0.0)
            winners = [i for i, candidate_scores in enumerate(subscores) if candidate_scores.get(task_id, 0.0) == best]
            if index in winners:
                pareto_count += 1
                if len(winners) == 1:
                    exclusive_scores.append(scores[task_id])

        candidate_parents = None
        if index < len(parents) and parents[index]:
            candidate_parents = parents[index]
        candidate_mean = _mean_list(values)
        hard = _hard_count(values)
        total = len(values)
        rows.append(
            {
                "cand [par]": _format_id_parents(index, candidate_parents),
                "mean": f"{candidate_mean:.3f}",
                "hard": f"{(hard / total if total else 0.0):.3f} ({hard}/{total})",
                "pareto": f"{pareto_count}/{len(task_ids)}",
                "exclusive": _format_exclusive(exclusive_scores),
                "Δ-seed": "-" if index == 0 else f"{candidate_mean - seed_mean:+.3f}",
                "_highlight": index == best_idx,
            }
        )
    return rows


def cost_rows(run_dir: str | Path) -> list[dict[str, Any]]:
    total_costs = aggregate_costs_from_log(
        Path(run_dir) / "cost_log.jsonl",
        role_order=_cost_role_order(),
        logical=False,
    )
    effective_costs = aggregate_costs_from_log(
        Path(run_dir) / "cost_log.jsonl",
        role_order=_cost_role_order(),
        logical=True,
    )
    efforts = _optimize_role_efforts(run_dir)
    total_costs = _with_model_efforts(total_costs, efforts)
    effective_costs = _with_model_efforts(effective_costs, efforts)
    costs = _cost_breakdowns(total_costs, effective_costs)
    rows = _grouped_cost_breakdown_rows(costs)
    if costs:
        if rows:
            rows.append(_cost_breakdown_spacer_row())
        rows.append(
            {
                "scope": "TOTAL",
                "model": "",
                "calls": f"{sum(cost['calls'] for cost in costs):,}",
                "prompt_tok": f"{sum(cost['prompt_tokens'] for cost in costs):,}",
                "completion_tok": f"{sum(cost['completion_tokens'] for cost in costs):,}",
                "total_cost": f"${sum(cost['total_cost_usd'] for cost in costs):.2f}",
                "repeat_cost": f"${sum(cost['repeat_cost_usd'] for cost in costs):.2f}",
                "effective_cost": f"${sum(cost['effective_cost_usd'] for cost in costs):.2f}",
            }
        )
    return rows


def eval_task_rows(run_dir: str | Path) -> list[dict[str, Any]]:
    report = load_eval_report(run_dir)
    rows: list[dict[str, Any]] = []
    for task in report.get("per_task") or []:
        cases = task.get("cases") or []
        passed_cases = sum(1 for case in cases if case.get("passed"))
        total_cases = len(cases)
        hard_rate = passed_cases / total_cases if total_cases else 0.0
        soft = float(task.get("soft") or 0.0)
        soft_counts = _soft_counts(cases)
        soft_label = f"{soft:.3f}"
        if soft_counts is not None:
            matched_cells, total_cells = soft_counts
            soft_label = f"{soft_label} ({matched_cells} /{total_cells})"
        rows.append(
            {
                "task": task.get("task_id", ""),
                "soft": soft_label,
                "hard": f"{hard_rate:.3f} ({passed_cases} /{total_cases})",
                "_align": {"soft": "left"},
            }
        )
    return rows


def _soft_counts(cases: list[dict[str, Any]]) -> tuple[int, int] | None:
    counts = [_case_soft_count(case) for case in cases]
    if not counts or any(count is None for count in counts):
        return None
    matched = sum(count[0] for count in counts if count is not None)
    total = sum(count[1] for count in counts if count is not None)
    return matched, total


def _case_soft_count(case: dict[str, Any]) -> tuple[int, int] | None:
    message = str(case.get("message") or "")
    exact_match = ALL_CELLS_MATCH_RE.search(message)
    if exact_match:
        total = int(exact_match.group(1))
        return total, total

    matches = CELL_MATCH_RE.findall(message)
    if not matches:
        return None
    matched, total = matches[-1]
    return int(matched), int(total)


def eval_cost_rows(run_dir: str | Path) -> list[dict[str, Any]]:
    report = load_eval_report(run_dir)
    costs = [
        LMCost(
            role=str(cost.get("role") or "unknown"),
            model=str(cost.get("model") or ""),
            calls=int(cost.get("calls") or 0),
            prompt_tokens=int(cost.get("prompt_tokens") or 0),
            completion_tokens=int(cost.get("completion_tokens") or 0),
            cost_usd=float(cost.get("cost_usd") or 0.0),
        )
        for cost in report.get("costs") or []
    ]
    costs = _with_model_efforts(costs, _eval_role_efforts(report))
    rows = _grouped_cost_rows(costs)
    if costs:
        if rows:
            rows.append(_cost_spacer_row())
        rows.append(
            {
                "scope": "TOTAL",
                "model": "",
                "calls": "",
                "prompt_tok": "",
                "completion_tok": "",
                "cost_usd": f"${sum(cost.cost_usd for cost in costs):.2f}",
            }
        )
    return rows


def _cost_role_order() -> list[str]:
    return [role for _stage, entries in COST_GROUPS for _label, roles in entries for role in roles]


def _optimize_role_efforts(run_dir: str | Path) -> dict[str, str | None]:
    config = load_run_metadata(run_dir).get("resolved_config") or {}
    executor = config.get("executor_reasoning_effort")
    executor_sub = config.get("executor_sub_lm_reasoning_effort")
    proposer = config.get("proposer_reasoning_effort")
    proposer_sub = config.get("proposer_sub_lm_reasoning_effort")
    return {
        "executor": executor,
        "main": executor,
        "merge_trace_executor": executor,
        "merge_trace_main": executor,
        "sub_lm": executor_sub,
        "sub": executor_sub,
        "merge_trace_sub_lm": executor_sub,
        "merge_trace_sub": executor_sub,
        "proposer": proposer,
        "reflection": proposer,
        "merge_proposer": proposer,
        "proposer_sub_lm": proposer_sub,
        "proposer_sub": proposer_sub,
        "merge_proposer_sub_lm": proposer_sub,
        "merge_proposer_sub": proposer_sub,
    }


def _eval_role_efforts(report: dict[str, Any]) -> dict[str, str | None]:
    config = report.get("config") or {}
    return {
        "main": config.get("reasoning_effort"),
        "sub": config.get("sub_lm_reasoning_effort", "none"),
    }


def _with_model_efforts(costs: list[LMCost], efforts_by_role: dict[str, str | None]) -> list[LMCost]:
    return [
        LMCost(
            role=cost.role,
            model=_model_with_effort(cost.model, efforts_by_role.get(cost.role)),
            calls=cost.calls,
            prompt_tokens=cost.prompt_tokens,
            completion_tokens=cost.completion_tokens,
            cost_usd=cost.cost_usd,
        )
        for cost in costs
    ]


def _model_with_effort(model: str, effort: str | None) -> str:
    effort_text = str(effort).strip() if effort is not None else ""
    if not model or not effort_text:
        return model
    suffix = f"-{effort_text}"
    if model.endswith(suffix):
        return model
    return f"{model}{suffix}"


def _format_iter_parents(iteration: Any, parents: Any) -> str:
    return _format_id_parents(iteration, parents)


def _format_id_parents(identifier: Any, parents: Any) -> str:
    return f"{identifier} [{_format_parent_text(parents)}]"


def _format_parent_text(parents: Any) -> str:
    if isinstance(parents, list | tuple):
        parent_values = [parent for parent in parents if parent is not None]
        if not parent_values:
            return "seed"
        return ", ".join(str(parent) for parent in parent_values)
    elif parents in (None, ""):
        return "seed"
    return str(parents)


def _cost_breakdowns(
    total_costs: list[LMCost],
    effective_costs: list[LMCost],
) -> list[dict[str, Any]]:
    effective_by_key = {(cost.role, cost.model): cost for cost in effective_costs}
    rows: list[dict[str, Any]] = []
    for total in total_costs:
        effective = effective_by_key.get((total.role, total.model))
        effective_cost = effective.cost_usd if effective is not None else 0.0
        rows.append(
            {
                "role": total.role,
                "model": total.model,
                "calls": total.calls,
                "prompt_tokens": total.prompt_tokens,
                "completion_tokens": total.completion_tokens,
                "total_cost_usd": total.cost_usd,
                "repeat_cost_usd": max(0.0, total.cost_usd - effective_cost),
                "effective_cost_usd": effective_cost,
            }
        )
    return rows


def _grouped_cost_breakdown_rows(costs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    emitted_roles: set[str] = set()
    for stage, entries in COST_GROUPS:
        stage_costs = [cost for cost in costs if any(cost["role"] in roles for _label, roles in entries)]
        if not stage_costs:
            continue
        if rows:
            rows.append(_cost_breakdown_spacer_row())
        rows.append(_cost_breakdown_section_row(stage))
        for label, roles in entries:
            matching = [cost for cost in stage_costs if cost["role"] in roles]
            if not matching:
                rows.append(_missing_cost_breakdown_row(label))
                continue
            for cost in matching:
                emitted_roles.add(cost["role"])
                rows.append(_cost_breakdown_row(cost, label=label))

    unknown_costs = [cost for cost in costs if cost["role"] not in emitted_roles]
    if unknown_costs:
        if rows:
            rows.append(_cost_breakdown_spacer_row())
        rows.append(_cost_breakdown_section_row("other"))
        rows.extend(_cost_breakdown_row(cost, label=cost["role"]) for cost in unknown_costs)
    return rows


def _cost_breakdown_section_row(stage: str) -> dict[str, Any]:
    return {
        "scope": stage,
        "model": "",
        "calls": "",
        "prompt_tok": "",
        "completion_tok": "",
        "total_cost": "",
        "repeat_cost": "",
        "effective_cost": "",
        "_category": True,
    }


def _missing_cost_breakdown_row(label: str) -> dict[str, Any]:
    return {
        "scope": f"  - {label}",
        "model": "-",
        "calls": "-",
        "prompt_tok": "-",
        "completion_tok": "-",
        "total_cost": "-",
        "repeat_cost": "-",
        "effective_cost": "-",
    }


def _cost_breakdown_spacer_row() -> dict[str, Any]:
    return {
        "scope": "",
        "model": "",
        "calls": "",
        "prompt_tok": "",
        "completion_tok": "",
        "total_cost": "",
        "repeat_cost": "",
        "effective_cost": "",
        "_spacer": True,
    }


def _cost_breakdown_row(cost: dict[str, Any], *, label: str) -> dict[str, Any]:
    return {
        "scope": f"  - {label}",
        "model": cost["model"] or "-",
        "calls": f"{cost['calls']:,}",
        "prompt_tok": f"{cost['prompt_tokens']:,}",
        "completion_tok": f"{cost['completion_tokens']:,}",
        "total_cost": f"${cost['total_cost_usd']:.2f}",
        "repeat_cost": f"${cost['repeat_cost_usd']:.2f}",
        "effective_cost": f"${cost['effective_cost_usd']:.2f}",
    }


def _grouped_cost_rows(costs: list[LMCost]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    emitted_roles: set[str] = set()
    for stage, entries in COST_GROUPS:
        stage_costs = [cost for cost in costs if any(cost.role in roles for _label, roles in entries)]
        if not stage_costs:
            continue
        if rows:
            rows.append(_cost_spacer_row())
        rows.append(_cost_section_row(stage))
        for label, roles in entries:
            matching = [cost for cost in stage_costs if cost.role in roles]
            if not matching:
                rows.append(_missing_cost_row(label))
                continue
            for cost in matching:
                emitted_roles.add(cost.role)
                rows.append(_cost_row(cost, label=label))

    unknown_costs = [cost for cost in costs if cost.role not in emitted_roles]
    if unknown_costs:
        if rows:
            rows.append(_cost_spacer_row())
        rows.append(_cost_section_row("other"))
        rows.extend(_cost_row(cost, label=cost.role) for cost in unknown_costs)
    return rows


def _cost_section_row(stage: str) -> dict[str, Any]:
    return {
        "scope": stage,
        "model": "",
        "calls": "",
        "prompt_tok": "",
        "completion_tok": "",
        "cost_usd": "",
        "_category": True,
    }


def _missing_cost_row(label: str) -> dict[str, Any]:
    return {
        "scope": f"  - {label}",
        "model": "-",
        "calls": "-",
        "prompt_tok": "-",
        "completion_tok": "-",
        "cost_usd": "-",
    }


def _cost_spacer_row() -> dict[str, Any]:
    return {
        "scope": "",
        "model": "",
        "calls": "",
        "prompt_tok": "",
        "completion_tok": "",
        "cost_usd": "",
        "_spacer": True,
    }


def render_table(rows: list[dict[str, Any]], output_format: str = "terminal") -> str:
    if not rows:
        return "(no rows)"
    rows = _compact_fractional_columns(rows)
    if output_format == "terminal":
        return _render_terminal_table(rows)
    if output_format == "markdown":
        return _render_markdown_table(rows)
    raise ValueError(f"unknown table output format: {output_format!r}")


def _compact_fractional_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    headers = [header for header in rows[0] if not str(header).startswith("_")]
    compact_headers = {
        header for header in headers if header not in TEXT_COLUMNS and _all_decimals_are_fractional(rows, header)
    }
    if not compact_headers:
        return rows
    compacted: list[dict[str, Any]] = []
    for row in rows:
        compacted_row = dict(row)
        for header in compact_headers:
            if header in compacted_row:
                compacted_row[header] = _compact_fractional_value(compacted_row[header])
        muted_prefix = compacted_row.get("_muted_prefix")
        if isinstance(muted_prefix, dict):
            compacted_row["_muted_prefix"] = {
                header: _compact_fractional_value(value) if header in compact_headers else value
                for header, value in muted_prefix.items()
            }
        muted_suffix = compacted_row.get("_muted_suffix")
        if isinstance(muted_suffix, dict):
            compacted_row["_muted_suffix"] = {
                header: _compact_fractional_value(value) if header in compact_headers else value
                for header, value in muted_suffix.items()
            }
        compacted.append(compacted_row)
    return compacted


def _all_decimals_are_fractional(rows: list[dict[str, Any]], header: str) -> bool:
    decimals: list[re.Match[str]] = []
    for row in rows:
        if row.get("_category") or row.get("_spacer"):
            continue
        value = row.get(header, "")
        if value in ("", "-"):
            continue
        decimals.extend(DECIMAL_RE.finditer(str(value)))
    return bool(decimals) and all(match.group("whole") == "0" for match in decimals)


def _compact_fractional_value(value: Any) -> Any:
    return LEADING_ZERO_DECIMAL_RE.sub(r"\g<sign>.\g<fraction>", str(value))


def _render_markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [header for header in rows[0] if not str(header).startswith("_")]
    rendered_rows = [_render_markdown_row(row, headers) for row in rows]
    widths = {
        header: max(len(str(header)), *(len(row[header]) for row in rendered_rows))
        for header in headers
    }
    header_line = "| " + " | ".join(str(header).ljust(widths[header]) for header in headers) + " |"
    rule = "| " + " | ".join("-" * max(3, widths[header]) for header in headers) + " |"
    body = [
        "| " + " | ".join(row[header].rjust(widths[header]) for header in headers) + " |"
        for row in rendered_rows
    ]
    return "\n".join([header_line, rule, *body])


def _render_terminal_table(rows: list[dict[str, Any]]) -> str:
    headers = [header for header in rows[0] if not str(header).startswith("_")]
    rendered_rows = [_render_terminal_row(row, headers) for row in rows]
    widths = {
        header: max(len(str(header)), *(len(row[header]) for row in rendered_rows))
        for header in headers
    }
    return "\n".join(
        [
            _terminal_rule("┌", "┬", "┐", headers, widths),
            _terminal_row({header: str(header) for header in headers}, headers, widths, align="left"),
            _terminal_rule("├", "┼", "┤", headers, widths),
            *[
                _terminal_row(
                    row,
                    headers,
                    widths,
                    align="right",
                    column_align=source_row.get("_align", {}),
                    highlight=source_row.get("_highlight", False),
                    category=source_row.get("_category", False),
                    muted_prefix=source_row.get("_muted_prefix", {}),
                    muted_suffix=source_row.get("_muted_suffix", {}),
                )
                for source_row, row in zip(rows, rendered_rows, strict=True)
            ],
            _terminal_rule("└", "┴", "┘", headers, widths),
        ]
    )


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", "<br>")


def _render_markdown_row(row: dict[str, Any], headers: list[str]) -> dict[str, str]:
    rendered = {header: _markdown_cell(row.get(header, "")) for header in headers}
    if row.get("_category") and headers:
        rendered[headers[0]] = f"*{rendered[headers[0]]}*"
    if row.get("_highlight"):
        rendered = {header: f"**{value}**" if value else value for header, value in rendered.items()}
    return rendered


def _terminal_cell(value: Any) -> str:
    return str(value).replace("\n", " ")


def _render_terminal_row(row: dict[str, Any], headers: list[str]) -> dict[str, str]:
    return {header: _terminal_cell(row.get(header, "")) for header in headers}


def _terminal_rule(
    left: str,
    middle: str,
    right: str,
    headers: list[str],
    widths: dict[str, int],
) -> str:
    return left + middle.join("─" * (widths[header] + 2) for header in headers) + right


def _terminal_row(
    row: dict[str, str],
    headers: list[str],
    widths: dict[str, int],
    *,
    align: str,
    column_align: dict[str, str] | None = None,
    highlight: bool = False,
    category: bool = False,
    muted_prefix: dict[str, str] | None = None,
    muted_suffix: dict[str, str] | None = None,
) -> str:
    cells = []
    for header in headers:
        value = row[header]
        cell_align = (column_align or {}).get(header, align)
        if header == headers[0] or cell_align == "left":
            value = value.ljust(widths[header])
        else:
            value = value.rjust(widths[header])
        value = _mute_terminal_prefix(value, row[header], (muted_prefix or {}).get(header, ""))
        value = _mute_terminal_suffix(value, row[header], (muted_suffix or {}).get(header, ""))
        if highlight and value.strip():
            value = f"{ANSI_BOLD_GOLD}{_restore_terminal_style(value, ANSI_BOLD_GOLD)}{ANSI_RESET}"
        elif category and header == headers[0] and value.strip():
            value = f"{ANSI_ITALIC}{value}{ANSI_RESET}"
        cells.append(f" {value} ")
    return "│" + "│".join(cells) + "│"


def _restore_terminal_style(value: str, style: str) -> str:
    return value.replace(ANSI_RESET, f"{ANSI_RESET}{style}")


def _mute_terminal_prefix(value: str, raw_value: str, prefix: str) -> str:
    if not prefix or not raw_value.startswith(prefix):
        return value
    start = value.find(prefix)
    if start < 0:
        return value
    end = start + len(prefix)
    return f"{value[:start]}{ANSI_MUTED}{value[start:end]}{ANSI_RESET}{value[end:]}"


def _mute_terminal_suffix(value: str, raw_value: str, suffix: str) -> str:
    if not suffix or not raw_value.endswith(suffix):
        return value
    start = value.rfind(suffix)
    if start < 0:
        return value
    end = start + len(suffix)
    return f"{value[:start]}{ANSI_MUTED}{value[start:end]}{ANSI_RESET}{value[end:]}"


def render_stats(run_dir: str | Path, table: str = "all", output_format: str = "terminal") -> str:
    if _is_eval_run(run_dir):
        return render_eval_stats(run_dir, table=table, output_format=output_format)

    sections: list[str] = [header_summary(run_dir)]
    if table in {"all", "iterations"}:
        sections.extend(["", "iterations:", render_table(iteration_rows(run_dir), output_format)])
    if table in {"all", "candidates"}:
        sections.extend(["", "candidates:", render_table(candidate_rows(run_dir), output_format)])
    if table in {"all", "costs"}:
        sections.extend(["", "costs:", render_table(cost_rows(run_dir), output_format)])
    return "\n".join(sections)


def render_eval_stats(run_dir: str | Path, table: str = "all", output_format: str = "terminal") -> str:
    sections: list[str] = [eval_header_summary(run_dir)]
    if table in {"all", "tasks"}:
        sections.extend(["", "tasks:", render_table(eval_task_rows(run_dir), output_format)])
    if table in {"all", "costs"}:
        sections.extend(["", "costs:", render_table(eval_cost_rows(run_dir), output_format)])
    if table in {"iterations", "candidates"}:
        sections.extend(["", f"{table}: not available for eval runs"])
    return "\n".join(sections)


def eval_header_summary(run_dir: str | Path) -> str:
    report = load_eval_report(run_dir)
    total_tasks = int(report.get("total_tasks") or 0)
    passing = int(report.get("tasks_all_passing") or 0)
    total_cost = float(report.get("total_cost_usd") or 0.0)
    duration_seconds = float(report.get("duration_seconds") or 0.0)
    minutes, seconds = divmod(int(duration_seconds), 60)
    return (
        f"eval: tasks={total_tasks}, soft={float(report.get('soft_restriction_avg') or 0.0):.3f}, "
        f"hard={float(report.get('hard_restriction_avg') or 0.0):.3f} "
        f"({passing}/{total_tasks}), cost=${total_cost:.2f}, duration={minutes}m {seconds}s"
    )


def _is_eval_run(run_dir: str | Path) -> bool:
    return (Path(run_dir) / "eval.json").exists() and not (Path(run_dir) / "gepa_state.bin").exists()


def _candidate_rows_from_artifact(run_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "all_candidates.json"
    if not path.exists():
        return []
    candidates = json.loads(path.read_text())
    best_idx = _best_candidate_idx_from_artifacts(run_dir, candidates)
    return [
        {
            "cand [par]": _format_id_parents(
                candidate.get("idx", index),
                candidate.get("parent", ""),
            ),
            "mean": f"{float(candidate.get('score', 0.0)):.3f}",
            "hard": "",
            "pareto": "",
            "exclusive": "",
            "Δ-seed": "",
            "_highlight": candidate.get("idx", index) == best_idx,
        }
        for index, candidate in enumerate(candidates)
    ]


def _best_candidate_idx(run_dir: str | Path, subscores: list[Any]) -> int | None:
    summary = load_summary(run_dir)
    if isinstance(summary.get("best_idx"), int):
        return summary["best_idx"]
    if not subscores:
        return None
    return max(range(len(subscores)), key=lambda index: _mean_scores(subscores[index]))


def _best_candidate_idx_from_artifacts(
    run_dir: str | Path, candidates: list[dict[str, Any]]
) -> int | None:
    summary = load_summary(run_dir)
    if isinstance(summary.get("best_idx"), int):
        return summary["best_idx"]
    if not candidates:
        return None
    best = max(candidates, key=lambda candidate: float(candidate.get("score", 0.0)))
    return int(best.get("idx", candidates.index(best)))


def _cost_row(cost: LMCost, *, label: str) -> dict[str, Any]:
    return {
        "scope": f"  - {label}",
        "model": cost.model or "-",
        "calls": cost.calls,
        "prompt_tok": f"{cost.prompt_tokens:,}",
        "completion_tok": f"{cost.completion_tokens:,}",
        "cost_usd": f"${cost.cost_usd:.2f}",
    }


def _iteration_scores(entry: dict[str, Any]) -> tuple[list[float], list[float], Any] | None:
    if "subsample_scores" in entry and "new_subsample_scores" in entry:
        return (
            _score_values(entry.get("subsample_scores") or []),
            _score_values(entry.get("new_subsample_scores") or []),
            entry.get("selected_program_candidate", ""),
        )
    if "id1_subsample_scores" in entry and "new_program_subsample_scores" in entry:
        id1_scores = _score_values(entry.get("id1_subsample_scores") or [])
        id2_scores = _score_values(entry.get("id2_subsample_scores") or [])
        parent_scores = [max(a, b) for a, b in zip(id1_scores, id2_scores, strict=False)]
        pair = entry.get("rlm_merge_candidate_pair") or entry.get("merged_entities") or "merge"
        return parent_scores, _score_values(entry.get("new_program_subsample_scores") or []), pair
    return None


def _mean_scores(scores: Any) -> float:
    return _mean_list(_score_values(scores))


def _score_values(scores: Any) -> list[float]:
    if isinstance(scores, dict):
        return [float(value) for value in scores.values()]
    return [float(value) for value in scores]


def _mean_list(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _hard_count(values: list[float]) -> int:
    return sum(1 for value in values if value >= HARD_THRESHOLD)


def _format_hard_change(parent_scores: list[float], new_scores: list[float]) -> tuple[str, str]:
    n = min(len(parent_scores), len(new_scores))
    parent_hard = _hard_count(parent_scores)
    new_hard = _hard_count(new_scores)
    parent_rate = parent_hard / n if n else 0.0
    new_rate = new_hard / n if n else 0.0
    delta = _format_delta(new_rate - parent_rate)
    secondary = f"{delta}; {parent_hard} → {new_hard} /{n}"
    return f"{parent_rate:.3f} → {new_rate:.3f} {secondary}", f" {secondary}"


def _format_soft_change(parent_scores: list[float], new_scores: list[float]) -> tuple[str, str]:
    parent_mean = _mean_list(parent_scores)
    new_mean = _mean_list(new_scores)
    secondary = _format_delta(new_mean - parent_mean)
    return f"{parent_mean:.3f} → {new_mean:.3f} {secondary}", f" {secondary}"


def _hard_rate(values: list[float]) -> float:
    return _hard_count(values) / len(values) if values else 0.0


def _format_delta(value: float) -> str:
    return f"{value:+.3f}"


def _hard_flips(parent_scores: list[float], new_scores: list[float]) -> tuple[int, int]:
    gains = losses = 0
    for parent, new in zip(parent_scores, new_scores, strict=False):
        parent_hard = parent >= HARD_THRESHOLD
        new_hard = new >= HARD_THRESHOLD
        if not parent_hard and new_hard:
            gains += 1
        elif parent_hard and not new_hard:
            losses += 1
    return gains, losses


def _format_flips(gains: int, losses: int) -> tuple[str, str]:
    net = gains - losses
    primary = f"+{gains}/-{losses}"
    secondary = f"{net:+d}"
    return f"{primary} {secondary}", f" {secondary}"


def _mcnemar_exact_p(gains: int, losses: int) -> float:
    total = gains + losses
    if total == 0:
        return 1.0
    smaller = min(gains, losses)
    cdf = sum(math.comb(total, i) for i in range(smaller + 1)) * (0.5 ** total)
    return min(1.0, 2 * cdf)


def _format_exclusive(scores: list[float]) -> str:
    if not scores:
        return "0"
    return f"{len(scores)} (avg {_mean_list(scores):.2f})"
