from __future__ import annotations

import json
import math
import pickle
import re
import shutil
import textwrap
from pathlib import Path
from statistics import mean
from typing import Any

from .cost import LMCost, aggregate_costs_from_log

HARD_THRESHOLD = 0.999
ANSI_ITALIC = "\033[3m"
ANSI_MUTED = "\033[38;5;248m"
ANSI_MUTED_GOLD = "\033[38;5;178m"
ANSI_BOLD_GOLD = "\033[1;38;5;220m"
ANSI_RESET = "\033[0m"
CELL_MATCH_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s+cells?\s+match")
ALL_CELLS_MATCH_RE = re.compile(r"All\s+(\d+)\s+cells?\s+match")
DECIMAL_RE = re.compile(r"(?<![\w.])(?P<sign>[+-]?)(?P<whole>\d+)\.(?P<fraction>\d+)")
LEADING_ZERO_DECIMAL_RE = re.compile(r"(?<![\w.])(?P<sign>[+-]?)0\.(?P<fraction>\d+)")
TEXT_COLUMNS = {"model", "outcome", "parent", "scope", "task"}
TERMINAL_HEADER_ALIASES = {
    "prompt_tok": "in_tok",
    "completion_tok": "out_tok",
    "total_cost": "total",
    "repeat_cost": "repeat",
    "effective_cost": "eff",
    "cost_usd": "cost",
    "soft: best(par) -> merge": "soft\nbest(par) -> merge",
    "hard: best(par) -> merge": "hard\nbest(par) -> merge",
}
MERGE_TERMINAL_HEADER_ALIASES = {
    "pair@anc": "pair\n@anc",
    "soft: best(par) -> merge": "soft\nbest(par) -> merge",
    "hard: best(par) -> merge": "hard\nbest(par) -> merge",
}
MERGE_METRIC_COLUMNS = (
    "soft: best(par) -> merge",
    "hard: best(par) -> merge",
    "flips",
)
TERMINAL_WRAP_COLUMNS = {
    "scope",
    "model",
    "iter",
    "pair@anc",
    "outcome",
    "soft: par → child",
    "hard: par → child",
    "soft: best(par) -> merge",
    "hard: best(par) -> merge",
    "flips",
}
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
    (
        "patch-merge",
        [
            ("main", {"patch_merge_proposer"}),
            ("sub", {"patch_merge_proposer_sub_lm", "patch_merge_proposer_sub"}),
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
        hard_change, hard_secondary = _format_hard_change(parent_scores, new_scores, include_total=False)
        hard_denominator = min(len(parent_scores), len(new_scores)) or None
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
                "_iteration_hard_denominator": hard_denominator,
            }
        )
    _apply_iteration_terminal_header_suffixes(rows)
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


def merge_rows(run_dir: str | Path) -> list[dict[str, Any]]:
    state = load_run_state(run_dir)
    rows: list[dict[str, Any]] = []
    for entry in state.get("full_program_trace") or []:
        if not _is_merge_entry(entry):
            continue
        status = str(entry.get("rlm_merge_status") or ("accepted" if entry.get("merged") else "scored"))
        score_vectors = _merge_score_vectors(entry)
        if score_vectors is None:
            soft_change = hard_change = flips = p_value = "-"
            muted_prefix = {}
            hard_denominator = None
        else:
            parent_scores, new_scores = score_vectors
            gains, losses = _hard_flips(parent_scores, new_scores)
            soft_change, soft_secondary = _format_soft_change(parent_scores, new_scores)
            hard_change, hard_secondary = _format_hard_change(parent_scores, new_scores, include_total=False)
            flips, flips_secondary = _format_flips(gains, losses)
            p_value = f"{_mcnemar_exact_p(gains, losses):.2f}"
            hard_denominator = min(len(parent_scores), len(new_scores))
            muted_prefix = {
                "soft: best(par) -> merge": soft_change.removesuffix(soft_secondary),
                "hard: best(par) -> merge": hard_change.removesuffix(hard_secondary),
                "flips": flips.removesuffix(flips_secondary),
            }
        rows.append(
            {
                "iter": str(entry.get("i", len(rows))),
                "pair@anc": _format_merge_pair(entry),
                "soft: best(par) -> merge": soft_change,
                "hard: best(par) -> merge": hard_change,
                "flips": flips,
                "p": p_value,
                "outcome": _format_merge_outcome(status),
                "_detail": _format_merge_detail(state, entry, status),
                "_muted_prefix": muted_prefix,
                "_merge_hard_denominator": hard_denominator,
            }
        )
    _apply_merge_terminal_header_aliases(rows)
    return rows


def _apply_merge_terminal_header_aliases(rows: list[dict[str, Any]]) -> None:
    aliases = dict(MERGE_TERMINAL_HEADER_ALIASES)
    denominator = _common_hard_denominator(rows, "_merge_hard_denominator")
    for row in rows:
        row["_terminal_header_aliases"] = {
            **row.get("_terminal_header_aliases", {}),
            **aliases,
        }
        if denominator is not None:
            row["_terminal_header_suffixes"] = {
                **row.get("_terminal_header_suffixes", {}),
                "hard: best(par) -> merge": f"/{denominator}",
            }


def _apply_iteration_terminal_header_suffixes(rows: list[dict[str, Any]]) -> None:
    denominator = _common_hard_denominator(rows, "_iteration_hard_denominator")
    if denominator is None:
        return
    for row in rows:
        row["_terminal_header_suffixes"] = {
            **row.get("_terminal_header_suffixes", {}),
            "hard: par → child": f"/{denominator}",
        }


def _common_hard_denominator(rows: list[dict[str, Any]], key: str) -> int | None:
    denominators = [row.get(key) for row in rows if row.get(key) is not None]
    if not denominators:
        return None
    unique_denominators = set(denominators)
    if len(unique_denominators) != 1:
        return None
    denominator = unique_denominators.pop()
    return int(denominator) if denominator else None


def merge_detail_lines(run_dir: str | Path, *, width: int = 80) -> list[str]:
    lines: list[str] = []
    for row in merge_rows(run_dir):
        detail = row.get("_detail")
        if not detail or detail == "-":
            continue
        label = f"iter {row['iter']}"
        if row.get("pair@anc") not in (None, "-"):
            label = f"{label} {row['pair@anc']}"
        prefix = f"  {label}: "
        lines.append(
            textwrap.fill(
                str(detail),
                width=width,
                initial_indent=prefix,
                subsequent_indent=" " * len(prefix),
            )
        )
    return lines


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
        "patch_merge_proposer": proposer,
        "proposer_sub_lm": proposer_sub,
        "proposer_sub": proposer_sub,
        "merge_proposer_sub_lm": proposer_sub,
        "merge_proposer_sub": proposer_sub,
        "patch_merge_proposer_sub_lm": proposer_sub,
        "patch_merge_proposer_sub": proposer_sub,
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


def _is_merge_entry(entry: dict[str, Any]) -> bool:
    return (
        bool(entry.get("invoked_merge"))
        or bool(entry.get("merged"))
        or "id1_subsample_scores" in entry
        or any(str(key).startswith("rlm_merge_") for key in entry)
    )


def _format_merge_pair(entry: dict[str, Any]) -> str:
    triplet = _coerce_sequence(entry.get("rlm_merge_triplet") or entry.get("merged_entities"))
    pair = _coerce_sequence(entry.get("rlm_merge_candidate_pair"))
    ancestor = entry.get("rlm_merge_ancestor")
    if pair is None and triplet is not None and len(triplet) >= 2:
        pair = triplet[:2]
    if ancestor is None and triplet is not None and len(triplet) >= 3:
        ancestor = triplet[2]
    if pair is None or len(pair) < 2:
        return "-"
    ancestor_text = str(ancestor) if ancestor not in (None, "") else "-"
    return f"{pair[0]}+{pair[1]}@{ancestor_text}"


def _format_merge_outcome(status: str) -> str:
    normalized = status.lower().replace("-", "_").replace(" ", "_")
    if normalized == "accepted":
        return "accepted"
    if normalized in {"pair_skipped", "attempt_cap_exhausted", "no_merge_candidate"}:
        return "skipped"
    if normalized in {"preflight_failed", "subsample_rejected"}:
        return "rejected"
    return status


def _coerce_sequence(value: Any) -> list[Any] | None:
    if isinstance(value, list | tuple):
        return list(value)
    return None


def _format_merge_detail(state: dict[str, Any], entry: dict[str, Any], status: str) -> str:
    child_idx = _merge_child_idx(entry)
    if status == "accepted" and child_idx is not None:
        details = [f"→ cand {child_idx}"]
        val_detail = _format_merge_val_detail(state, entry)
        if val_detail != "-":
            details.append(val_detail)
        return "; ".join(details)
    if entry.get("rlm_merge_reject_reason"):
        return str(entry["rlm_merge_reject_reason"])
    if entry.get("rlm_merge_error_type"):
        return str(entry["rlm_merge_error_type"])
    return "-"


def _format_merge_val_detail(state: dict[str, Any], entry: dict[str, Any]) -> str:
    child_idx = _merge_child_idx(entry)
    parent_ids = _merge_parent_ids(entry)
    subscores = state.get("prog_candidate_val_subscores") or []
    if child_idx is None or not parent_ids or not _has_candidate_scores(subscores, child_idx):
        return "-"
    child_scores = subscores[int(child_idx)]
    parts: list[str] = []
    for parent_id in parent_ids:
        if not _has_candidate_scores(subscores, parent_id):
            continue
        parent_values, child_values = _aligned_score_values(subscores[parent_id], child_scores)
        if not parent_values or not child_values:
            continue
        gains, losses = _hard_flips(parent_values, child_values)
        parent_mean = _mean_list(parent_values)
        child_mean = _mean_list(child_values)
        parts.append(
            f"vs {parent_id}: "
            f"{parent_mean:.3f}→{child_mean:.3f} {_format_delta(child_mean - parent_mean)}, "
            f"hard {_hard_count(parent_values)}→{_hard_count(child_values)}/{len(child_values)}, "
            f"flips +{gains}/-{losses}"
        )
    if not parts:
        return "-"
    return "full val " + "; ".join(parts)


def _merge_child_idx(entry: dict[str, Any]) -> Any:
    if "rlm_merge_new_program_idx" in entry:
        return entry.get("rlm_merge_new_program_idx")
    if entry.get("rlm_merge_status") == "accepted":
        return entry.get("new_program_idx")
    return None


def _merge_parent_ids(entry: dict[str, Any]) -> list[int]:
    raw_ids: list[Any] = []
    if entry.get("rlm_merge_base_parent") is not None:
        raw_ids.append(entry.get("rlm_merge_base_parent"))
    if entry.get("rlm_merge_patch_source_parent") is not None:
        raw_ids.append(entry.get("rlm_merge_patch_source_parent"))
    if not raw_ids:
        pair = _coerce_sequence(entry.get("rlm_merge_candidate_pair") or entry.get("merged_entities"))
        if pair is not None:
            raw_ids.extend(pair[:2])
    parent_ids: list[int] = []
    for raw_id in raw_ids:
        try:
            parent_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if parent_id not in parent_ids:
            parent_ids.append(parent_id)
    return parent_ids


def _has_candidate_scores(subscores: Any, candidate_idx: Any) -> bool:
    try:
        index = int(candidate_idx)
    except (TypeError, ValueError):
        return False
    return isinstance(subscores, list | tuple) and 0 <= index < len(subscores) and bool(subscores[index])


def _aligned_score_values(parent_scores: Any, child_scores: Any) -> tuple[list[float], list[float]]:
    if isinstance(parent_scores, dict) and isinstance(child_scores, dict):
        keys = [key for key in parent_scores if key in child_scores]
        return [float(parent_scores[key]) for key in keys], [float(child_scores[key]) for key in keys]
    pairs = list(zip(_score_values(parent_scores), _score_values(child_scores), strict=False))
    return [parent for parent, _child in pairs], [child for _parent, child in pairs]


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
    header_labels = _terminal_header_labels(rows, headers)
    header_suffixes = _terminal_header_suffixes(rows, headers)
    terminal_width = shutil.get_terminal_size(fallback=(120, 24)).columns
    widths = _terminal_widths(headers, rendered_rows, header_labels)
    if _terminal_table_width(headers, widths) > terminal_width:
        rendered_rows = _compact_terminal_count_columns(headers, rendered_rows)
        widths = _terminal_widths(headers, rendered_rows, header_labels)
    header_labels, rendered_rows, widths = _wrap_terminal_rows_to_width(
        headers,
        header_labels,
        rendered_rows,
        widths,
        terminal_width=terminal_width,
        source_rows=rows,
    )
    header_labels, widths = _apply_terminal_header_suffixes(header_labels, header_suffixes, headers, widths)
    body = [
        line
        for source_row, row in zip(rows, rendered_rows, strict=True)
        for line in _terminal_row_lines(
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
    ]
    return "\n".join(
        [
            _terminal_rule("┌", "┬", "┐", headers, widths),
            *_terminal_row_lines(header_labels, headers, widths, align="left"),
            _terminal_rule("├", "┼", "┤", headers, widths),
            *body,
            _terminal_rule("└", "┴", "┘", headers, widths),
        ]
    )


def _terminal_header_labels(rows: list[dict[str, Any]], headers: list[str]) -> dict[str, str]:
    header_labels = {header: TERMINAL_HEADER_ALIASES.get(header, str(header)) for header in headers}
    for row in rows:
        aliases = row.get("_terminal_header_aliases")
        if isinstance(aliases, dict):
            header_labels.update({header: str(label) for header, label in aliases.items() if header in headers})
    return header_labels


def _terminal_header_suffixes(rows: list[dict[str, Any]], headers: list[str]) -> dict[str, str]:
    header_suffixes: dict[str, str] = {}
    for row in rows:
        suffixes = row.get("_terminal_header_suffixes")
        if isinstance(suffixes, dict):
            header_suffixes.update(
                {
                    header: str(suffix).strip()
                    for header, suffix in suffixes.items()
                    if header in headers and str(suffix).strip()
                }
            )
    return header_suffixes


def _apply_terminal_header_suffixes(
    header_labels: dict[str, str],
    header_suffixes: dict[str, str],
    headers: list[str],
    widths: dict[str, int],
) -> tuple[dict[str, str], dict[str, int]]:
    if not header_suffixes:
        return header_labels, widths

    labels = dict(header_labels)
    adjusted_widths = dict(widths)
    for header in headers:
        suffix = header_suffixes.get(header)
        if not suffix:
            continue
        label_lines = labels[header].splitlines() or [""]
        bottom_line = label_lines[-1]
        min_width = len(suffix) if not bottom_line else len(bottom_line) + len(suffix) + 1
        width = max(adjusted_widths[header], min_width)
        if bottom_line:
            label_lines[-1] = bottom_line + " " * (width - len(bottom_line) - len(suffix)) + suffix
        else:
            label_lines[-1] = suffix.rjust(width)
        labels[header] = "\n".join(label_lines)
        adjusted_widths[header] = max(width, _max_line_len(labels[header]))
    return labels, adjusted_widths


def _wrap_terminal_rows_to_width(
    headers: list[str],
    header_labels: dict[str, str],
    rendered_rows: list[dict[str, str]],
    widths: dict[str, int],
    *,
    terminal_width: int,
    source_rows: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, str], list[dict[str, str]], dict[str, int]]:
    if _terminal_table_width(headers, widths) <= terminal_width:
        return header_labels, rendered_rows, widths

    if _is_merge_terminal_table(headers):
        return _wrap_merge_terminal_headers(headers, header_labels, rendered_rows)

    header_labels, rendered_rows, widths = _collapse_pair_ancestor_column(
        headers,
        header_labels,
        rendered_rows,
        widths,
    )
    if _terminal_table_width(headers, widths) <= terminal_width:
        return header_labels, rendered_rows, widths

    rendered_rows, widths, metrics_collapsed = _collapse_merge_metric_columns(
        headers,
        header_labels,
        rendered_rows,
        widths,
        source_rows or [],
    )
    if _terminal_table_width(headers, widths) <= terminal_width:
        return header_labels, rendered_rows, widths

    wrap_headers = [header for header in headers if header in TERMINAL_WRAP_COLUMNS]
    if not wrap_headers:
        return header_labels, rendered_rows, widths

    fixed_width = sum(width for header, width in widths.items() if header not in wrap_headers)
    available = terminal_width - fixed_width - (3 * len(headers) + 1)
    min_widths = _terminal_min_wrap_widths(
        wrap_headers,
        header_labels,
        rendered_rows,
        metrics_collapsed,
        available,
    )
    min_wrap_width = sum(min_widths.values())
    available = max(available, min_wrap_width)

    wrapped_widths = dict(widths)
    remaining = available - min_wrap_width
    for header in wrap_headers:
        wrapped_widths[header] = min(widths[header], min_widths[header])
    while remaining > 0 and any(wrapped_widths[header] < widths[header] for header in wrap_headers):
        for header in wrap_headers:
            if remaining <= 0:
                break
            if wrapped_widths[header] < widths[header]:
                wrapped_widths[header] += 1
                remaining -= 1

    wrapped_header_labels = dict(header_labels)
    for header in wrap_headers:
        wrapped_header_labels[header] = _wrap_terminal_cell(
            header_labels[header],
            wrapped_widths[header],
            header,
        )

    wrapped_rows: list[dict[str, str]] = []
    for row in rendered_rows:
        wrapped_row = dict(row)
        for header in wrap_headers:
            wrapped_row[header] = _wrap_terminal_cell(row[header], wrapped_widths[header], header)
        wrapped_rows.append(wrapped_row)
    return wrapped_header_labels, wrapped_rows, wrapped_widths


def _is_merge_terminal_table(headers: list[str]) -> bool:
    return "pair@anc" in headers and all(header in headers for header in MERGE_METRIC_COLUMNS)


def _wrap_merge_terminal_headers(
    headers: list[str],
    header_labels: dict[str, str],
    rendered_rows: list[dict[str, str]],
) -> tuple[dict[str, str], list[dict[str, str]], dict[str, int]]:
    wrapped_header_labels = dict(header_labels)
    if "pair@anc" in headers:
        wrapped_header_labels["pair@anc"] = "pair\n@anc"
    return wrapped_header_labels, rendered_rows, _terminal_widths(headers, rendered_rows, wrapped_header_labels)


def _collapse_pair_ancestor_column(
    headers: list[str],
    header_labels: dict[str, str],
    rendered_rows: list[dict[str, str]],
    widths: dict[str, int],
) -> tuple[dict[str, str], list[dict[str, str]], dict[str, int]]:
    if "pair@anc" not in headers:
        return header_labels, rendered_rows, widths
    header_labels = dict(header_labels)
    header_labels["pair@anc"] = "pair\n@anc"
    rendered_rows = [
        {**row, "pair@anc": _collapse_pair_ancestor_cell(row["pair@anc"])} for row in rendered_rows
    ]
    return header_labels, rendered_rows, _terminal_widths(headers, rendered_rows, header_labels)


def _collapse_pair_ancestor_cell(value: str) -> str:
    text = str(value)
    if "@" not in text or text == "@":
        return text
    pair, ancestor = text.rsplit("@", 1)
    if not pair or not ancestor:
        return text
    return f"{pair}\n@\n{ancestor}"


def _collapse_merge_metric_columns(
    headers: list[str],
    header_labels: dict[str, str],
    rendered_rows: list[dict[str, str]],
    widths: dict[str, int],
    source_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, int], bool]:
    metric_headers = [header for header in MERGE_METRIC_COLUMNS if header in headers]
    if not metric_headers or not source_rows:
        return rendered_rows, widths, False

    wrapped_rows: list[dict[str, str]] = []
    changed = False
    for row, source_row in zip(rendered_rows, source_rows, strict=False):
        muted_prefix = source_row.get("_muted_prefix", {})
        wrapped_row = dict(row)
        for header in metric_headers:
            prefix = muted_prefix.get(header) if isinstance(muted_prefix, dict) else None
            if not prefix:
                continue
            collapsed = _collapse_at_muted_prefix_boundary(wrapped_row[header], str(prefix))
            if collapsed != wrapped_row[header]:
                changed = True
                wrapped_row[header] = collapsed
        wrapped_rows.append(wrapped_row)
    if not changed:
        return rendered_rows, widths, False
    return wrapped_rows, _terminal_widths(headers, wrapped_rows, header_labels), True


def _collapse_at_muted_prefix_boundary(value: str, prefix: str) -> str:
    text = str(value)
    if not text.startswith(prefix) or len(text) <= len(prefix):
        return text
    secondary = text[len(prefix) :].strip()
    if not secondary:
        return text
    return f"{prefix}\n{secondary}"


def _terminal_min_wrap_widths(
    wrap_headers: list[str],
    header_labels: dict[str, str],
    rendered_rows: list[dict[str, str]],
    metrics_collapsed: bool,
    available: int,
) -> dict[str, int]:
    min_widths = {header: _min_terminal_wrap_width(header_labels[header]) for header in wrap_headers}
    if not metrics_collapsed:
        return min_widths

    protected_widths = dict(min_widths)
    for header in MERGE_METRIC_COLUMNS:
        if header in protected_widths:
            protected_widths[header] = max(
                protected_widths[header],
                _max_line_len(header_labels[header]),
                *(_max_line_len(row[header]) for row in rendered_rows),
            )
    if sum(protected_widths.values()) <= available:
        return protected_widths
    return min_widths


def _min_terminal_wrap_width(label: str) -> int:
    return max((min(len(part), 10) for line in str(label).splitlines() for part in line.split()), default=1)


def _terminal_widths(
    headers: list[str],
    rendered_rows: list[dict[str, str]],
    header_labels: dict[str, str],
) -> dict[str, int]:
    return {
        header: max(
            _max_line_len(header_labels[header]),
            *(_max_line_len(row[header]) for row in rendered_rows),
        )
        for header in headers
    }


def _compact_terminal_count_columns(
    headers: list[str],
    rendered_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    count_columns = {"calls", "prompt_tok", "completion_tok"}.intersection(headers)
    if not count_columns:
        return rendered_rows
    return [
        {
            header: _compact_count_cell(value) if header in count_columns else value
            for header, value in row.items()
        }
        for row in rendered_rows
    ]


def _compact_count_cell(value: str) -> str:
    value = value.strip()
    if value == "-":
        return value
    try:
        count = int(value.replace(",", ""))
    except ValueError:
        return value
    if abs(count) >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if abs(count) >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)


def _terminal_table_width(headers: list[str], widths: dict[str, int]) -> int:
    return sum(widths[header] for header in headers) + (3 * len(headers) + 1)


def _max_line_len(value: str) -> int:
    return max((len(line) for line in str(value).splitlines()), default=0)


def _wrap_terminal_cell(value: str, width: int, header: str) -> str:
    text = str(value)
    if not text.strip() or len(text) <= width:
        return text
    if "\n" in text:
        return "\n".join(_wrap_terminal_cell(line, width, header) for line in text.splitlines())
    if header == "scope":
        bullet_match = re.match(r"^(\s*-\s+)(.*)$", text)
        if bullet_match and len(bullet_match.group(1)) < width:
            prefix = bullet_match.group(1)
            return "\n".join(
                textwrap.wrap(
                    bullet_match.group(2),
                    width=width,
                    initial_indent=prefix,
                    subsequent_indent=" " * len(prefix),
                    break_long_words=True,
                    break_on_hyphens=True,
                )
            )
    return "\n".join(
        textwrap.wrap(
            text,
            width=width,
            break_long_words=True,
            break_on_hyphens=True,
        )
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
    return _terminal_row_lines(
        row,
        headers,
        widths,
        align=align,
        column_align=column_align,
        highlight=highlight,
        category=category,
        muted_prefix=muted_prefix,
        muted_suffix=muted_suffix,
    )[0]


def _terminal_row_lines(
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
) -> list[str]:
    cell_lines = {header: row[header].splitlines() or [""] for header in headers}
    height = max(len(lines) for lines in cell_lines.values())
    return [
        _terminal_row_line(
            row,
            cell_lines,
            line_index,
            headers,
            widths,
            align=align,
            column_align=column_align,
            highlight=highlight,
            category=category,
            muted_prefix=muted_prefix,
            muted_suffix=muted_suffix,
        )
        for line_index in range(height)
    ]


def _terminal_row_line(
    row: dict[str, str],
    cell_lines: dict[str, list[str]],
    line_index: int,
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
        value = cell_lines[header][line_index] if line_index < len(cell_lines[header]) else ""
        cell_align = (column_align or {}).get(header, align)
        if header == headers[0] or cell_align == "left":
            value = value.ljust(widths[header])
        else:
            value = value.rjust(widths[header])
        raw_line = cell_lines[header][line_index] if line_index < len(cell_lines[header]) else ""
        muted_style = ANSI_MUTED_GOLD if highlight else ANSI_MUTED
        value = _mute_terminal_prefix(value, raw_line, (muted_prefix or {}).get(header, ""), muted_style)
        value = _mute_terminal_suffix(value, raw_line, (muted_suffix or {}).get(header, ""), muted_style)
        if highlight and value.strip():
            value = f"{ANSI_BOLD_GOLD}{_restore_terminal_style(value, ANSI_BOLD_GOLD)}{ANSI_RESET}"
        elif category and header == headers[0] and value.strip():
            value = f"{ANSI_ITALIC}{value}{ANSI_RESET}"
        cells.append(f" {value} ")
    return "│" + "│".join(cells) + "│"


def _restore_terminal_style(value: str, style: str) -> str:
    return value.replace(ANSI_RESET, f"{ANSI_RESET}{style}")


def _mute_terminal_prefix(value: str, raw_value: str, prefix: str, style: str = ANSI_MUTED) -> str:
    if not prefix or not raw_value.startswith(prefix):
        return value
    start = value.find(prefix)
    if start < 0:
        return value
    end = start + len(prefix)
    return f"{value[:start]}{style}{value[start:end]}{ANSI_RESET}{value[end:]}"


def _mute_terminal_suffix(value: str, raw_value: str, suffix: str, style: str = ANSI_MUTED) -> str:
    if not suffix or not raw_value.endswith(suffix):
        return value
    start = value.rfind(suffix)
    if start < 0:
        return value
    end = start + len(suffix)
    return f"{value[:start]}{style}{value[start:end]}{ANSI_RESET}{value[end:]}"


def render_stats(run_dir: str | Path, table: str = "all", output_format: str = "terminal") -> str:
    if _is_eval_run(run_dir):
        return render_eval_stats(run_dir, table=table, output_format=output_format)

    sections: list[str] = [header_summary(run_dir)]
    if table in {"all", "iterations"}:
        sections.extend(["", "iterations:", render_table(iteration_rows(run_dir), output_format)])
    if table == "merges":
        _append_merge_section(sections, run_dir, output_format)
    elif table == "all":
        merges = merge_rows(run_dir)
        if merges:
            _append_merge_section(sections, run_dir, output_format, rows=merges)
    if table in {"all", "candidates"}:
        sections.extend(["", "candidates:", render_table(candidate_rows(run_dir), output_format)])
    if table in {"all", "costs"}:
        sections.extend(["", "costs:", render_table(cost_rows(run_dir), output_format)])
    return "\n".join(sections)


def _append_merge_section(
    sections: list[str],
    run_dir: str | Path,
    output_format: str,
    *,
    rows: list[dict[str, Any]] | None = None,
) -> None:
    rows = merge_rows(run_dir) if rows is None else rows
    sections.extend(["", "merges:", render_table(rows, output_format)])
    details = merge_detail_lines(run_dir)
    if details:
        sections.extend(["", "merge details:", *details])


def render_eval_stats(run_dir: str | Path, table: str = "all", output_format: str = "terminal") -> str:
    sections: list[str] = [eval_header_summary(run_dir)]
    if table in {"all", "tasks"}:
        sections.extend(["", "tasks:", render_table(eval_task_rows(run_dir), output_format)])
    if table in {"all", "costs"}:
        sections.extend(["", "costs:", render_table(eval_cost_rows(run_dir), output_format)])
    if table in {"iterations", "candidates", "merges"}:
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
    merge_vectors = _merge_score_vectors(entry)
    if merge_vectors is not None:
        parent_scores, new_scores = merge_vectors
        pair = entry.get("rlm_merge_candidate_pair") or entry.get("merged_entities") or "merge"
        return parent_scores, new_scores, pair
    return None


def _merge_score_vectors(entry: dict[str, Any]) -> tuple[list[float], list[float]] | None:
    if "id1_subsample_scores" not in entry or "new_program_subsample_scores" not in entry:
        return None
    id1_scores = _score_values(entry.get("id1_subsample_scores") or [])
    id2_scores = _score_values(entry.get("id2_subsample_scores") or [])
    new_scores = _score_values(entry.get("new_program_subsample_scores") or [])
    if not id1_scores or not new_scores:
        return None
    parent_scores = id1_scores if sum(id1_scores) >= sum(id2_scores) else id2_scores
    if not parent_scores:
        return None
    return parent_scores, new_scores


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


def _format_hard_change(
    parent_scores: list[float],
    new_scores: list[float],
    *,
    include_total: bool = True,
) -> tuple[str, str]:
    n = min(len(parent_scores), len(new_scores))
    parent_hard = _hard_count(parent_scores)
    new_hard = _hard_count(new_scores)
    parent_rate = parent_hard / n if n else 0.0
    new_rate = new_hard / n if n else 0.0
    delta = _format_delta(new_rate - parent_rate)
    total = f" /{n}" if include_total else ""
    secondary = f"{delta}; {parent_hard} → {new_hard}{total}"
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
    cdf = sum(math.comb(total, i) for i in range(smaller + 1)) * (0.5**total)
    return min(1.0, 2 * cdf)


def _format_exclusive(scores: list[float]) -> str:
    if not scores:
        return "0"
    return f"{len(scores)} (avg {_mean_list(scores):.2f})"
