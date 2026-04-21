"""Run multi-component GEPA optimization on the SpreadsheetRLM prompts.

Thin CLI wrapper around :func:`scripts.lib.optimize.run_optimization`.
All real logic lives in ``scripts/lib/optimize.py`` so notebooks and
the future eval-with-evolved-skill flow can drive the same loop
in-process.

Examples:

    # First smoke run with a small budget.
    uv run --env-file .env.development \\
        python examples/spreadbench/scripts/optimize.py \\
        --max_metric_calls 100 --rlm_proposer

    # Resume an interrupted run by pointing at the existing run dir.
    uv run --env-file .env.development \\
        python examples/spreadbench/scripts/optimize.py \\
        --run_dir examples/spreadbench/runs/optimize_20260413_171500 \\
        --max_metric_calls 500 --rlm_proposer
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import shlex
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# On-demand stack dumps: ``kill -USR1 <pid>`` dumps every Python thread's
# stack to stderr (which lands in the run's log file). Essential for
# diagnosing stalls where the process goes 0% CPU with no log activity
# and neither ``sample``, ``lldb``, nor ``py-spy`` can attach without
# sudo on macOS 14+. ``PYTHONFAULTHANDLER=1`` alone only covers crashes;
# the explicit register() call adds on-demand live-process dumps.
faulthandler.register(signal.SIGUSR1, all_threads=True)

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from lib.lm_config import SUB_LM  # noqa: E402
from lib.optimize import OptimizeConfig, run_optimization  # noqa: E402


def _load_previous_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _extract_cost_rows(summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not summary:
        return []
    rows = summary.get("cumulative_costs")
    if not isinstance(rows, list):
        rows = summary.get("costs")
    return [r for r in rows or [] if isinstance(r, dict)]


def _merge_cost_rows(
    previous: list[dict[str, Any]],
    current: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    totals: dict[tuple[str, str], dict[str, Any]] = {}

    def _add(row: dict[str, Any]) -> None:
        role = str(row.get("role") or "unknown")
        model = str(row.get("model") or "unknown")
        key = (role, model)
        if key not in totals:
            totals[key] = {
                "role": role,
                "model": model,
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd": 0.0,
            }
        dst = totals[key]
        dst["calls"] += int(row.get("calls") or 0)
        dst["prompt_tokens"] += int(row.get("prompt_tokens") or 0)
        dst["completion_tokens"] += int(row.get("completion_tokens") or 0)
        dst["cost_usd"] += float(row.get("cost_usd") or 0.0)

    for row in previous:
        _add(row)
    for row in current:
        _add(row)

    return sorted(totals.values(), key=lambda r: (r["role"], r["model"]))


def _build_summary_payload(
    report_dict: dict[str, Any],
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    current_rows = [r for r in report_dict.get("costs", []) if isinstance(r, dict)]
    cumulative_rows = _merge_cost_rows(_extract_cost_rows(previous), current_rows)

    cumulative_rollout_cost = sum(
        float(r.get("cost_usd") or 0.0)
        for r in cumulative_rows
        if r.get("role") in {"main", "sub"}
    )
    cumulative_total_cost = sum(float(r.get("cost_usd") or 0.0) for r in cumulative_rows)

    report_dict["cumulative_costs"] = cumulative_rows
    report_dict["cumulative_rollout_cost_usd"] = cumulative_rollout_cost
    report_dict["cumulative_optimization_cost_usd"] = (
        cumulative_total_cost - cumulative_rollout_cost
    )
    report_dict["cumulative_total_cost_usd"] = cumulative_total_cost
    return report_dict


def _slug_lm(model: str | None) -> str:
    """Short filesystem-safe slug for an LM model identifier.

    ``openai/gpt-5.4-mini`` → ``gpt-5.4-mini``
    ``gemini/gemini-3.0-flash`` → ``gemini-3.0-flash``
    ``anthropic/claude-opus-4-6`` → ``claude-opus-4-6``
    """
    if not model:
        return "unknown"
    tail = model.split("/")[-1]
    # Collapse any remaining filesystem-unfriendly characters.
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in tail)


def _append_launch_history(run_dir: Path) -> None:
    """Record this invocation in ``<run_dir>/launch_history.log``.

    Each line:
      ``<iso-timestamp>\\t<cwd>\\t<shlex-joined argv>``

    Appending (rather than overwriting) preserves the full invocation
    history across resumes — handy when a run was relaunched several
    times with slightly different flags and you want to see which
    configuration produced which iterations in the state bin.
    """
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = shlex.join([sys.executable, *sys.argv])
        line = (
            f"{datetime.now().isoformat(timespec='seconds')}\t"
            f"{Path.cwd()}\t{cmd}\n"
        )
        with (run_dir / "launch_history.log").open("a") as f:
            f.write(line)
    except Exception:
        pass  # best-effort; a missing log shouldn't block a real run


def _resolve_run_dir_for_cli(config: OptimizeConfig) -> Path:
    """Resolve and pin ``config.run_dir`` before optimization starts."""
    if config.run_dir is not None:
        return Path(config.run_dir)
    example_dir = Path(__file__).resolve().parent.parent
    runs_dir = example_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Encode the three LM roles into the directory name so a glance at
    # runs/ tells you what model mix produced each artifact. The order
    # matches the flag order: main LM, sub LM, reflection/proposer LM.
    main_slug = _slug_lm(getattr(config, "lm", None))
    sub_slug = _slug_lm(getattr(config, "sub_lm", None))
    refl_slug = _slug_lm(getattr(config, "reflection_lm", None))
    run_dir = runs_dir / (
        f"optimize_{timestamp}_{main_slug}__sub-{sub_slug}__prop-{refl_slug}"
    )
    config.run_dir = run_dir
    return run_dir


def _build_resume_argv(run_dir: Path, argv: list[str]) -> list[str]:
    """Return ``argv`` with ``--run_dir`` set to *run_dir*."""
    updated: list[str] = []
    i = 0
    saw_run_dir = False
    while i < len(argv):
        token = argv[i]
        if token == "--run_dir":
            saw_run_dir = True
            updated.extend(["--run_dir", str(run_dir)])
            i += 2
            continue
        if token.startswith("--run_dir="):
            saw_run_dir = True
            updated.append(f"--run_dir={run_dir}")
            i += 1
            continue
        updated.append(token)
        i += 1
    if not saw_run_dir:
        updated.extend(["--run_dir", str(run_dir)])
    return updated


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-component GEPA optimization of the SpreadsheetRLM prompts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Task LMs
    p.add_argument("--lm", default="openai/gpt-5.4", help="task LM")
    p.add_argument(
        "--reasoning_effort",
        default="low",
        help="reasoning effort for the task LM (e.g. low, medium, high). "
        "Pass the empty string to omit it.",
    )
    p.add_argument(
        "--sub_lm", default=SUB_LM, help="sub LM for predict() calls"
    )

    # Reflection / proposer LM
    p.add_argument(
        "--reflection_lm",
        default="openai/gpt-5.4",
        help="reflection LM (also used as the SURGICAL RLM proposer's main LM "
        "when --rlm_proposer is set)",
    )
    p.add_argument(
        "--reflection_reasoning_effort",
        default="medium",
        help="reasoning effort for the reflection LM (e.g. low, medium, high). "
        "Pass '' or 'none' to omit. On Claude 4.6 any non-empty value maps to "
        "adaptive thinking, which increases proposer cost ~2-3x.",
    )
    p.add_argument(
        "--proposer_sub_lm",
        default="openai/gpt-5.4",
        help="sub LM for the RLM proposer's predict() calls when "
        "--rlm_proposer is set. Independent of --sub_lm (which is the "
        "executor's sub LM). Used only when --rlm_proposer is on.",
    )
    p.add_argument(
        "--proposer_sub_lm_reasoning_effort",
        default="medium",
        help="reasoning effort for the proposer's sub LM. Pass '' or 'none' "
        "to omit. Used only when --rlm_proposer is on.",
    )

    # Datasets
    p.add_argument(
        "--train_dataset",
        default="trainset",
        help="dataset folder under data/ to draw train+val from "
        "(e.g. trainset, testset)",
    )
    p.add_argument(
        "--val_ratio",
        type=float,
        default=0.20,
        help="fraction of the train_dataset held out as the GEPA inner-loop val set",
    )
    p.add_argument(
        "--val_limit",
        type=int,
        default=None,
        help="cap val tasks during the GEPA inner loop (None = no cap, "
        "use the full held-out val set; cap is still useful for smoke runs)",
    )
    p.add_argument(
        "--cases_per_task",
        type=int,
        default=1,
        help="max test cases per task during optimization (1 = first case only)",
    )

    # GEPA budget + sampling
    p.add_argument(
        "--max_metric_calls",
        type=int,
        default=2000,
        help="GEPA budget — total candidate evaluations (each evaluation = "
        "one minibatch eval)",
    )
    p.add_argument(
        "--minibatch_size",
        type=int,
        default=50,
        help="minibatch size for GEPA's inner-loop reflection — each "
        "iteration evaluates the new candidate on this many tasks "
        "before gating against the parent's score",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--module_selector",
        choices=["round_robin", "all"],
        default="round_robin",
        help="which components GEPA updates per round: round_robin alternates "
        "sig and skill, all updates both every round (2x reflection cost)",
    )

    p.add_argument(
        "--candidate_selection_strategy",
        choices=["pareto", "current_best", "epsilon_greedy"],
        default="pareto",
        help="GEPA parent selection: pareto (diversity), current_best (exploit), "
        "epsilon_greedy (90%% exploit / 10%% random)",
    )

    # Proposer
    p.add_argument(
        "--rlm_proposer",
        action="store_true",
        help="use the SURGICAL RLM proposer (a PredictRLM that iteratively "
        "analyzes traces) instead of GEPA's stock single-shot reflection LM",
    )
    p.add_argument(
        "--proposer_max_iterations",
        type=int,
        default=20,
        help="max iterations for the RLM proposer when --rlm_proposer is set",
    )
    p.add_argument(
        "--proposer_timeout",
        type=int,
        default=600,
        help="wall-time timeout (seconds) for each proposer acall (default: 600)",
    )
    p.add_argument(
        "--use_optimize_gen",
        action="store_true",
        help="use the generic skill-evolution proposer from "
        "lib/optimize_gen.py (discovery-first: 4 rule shapes, signal "
        "processing, 3-bucket trace pass) instead of the hardcoded "
        "ImproveInstructions class. Requires --rlm_proposer.",
    )

    # Eval-side budget (per task case)
    p.add_argument("--concurrency", type=int, default=30)
    p.add_argument("--max_iterations", type=int, default=50)
    p.add_argument("--task_timeout", type=int, default=300)
    p.add_argument(
        "--with_cache",
        action="store_true",
        help="enable dspy.LM's per-call response cache. Off by default — "
        "cache hits return response.usage=0 which makes per-run cost "
        "accounting lie; GEPA already stores candidate scores in "
        "gepa_state.bin, so LM-call caching adds no value for resumes. "
        "Opt in for dev iteration where you're repeating identical prompts.",
    )
    p.add_argument(
        "--verbose_rlm",
        action="store_true",
        help="stream per-iteration reasoning/code/output from the eval-side "
        "PredictRLM to stdout. Noisy with concurrency>1 since many cases "
        "interleave, but useful for smoke runs. The same information is "
        "always persisted to task_traces/evaluate_NNNN.jsonl regardless.",
    )

    # Run directory (resume support)
    p.add_argument(
        "--run_dir",
        default=None,
        help="GEPA run directory (default: examples/spreadbench/runs/optimize_<timestamp>). "
        "Pass an existing run dir to RESUME from its checkpointed state.",
    )

    return p.parse_args()


def _normalize_effort(value: str | None) -> str | None:
    if not value:
        return None
    if value.strip().lower() == "none":
        return None
    return value


def main() -> int:
    args = _parse_args()

    config = OptimizeConfig(
        lm=args.lm,
        sub_lm=args.sub_lm,
        reasoning_effort=_normalize_effort(args.reasoning_effort),
        reflection_lm=args.reflection_lm,
        reflection_reasoning_effort=_normalize_effort(args.reflection_reasoning_effort),
        proposer_sub_lm=args.proposer_sub_lm,
        proposer_sub_lm_reasoning_effort=_normalize_effort(
            args.proposer_sub_lm_reasoning_effort
        ),
        train_dataset=args.train_dataset,
        val_ratio=args.val_ratio,
        val_limit=args.val_limit,
        cases_per_task=args.cases_per_task,
        max_metric_calls=args.max_metric_calls,
        minibatch_size=args.minibatch_size,
        seed=args.seed,
        module_selector=args.module_selector,
        candidate_selection_strategy=args.candidate_selection_strategy,
        rlm_proposer=args.rlm_proposer,
        proposer_max_iterations=args.proposer_max_iterations,
        proposer_timeout=args.proposer_timeout,
        use_optimize_gen=args.use_optimize_gen,
        concurrency=args.concurrency,
        max_iterations=args.max_iterations,
        task_timeout=args.task_timeout,
        cache=args.with_cache,
        verbose_rlm=args.verbose_rlm,
        run_dir=Path(args.run_dir) if args.run_dir else None,
    )

    run_dir = _resolve_run_dir_for_cli(config)
    resume_argv = _build_resume_argv(run_dir, sys.argv[1:])
    _append_launch_history(run_dir)

    try:
        report = run_optimization(config)
    except KeyboardInterrupt:
        print()
        print("Optimization interrupted (Ctrl+C).")
        print(f"Run dir:           {run_dir}")
        print("To resume from this checkpoint, run:")
        print(
            "  python examples/spreadbench/scripts/optimize.py "
            + shlex.join(resume_argv)
        )
        print("(prepend your usual uv/env-file wrapper if needed)")
        return 130

    # Persist the report alongside the run dir
    report_path = Path(report.run_dir) / "optimization_summary.json"
    previous_summary = _load_previous_summary(report_path)
    summary_payload = _build_summary_payload(report.to_dict(), previous_summary)
    report_path.write_text(json.dumps(summary_payload, indent=2, default=str))

    mins, secs = divmod(int(report.duration_seconds), 60)

    print()
    print("=" * 60)
    print("GEPA OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Run dir:           {report.run_dir}")
    print(f"Duration:          {mins}m {secs}s")
    print(f"Candidates tried:  {report.total_candidates}")
    print(f"Metric calls:      {report.total_metric_calls}")
    print(f"Best val score:    {report.best_val_score:.4f}  (idx={report.best_idx})")
    print()
    print("Costs:")

    # Role-to-aggregate mapping — keep in sync with the report's
    # rollout_cost_usd / optimization_cost_usd computation so the
    # per-row sums match the summary totals.
    _ROLLOUT_ROLES = {"main", "sub"}
    _OPTIMIZE_ROLES = {"proposer", "proposer_sub", "reflection"}

    def _sum_rows(rows, keys, field):
        return sum(getattr(r, field) for r in rows if r.role in keys)

    rollout_calls = _sum_rows(report.costs, _ROLLOUT_ROLES, "calls")
    rollout_in = _sum_rows(report.costs, _ROLLOUT_ROLES, "prompt_tokens")
    rollout_out = _sum_rows(report.costs, _ROLLOUT_ROLES, "completion_tokens")
    optimize_calls = _sum_rows(report.costs, _OPTIMIZE_ROLES, "calls")
    optimize_in = _sum_rows(report.costs, _OPTIMIZE_ROLES, "prompt_tokens")
    optimize_out = _sum_rows(report.costs, _OPTIMIZE_ROLES, "completion_tokens")
    total_calls = sum(c.calls for c in report.costs)
    total_in = sum(c.prompt_tokens for c in report.costs)
    total_out = sum(c.completion_tokens for c in report.costs)

    cum_rows = summary_payload.get("cumulative_costs") or []
    cum_calls = sum(int(r.get("calls") or 0) for r in cum_rows)
    cum_in = sum(int(r.get("prompt_tokens") or 0) for r in cum_rows)
    cum_out = sum(int(r.get("completion_tokens") or 0) for r in cum_rows)

    for c in report.costs:
        print(
            f"  {c.role:<12s} {c.model:<32s} "
            f"{c.calls:>6,d} calls  "
            f"{c.prompt_tokens:>13,d} in / {c.completion_tokens:>12,d} out  "
            f"${c.cost_usd:>10.4f}"
        )
    print(
        f"  {'rollouts':<12s} {'':<32s} "
        f"{rollout_calls:>6,d} calls  "
        f"{rollout_in:>13,d} in / {rollout_out:>12,d} out  "
        f"${report.rollout_cost_usd:>10.4f}"
    )
    print(
        f"  {'optimize':<12s} {'':<32s} "
        f"{optimize_calls:>6,d} calls  "
        f"{optimize_in:>13,d} in / {optimize_out:>12,d} out  "
        f"${report.optimization_cost_usd:>10.4f}"
    )
    print(
        f"  {'total':<12s} {'':<32s} "
        f"{total_calls:>6,d} calls  "
        f"{total_in:>13,d} in / {total_out:>12,d} out  "
        f"${report.total_cost_usd:>10.4f}"
    )
    print(
        f"  {'cum_total':<12s} {'':<32s} "
        f"{cum_calls:>6,d} calls  "
        f"{cum_in:>13,d} in / {cum_out:>12,d} out  "
        f"${summary_payload['cumulative_total_cost_usd']:>10.4f}"
    )
    print()
    print("Best candidate written:")
    print(f"  {report.run_dir}/best_signature_docstring.txt")
    print(f"  {report.run_dir}/best_skill_instructions.txt")
    print(f"  {report.run_dir}/all_candidates.json")
    print(f"Summary:           {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
