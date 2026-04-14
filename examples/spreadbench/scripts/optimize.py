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
import json
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from lib.lm_config import SUB_LM  # noqa: E402
from lib.optimize import OptimizeConfig, run_optimization  # noqa: E402


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
        default="anthropic/claude-opus-4-6",
        help="reflection LM (also used as the SURGICAL RLM proposer's main LM "
        "when --rlm_proposer is set)",
    )
    p.add_argument(
        "--reflection_reasoning_effort",
        default="medium",
        help="reasoning effort for the reflection LM. Pass the empty string to omit.",
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
        default=50,
        help="cap val tasks during the GEPA inner loop (None = no cap)",
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
    p.add_argument("--minibatch_size", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--module_selector",
        choices=["round_robin", "all"],
        default="round_robin",
        help="which components GEPA updates per round: round_robin alternates "
        "sig and skill, all updates both every round (2x reflection cost)",
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

    # Eval-side budget (per task case)
    p.add_argument("--concurrency", type=int, default=30)
    p.add_argument("--max_iterations", type=int, default=50)
    p.add_argument("--task_timeout", type=int, default=300)
    p.add_argument(
        "--no_cache",
        action="store_true",
        help="disable dspy.LM caching",
    )

    # Run directory (resume support)
    p.add_argument(
        "--run_dir",
        default=None,
        help="GEPA run directory (default: examples/spreadbench/runs/optimize_<timestamp>). "
        "Pass an existing run dir to RESUME from its checkpointed state.",
    )

    return p.parse_args()


def main() -> int:
    args = _parse_args()

    config = OptimizeConfig(
        lm=args.lm,
        sub_lm=args.sub_lm,
        reasoning_effort=args.reasoning_effort or None,
        reflection_lm=args.reflection_lm,
        reflection_reasoning_effort=args.reflection_reasoning_effort or None,
        train_dataset=args.train_dataset,
        val_ratio=args.val_ratio,
        val_limit=args.val_limit,
        cases_per_task=args.cases_per_task,
        max_metric_calls=args.max_metric_calls,
        minibatch_size=args.minibatch_size,
        seed=args.seed,
        module_selector=args.module_selector,
        rlm_proposer=args.rlm_proposer,
        proposer_max_iterations=args.proposer_max_iterations,
        concurrency=args.concurrency,
        max_iterations=args.max_iterations,
        task_timeout=args.task_timeout,
        cache=not args.no_cache,
        run_dir=Path(args.run_dir) if args.run_dir else None,
    )

    report = run_optimization(config)

    # Persist the report alongside the run dir
    report_path = Path(report.run_dir) / "optimization_summary.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2, default=str))

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
    for c in report.costs:
        print(
            f"  {c.role:<10s} {c.model:<32s} "
            f"{c.calls:>5d} calls  "
            f"{c.prompt_tokens:>11,d} in / {c.completion_tokens:>10,d} out  "
            f"${c.cost_usd:>9.4f}"
        )
    print(
        f"  {'total':<10s} {'':<32s} {'':<5s}         "
        f"{'':<27s}  ${report.total_cost_usd:>9.4f}"
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
