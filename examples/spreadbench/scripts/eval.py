"""Run the SpreadsheetBench eval on a dataset.

Thin CLI wrapper around :func:`scripts.lib.evaluation.run_evaluation`.
All the real work lives in ``scripts/lib/evaluation.py`` so notebooks
and the future optimize.py can drive the same eval loop in-process.

Examples:
    # Default: gpt-5.4 low, seed prompt, testset (verified 400).
    uv run python scripts/eval.py

    # Same run against a GEPA-optimized prompt.
    uv run python scripts/eval.py --run_dir runs/gepa_20260410_151500

    # Smoke test on 5 tasks.
    uv run python scripts/eval.py --limit 5

    # Different provider.
    uv run python scripts/eval.py --lm anthropic/claude-opus-4-6 \\
        --reasoning_effort high
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from lib.evaluation import EvalConfig, run_evaluation  # noqa: E402
from lib.lm_config import SUB_LM  # noqa: E402

_EXAMPLE_DIR = _SCRIPTS_DIR.parent
DEFAULT_OUTPUT_DIR = _EXAMPLE_DIR / "runs"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate SpreadsheetRLM on a dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--lm", default="openai/gpt-5.4", help="main LM")
    p.add_argument(
        "--reasoning_effort",
        default="low",
        help="reasoning effort for the main LM (e.g. low, medium, high). "
        "Pass '' or 'none' to omit it (required on Claude 4.6 to disable "
        "extended thinking — any non-empty value maps to adaptive thinking).",
    )
    p.add_argument("--sub_lm", default=SUB_LM, help="sub LM for predict() calls")
    p.add_argument(
        "--dataset",
        default="testset",
        help="dataset folder under data/ (e.g. testset, trainset)",
    )
    p.add_argument(
        "--run_dir",
        default=None,
        help="optional GEPA run dir to extract evolved components from; "
        "when omitted, the eval uses the seed ManipulateSpreadsheet docstring "
        "and seed skill instructions. By default both evolved components "
        "(signature + skill) are loaded; use --only to apply just one.",
    )
    p.add_argument(
        "--only",
        choices=["signature", "skill"],
        default=None,
        help="when --run_dir is set, apply only one evolved component and "
        "use the seed value for the other (useful for A/B'ing each component's "
        "individual contribution). Default: apply both.",
    )
    p.add_argument("--limit", type=int, default=None, help="cap tasks for smoke tests")
    p.add_argument(
        "--task_ids",
        default=None,
        help="comma-separated task IDs to run (subset of the dataset). "
        "Prefix with '@' to read from a file with one ID per line, "
        "e.g. --task_ids @failing_ids.txt",
    )
    p.add_argument("--concurrency", type=int, default=30)
    p.add_argument("--max_iterations", type=int, default=50)
    p.add_argument("--task_timeout", type=int, default=300)
    p.add_argument(
        "--output",
        default=None,
        help=f"results JSON path (default: {DEFAULT_OUTPUT_DIR}/eval_<timestamp>.json)",
    )
    p.add_argument(
        "--log_dir",
        default=None,
        help="per-case RLM log directory (default: sibling of --output, "
        "named after it without the .json suffix)",
    )
    p.add_argument(
        "--no_logs",
        action="store_true",
        help="disable per-case RLM logging (logs are on by default)",
    )
    p.add_argument(
        "--no_cache",
        action="store_true",
        help="disable dspy.LM caching (useful when re-running the same prompt)",
    )
    return p.parse_args()


def _resolve_output(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg)
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_DIR / f"eval_{ts}.json"


def _resolve_log_dir(
    arg: str | None, output_path: Path, disabled: bool
) -> Path | None:
    if disabled:
        return None
    if arg:
        return Path(arg)
    return output_path.with_suffix("")


def _parse_task_ids(raw: str | None) -> tuple[str, ...] | None:
    if not raw:
        return None
    if raw.startswith("@"):
        text = Path(raw[1:]).read_text()
    else:
        text = raw
    ids = [s.strip() for s in text.replace("\n", ",").split(",")]
    return tuple(s for s in ids if s) or None


def main() -> int:
    args = _parse_args()

    output_path = _resolve_output(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = _resolve_log_dir(args.log_dir, output_path, args.no_logs)

    effort = args.reasoning_effort
    if effort and effort.strip().lower() == "none":
        effort = None

    config = EvalConfig(
        lm=args.lm,
        sub_lm=args.sub_lm,
        reasoning_effort=effort or None,
        dataset=args.dataset,
        run_dir=args.run_dir,
        only=args.only,
        limit=args.limit,
        task_ids=_parse_task_ids(args.task_ids),
        concurrency=args.concurrency,
        max_iterations=args.max_iterations,
        task_timeout=args.task_timeout,
        cache=not args.no_cache,
        log_dir=log_dir,
    )

    report = run_evaluation(config)

    output_path.write_text(json.dumps(report.to_dict(), indent=2))

    mins, secs = divmod(int(report.duration_seconds), 60)
    print()
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Signature:      {report.signature_source} ({report.signature_length} chars)")
    print(f"Skill:          {report.skill_source} ({report.skill_length} chars)")
    _effort = config.reasoning_effort if config.reasoning_effort else "none"
    print(f"Model:          {config.lm}  (reasoning_effort={_effort})")
    print(f"Dataset:        {config.dataset}")
    print(f"Tasks:          {report.total_tasks}")
    print(f"Duration:       {mins}m {secs}s")
    print(f"Soft (avg):     {report.soft_restriction_avg:.4f}")
    print(
        f"Hard (avg):     {report.hard_restriction_avg:.4f}  "
        f"({report.tasks_all_passing}/{report.total_tasks} all passing)"
    )
    print()
    print("Costs:")
    for c in report.costs:
        print(
            f"  {c.role:<8s} {c.model:<32s} "
            f"{c.calls:>5d} calls  "
            f"{c.prompt_tokens:>10,d} in / {c.completion_tokens:>10,d} out  "
            f"${c.cost_usd:>9.4f}"
        )
    print(f"  {'total':<8s} {'':<32s} {'':<5s}         "
          f"{'':<27s}  ${report.total_cost_usd:>9.4f}")
    print()
    print(f"Saved to:       {output_path}")
    if config.log_dir is not None:
        print(f"Per-case logs:  {config.log_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
