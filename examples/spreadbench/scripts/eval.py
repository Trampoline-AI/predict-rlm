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
import faulthandler
import json
import signal
import sys
from datetime import datetime
from pathlib import Path

# On-demand stack dumps: ``kill -USR1 <pid>`` dumps every Python thread's
# stack to stderr (which lands in the run's log file). Essential for
# diagnosing stalls like the gemini-medium hang on 2026-04-18 where the
# process went 0% CPU with no log activity for 20+ minutes and neither
# ``sample``, ``lldb``, nor ``py-spy`` could attach without sudo on
# macOS 14+. ``PYTHONFAULTHANDLER=1`` alone only covers crashes; the
# explicit register() call adds the on-demand dump.
faulthandler.register(signal.SIGUSR1, all_threads=True)

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
        "--thinking_budget",
        type=int,
        default=None,
        help="explicit thinking-token budget for the main LM (e.g. 0 to "
        "attempt disabling reasoning). Passed through to LiteLLM's "
        "``thinking_budget`` kwarg; for Gemini 3 non-flash models this "
        "maps to the minimum tier rather than fully off. Overrides the "
        "``--reasoning_effort`` → budget mapping when set.",
    )
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
    p.add_argument(
        "--cases_per_task",
        type=int,
        default=0,
        help="max test cases per task (0 = all cases, default). "
        "Pass 1 to match the optimize loop's default and run one case per task.",
    )
    p.add_argument("--concurrency", type=int, default=30)
    p.add_argument("--max_iterations", type=int, default=50)
    p.add_argument("--task_timeout", type=int, default=300)
    p.add_argument(
        "--output",
        default=None,
        help=f"output directory path — the result JSON is written to "
        f"<dir>/eval.json and per-case logs to <dir>/<task_id>/case_<idx>.log "
        f"(default: {DEFAULT_OUTPUT_DIR}/eval_<timestamp>_<lm>__eff-<e>__sub-<s>__sk-<skill>). "
        f"Legacy: a path ending in .json has the suffix stripped.",
    )
    p.add_argument(
        "--log_dir",
        default=None,
        help="per-case RLM log directory (default: same as --output dir, "
        "so the result JSON and per-case logs share a folder)",
    )
    p.add_argument(
        "--no_logs",
        action="store_true",
        help="disable per-case RLM logging (logs are on by default)",
    )
    p.add_argument(
        "--verbose_rlm",
        action="store_true",
        help="stream per-iteration reasoning/code/output from PredictRLM "
        "to stdout in addition to the per-case log file. Noisy with "
        "concurrency>1 since many cases interleave, but useful for "
        "smoke runs. The same information is always persisted to the "
        "per-case case_1.log regardless.",
    )
    p.add_argument(
        "--no_cache",
        action="store_true",
        help="disable dspy.LM caching (useful when re-running the same prompt)",
    )
    return p.parse_args()


def _slug_lm(model: str | None) -> str:
    """Short filesystem-safe slug for an LM model identifier (matches
    the convention in optimize.py::_slug_lm). Strips the ``provider/``
    prefix and replaces any filesystem-unfriendly chars with ``_``.
    """
    if not model:
        return "unknown"
    tail = model.split("/", 1)[-1]
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in tail)


def _resolve_output_dir(
    path_arg: str | None, args: argparse.Namespace | None = None
) -> Path:
    """Build the eval output directory path.

    The directory contains both the result JSON (``eval.json`` inside)
    and the per-case log tree. Co-locating them makes an eval a single
    self-contained artifact and keeps ``runs/`` from developing twin
    ``foo.json`` + ``foo/`` siblings.

    When ``--output`` isn't given, encode the model mix into the
    directory name so a glance at ``runs/`` tells you what each eval
    measured (effort tier, main LM, sub LM, source of evolved skill).
    Matches the ``optimize_<ts>_<main>__sub-<sub>__prop-<refl>``
    convention on the optimize side.

    ``--output`` accepts either a directory path or a legacy
    ``.json`` path (the ``.json`` suffix is stripped so existing
    scripts keep working).
    """
    if path_arg:
        p = Path(path_arg)
        if p.suffix == ".json":
            p = p.with_suffix("")
        return p
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args is None:
        return DEFAULT_OUTPUT_DIR / f"eval_{ts}"

    # Match optimize.py's convention:
    #   optimize_<ts>_<main>__sub-<sub>__prop-<refl>
    # becomes, on the eval side:
    #   eval_<ts>_<main>__eff-<effort>__sub-<sub>__sk-<opt-ts>
    main_slug = _slug_lm(getattr(args, "lm", None))
    head = f"eval_{ts}_{main_slug}"
    tail_parts = []
    effort = getattr(args, "reasoning_effort", None) or "none"
    tail_parts.append(f"eff-{effort}")
    sub = getattr(args, "sub_lm", None)
    if sub:
        tail_parts.append(f"sub-{_slug_lm(sub)}")
    run_dir = getattr(args, "run_dir", None)
    if run_dir:
        rd_name = Path(run_dir).name
        # Extract ``optimize_<YYYYMMDD_HHMMSS>`` prefix if present;
        # fall back to the full name otherwise.
        import re
        m = re.match(r"optimize_(\d{8}_\d{6})", rd_name)
        tail_parts.append(f"sk-{m.group(1) if m else _slug_lm(rd_name)}")
    dname = head + ("__" + "__".join(tail_parts) if tail_parts else "")
    return DEFAULT_OUTPUT_DIR / dname


def _resolve_log_dir(
    arg: str | None, output_dir: Path, disabled: bool
) -> Path | None:
    if disabled:
        return None
    if arg:
        return Path(arg)
    # Log tree lives inside the output dir alongside the result JSON.
    return output_dir


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

    output_dir = _resolve_output_dir(args.output, args)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "eval.json"
    log_dir = _resolve_log_dir(args.log_dir, output_dir, args.no_logs)

    # Optional stdout stream of per-iteration reasoning/code/output.
    # Per-case logs always capture this info; --verbose_rlm additionally
    # fans it out to stdout so you can follow the RLM live while a run
    # is executing. Concurrency>1 causes interleaving; read at your own
    # peril for smoke runs with concurrency=1.
    if args.verbose_rlm:
        import logging as _logging
        import sys as _sys

        rlm_logger = _logging.getLogger("dspy.predict.rlm")
        rlm_logger.setLevel(_logging.INFO)
        # Avoid duplicate handlers if main() is called twice (e.g. tests).
        if not any(
            isinstance(h, _logging.StreamHandler)
            and getattr(h, "stream", None) is _sys.stdout
            for h in rlm_logger.handlers
        ):
            h = _logging.StreamHandler(_sys.stdout)
            h.setLevel(_logging.INFO)
            h.setFormatter(_logging.Formatter("%(asctime)s %(message)s"))
            rlm_logger.addHandler(h)

    effort = args.reasoning_effort
    if effort and effort.strip().lower() == "none":
        effort = None

    config = EvalConfig(
        lm=args.lm,
        sub_lm=args.sub_lm,
        reasoning_effort=effort or None,
        thinking_budget=args.thinking_budget,
        dataset=args.dataset,
        run_dir=args.run_dir,
        only=args.only,
        limit=args.limit,
        task_ids=_parse_task_ids(args.task_ids),
        cases_per_task=args.cases_per_task,
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
    total_calls = sum(c.calls for c in report.costs)
    total_in = sum(c.prompt_tokens for c in report.costs)
    total_out = sum(c.completion_tokens for c in report.costs)
    for c in report.costs:
        print(
            f"  {c.role:<8s} {c.model:<32s} "
            f"{c.calls:>6,d} calls  "
            f"{c.prompt_tokens:>12,d} in / {c.completion_tokens:>12,d} out  "
            f"${c.cost_usd:>9.4f}"
        )
    print(
        f"  {'total':<8s} {'':<32s} "
        f"{total_calls:>6,d} calls  "
        f"{total_in:>12,d} in / {total_out:>12,d} out  "
        f"${report.total_cost_usd:>9.4f}"
    )
    print()
    print(f"Saved to:       {output_path}")
    if config.log_dir is not None:
        print(f"Per-case logs:  {config.log_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
