from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from .config import DEFAULT_EVAL_LM, DEFAULT_EVAL_SUB_LM, EvalConfig
from .evaluation import (
    extract_candidate,
    format_component_source,
    run_evaluation,
)

SPREADBENCH_DIR = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_OUTPUT_DIR = SPREADBENCH_DIR / "runs"


def add_eval_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "eval",
        description="Evaluate SpreadsheetRLM on a dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_eval_args(parser)


def handle_eval_command(args: argparse.Namespace) -> int | None:
    if args.command == "eval":
        return run_eval_args(args)
    return None


def run_eval_cli(argv: list[str] | None = None) -> int:
    return run_eval_args(build_eval_parser().parse_args(argv))


def run_eval_args(args: argparse.Namespace) -> int:
    output_dir = resolve_eval_output_dir(args.output, args)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "eval.json"
    log_dir = None if args.no_logs else Path(args.log_dir) if args.log_dir else output_dir

    install_codex_lm(args)

    if args.verbose_rlm:
        rlm_logger = logging.getLogger("dspy.predict.rlm")
        rlm_logger.setLevel(logging.INFO)
        if not any(
            isinstance(handler, logging.StreamHandler)
            and getattr(handler, "stream", None) is sys.stdout
            for handler in rlm_logger.handlers
        ):
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.INFO)
            handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            rlm_logger.addHandler(handler)

    effort = args.reasoning_effort
    if effort and effort.strip().lower() == "none":
        effort = None

    config = EvalConfig(
        lm=args.lm,
        sub_lm=args.sub_lm,
        reasoning_effort=effort or None,
        thinking_budget=args.thinking_budget,
        dataset=args.dataset,
        run_dir=str(args.run_dir) if args.run_dir is not None else None,
        only=args.only,
        cand_idx=args.cand_idx,
        limit=args.limit,
        task_ids=parse_eval_task_ids(args.task_ids),
        cases_per_task=args.cases_per_task,
        concurrency=args.concurrency,
        max_iterations=args.max_iterations,
        task_timeout=args.task_timeout,
        cache=args.cache,
        log_dir=log_dir,
    )
    report = run_evaluation(config)
    output_path.write_text(json.dumps(report.to_dict(), indent=2))

    mins, secs = divmod(int(report.duration_seconds), 60)
    print()
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(
        "Signature:      "
        f"{format_component_source(report.signature_source, report.signature_length, report.signature_id)}"
    )
    print(
        "Skill:          "
        f"{format_component_source(report.skill_source, report.skill_length, report.skill_id)}"
    )
    effort_label = config.reasoning_effort if config.reasoning_effort else "none"
    print(f"Model:          {config.lm}  (reasoning_effort={effort_label})")
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
    total_calls = sum(cost.calls for cost in report.costs)
    total_in = sum(cost.prompt_tokens for cost in report.costs)
    total_out = sum(cost.completion_tokens for cost in report.costs)
    for cost in report.costs:
        print(
            f"  {cost.role:<8s} {cost.model:<32s} "
            f"{cost.calls:>6,d} calls  "
            f"{cost.prompt_tokens:>12,d} in / {cost.completion_tokens:>12,d} out  "
            f"${cost.cost_usd:>9.4f}"
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


def build_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlm-gepa eval",
        description="Evaluate SpreadsheetRLM on a dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_eval_args(parser)
    return parser


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lm", default=DEFAULT_EVAL_LM, help="main LM")
    parser.add_argument(
        "--reasoning-effort",
        "--reasoning_effort",
        dest="reasoning_effort",
        default="low",
        help="reasoning effort for the main LM; pass 'none' to omit it",
    )
    parser.add_argument(
        "--sub-lm",
        "--sub_lm",
        dest="sub_lm",
        default=DEFAULT_EVAL_SUB_LM,
        help="sub LM for predict() calls",
    )
    parser.add_argument(
        "--thinking-budget",
        "--thinking_budget",
        dest="thinking_budget",
        type=int,
        default=None,
        help="explicit thinking-token budget for the main LM",
    )
    parser.add_argument("--dataset", default="testset", help="dataset folder under data/")
    parser.add_argument(
        "--run-dir",
        "--run_dir",
        dest="run_dir",
        type=Path,
        default=None,
        help="optional GEPA run dir to extract evolved components from",
    )
    parser.add_argument(
        "--only",
        choices=["signature", "skill"],
        default=None,
        help="with --run-dir, apply only one evolved component",
    )
    parser.add_argument(
        "--cand-idx",
        "--cand_idx",
        dest="cand_idx",
        type=int,
        default=None,
        help="specific candidate index; defaults to best-by-mean",
    )
    parser.add_argument("--limit", type=int, default=None, help="cap tasks for smoke tests")
    parser.add_argument(
        "--task-ids",
        "--task_ids",
        dest="task_ids",
        default=None,
        help="comma-separated task IDs, or @file with one task ID per line",
    )
    parser.add_argument(
        "--cases-per-task",
        "--cases_per_task",
        dest="cases_per_task",
        type=int,
        default=0,
        help="max test cases per task (0 = all cases)",
    )
    parser.add_argument("--concurrency", type=int, default=30)
    parser.add_argument(
        "--max-iterations",
        "--max_iterations",
        dest="max_iterations",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--task-timeout",
        "--task_timeout",
        dest="task_timeout",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--output",
        default=None,
        help="output directory; writes <dir>/eval.json and per-case logs",
    )
    parser.add_argument(
        "--log-dir",
        "--log_dir",
        dest="log_dir",
        default=None,
        help="per-case RLM log directory",
    )
    parser.add_argument(
        "--no-logs",
        "--no_logs",
        dest="no_logs",
        action="store_true",
        help="disable per-case logs",
    )
    parser.add_argument("--verbose-rlm", "--verbose_rlm", dest="verbose_rlm", action="store_true")
    codex_group = parser.add_mutually_exclusive_group()
    codex_group.add_argument(
        "--codex-lm",
        dest="codex_lm",
        action="store_true",
        default=None,
        help="force routing OpenAI-family dspy.LM constructions through dspy-codex-lm",
    )
    codex_group.add_argument(
        "--no-codex-lm",
        dest="codex_lm",
        action="store_false",
        default=None,
        help="disable automatic dspy-codex-lm routing",
    )
    parser.add_argument(
        "--codex-lm-exclude",
        action="append",
        default=[],
        help="model substring to leave unpatched when --codex-lm is enabled; repeatable",
    )
    parser.add_argument("--cache", action="store_true", help="enable dspy.LM caching")


def resolve_eval_output_dir(path_arg: str | None, args: argparse.Namespace) -> Path:
    if path_arg:
        path = Path(path_arg)
        if path.suffix == ".json":
            path = path.with_suffix("")
        return path
    DEFAULT_EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    head = f"eval_{timestamp}_{slug_lm(args.lm)}"
    tail_parts = [f"eff-{args.reasoning_effort or 'none'}"]
    if args.sub_lm:
        tail_parts.append(f"sub-{slug_lm(args.sub_lm)}")
    if args.run_dir:
        run_name = Path(args.run_dir).name
        match = __import__("re").match(r"optimize_(\d{8}_\d{6})", run_name)
        skill_tag = match.group(1) if match else slug_lm(run_name)
        _candidate, candidate_idx = extract_candidate(args.run_dir, cand_idx=args.cand_idx)
        candidate_tag = f"cand{candidate_idx}"
        if args.cand_idx is None:
            candidate_tag = f"best-{candidate_tag}"
        skill_tag += f"-{candidate_tag}"
        tail_parts.append(f"skill-{skill_tag}")
    return DEFAULT_EVAL_OUTPUT_DIR / (head + "__" + "__".join(tail_parts))


def install_codex_lm(args: argparse.Namespace) -> None:
    codex_available = importlib.util.find_spec("dspy_codex_lm") is not None
    if args.codex_lm is False or (args.codex_lm is None and not codex_available):
        return
    if not codex_available:
        raise RuntimeError(
            "--codex-lm requires dspy-codex-lm in the uv run environment. "
            "Use: uv run --project examples/spreadbench "
            "--with-editable /Users/gabriel/Workspace/dspy-codex-lm "
            "python -m spreadsheet_rlm.gepa ..."
        )

    from dspy_codex_lm.cli import install_monkeypatch

    install_monkeypatch(exclude=args.codex_lm_exclude)
    os.environ.setdefault("OPENAI_API_KEY", "codex-lm")


def parse_eval_task_ids(raw: str | None) -> tuple[str, ...] | None:
    if not raw:
        return None
    text = Path(raw[1:]).read_text() if raw.startswith("@") else raw
    ids = [item.strip() for item in text.replace("\n", ",").split(",")]
    return tuple(item for item in ids if item) or None


def slug_lm(model: str | None) -> str:
    if not model:
        return "unknown"
    tail = model.split("/", 1)[-1]
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in tail)
