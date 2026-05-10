from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

from .config import DEFAULT_CONCURRENCY, DEFAULT_TASK_TIMEOUT, EvalConfig
from .evaluation import run_evaluation_sync


def add_eval_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("eval", help="Evaluate an AppWorld RLM candidate")
    parser.add_argument("--dataset", default="validation")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cand-idx", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--task-id", dest="task_ids", action="append")
    parser.add_argument("--lm", default="openai/gpt-5.4")
    parser.add_argument("--sub-lm", default="openai/gpt-5.4-mini")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--task-timeout", type=int, default=DEFAULT_TASK_TIMEOUT)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--verbose-rlm", action="store_true")
    add_codex_lm_args(parser)


def handle_eval_command(args: argparse.Namespace) -> int | None:
    if args.command != "eval":
        return None
    install_codex_lm(args)
    report = run_evaluation_sync(
        EvalConfig(
            lm=args.lm,
            sub_lm=args.sub_lm,
            reasoning_effort=args.reasoning_effort,
            dataset=args.dataset,
            data_root=args.data_root,
            run_dir=args.run_dir,
            output_dir=args.output_dir,
            cand_idx=args.cand_idx,
            limit=args.limit,
            task_ids=tuple(args.task_ids) if args.task_ids else None,
            concurrency=args.concurrency,
            max_iterations=args.max_iterations,
            task_timeout=args.task_timeout,
            val_ratio=args.val_ratio,
            seed=args.seed,
            verbose_rlm=args.verbose_rlm,
        )
    )
    print(report.to_json())
    print(appworld_eval_header_summary(report.run_dir))
    return 0


def appworld_eval_header_summary(run_dir: str | Path) -> str:
    report = json.loads((Path(run_dir) / "eval.json").read_text())
    total_tasks = int(report.get("total_tasks") or 0)
    passing = int(report.get("tasks_all_passing") or 0)
    tgc_passing = passing
    tgc_total = total_tasks
    sgc_passing = int(report.get("scenarios_all_passing") or 0)
    sgc_total = int(report.get("total_scenarios") or 0)
    total_cost = float(report.get("total_cost_usd") or 0.0)
    duration_seconds = float(report.get("duration_seconds") or 0.0)
    minutes, seconds = divmod(int(duration_seconds), 60)
    tgc = tgc_passing / tgc_total if tgc_total else 0.0
    sgc = sgc_passing / sgc_total if sgc_total else 0.0
    return (
        f"eval: tasks={total_tasks}, soft={float(report.get('soft_restriction_avg') or 0.0):.3f}, "
        f"hard={float(report.get('hard_restriction_avg') or 0.0):.3f} "
        f"({passing}/{total_tasks}), TGC={100 * tgc:.1f}% ({tgc_passing}/{tgc_total}), "
        f"SGC={100 * sgc:.1f}% ({sgc_passing}/{sgc_total}), "
        f"cost=${total_cost:.2f}, duration={minutes}m {seconds}s"
    )


def add_codex_lm_args(parser: argparse.ArgumentParser) -> None:
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


def install_codex_lm(args: argparse.Namespace) -> None:
    codex_available = importlib.util.find_spec("dspy_codex_lm") is not None
    if args.codex_lm is False or (args.codex_lm is None and not codex_available):
        return
    if not codex_available:
        raise RuntimeError(
            "--codex-lm requires dspy-codex-lm in the uv run environment. "
            "Use: uv run --project examples/appworld "
            "--with-editable /Users/gabriel/Workspace/dspy-codex-lm "
            "rlm-gepa ..."
        )

    from dspy_codex_lm.cli import install_monkeypatch

    install_monkeypatch(exclude=args.codex_lm_exclude)
    os.environ.setdefault("OPENAI_API_KEY", "codex-lm")
