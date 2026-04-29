from __future__ import annotations

import argparse
import copy
import inspect
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .reporting.stats import render_stats
from .schema import OptimizeConfig
from .service import check_optimization, run_optimization


def run_project_cli(
    build_project: Callable[[], Any],
    default_config: OptimizeConfig,
    *,
    argv: list[str] | None = None,
    add_project_args: Callable[[argparse.ArgumentParser], None] | None = None,
    apply_project_args: Callable[[OptimizeConfig, argparse.Namespace], OptimizeConfig] | None = None,
    add_project_subcommands: Callable[[Any], None] | None = None,
    handle_project_command: Callable[[argparse.Namespace], int | None] | None = None,
) -> int:
    parser = build_parser(
        add_project_args=add_project_args,
        add_project_subcommands=add_project_subcommands,
    )
    args = parser.parse_args(argv)

    if args.command == "stats":
        print(render_stats(args.run_dir, table=args.table, output_format=args.format))
        return 0
    if args.command == "plot":
        from .reporting.plots import write_plots

        outputs = write_plots(args.run_dir, output=args.output)
        for output in outputs:
            print(output)
        return 0

    if args.command != "optimize":
        if handle_project_command is not None:
            result = handle_project_command(args)
            if result is not None:
                return result
        parser.print_help()
        return 2

    config = apply_optimize_args(default_config, args)
    if apply_project_args is not None:
        config = apply_project_args(config, args)

    project = _build_project(build_project, config)
    print_resolved_config(config)
    if args.check:
        validation = check_optimization(project, config)
        print(
            f"check ok: {len(validation.trainset)} train examples, "
            f"{len(validation.valset)} val examples"
        )
        return 0

    report = run_optimization(project, config, command=" ".join(sys.argv if argv is None else argv))
    print(f"run_dir: {report.run_dir}")
    print(f"best: candidate {report.best_idx} score={report.best_val_score:.4f}")
    print(f"total_cost_usd: ${report.total_cost_usd:.2f}")
    return 0


def build_parser(
    *,
    add_project_args: Callable[[argparse.ArgumentParser], None] | None = None,
    add_project_subcommands: Callable[[Any], None] | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rlm-gepa")
    subparsers = parser.add_subparsers(dest="command", required=True)

    optimize = subparsers.add_parser("optimize")
    optimize.add_argument("--check", action="store_true")
    optimize.add_argument("--executor-lm")
    optimize.add_argument("--executor-sub-lm")
    optimize.add_argument("--executor-reasoning-effort")
    optimize.add_argument("--executor-sub-lm-reasoning-effort")
    optimize.add_argument("--proposer-lm")
    optimize.add_argument("--proposer-sub-lm")
    optimize.add_argument("--proposer-reasoning-effort")
    optimize.add_argument("--proposer-sub-lm-reasoning-effort")
    optimize.add_argument("--max-metric-calls", type=int)
    optimize.add_argument("--minibatch-size", type=int)
    optimize.add_argument("--concurrency", type=int)
    optimize.add_argument("--max-iterations", type=int)
    optimize.add_argument("--task-timeout", type=int)
    optimize.add_argument("--proposer-timeout", type=int)
    optimize.add_argument("--heartbeat-interval-seconds", type=float)
    optimize.add_argument("--run-dir", type=Path)
    optimize.add_argument("--resume", action="store_true")
    optimize.add_argument("--cache", action="store_true")
    optimize.add_argument("--verbose-rlm", action="store_true")
    optimize.add_argument("--merge-proposer", action="store_true")
    optimize.add_argument("--max-merge-attempts", type=int)
    optimize.add_argument("--candidate-selection-strategy", choices=["pareto", "current_best", "epsilon_greedy"])
    optimize.add_argument("--component-selection-strategy", choices=["round_robin", "all"])
    if add_project_args is not None:
        add_project_args(optimize)

    stats = subparsers.add_parser("stats")
    stats.add_argument("run_dir", type=Path)
    stats.add_argument(
        "--table",
        choices=["all", "iterations", "candidates", "tasks", "costs"],
        default="all",
    )
    stats.add_argument("--format", choices=["terminal", "markdown"], default="terminal")

    plot = subparsers.add_parser("plot")
    plot.add_argument("run_dir", type=Path)
    plot.add_argument("-o", "--output", type=Path, help="output directory or file prefix")
    if add_project_subcommands is not None:
        add_project_subcommands(subparsers)
    return parser


def apply_optimize_args(config: OptimizeConfig, args: argparse.Namespace) -> OptimizeConfig:
    config = copy.copy(config)
    updates = {
        "executor_lm": args.executor_lm,
        "executor_sub_lm": args.executor_sub_lm,
        "executor_reasoning_effort": args.executor_reasoning_effort,
        "executor_sub_lm_reasoning_effort": args.executor_sub_lm_reasoning_effort,
        "proposer_lm": args.proposer_lm,
        "proposer_sub_lm": args.proposer_sub_lm,
        "proposer_reasoning_effort": args.proposer_reasoning_effort,
        "proposer_sub_lm_reasoning_effort": args.proposer_sub_lm_reasoning_effort,
        "max_metric_calls": args.max_metric_calls,
        "minibatch_size": args.minibatch_size,
        "concurrency": args.concurrency,
        "max_iterations": args.max_iterations,
        "task_timeout": args.task_timeout,
        "proposer_timeout": args.proposer_timeout,
        "heartbeat_interval_seconds": args.heartbeat_interval_seconds,
        "run_dir": args.run_dir,
        "candidate_selection_strategy": args.candidate_selection_strategy,
        "component_selection_strategy": args.component_selection_strategy,
        "max_merge_attempts": args.max_merge_attempts,
    }
    for key, value in updates.items():
        if value is not None:
            setattr(config, key, value)
    if args.resume:
        config.resume = True
    if args.cache:
        config.cache = True
    if args.verbose_rlm:
        config.verbose_rlm = True
    if args.merge_proposer:
        config.merge_proposer = True
    return config


def print_resolved_config(config: OptimizeConfig) -> None:
    print("resolved config:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")


def _build_project(build_project: Callable[..., Any], config: OptimizeConfig) -> Any:
    signature = inspect.signature(build_project)
    if len(signature.parameters) == 0:
        return build_project()
    return build_project(config)
