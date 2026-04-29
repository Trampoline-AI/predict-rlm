from __future__ import annotations

import argparse
import sys
from typing import Any

from rlm_gepa import OptimizeConfig
from rlm_gepa.cli import run_project_cli

from ..bench.cli import add_eval_subcommand, handle_eval_command, install_codex_lm
from .config import SpreadsheetGepaConfig, default_config
from .project import build_project


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    return run_project_cli(
        build_project,
        default_config(),
        argv=argv,
        add_project_args=_add_project_args,
        apply_project_args=_apply_project_args,
        add_project_subcommands=add_eval_subcommand,
        handle_project_command=handle_eval_command,
    )


def _add_project_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-dataset")
    parser.add_argument("--val-ratio", type=float)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--cases-per-task", type=int)
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


def _apply_project_args(config: OptimizeConfig, args: Any) -> SpreadsheetGepaConfig:
    install_codex_lm(args)
    if not isinstance(config, SpreadsheetGepaConfig):
        config = SpreadsheetGepaConfig(**config.to_dict())
    if args.train_dataset is not None:
        config.train_dataset = args.train_dataset
    if args.val_ratio is not None:
        config.val_ratio = args.val_ratio
    if args.val_limit is not None:
        config.val_limit = args.val_limit
    if args.cases_per_task is not None:
        config.cases_per_task = args.cases_per_task
    return config
