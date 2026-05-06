from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from rlm_gepa import OptimizeConfig
from rlm_gepa.cli import run_project_cli

from ..bench.cli import (
    add_codex_lm_args,
    add_eval_subcommand,
    handle_eval_command,
    install_codex_lm,
)
from .config import AppWorldGepaConfig, default_config
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
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--val-ratio", type=float)
    parser.add_argument("--val-limit", type=int)
    add_codex_lm_args(parser)


def _apply_project_args(config: OptimizeConfig, args: Any) -> AppWorldGepaConfig:
    install_codex_lm(args)
    if not isinstance(config, AppWorldGepaConfig):
        config = AppWorldGepaConfig(**config.to_dict())
    if args.train_dataset is not None:
        config.train_dataset = args.train_dataset
    if args.data_root is not None:
        config.data_root = args.data_root
    if args.val_ratio is not None:
        config.val_ratio = args.val_ratio
    if args.val_limit is not None:
        config.val_limit = args.val_limit
    return config
