from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from rlm_gepa import OptimizeConfig
from rlm_gepa.cli import run_project_cli

from .config import TerminalBenchGepaConfig, coerce_task_ids, default_config
from .project import build_project


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    return run_project_cli(
        build_project,
        default_config(),
        argv=argv,
        add_project_args=_add_project_args,
        apply_project_args=_apply_project_args,
    )


def _add_project_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-name")
    parser.add_argument("--dataset-version")
    parser.add_argument("--train-task-id", action="append", dest="train_task_ids")
    parser.add_argument("--val-task-id", action="append", dest="val_task_ids")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--terminal-bench-output-dir", type=Path)
    parser.add_argument("--terminal-bench-executable")
    parser.add_argument("--harbor-executable")
    parser.add_argument("--harbor-dataset")
    parser.add_argument("--harness-backend", choices=("harbor", "python", "cli"))
    parser.add_argument("--timeout-cleanup-grace-sec", type=int)
    parser.add_argument("--n-attempts", type=int)
    parser.add_argument("--n-concurrent-trials", type=int)
    parser.add_argument("--no-rebuild", action="store_true")
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--upload-results", action="store_true")
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


def _apply_project_args(config: OptimizeConfig, args: Any) -> TerminalBenchGepaConfig:
    codex_lm_enabled = _install_codex_lm(args)
    if not isinstance(config, TerminalBenchGepaConfig):
        config = TerminalBenchGepaConfig(**config.to_dict())
    if args.dataset_name is not None:
        config.dataset_name = args.dataset_name
    if args.dataset_version is not None:
        config.dataset_version = args.dataset_version
    if args.train_task_ids is not None:
        config.train_task_ids = coerce_task_ids(args.train_task_ids)
    if args.val_task_ids is not None:
        config.val_task_ids = coerce_task_ids(args.val_task_ids)
    if args.train_limit is not None:
        config.train_limit = args.train_limit
    if args.val_limit is not None:
        config.val_limit = args.val_limit
    if args.terminal_bench_output_dir is not None:
        config.terminal_bench_output_dir = args.terminal_bench_output_dir
    if args.terminal_bench_executable is not None:
        config.terminal_bench_executable = args.terminal_bench_executable
    if args.harbor_executable is not None:
        config.harbor_executable = args.harbor_executable
    if args.harbor_dataset is not None:
        config.harbor_dataset = args.harbor_dataset
    if args.harness_backend is not None:
        config.harness_backend = args.harness_backend
    if args.timeout_cleanup_grace_sec is not None:
        config.timeout_cleanup_grace_sec = args.timeout_cleanup_grace_sec
    if args.n_attempts is not None:
        config.n_attempts = args.n_attempts
    if args.n_concurrent_trials is not None:
        config.n_concurrent_trials = args.n_concurrent_trials
    if args.no_rebuild:
        config.no_rebuild = True
    if args.no_cleanup:
        config.cleanup = False
    if args.upload_results:
        config.upload_results = True
    config.codex_lm = codex_lm_enabled
    config.codex_lm_exclude = tuple(args.codex_lm_exclude or ())
    return config


def _install_codex_lm(args: Any) -> bool:
    codex_available = importlib.util.find_spec("dspy_codex_lm") is not None
    if args.codex_lm is False or (args.codex_lm is None and not codex_available):
        return False
    if not codex_available:
        raise RuntimeError(
            "--codex-lm requires dspy-codex-lm in the uv run environment. "
            "Use: uv run --project examples/terminal_bench "
            "--with-editable /Users/gabriel/Workspace/dspy-codex-lm "
            "rlm-gepa optimize --codex-lm ..."
        )

    from dspy_codex_lm.cli import install_monkeypatch

    install_monkeypatch(exclude=args.codex_lm_exclude)
    os.environ.setdefault("OPENAI_API_KEY", "codex-lm")
    return True
