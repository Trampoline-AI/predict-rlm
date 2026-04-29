from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_EVAL_LM = "openai/gpt-5.4"
DEFAULT_EVAL_SUB_LM = "openai/gpt-5.1"


@dataclass
class EvalConfig:
    """Configuration for a single SpreadsheetBench evaluation run."""

    lm: str = DEFAULT_EVAL_LM
    sub_lm: str = DEFAULT_EVAL_SUB_LM
    reasoning_effort: str | None = "low"
    thinking_budget: int | None = None
    dataset: str = "testset"
    run_dir: str | None = None
    only: str | None = None
    cand_idx: int | None = None
    limit: int | None = None
    task_ids: tuple[str, ...] | None = None
    cases_per_task: int = 0
    concurrency: int = 30
    max_iterations: int = 50
    task_timeout: int = 300
    cache: bool = False
    log_dir: Path | None = None
