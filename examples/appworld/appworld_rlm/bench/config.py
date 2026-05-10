from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_EVAL_LM = "openai/gpt-5.4"
DEFAULT_EVAL_SUB_LM = "openai/gpt-5.4-mini"
DEFAULT_CONCURRENCY = 10
DEFAULT_TASK_TIMEOUT = 600
DEFAULT_SEED = 13


@dataclass
class EvalConfig:
    lm: str = DEFAULT_EVAL_LM
    sub_lm: str = DEFAULT_EVAL_SUB_LM
    reasoning_effort: str | None = "low"
    dataset: str = "validation"
    data_root: Path = Path("data")
    run_dir: Path | None = None
    output_dir: Path | None = None
    cand_idx: int | None = None
    limit: int | None = None
    task_ids: tuple[str, ...] | None = None
    concurrency: int = DEFAULT_CONCURRENCY
    max_iterations: int = 50
    task_timeout: int = DEFAULT_TASK_TIMEOUT
    val_ratio: float = 0.20
    seed: int = DEFAULT_SEED
    verbose_rlm: bool = False
