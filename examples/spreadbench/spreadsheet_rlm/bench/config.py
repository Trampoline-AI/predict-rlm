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
    sandbox_backend: str = "jspi"
    sbx_pool_size: int | None = None
    sbx_template: str | None = None
    sbx_preinstall_packages: bool = True
    verbose_rlm: bool = False
    debug_rlm: bool = False

    def __post_init__(self) -> None:
        self.sandbox_backend = str(self.sandbox_backend).lower()
        if self.sandbox_backend not in {"jspi", "sbx"}:
            raise ValueError(
                f"sandbox_backend must be 'jspi' or 'sbx', got {self.sandbox_backend!r}"
            )
        if self.sbx_pool_size is not None and self.sbx_pool_size <= 0:
            self.sbx_pool_size = None
        if self.sbx_pool_size is not None and self.sandbox_backend != "sbx":
            raise ValueError("sbx_pool_size requires sandbox_backend='sbx'.")
