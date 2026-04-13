"""Eval loop for the SpreadsheetBench pipeline.

Call :func:`run_evaluation` with an :class:`EvalConfig` to run the
SpreadsheetRLM over a dataset, recalculate formulas on every produced
output via the local formulas+LO pipeline, and score against ground
truth. Works with either the seed docstring or a GEPA-optimized prompt
extracted from a run directory.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import pickle
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import dspy
import nest_asyncio
from spreadsheet_rlm.recalculate import recalculate
from spreadsheet_rlm.signature import ManipulateSpreadsheet
from spreadsheet_rlm.skills import libreoffice_spreadsheet_skill
from tqdm import tqdm

from predict_rlm import File, PredictRLM

from .dataset import SpreadsheetTask, load_dataset
from .lm_config import SUB_LM, compute_lm_cost, get_lm_config, get_sub_lm_config
from .scoring import score_workbooks

nest_asyncio.apply()

GEPA_COMPONENT = "signature_docstring"

RLM_LOGGER_NAME = "dspy.predict.rlm"

_current_log_file: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "spreadbench_eval_log_file", default=None
)


class _TaskFileHandler(logging.Handler):
    """Routes ``dspy.predict.rlm`` records to each async task's own log file.

    Every coroutine running concurrently under :func:`run_evaluation`
    lives in its own asyncio context, so the :class:`ContextVar` set by
    :func:`_run_case` is task-local. This handler reads that var on
    every emit and appends to the file the current task owns. When the
    var is ``None`` (logs disabled, or code running outside a task
    context), the record is dropped silently.
    """

    def emit(self, record: logging.LogRecord) -> None:
        log_file = _current_log_file.get()
        if log_file is None:
            return
        try:
            with open(log_file, "a") as f:
                f.write(self.format(record) + "\n")
        except Exception:
            self.handleError(record)


def _install_log_handler() -> _TaskFileHandler:
    """Attach the per-task file handler to ``dspy.predict.rlm``, idempotently.

    Clears any previously installed ``_TaskFileHandler`` first so that
    repeated :func:`run_evaluation` calls (from a notebook, say) don't
    stack handlers and duplicate records.
    """
    rlm_logger = logging.getLogger(RLM_LOGGER_NAME)
    rlm_logger.setLevel(logging.INFO)
    for existing in list(rlm_logger.handlers):
        if isinstance(existing, _TaskFileHandler):
            rlm_logger.removeHandler(existing)
    handler = _TaskFileHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    rlm_logger.addHandler(handler)
    return handler


@dataclass
class EvalConfig:
    """Knobs for a single eval run."""

    lm: str = "openai/gpt-5.4"
    sub_lm: str = SUB_LM
    reasoning_effort: str | None = "low"
    dataset: str = "testset"
    run_dir: str | None = None
    limit: int | None = None
    concurrency: int = 30
    max_iterations: int = 50
    task_timeout: int = 300
    cache: bool = True
    log_dir: Path | None = None


@dataclass
class LMCost:
    """Aggregate cost + token usage for a single ``dspy.LM``."""

    role: str  # "main", "sub", "proposer", ...
    model: str
    calls: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "model": self.model,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": self.cost_usd,
        }


def summarize_lm_cost(role: str, lm: dspy.LM) -> LMCost:
    """Aggregate cost + tokens from a ``dspy.LM``'s call history.

    DSPy's ``LM.history`` records one dict per call (via LiteLLM), with
    ``cost`` and ``usage.prompt_tokens`` / ``usage.completion_tokens``.
    Models that LiteLLM doesn't have pricing for (e.g. mercury-2 via
    Inception Labs) report ``cost = None`` per call; for those models we
    fall back to :func:`compute_lm_cost`, which applies the per-model
    USD/MTok overrides declared in ``lm_config``.
    """
    history = getattr(lm, "history", []) or []
    calls = len(history)
    prompt_tokens = 0
    completion_tokens = 0
    cost_from_litellm = 0.0
    for entry in history:
        usage = entry.get("usage") or {}
        prompt_tokens += int(usage.get("prompt_tokens") or 0)
        completion_tokens += int(usage.get("completion_tokens") or 0)
        cost_from_litellm += float(entry.get("cost") or 0.0)

    model = getattr(lm, "model", "unknown")
    override_cost = compute_lm_cost(model, prompt_tokens, completion_tokens)
    cost = override_cost if override_cost is not None else cost_from_litellm

    return LMCost(
        role=role,
        model=model,
        calls=calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost,
    )


@dataclass
class CaseResult:
    idx: int
    score: float
    passed: bool
    message: str
    recalc_source: str | None = None
    log_file: str | None = None


@dataclass
class TaskResult:
    task_id: str
    soft: float  # fraction of test cases that passed
    hard: int  # 1 iff every test case passed
    cases: list[CaseResult] = field(default_factory=list)


@dataclass
class EvalReport:
    config: EvalConfig
    prompt_source: str
    prompt_length: int
    total_tasks: int
    soft_restriction_avg: float
    hard_restriction_avg: float
    tasks_all_passing: int
    duration_seconds: float
    per_task: list[TaskResult]
    costs: list[LMCost]

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.costs)

    def to_dict(self) -> dict:
        return {
            "config": {
                "lm": self.config.lm,
                "sub_lm": self.config.sub_lm,
                "reasoning_effort": self.config.reasoning_effort,
                "dataset": self.config.dataset,
                "run_dir": self.config.run_dir,
                "limit": self.config.limit,
                "concurrency": self.config.concurrency,
                "max_iterations": self.config.max_iterations,
                "task_timeout": self.config.task_timeout,
            },
            "prompt_source": self.prompt_source,
            "prompt_length": self.prompt_length,
            "total_tasks": self.total_tasks,
            "soft_restriction_avg": self.soft_restriction_avg,
            "hard_restriction_avg": self.hard_restriction_avg,
            "tasks_all_passing": self.tasks_all_passing,
            "duration_seconds": self.duration_seconds,
            "costs": [c.to_dict() for c in self.costs],
            "total_cost_usd": self.total_cost_usd,
            "per_task": [
                {
                    "task_id": t.task_id,
                    "soft": t.soft,
                    "hard": t.hard,
                    "cases": [
                        {
                            "idx": c.idx,
                            "score": c.score,
                            "passed": c.passed,
                            "message": c.message,
                            "recalc_source": c.recalc_source,
                            "log_file": c.log_file,
                        }
                        for c in t.cases
                    ],
                }
                for t in self.per_task
            ],
        }


def make_dynamic_signature(docstring: str) -> type[dspy.Signature]:
    """Return a ``ManipulateSpreadsheet`` variant with a custom docstring."""
    return ManipulateSpreadsheet.with_instructions(docstring)


def extract_best_prompt(run_dir: str | Path) -> tuple[str, int, float]:
    """Pull the best-by-mean candidate prompt from a GEPA ``run_dir``.

    Reads ``{run_dir}/gepa_state.bin`` (a pickled GEPA state dict),
    picks the candidate whose mean validation subscore is highest,
    and returns ``(prompt_text, candidate_index, mean_score)``.
    """
    state_path = Path(run_dir) / "gepa_state.bin"
    with state_path.open("rb") as f:
        state = pickle.load(f)
    subs = state["prog_candidate_val_subscores"]
    cands = state["program_candidates"]
    best_idx = max(
        range(len(cands)),
        key=lambda i: (sum(subs[i].values()) / len(subs[i])) if subs[i] else 0.0,
    )
    best_mean = sum(subs[best_idx].values()) / len(subs[best_idx])
    return cands[best_idx][GEPA_COMPONENT], best_idx, best_mean


async def _run_case(
    task: SpreadsheetTask,
    idx: int,
    input_path: str,
    answer_path: str | None,
    sig_cls: type[dspy.Signature],
    lm: dspy.LM,
    sub_lm: dspy.LM,
    sem: asyncio.Semaphore,
    tmp_dir: str,
    config: EvalConfig,
) -> CaseResult:
    output_path = os.path.join(tmp_dir, f"{idx}_{task.task_id}_output.xlsx")

    log_file: Path | None = None
    log_file_str: str | None = None
    if config.log_dir is not None:
        log_file = config.log_dir / task.task_id / f"case_{idx}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(
            f"task: {task.task_id}  case: {idx}\n"
            f"instruction_type: {task.instruction_type}\n"
            f"answer_position: {task.answer_position}\n"
            f"input:  {input_path}\n"
            f"answer: {answer_path}\n"
            f"{'=' * 60}\n"
        )
        _current_log_file.set(log_file)
        log_file_str = str(log_file)

    async with sem:
        try:
            predictor = PredictRLM(
                sig_cls,
                lm=lm,
                sub_lm=sub_lm,
                skills=[libreoffice_spreadsheet_skill],
                max_iterations=config.max_iterations,
                verbose=config.log_dir is not None,
                debug=False,
            )
            result = await asyncio.wait_for(
                predictor.acall(
                    input_spreadsheet=File(path=input_path),
                    instruction=task.instruction,
                ),
                timeout=config.task_timeout,
            )
            if not (
                result
                and result.output_spreadsheet
                and result.output_spreadsheet.path
                and os.path.exists(result.output_spreadsheet.path)
            ):
                return CaseResult(idx, 0.0, False, "No output", log_file=log_file_str)
            shutil.copy2(result.output_spreadsheet.path, output_path)
        except asyncio.TimeoutError:
            return CaseResult(
                idx, 0.0, False, f"Timeout ({config.task_timeout}s)",
                log_file=log_file_str,
            )
        except Exception as e:
            return CaseResult(
                idx, 0.0, False, f"RLM error: {e}", log_file=log_file_str,
            )

    recalc_source: str | None = None
    try:
        recalc_result = await asyncio.to_thread(recalculate, output_path)
        recalc_source = recalc_result.source
    except Exception as e:
        recalc_source = f"failed: {e}"

    if answer_path is None:
        return CaseResult(
            idx, 0.0, False, "Answer file not found",
            recalc_source=recalc_source, log_file=log_file_str,
        )

    try:
        ratio, msg = await asyncio.to_thread(
            score_workbooks,
            answer_path,
            output_path,
            task.instruction_type,
            task.answer_position,
        )
        if log_file is not None:
            with log_file.open("a") as f:
                f.write(
                    f"\n{'=' * 60}\n"
                    f"recalc_source: {recalc_source}\n"
                    f"score:         {ratio:.4f}  "
                    f"{'PASS' if ratio == 1.0 else 'FAIL'}\n"
                    f"{msg}\n"
                )
        return CaseResult(
            idx, ratio, ratio == 1.0, msg,
            recalc_source=recalc_source, log_file=log_file_str,
        )
    except Exception as e:
        return CaseResult(
            idx, 0.0, False, f"Comparison error: {e}",
            recalc_source=recalc_source, log_file=log_file_str,
        )


async def _run_tasks_async(
    tasks: list[SpreadsheetTask],
    sig_cls: type[dspy.Signature],
    lm: dspy.LM,
    sub_lm: dspy.LM,
    config: EvalConfig,
) -> list[TaskResult]:
    tmp_dir = tempfile.mkdtemp(prefix="spreadbench_eval_")
    sem = asyncio.Semaphore(config.concurrency)
    pbar = tqdm(total=len(tasks), desc="Evaluating", unit="task")

    async def _process(task: SpreadsheetTask) -> TaskResult:
        case_coros = [
            _run_case(
                task, idx, input_path, answer_path,
                sig_cls, lm, sub_lm, sem, tmp_dir, config,
            )
            for idx, input_path, answer_path in task.test_cases
        ]
        cases = await asyncio.gather(*case_coros)
        soft = (
            sum(c.score for c in cases) / len(cases) if cases else 0.0
        )
        hard = 1 if cases and all(c.passed for c in cases) else 0
        pbar.set_postfix_str(f"{task.task_id}={soft:.0%}")
        pbar.update(1)
        return TaskResult(task_id=task.task_id, soft=soft, hard=hard, cases=list(cases))

    try:
        results = await asyncio.gather(*(_process(t) for t in tasks))
    finally:
        pbar.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return list(results)


def run_evaluation(config: EvalConfig) -> EvalReport:
    """Run the eval loop described by *config* and return a report.

    Resolves the prompt (seed docstring by default, GEPA best when
    ``config.run_dir`` is set), builds the main/sub LMs, loads the
    dataset, runs every task's cases concurrently, recalculates each
    produced output through the formulas+LO pipeline, and scores
    against ground truth.
    """
    if config.run_dir:
        prompt, best_idx, best_mean = extract_best_prompt(config.run_dir)
        prompt_source = f"{Path(config.run_dir).name}#cand{best_idx}"
        print(
            f"Loaded GEPA best prompt: {prompt_source} "
            f"(val_mean={best_mean:.4f}, {len(prompt)} chars)"
        )
    else:
        prompt = ManipulateSpreadsheet.__doc__ or ""
        prompt_source = "seed"
        print(f"Using seed prompt ({len(prompt)} chars)")

    sig_cls = make_dynamic_signature(prompt)

    lm_cfg = get_lm_config(config.lm, config.reasoning_effort)
    sub_lm_cfg = get_sub_lm_config(config.sub_lm)
    lm = dspy.LM(**lm_cfg, cache=config.cache)
    sub_lm = dspy.LM(**sub_lm_cfg, cache=config.cache)

    effort_tag = (
        f" (reasoning_effort={config.reasoning_effort})"
        if config.reasoning_effort
        else ""
    )
    print(f"Main LM: {config.lm}{effort_tag}")
    print(f"Sub LM:  {config.sub_lm}")

    tasks = load_dataset(config.dataset, max_cases_per_task=0)
    if config.limit:
        tasks = tasks[: config.limit]
    print(f"Dataset: {config.dataset} ({len(tasks)} tasks)")

    if config.log_dir is not None:
        config.log_dir.mkdir(parents=True, exist_ok=True)
        _install_log_handler()
        print(f"Per-case RLM logs: {config.log_dir}/<task_id>/case_<idx>.log")

    print(
        f"Running {len(tasks)} tasks with concurrency={config.concurrency}, "
        f"max_iterations={config.max_iterations}, "
        f"task_timeout={config.task_timeout}s..."
    )
    t0 = time.time()
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(
        _run_tasks_async(tasks, sig_cls, lm, sub_lm, config)
    )
    elapsed = time.time() - t0

    total = len(results)
    soft_avg = sum(r.soft for r in results) / total if total else 0.0
    hard_avg = sum(r.hard for r in results) / total if total else 0.0
    hard_count = sum(r.hard for r in results)

    costs = [
        summarize_lm_cost("main", lm),
        summarize_lm_cost("sub", sub_lm),
    ]

    return EvalReport(
        config=config,
        prompt_source=prompt_source,
        prompt_length=len(prompt),
        total_tasks=total,
        soft_restriction_avg=soft_avg,
        hard_restriction_avg=hard_avg,
        tasks_all_passing=hard_count,
        duration_seconds=elapsed,
        per_task=results,
        costs=costs,
    )
