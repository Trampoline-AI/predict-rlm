from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rlm_gepa.reporting.cost import aggregate_costs_from_log
from rlm_gepa.runtime.adapter import RLMGepaAdapter
from rlm_gepa.runtime.lm_config import build_lm
from rlm_gepa.runtime.utils import atomic_write_json

from ..agent.skills import appworld_skill
from ..gepa.config import AppWorldGepaConfig
from ..gepa.project import COMPONENT_SKILL, AppWorldGepaProject
from .config import EvalConfig
from .dataset import AppWorldExample, load_dataset, load_train_validation

HARD_PASS_THRESHOLD = 0.999


@dataclass(frozen=True)
class EvalExampleResult:
    task_id: str
    score: float
    feedback: str


@dataclass(frozen=True)
class EvalReport:
    dataset: str
    count: int
    mean_score: float
    results: tuple[EvalExampleResult, ...]
    run_dir: str
    duration_seconds: float
    costs: tuple[Any, ...]

    @property
    def total_cost_usd(self) -> float:
        return sum(float(getattr(cost, "cost_usd", 0.0)) for cost in self.costs)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["costs"] = [cost.to_dict() for cost in self.costs]
        payload["total_cost_usd"] = self.total_cost_usd
        return payload


def load_eval_dataset(config: EvalConfig) -> list[AppWorldExample]:
    if config.dataset == "validation":
        _train, examples = load_train_validation(config.data_root, config.val_ratio, config.seed)
    else:
        examples = load_dataset(config.dataset, config.data_root)
    if config.task_ids:
        wanted = set(config.task_ids)
        examples = [example for example in examples if example.task_id in wanted]
    if config.limit is not None:
        examples = examples[: config.limit]
    return examples


def extract_candidate(run_dir: str | Path, cand_idx: int | None = None) -> tuple[dict[str, str], int]:
    path = Path(run_dir) / "all_candidates.json"
    candidates = json.loads(path.read_text())
    index = cand_idx if cand_idx is not None else len(candidates) - 1
    candidate = candidates[index]
    if "candidate" in candidate:
        candidate = candidate["candidate"]
    return {COMPONENT_SKILL: str(candidate[COMPONENT_SKILL])}, index


async def run_evaluation(config: EvalConfig) -> EvalReport:
    examples = load_eval_dataset(config)
    candidate = {COMPONENT_SKILL: appworld_skill.instructions}
    if config.run_dir is not None:
        candidate, _idx = extract_candidate(config.run_dir, config.cand_idx)

    run_dir, run_id = _prepare_eval_run_dir(config, len(examples))
    project = AppWorldGepaProject(
        AppWorldGepaConfig(
            data_root=config.data_root,
            val_ratio=config.val_ratio,
            seed=config.seed,
        )
    )
    adapter = RLMGepaAdapter(
        project=project,
        lm=build_lm(config.lm, reasoning_effort=config.reasoning_effort),
        sub_lm=build_lm(config.sub_lm, reasoning_effort=None),
        max_iterations=config.max_iterations,
        concurrency=config.concurrency,
        task_timeout=config.task_timeout,
        output_dir=run_dir,
        run_id=run_id,
        verbose_rlm=config.verbose_rlm,
        display_progress_bar=True,
        valset_size=len(examples),
    )

    started = time.time()
    with adapter.progress_label("EVAL 0000"):
        await adapter.aevaluate(examples, candidate, capture_traces=True, kind="eval")
    duration_seconds = time.time() - started

    results = tuple(_read_eval_results(run_dir))
    mean_score = sum(result.score for result in results) / len(results) if results else 0.0
    costs = tuple(
        aggregate_costs_from_log(
            run_dir / "cost_log.jsonl",
            role_order=["executor", "sub_lm"],
        )
    )
    report = EvalReport(
        dataset=config.dataset,
        count=len(results),
        mean_score=mean_score,
        results=results,
        run_dir=str(run_dir),
        duration_seconds=duration_seconds,
        costs=costs,
    )
    atomic_write_json(run_dir / "eval.json", _stats_payload(config, report))
    return report


def run_evaluation_sync(config: EvalConfig) -> EvalReport:
    return asyncio.run(run_evaluation(config))



def _prepare_eval_run_dir(config: EvalConfig, example_count: int) -> tuple[Path, str]:
    run_dir = config.output_dir or _default_eval_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "task_traces").mkdir()
    (run_dir / "proposer_traces").mkdir()
    run_id = f"eval_{uuid.uuid4().hex[:8]}"
    atomic_write_json(
        run_dir / "run_metadata.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "project_name": "appworld-rlm",
            "run_kind": "eval",
            "created_at": datetime.now().isoformat(),
            "example_count": example_count,
            "resolved_config": _config_payload(config),
        },
    )
    return run_dir, run_id


def _default_eval_run_dir(config: EvalConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("runs") / f"eval-{config.dataset}-{timestamp}-{uuid.uuid4().hex[:6]}"


def _read_eval_results(run_dir: Path) -> list[EvalExampleResult]:
    trace_files = sorted((run_dir / "task_traces").glob("*_eval.jsonl"))
    if len(trace_files) != 1:
        raise RuntimeError(f"expected one eval task trace file, found {len(trace_files)} in {run_dir}")
    results = []
    for line in trace_files[0].read_text().splitlines():
        record = json.loads(line)
        results.append(
            EvalExampleResult(
                task_id=str(record["example_id"]),
                score=float(record["score"]),
                feedback=str(record.get("feedback") or record.get("error") or ""),
            )
        )
    return results


def _stats_payload(config: EvalConfig, report: EvalReport) -> dict[str, Any]:
    per_task = []
    grouped: dict[str, list[EvalExampleResult]] = defaultdict(list)
    for result in report.results:
        grouped[_task_group(result.task_id)].append(result)
    for task_id, cases in grouped.items():
        scores = [case.score for case in cases]
        per_task.append(
            {
                "task_id": task_id,
                "soft": sum(scores) / len(scores) if scores else 0.0,
                "hard": 1.0 if cases and all(case.score >= HARD_PASS_THRESHOLD for case in cases) else 0.0,
                "cases": [
                    {
                        "idx": idx,
                        "score": case.score,
                        "passed": case.score >= HARD_PASS_THRESHOLD,
                        "message": case.feedback,
                    }
                    for idx, case in enumerate(cases)
                ],
            }
        )
    total_tasks = len(report.results)
    tasks_all_passing = sum(
        1 for result in report.results if result.score >= HARD_PASS_THRESHOLD
    )
    total_scenarios = len(per_task)
    scenarios_all_passing = sum(1 for task in per_task if float(task["hard"]) >= 1.0)
    return {
        "config": _config_payload(config),
        "dataset": report.dataset,
        "run_dir": report.run_dir,
        "total_tasks": total_tasks,
        "soft_restriction_avg": report.mean_score,
        "hard_restriction_avg": tasks_all_passing / total_tasks if total_tasks else 0.0,
        "task_goal_completion": tasks_all_passing / total_tasks if total_tasks else 0.0,
        "scenario_goal_completion": (
            scenarios_all_passing / total_scenarios if total_scenarios else 0.0
        ),
        "tasks_all_passing": tasks_all_passing,
        "scenarios_all_passing": scenarios_all_passing,
        "total_scenarios": total_scenarios,
        "duration_seconds": report.duration_seconds,
        "costs": [cost.to_dict() for cost in report.costs],
        "total_cost_usd": report.total_cost_usd,
        "per_task": per_task,
    }


def _config_payload(config: EvalConfig) -> dict[str, Any]:
    return {
        "lm": config.lm,
        "sub_lm": config.sub_lm,
        "reasoning_effort": config.reasoning_effort,
        "dataset": config.dataset,
        "data_root": str(config.data_root),
        "run_dir": str(config.run_dir) if config.run_dir is not None else None,
        "output_dir": str(config.output_dir) if config.output_dir is not None else None,
        "cand_idx": config.cand_idx,
        "limit": config.limit,
        "task_ids": list(config.task_ids) if config.task_ids else None,
        "concurrency": config.concurrency,
        "max_iterations": config.max_iterations,
        "task_timeout": config.task_timeout,
        "val_ratio": config.val_ratio,
        "seed": config.seed,
        "verbose_rlm": config.verbose_rlm,
    }


def _task_group(task_id: str) -> str:
    return task_id.rsplit("_", 1)[0]
