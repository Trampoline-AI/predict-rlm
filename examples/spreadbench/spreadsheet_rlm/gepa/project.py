from __future__ import annotations

import asyncio
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any

from predict_rlm import File, PredictRLM
from predict_rlm.trace import RunTrace, extract_trace_from_exc
from rlm_gepa import EvaluationContext, RLMGepaExampleResult, RLMGepaProject

from ..agent.signature import ManipulateSpreadsheet
from ..agent.skills import libreoffice_spreadsheet_skill
from ..bench.dataset import SpreadsheetTask, load_dataset
from ..bench.evaluation import _build_instruction, parse_answer_position
from ..bench.scoring import score_workbooks
from ..tools.recalculate import recalculate
from .config import SPREADSHEET_SPEC, SpreadsheetGepaConfig, default_config

COMPONENT_SKILL = "skill_instructions"


class SpreadsheetGepaProject(RLMGepaProject):
    project_name = "spreadsheet-rlm"
    components = (COMPONENT_SKILL,)
    agent_spec = SPREADSHEET_SPEC

    def __init__(self, config: SpreadsheetGepaConfig):
        self.config = config
        self._split: tuple[list[SpreadsheetTask], list[SpreadsheetTask]] | None = None

    def seed_candidate(self) -> dict[str, str]:
        return {COMPONENT_SKILL: libreoffice_spreadsheet_skill.instructions}

    def load_trainset(self) -> list[SpreadsheetTask]:
        train, _val = self._load_split()
        return train

    def load_valset(self) -> list[SpreadsheetTask]:
        _train, val = self._load_split()
        return val

    async def evaluate_example(
        self,
        candidate: dict[str, str],
        example: SpreadsheetTask,
        context: EvaluationContext,
    ) -> RLMGepaExampleResult:
        skill = libreoffice_spreadsheet_skill.model_copy(
            update={"instructions": candidate[COMPONENT_SKILL]}
        )
        case_scores: list[float] = []
        feedback_lines: list[str] = []
        traces: list[RunTrace] = []

        with tempfile.TemporaryDirectory(prefix="rlm_gepa_spreadsheet_") as tmp_dir:
            for case_idx, input_path, answer_path in example.test_cases:
                score, feedback, trace = await self._run_case(
                    example,
                    case_idx,
                    input_path,
                    answer_path,
                    skill,
                    context,
                    Path(tmp_dir),
                )
                case_scores.append(score)
                feedback_lines.append(feedback)
                if trace is not None:
                    traces.append(trace)

        score = sum(case_scores) / len(case_scores) if case_scores else 0.0
        return RLMGepaExampleResult(
            score=score,
            feedback="\n".join(feedback_lines),
            traces=traces,
            rlm_inputs={
                "task_id": example.task_id,
                "instruction_type": example.instruction_type,
                "instruction": example.instruction,
                "cases": len(example.test_cases),
            },
            example_id=example.task_id,
            error=None if traces else "no RunTrace captured for any case",
        )

    async def _run_case(
        self,
        task: SpreadsheetTask,
        case_idx: int,
        input_path: str,
        answer_path: str | None,
        skill: Any,
        context: EvaluationContext,
        tmp_dir: Path,
    ) -> tuple[float, str, RunTrace | None]:
        output_path = tmp_dir / f"{case_idx}_{task.task_id}_output.xlsx"
        trace: RunTrace | None = None
        try:
            answer_sheet, answer_range = await asyncio.to_thread(
                parse_answer_position,
                task.answer_position,
                input_path,
            )
            formatted_instruction = _build_instruction(
                task.instruction,
                answer_range,
                answer_sheet,
                task.instruction_type,
            )
            predictor = PredictRLM(
                ManipulateSpreadsheet,
                lm=context.lm,
                sub_lm=context.sub_lm,
                skills=[skill],
                max_iterations=context.max_iterations,
                verbose=context.verbose_rlm,
                debug=False,
            )
            result = await asyncio.wait_for(
                predictor.acall(
                    input_spreadsheet=File(path=input_path),
                    instruction=formatted_instruction,
                ),
                timeout=context.task_timeout,
            )
            trace = getattr(result, "trace", None)
            if not (
                result
                and result.output_spreadsheet
                and result.output_spreadsheet.path
                and os.path.exists(result.output_spreadsheet.path)
            ):
                return 0.0, f"case {case_idx}: RLM returned no output workbook", trace
            shutil.copy2(result.output_spreadsheet.path, output_path)
        except asyncio.TimeoutError as exc:
            return (
                0.0,
                f"case {case_idx}: RLM timeout at {context.task_timeout}s",
                extract_trace_from_exc(exc),
            )
        except Exception as exc:
            return (
                0.0,
                f"case {case_idx}: RLM {type(exc).__name__}: {exc}",
                extract_trace_from_exc(exc),
            )

        await asyncio.to_thread(_best_effort_recalculate, output_path)
        if answer_path is None:
            return 0.0, f"case {case_idx}: answer file not found", trace
        try:
            score, message = await asyncio.to_thread(
                score_workbooks,
                answer_path,
                str(output_path),
                task.instruction_type,
                task.answer_position,
            )
            status = "PASS" if score == 1.0 else "FAIL"
            return score, f"case {case_idx}: score={score:.3f} {status}\n{message}", trace
        except Exception as exc:
            return 0.0, f"case {case_idx}: comparison error: {exc}", trace

    def _load_split(self) -> tuple[list[SpreadsheetTask], list[SpreadsheetTask]]:
        if self._split is not None:
            return self._split
        tasks = load_dataset(
            self.config.train_dataset,
            max_cases_per_task=self.config.cases_per_task,
        )
        train, val = split_train_val(tasks, self.config.val_ratio, self.config.seed)
        if self.config.val_limit and len(val) > self.config.val_limit:
            rng = random.Random(self.config.seed)
            val = list(val)
            rng.shuffle(val)
            val = val[: self.config.val_limit]
        self._split = (train, val)
        return self._split


def build_project(config: SpreadsheetGepaConfig | None = None) -> RLMGepaProject:
    return SpreadsheetGepaProject(config or default_config())


def split_train_val(
    tasks: list[SpreadsheetTask],
    val_ratio: float,
    seed: int,
) -> tuple[list[SpreadsheetTask], list[SpreadsheetTask]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    rng = random.Random(seed)
    indices = list(range(len(tasks)))
    rng.shuffle(indices)
    val_size = int(round(len(tasks) * val_ratio))
    val_indices = set(indices[:val_size])
    train = [task for index, task in enumerate(tasks) if index not in val_indices]
    val = [task for index, task in enumerate(tasks) if index in val_indices]
    return train, val


def _best_effort_recalculate(path: Path) -> None:
    try:
        recalculate(str(path))
    except Exception:
        return
