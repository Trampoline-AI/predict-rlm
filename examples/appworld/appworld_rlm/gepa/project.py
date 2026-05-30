from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

from predict_rlm import Skill
from predict_rlm.trace import RunTrace, extract_trace_from_exc
from rlm_gepa import EvaluationContext, RLMGepaExampleResult, RLMGepaProject

from ..agent.service import AppWorldRLM
from ..agent.skills import get_appworld_skill
from ..bench.dataset import AppWorldExample, load_dataset, split_train_validation
from .config import APPWORLD_SPEC, AppWorldGepaConfig, default_config

COMPONENT_SKILL = "skill_instructions"


def score_runner_result(payload: str | dict[str, object]) -> tuple[float, str]:
    data = json.loads(payload) if isinstance(payload, str) else payload
    score = _appworld_soft_score(data)
    success = bool(data.get("success", score >= 1.0))
    feedback = str(data.get("feedback") or "")
    stderr = str(data.get("stderr") or "")
    if success and score >= 1.0:
        return 1.0, feedback or "AppWorld evaluator reported success"
    parts = [f"AppWorld score={score:.3f}"]
    if feedback:
        parts.append(feedback)
    if stderr:
        parts.append(stderr)
    return score, "\n".join(parts)


def _appworld_soft_score(data: Mapping[str, Any]) -> float:
    result = data.get("result")
    counted_score = _appworld_counted_score(data)
    if counted_score is None and isinstance(result, Mapping):
        counted_score = _appworld_counted_score(result)
    if counted_score is not None:
        return counted_score
    return max(0.0, min(1.0, float(data.get("score", 0.0) or 0.0)))


def _appworld_counted_score(data: Mapping[Any, Any]) -> float | None:
    num_tests = data.get("num_tests")
    passes = data.get("passes")
    if not isinstance(num_tests, int | float) or num_tests <= 0 or not isinstance(passes, list):
        return None
    return max(0.0, min(1.0, len(passes) / float(num_tests)))


class AppWorldGepaProject(RLMGepaProject):
    project_name = "appworld-rlm"
    components = (COMPONENT_SKILL,)
    agent_spec = APPWORLD_SPEC

    def __init__(self, config: AppWorldGepaConfig):
        self.config = config
        self._split: tuple[list[AppWorldExample], list[AppWorldExample]] | None = None

    def seed_candidate(self) -> dict[str, str]:
        return {COMPONENT_SKILL: get_appworld_skill().instructions}

    def load_trainset(self) -> list[AppWorldExample]:
        train, _validation = self._load_split()
        return train

    def load_valset(self) -> list[AppWorldExample]:
        _train, validation = self._load_split()
        return validation

    async def evaluate_example(
        self,
        candidate: dict[str, str],
        example: AppWorldExample,
        context: EvaluationContext,
    ) -> RLMGepaExampleResult:
        skill = Skill(name="appworld", instructions=candidate[COMPONENT_SKILL])
        trace: RunTrace | None = None
        agent = AppWorldRLM(
            lm=context.lm,
            sub_lm=context.sub_lm,
            max_iterations=context.max_iterations,
            verbose=context.verbose_rlm,
            skill=skill,
            data_root=self.config.data_root,
        )
        try:
            result = await asyncio.wait_for(
                agent.acall(
                    task_id=example.task_id,
                    instruction=example.instruction or example.task_id,
                    supervisor_name=example.supervisor_name,
                    supervisor_email=example.supervisor_email,
                    supervisor_phone_number=example.supervisor_phone_number,
                ),
                timeout=context.task_timeout,
            )
            trace = getattr(result, "trace", None)
            evaluation_payload = agent.appworld_client.evaluate_appworld_task(example.task_id)
            score, feedback = score_runner_result(evaluation_payload)
            return RLMGepaExampleResult(
                score=score,
                feedback=feedback,
                traces=[trace] if trace is not None else [],
                rlm_inputs={"task_id": example.task_id, "dataset": example.dataset},
                example_id=example.task_id,
                error=None if trace is not None else "no RunTrace captured",
            )
        except asyncio.TimeoutError as exc:
            return self._error_result(example, f"RLM timeout at {context.task_timeout}s", exc)
        except Exception as exc:
            return self._error_result(example, f"RLM {type(exc).__name__}: {exc}", exc)
        finally:
            try:
                agent.appworld_client.close_appworld_task(example.task_id)
            except Exception:
                pass
            close_client = getattr(agent.appworld_client, "close", None)
            if callable(close_client):
                try:
                    close_client()
                except Exception:
                    pass

    def component_focus(self, component_name: str) -> str:
        if component_name == COMPONENT_SKILL:
            return "Improve AppWorld execution strategy, API probing, retry discipline, supervisor completion, and final-state verification without evaluator access during solving."
        return ""

    def minibatch_group_id(self, example: AppWorldExample) -> str | None:
        return example.group_id

    def _load_split(self) -> tuple[list[AppWorldExample], list[AppWorldExample]]:
        if self._split is not None:
            return self._split
        train_pool = load_dataset(self.config.train_dataset, self.config.data_root)
        if self.config.val_dataset is None:
            train, validation = split_train_validation(
                train_pool,
                val_ratio=self.config.val_ratio,
                seed=self.config.seed,
            )
        else:
            train = train_pool
            validation = load_dataset(self.config.val_dataset, self.config.data_root)
        if self.config.val_limit is not None:
            validation = validation[: self.config.val_limit]
        self._split = (train, validation)
        return self._split

    def _error_result(
        self,
        example: AppWorldExample,
        feedback: str,
        exc: BaseException,
    ) -> RLMGepaExampleResult:
        trace = extract_trace_from_exc(exc)
        return RLMGepaExampleResult(
            score=0.0,
            feedback=feedback,
            traces=[trace] if trace is not None else [],
            rlm_inputs={"task_id": example.task_id, "dataset": example.dataset},
            example_id=example.task_id,
            error=None if trace is not None else feedback,
        )


def build_project(config: AppWorldGepaConfig | None = None) -> RLMGepaProject:
    return AppWorldGepaProject(config or default_config())
