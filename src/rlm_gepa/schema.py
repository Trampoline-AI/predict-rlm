from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from predict_rlm.trace import RunTrace

SCHEMA_VERSION = 1

COUNTERFACTUAL_AXIS_SINGULAR = {
    "domains": "domain",
    "task shapes": "task shape",
    "failure modes": "failure mode",
    "task types": "task type",
    "problem classes": "problem class",
}

_PROJECT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass
class AgentSpec:
    agent_type: str
    use_cases: list[str]
    runtime_grounding_examples: dict[str, list[str]]
    tool_signatures: str
    target_signature: str
    scoring_description: str
    counterfactual_axis_name: str = "domains"
    domain_conventions_note: str = ""
    traces_file_mount: str = "/sandbox/input/traces_file/"
    paired_traces_file_mount: str = "/sandbox/input/paired_traces_file/"

    def __post_init__(self) -> None:
        if not self.agent_type.strip():
            raise ValueError("AgentSpec.agent_type must be a non-empty string")
        if len(set(self.use_cases)) < 2:
            raise ValueError("AgentSpec.use_cases must list at least 2 distinct entries")
        if len(self.runtime_grounding_examples) < 3:
            raise ValueError("AgentSpec.runtime_grounding_examples must have at least 3 groups")
        for category, surfaces in self.runtime_grounding_examples.items():
            if not category.strip() or not surfaces:
                raise ValueError("AgentSpec.runtime_grounding_examples groups must be non-empty")
        if not self.tool_signatures.strip():
            raise ValueError("AgentSpec.tool_signatures must be a non-empty string")
        if not self.target_signature.strip():
            raise ValueError("AgentSpec.target_signature must be a non-empty string")
        if not self.scoring_description.strip():
            raise ValueError("AgentSpec.scoring_description must be a non-empty string")
        if self.counterfactual_axis_name not in COUNTERFACTUAL_AXIS_SINGULAR:
            raise ValueError(
                f"AgentSpec.counterfactual_axis_name={self.counterfactual_axis_name!r} "
                "is not registered in COUNTERFACTUAL_AXIS_SINGULAR"
            )


@dataclass(frozen=True)
class LMCost:
    role: str
    model: str
    calls: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "model": self.model,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": self.cost_usd,
        }


@dataclass(frozen=True)
class CostRow:
    event_id: str
    operation_id: str
    attempt_id: str
    event: str
    role: str
    model: str
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cache_hits: int = 0
    ts: str | None = None
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ts": self.ts or datetime.now().isoformat(),
            "event_id": self.event_id,
            "operation_id": self.operation_id,
            "attempt_id": self.attempt_id,
            "event": self.event,
            "role": self.role,
            "model": self.model,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "cache_hits": self.cache_hits,
        }


@dataclass
class OptimizeConfig:
    executor_lm: Any = "openai/gpt-5.4"
    executor_sub_lm: Any = "openai/gpt-5.1"
    executor_reasoning_effort: str | None = "low"
    executor_sub_lm_reasoning_effort: str | None = "none"

    proposer_lm: Any = "openai/gpt-5.4"
    proposer_reasoning_effort: str | None = "medium"
    proposer_sub_lm: Any = "openai/gpt-5.4"
    proposer_sub_lm_reasoning_effort: str | None = "medium"
    proposer_max_iterations: int = 20
    proposer_timeout: int = 600
    heartbeat_interval_seconds: float = 30.0

    max_metric_calls: int = 2000
    minibatch_size: int = 50
    concurrency: int = 30
    max_iterations: int = 50
    task_timeout: int = 300
    seed: int = 42

    candidate_selection_strategy: str = "pareto"
    component_selection_strategy: str = "round_robin"

    merge_proposer: bool = False
    max_merge_invocations: int = 10
    max_merge_attempts: int = 12
    merge_minibatch_size: int = 50
    merge_min_each: int = 3

    cache: bool = False
    verbose_rlm: bool = False
    display_progress_bar: bool = True
    run_dir: Path | None = None
    resume: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.run_dir is not None:
            payload["run_dir"] = str(self.run_dir)
        for key in (
            "executor_lm",
            "executor_sub_lm",
            "proposer_lm",
            "proposer_sub_lm",
        ):
            payload[key] = _serialize_lm(payload[key])
        return payload


@dataclass
class OptimizeReport:
    config: OptimizeConfig
    run_dir: str
    best_idx: int
    best_val_score: float
    total_candidates: int
    total_metric_calls: int
    duration_seconds: float
    best_candidate: dict[str, str]
    val_aggregate_scores: list[float]
    costs: list[LMCost] = field(default_factory=list)

    @property
    def rollout_cost_usd(self) -> float:
        return sum(
            cost.cost_usd
            for cost in self.costs
            if cost.role in {"executor", "sub_lm", "merge_trace_executor", "merge_trace_sub_lm"}
        )

    @property
    def optimization_cost_usd(self) -> float:
        return self.total_cost_usd - self.rollout_cost_usd

    @property
    def total_cost_usd(self) -> float:
        return sum(cost.cost_usd for cost in self.costs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "run_dir": self.run_dir,
            "best_idx": self.best_idx,
            "best_val_score": self.best_val_score,
            "total_candidates": self.total_candidates,
            "total_metric_calls": self.total_metric_calls,
            "duration_seconds": self.duration_seconds,
            "best_candidate": self.best_candidate,
            "val_aggregate_scores": self.val_aggregate_scores,
            "costs": [cost.to_dict() for cost in self.costs],
            "rollout_cost_usd": self.rollout_cost_usd,
            "optimization_cost_usd": self.optimization_cost_usd,
            "total_cost_usd": self.total_cost_usd,
        }


@dataclass(frozen=True)
class EvaluationContext:
    lm: Any
    sub_lm: Any
    max_iterations: int
    task_timeout: int
    output_dir: Path
    kind: str
    verbose_rlm: bool = False


@dataclass
class RLMGepaExampleResult:
    score: float
    feedback: str
    traces: list[RunTrace]
    rlm_inputs: Mapping[str, Any] = field(default_factory=dict)
    example_id: str | None = None
    objective_scores: Any | None = None
    error: str | None = None


class RLMGepaProject(ABC):
    project_name: str
    components: tuple[str, ...]
    agent_spec: AgentSpec

    @abstractmethod
    def seed_candidate(self) -> dict[str, str]: ...

    @abstractmethod
    def load_trainset(self) -> Sequence[Any]: ...

    @abstractmethod
    def load_valset(self) -> Sequence[Any]: ...

    @abstractmethod
    async def evaluate_example(
        self,
        candidate: dict[str, str],
        example: Any,
        context: EvaluationContext,
    ) -> RLMGepaExampleResult: ...

    def component_focus(self, component_name: str) -> str:
        return ""


@dataclass(frozen=True)
class ProjectValidation:
    seed_candidate: dict[str, str]
    trainset: Sequence[Any]
    valset: Sequence[Any]


def validate_project(project: RLMGepaProject) -> ProjectValidation:
    project_name = getattr(project, "project_name", "")
    if not isinstance(project_name, str) or not _PROJECT_NAME_RE.match(project_name):
        raise ValueError(
            "project_name must be a non-empty filesystem-safe string using "
            "letters, numbers, dots, dashes, or underscores"
        )

    components = getattr(project, "components", ())
    if not isinstance(components, tuple) or not components:
        raise ValueError("components must be a non-empty tuple of component names")
    if any(not isinstance(component, str) or not component.strip() for component in components):
        raise ValueError("components must contain non-empty strings")
    if len(set(components)) != len(components):
        raise ValueError("components must contain unique component names")

    agent_spec = getattr(project, "agent_spec", None)
    if not isinstance(agent_spec, AgentSpec):
        raise ValueError("agent_spec must be an rlm_gepa.AgentSpec")

    seed = project.seed_candidate()
    if set(seed) != set(components):
        raise ValueError(
            "seed_candidate() must return exactly the declared component keys "
            f"{components!r}; got {tuple(seed)!r}"
        )
    for component, text in seed.items():
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"seed component {component!r} must be a non-empty string")

    trainset = project.load_trainset()
    valset = project.load_valset()
    if not _is_non_empty_sequence(trainset):
        raise ValueError("load_trainset() must return a non-empty sequence")
    if not _is_non_empty_sequence(valset):
        raise ValueError("load_valset() must return a non-empty sequence")

    for component in components:
        focus = project.component_focus(component)
        if not isinstance(focus, str):
            raise ValueError(f"component_focus({component!r}) must return a string")

    return ProjectValidation(seed_candidate=seed, trainset=trainset, valset=valset)


def validate_example_result(result: RLMGepaExampleResult) -> None:
    if not isinstance(result, RLMGepaExampleResult):
        raise ValueError("evaluate_example(...) must return RLMGepaExampleResult")
    if not isinstance(result.score, int | float) or not math.isfinite(result.score):
        raise ValueError("RLMGepaExampleResult.score must be finite")
    if not isinstance(result.feedback, str):
        raise ValueError("RLMGepaExampleResult.feedback must be a string")
    if result.score < 1.0 and not result.feedback.strip():
        raise ValueError("RLMGepaExampleResult.feedback must be non-empty for imperfect scores")
    if not isinstance(result.traces, list):
        raise ValueError("RLMGepaExampleResult.traces must be a list")
    if not result.traces and not result.error:
        raise ValueError(
            "RLMGepaExampleResult.traces must contain at least one RunTrace "
            "unless error is populated"
        )


def _serialize_lm(value: Any) -> str:
    if isinstance(value, str):
        return value
    return str(getattr(value, "model", value))


def _is_non_empty_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes) and len(value) > 0
