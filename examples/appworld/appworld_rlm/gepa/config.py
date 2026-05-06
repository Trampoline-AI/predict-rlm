from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlm_gepa import AgentSpec, OptimizeConfig, agent_spec_from_rlm

from ..agent.service import AppWorldRLM


@dataclass
class AppWorldGepaConfig(OptimizeConfig):
    train_dataset: str = "train"
    data_root: Path = Path("data")
    val_ratio: float = 0.20
    val_limit: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "train_dataset": self.train_dataset,
                "data_root": str(self.data_root),
                "val_ratio": self.val_ratio,
                "val_limit": self.val_limit,
            }
        )
        return payload


def default_config() -> AppWorldGepaConfig:
    return AppWorldGepaConfig(
        executor_lm="openai/gpt-5.4-mini",
        executor_sub_lm="openai/gpt-5.4-mini",
        proposer_lm="anthropic/claude-sonnet-4-6",
        proposer_sub_lm="openai/gpt-5.4-mini",
        seed=13,
        minibatch_size=5,
        concurrency=2,
    )


def build_appworld_spec() -> AgentSpec:
    rlm = AppWorldRLM(max_iterations=1).build_predictor()
    return agent_spec_from_rlm(
        rlm,
        agent_type=(
            "an AppWorld task-solving RLM that writes Python against clean, "
            "direct AppWorld API tools in an isolated app environment"
        ),
        use_cases=[
            "realistic multi-app personal-assistant workflows with persistent state",
            "API-grounded planning tasks requiring exact entity IDs and side effects",
            "long-horizon tool-use benchmarks where task completion is scored by a harness-side evaluator",
        ],
        runtime_grounding_examples={
            "AppWorld split hygiene": [
                "official train is split into train/validation for optimization",
                "test_normal and test_challenge remain held out for reporting",
            ],
            "runner contract": [
                "direct AppWorld tools keep persistent task state and return JSON strings",
                "the RLM must complete tasks through the supervisor API, not evaluator feedback",
                "the AppWorld evaluator is called only by the GEPA/eval harness after the attempt finishes",
            ],
            "failure modes": [
                "wrong state mutation",
                "missing or malformed supervisor completion answer",
                "invalid API call",
                "timeout",
            ],
            "environment facts": [
                "AppWorld runs in a separate Python environment because of Pydantic v1",
            ],
        },
        scoring_description=(
            "Score is the harness-side normalized AppWorld evaluator score for a completed task. "
            "Feedback names failed answer checks, state assertions, changed-record checks, "
            "invalid calls, exceptions, timeouts, and missing completion. The RLM does not see "
            "this feedback during the same task attempt; GEPA uses it afterward to improve the "
            "skill instructions across training examples."
        ),
        domain_conventions_note=(
            "Only AppWorld strategies grounded in train/validation execution traces should transfer; "
            "never encode task ids, split membership, private data, or reference answers in candidate instructions."
        ),
        counterfactual_axis_name="task types",
    )


APPWORLD_SPEC = build_appworld_spec()
