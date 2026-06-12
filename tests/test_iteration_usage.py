"""Per-iteration token usage/cost accounting on IterationStep.usage."""

from __future__ import annotations

import dspy

from predict_rlm import PredictRLM
from predict_rlm.trace import IterationStep, LMUsage, PredictCallGroup, TokenUsage


def test_iteration_step_usage_defaults_to_empty_lm_usage():
    step = IterationStep(
        iteration=1,
        reasoning="r",
        code="c",
        output="o",
        untruncated_output="o",
        duration_ms=1,
    )
    assert isinstance(step.usage, LMUsage)
    assert step.usage.main.input_tokens == 0
    assert step.usage.sub.cost == 0.0


def test_build_iteration_usage_combines_main_and_sub():
    rlm = PredictRLM("q -> a", sub_lm=dspy.LM("openai/gpt-4o", api_key="x"))
    # Main action-LM usage stashed by _record_action_generation_ok.
    rlm._last_action_lm_usage = TokenUsage(input_tokens=2000, output_tokens=100, cost=0.012)
    predict_calls = [
        PredictCallGroup(
            signature="x -> y",
            instructions=None,
            model="openai/gpt-4o",
            total_usage=TokenUsage(input_tokens=50, output_tokens=10, cost=0.001),
            calls=[],
        ),
        PredictCallGroup(
            signature="x -> z",
            instructions=None,
            model="openai/gpt-4o",
            total_usage=TokenUsage(input_tokens=30, output_tokens=5, cost=0.0005),
            calls=[],
        ),
    ]

    usage = rlm._build_iteration_usage(predict_calls)

    assert usage.main.input_tokens == 2000
    assert usage.main.output_tokens == 100
    assert usage.main.cost == 0.012
    assert usage.sub.input_tokens == 80  # 50 + 30
    assert usage.sub.output_tokens == 15
    assert round(usage.sub.cost, 4) == 0.0015
    # Stash is consumed so it cannot leak into the next iteration.
    assert rlm._last_action_lm_usage is None


def test_build_iteration_usage_without_stash_is_empty_main():
    rlm = PredictRLM("q -> a", sub_lm=dspy.LM("openai/gpt-4o", api_key="x"))
    usage = rlm._build_iteration_usage([])
    assert usage.main.input_tokens == 0
    assert usage.sub.input_tokens == 0
