"""Tests for CtxStr prompt-injected string inputs."""

import asyncio
from unittest.mock import MagicMock

import dspy
import pytest

import predict_rlm
import predict_rlm.in_context as in_context_module
from predict_rlm import CtxStr, PredictRLM, Skill
from predict_rlm.in_context import _build_in_context_instructions
from predict_rlm.runtime import InputAdapter, PreparedInput, use_run_context


class InContextSignature(dspy.Signature):
    """Answer the query using the provided criteria."""

    criteria: CtxStr = dspy.InputField(desc="Full rubric to apply")
    query: str = dspy.InputField(desc="Question to answer")
    answer: str = dspy.OutputField(desc="Answer to the query")


class PrefixStringInputAdapter(InputAdapter[str]):
    name = "prefixed_string"
    value_type = str

    async def prepare(self, field, value, ctx):
        return PreparedInput(model_value=f"prepared:{value}")


async def _prepare_run(
    rlm: PredictRLM,
    input_values: dict[str, object],
    *,
    file_instructions: str = "",
):
    ctx = rlm._new_run_context(input_values)
    await rlm._prepare_runtime_inputs(ctx, input_values)
    file_plan = {"instructions": file_instructions} if file_instructions else None
    rlm._configure_run_predictors(ctx, file_plan)
    return ctx


def test_in_context_is_pydantic_string_schema():
    field = InContextSignature.input_fields["criteria"]

    assert field.annotation is CtxStr
    assert InContextSignature.model_json_schema()["properties"]["criteria"]["type"] == "string"


def test_in_context_public_surface_is_only_ctx_str():
    assert in_context_module.__all__ == ["CtxStr"]
    assert "CtxStr" in predict_rlm.__all__
    assert "PromptContributor" not in predict_rlm.__all__
    assert "discover_runtime_modules" not in predict_rlm.__all__


def test_build_in_context_instructions_includes_only_marked_fields():
    criteria = "Use every cited fact.\nPrefer concise answers."

    instructions = _build_in_context_instructions(
        InContextSignature,
        {
            "criteria": criteria,
            "query": "This ordinary string input should not be injected.",
        },
    )

    assert instructions.startswith("## In-Context Inputs")
    assert instructions.count("## In-Context Inputs") == 1
    assert "### `criteria`" in instructions
    assert "Full rubric to apply" not in instructions
    assert criteria in instructions
    assert '<BEGIN_IN_CONTEXT_INPUT name="criteria">' in instructions
    assert '<END_IN_CONTEXT_INPUT name="criteria">' in instructions
    assert "This ordinary string input should not be injected." not in instructions


@pytest.mark.asyncio
async def test_in_context_delimiters_avoid_prepared_value_collisions():
    nominal_closing_marker = '<END_IN_CONTEXT_INPUT name="criteria">'
    raw_value = f"Keep this exact marker:\n{nominal_closing_marker}\nwithout changing it."
    prepared_value = f"prepared:{raw_value}"
    rlm = PredictRLM(
        InContextSignature,
        sub_lm=MagicMock(),
        adapters=[PrefixStringInputAdapter()],
        max_iterations=1,
    )

    ctx = await _prepare_run(
        rlm,
        {"criteria": raw_value, "query": "What matters?"},
    )
    instructions = str(ctx.state["generate_action"].signature.instructions)

    assert ctx.input_bindings["criteria"].prepared.model_value == prepared_value
    assert prepared_value in instructions
    assert instructions.count(nominal_closing_marker) == 1
    assert not instructions.rstrip().endswith(nominal_closing_marker)


@pytest.mark.asyncio
async def test_runtime_in_context_input_uses_run_local_predictors_and_repl_value():
    rlm = PredictRLM(InContextSignature, sub_lm=MagicMock(), max_iterations=1)
    original_action = rlm.generate_action
    original_extract = rlm.extract
    criteria = "Always mention the controlling rule."

    ctx = await _prepare_run(
        rlm,
        {"criteria": criteria, "query": "What matters?"},
    )

    action = str(ctx.state["generate_action"].signature.instructions)
    extract = str(ctx.state["extract"].signature.instructions)
    assert ctx.input_bindings["criteria"].prepared.model_value == criteria
    assert action.count(criteria) == 1
    assert extract.count(criteria) == 1
    assert action.rstrip().endswith('<END_IN_CONTEXT_INPUT name="criteria">')
    assert extract.rstrip().endswith('<END_IN_CONTEXT_INPUT name="criteria">')
    assert rlm.generate_action is original_action
    assert rlm.extract is original_extract


@pytest.mark.asyncio
async def test_runtime_in_context_instructions_follow_files_and_skills():
    skill = Skill(name="domain", instructions="Skill block")
    rlm = PredictRLM(
        InContextSignature,
        sub_lm=MagicMock(),
        max_iterations=1,
        skills=[skill],
    )

    ctx = await _prepare_run(
        rlm,
        {"criteria": "Criteria block", "query": "What matters?"},
        file_instructions="File block",
    )
    action = str(ctx.state["generate_action"].signature.instructions)
    extract = str(ctx.state["extract"].signature.instructions)

    assert action.index("File block") < action.index("Skill block")
    assert action.index("Skill block") < action.index("Criteria block")
    assert extract.index("File block") < extract.index("Skill block")
    assert extract.index("Skill block") < extract.index("Criteria block")


@pytest.mark.asyncio
async def test_in_context_preserves_invocation_signature_builder_override():
    class TrackingPredictRLM(PredictRLM):
        def _build_signatures_with_files(self, file_instructions):
            self.file_builder_calls = getattr(self, "file_builder_calls", 0) + 1
            action, extract = super()._build_signatures_with_files(file_instructions)
            action.builder_marker = "action"
            extract.builder_marker = "extract"
            return action, extract

    rlm = TrackingPredictRLM(
        InContextSignature,
        sub_lm=MagicMock(),
        max_iterations=1,
    )

    ctx = await _prepare_run(
        rlm,
        {"criteria": "Criteria block", "query": "What matters?"},
        file_instructions="File block",
    )

    assert rlm.file_builder_calls == 1
    assert ctx.state["generate_action"].builder_marker == "action"
    assert ctx.state["extract"].builder_marker == "extract"


@pytest.mark.asyncio
async def test_concurrent_in_context_runs_do_not_cross_contaminate_predictors():
    rlm = PredictRLM(InContextSignature, sub_lm=MagicMock(), max_iterations=1)

    first, second = await asyncio.gather(
        _prepare_run(rlm, {"criteria": "FIRST-RULE", "query": "one"}),
        _prepare_run(rlm, {"criteria": "SECOND-RULE", "query": "two"}),
    )

    first_action = str(first.state["generate_action"].signature.instructions)
    second_action = str(second.state["generate_action"].signature.instructions)
    assert "FIRST-RULE" in first_action
    assert "SECOND-RULE" not in first_action
    assert "SECOND-RULE" in second_action
    assert "FIRST-RULE" not in second_action
    assert first.state["generate_action"] is not second.state["generate_action"]


@pytest.mark.asyncio
async def test_in_context_rejects_non_string_runtime_value():
    rlm = PredictRLM(InContextSignature, sub_lm=MagicMock(), max_iterations=1)

    with pytest.raises(TypeError, match="expects a string"):
        await _prepare_run(rlm, {"criteria": 123, "query": "What matters?"})


@pytest.mark.asyncio
async def test_string_input_adapter_owns_in_context_prompt_value():
    rlm = PredictRLM(
        InContextSignature,
        sub_lm=MagicMock(),
        adapters=[PrefixStringInputAdapter()],
        max_iterations=1,
    )

    ctx = await _prepare_run(
        rlm,
        {"criteria": "RAW-RULE", "query": "What matters?"},
    )
    action = str(ctx.state["generate_action"].signature.instructions)

    assert ctx.input_bindings["criteria"].prepared.model_value == "prepared:RAW-RULE"
    assert "prepared:RAW-RULE" in action
    assert "\nRAW-RULE\n" not in action


@pytest.mark.asyncio
async def test_delegating_sync_iteration_uses_run_local_in_context_predictor():
    class DelegatingPredictRLM(PredictRLM):
        def _execute_iteration(self, *args, **kwargs):
            return super()._execute_iteration(*args, **kwargs)

    rlm = DelegatingPredictRLM(
        InContextSignature,
        sub_lm=MagicMock(),
        max_iterations=1,
    )
    ctx = await _prepare_run(
        rlm,
        {"criteria": "RUN-RULE", "query": "What matters?"},
    )
    run_action = MagicMock(
        return_value=dspy.Prediction(reasoning="done", code="SUBMIT(answer='done')")
    )
    ctx.state["generate_action"] = run_action
    rlm.generate_action = MagicMock(side_effect=AssertionError("used shared predictor"))
    rlm._record_action_generation_ok = MagicMock(return_value=None)
    rlm._prepare_iteration_execution = MagicMock(return_value=("code", False))
    rlm._execute_iteration_code = MagicMock(return_value="result")
    rlm._complete_iteration_execution = MagicMock(return_value="done")

    async with use_run_context(ctx):
        result = await rlm._aexecute_iteration(None, [], [], 0, {}, ["answer"])

    assert result == "done"
    run_action.assert_called_once()
    rlm.generate_action.assert_not_called()


@pytest.mark.asyncio
async def test_delegating_extract_fallback_uses_run_local_in_context_predictor():
    class DelegatingPredictRLM(PredictRLM):
        def _extract_fallback(self, *args, **kwargs):
            return super()._extract_fallback(*args, **kwargs)

    rlm = DelegatingPredictRLM(
        InContextSignature,
        sub_lm=MagicMock(),
        max_iterations=1,
    )
    ctx = await _prepare_run(
        rlm,
        {"criteria": "RUN-RULE", "query": "What matters?"},
    )
    run_extract = MagicMock(return_value=dspy.Prediction(answer="run-local"))
    ctx.state["extract"] = run_extract
    rlm.extract = MagicMock(side_effect=AssertionError("used shared predictor"))

    async with use_run_context(ctx):
        result = await rlm._aextract_fallback_for_run([], [], ["answer"])

    assert result.answer == "run-local"
    run_extract.assert_called_once()
    rlm.extract.assert_not_called()


@pytest.mark.asyncio
async def test_sync_extract_fallback_calls_cooperative_later_mro_with_run_local_copy():
    fallback_instances = []

    class CooperativeFallbackRLM(dspy.RLM):
        def _extract_fallback(self, variables, history, output_field_names):
            fallback_instances.append(self)
            return super()._extract_fallback(variables, history, output_field_names)

    class CooperativePredictRLM(PredictRLM, CooperativeFallbackRLM):
        pass

    rlm = CooperativePredictRLM(
        InContextSignature,
        sub_lm=MagicMock(),
        max_iterations=1,
    )
    ctx = await _prepare_run(
        rlm,
        {"criteria": "RUN-RULE", "query": "What matters?"},
    )
    run_extract = MagicMock(return_value=dspy.Prediction(answer="run-local"))
    ctx.state["extract"] = run_extract
    rlm.extract = MagicMock(side_effect=AssertionError("used shared predictor"))

    async with use_run_context(ctx):
        result = rlm._extract_fallback_for_run([], [], ["answer"])

    assert result.answer == "run-local"
    assert result.trajectory == []
    assert result.final_reasoning == "Extract forced final output"
    assert len(fallback_instances) == 1
    assert type(fallback_instances[0]) is CooperativePredictRLM
    assert fallback_instances[0] is not rlm
    assert fallback_instances[0].extract is run_extract
    run_extract.assert_called_once_with(variables_info=[], repl_history=[])
    rlm.extract.assert_not_called()


def test_in_context_is_input_only():
    class BadOutput(dspy.Signature):
        prompt: str = dspy.InputField()
        answer: CtxStr = dspy.OutputField()

    with pytest.raises(TypeError, match="CtxStr fields are input-only"):
        PredictRLM(BadOutput, sub_lm=MagicMock(), max_iterations=1)


def test_in_context_rejects_optional_input_annotation():
    class BadOptionalInput(dspy.Signature):
        criteria: CtxStr | None = dspy.InputField()
        answer: str = dspy.OutputField()

    with pytest.raises(TypeError, match="annotated directly"):
        PredictRLM(BadOptionalInput, sub_lm=MagicMock(), max_iterations=1)


def test_in_context_rejects_list_input_annotation():
    class BadListInput(dspy.Signature):
        criteria: list[CtxStr] = dspy.InputField()
        answer: str = dspy.OutputField()

    with pytest.raises(TypeError, match="annotated directly"):
        PredictRLM(BadListInput, sub_lm=MagicMock(), max_iterations=1)
