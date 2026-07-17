"""Tests for CtxStr prompt-injected string inputs."""

import asyncio
from types import MethodType
from unittest.mock import MagicMock

import dspy
import pytest

import predict_rlm
import predict_rlm.in_context as in_context_module
from predict_rlm import CtxStr, CtxStrInputAdapter, PredictRLM, Skill
from predict_rlm.runtime import (
    BoundInput,
    FieldDescriptor,
    InputAdapter,
    PreparedInput,
    RunContext,
    use_run_context,
)


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
    assert in_context_module.__all__ == ["CtxStr", "CtxStrInputAdapter"]
    assert "CtxStr" in predict_rlm.__all__
    assert "CtxStrInputAdapter" in predict_rlm.__all__
    assert "PromptContributor" not in predict_rlm.__all__
    assert "discover_runtime_modules" not in predict_rlm.__all__


@pytest.mark.asyncio
async def test_ctx_str_resolves_to_builtin_adapter_ahead_of_generic_str_adapter():
    rlm = PredictRLM(
        InContextSignature,
        sub_lm=MagicMock(),
        adapters=[PrefixStringInputAdapter()],
        max_iterations=1,
    )

    ctx = await _prepare_run(rlm, {"criteria": "RULE", "query": "QUESTION"})

    assert isinstance(ctx.input_bindings["criteria"].adapter, CtxStrInputAdapter)
    assert ctx.input_bindings["criteria"].prepared.model_value == "RULE"
    assert ctx.input_bindings["query"].prepared.model_value == "prepared:QUESTION"


@pytest.mark.asyncio
async def test_ctx_str_subclass_resolves_to_builtin_ahead_of_generic_str_adapter():
    class SpecializedCtxStr(CtxStr):
        pass

    class SpecializedSignature(dspy.Signature):
        criteria: SpecializedCtxStr = dspy.InputField()
        answer: str = dspy.OutputField()

    rlm = PredictRLM(
        SpecializedSignature,
        sub_lm=MagicMock(),
        adapters=[PrefixStringInputAdapter()],
        max_iterations=1,
    )

    ctx = await _prepare_run(rlm, {"criteria": "RULE"})

    assert isinstance(ctx.input_bindings["criteria"].adapter, CtxStrInputAdapter)
    assert "RULE" in str(ctx.state["generate_action"].signature.instructions)


@pytest.mark.asyncio
async def test_multiple_ctx_str_adapter_instances_append_one_ordered_section():
    class SpecializedCtxStr(CtxStr):
        pass

    class MultipleCtxStrSignature(dspy.Signature):
        first: CtxStr = dspy.InputField()
        second: SpecializedCtxStr = dspy.InputField()
        answer: str = dspy.OutputField()

    class SpecializedCtxStrAdapter(CtxStrInputAdapter):
        name = "specialized_ctx_str"
        value_type = SpecializedCtxStr

    rlm = PredictRLM(
        MultipleCtxStrSignature,
        sub_lm=MagicMock(),
        adapters=[SpecializedCtxStrAdapter()],
        max_iterations=1,
    )

    ctx = await _prepare_run(rlm, {"first": "FIRST", "second": "SECOND"})

    for predictor_name in ("generate_action", "extract"):
        prompt = str(ctx.state[predictor_name].signature.instructions)
        assert prompt.count("## In-Context Inputs") == 1
        assert prompt.index("FIRST") < prompt.index("SECOND")


@pytest.mark.asyncio
async def test_default_ctx_str_prompt_excludes_custom_prompt_hook_fields():
    class SpecializedCtxStr(CtxStr):
        pass

    class MultipleCtxStrSignature(dspy.Signature):
        default: CtxStr = dspy.InputField()
        custom: SpecializedCtxStr = dspy.InputField()
        answer: str = dspy.OutputField()

    class CustomPromptAdapter(CtxStrInputAdapter):
        name = "custom_prompt_ctx_str"
        value_type = SpecializedCtxStr

        def append_prompt(self, prompt, field, prepared, ctx):
            return f"{prompt}\n\nCUSTOM:{prepared.model_value}"

    rlm = PredictRLM(
        MultipleCtxStrSignature,
        sub_lm=MagicMock(),
        adapters=[CustomPromptAdapter()],
        max_iterations=1,
    )

    ctx = await _prepare_run(rlm, {"default": "DEFAULT", "custom": "CUSTOM-VALUE"})

    for predictor_name in ("generate_action", "extract"):
        prompt = str(ctx.state[predictor_name].signature.instructions)
        assert prompt.count("## In-Context Inputs") == 1
        assert prompt.count("CUSTOM-VALUE") == 1
        assert "CUSTOM:CUSTOM-VALUE" in prompt


def test_independent_same_name_ctx_str_adapter_is_rejected_at_construction():
    class UnsafeCtxStrAdapter(InputAdapter[CtxStr]):
        name = "ctx_str"
        value_type = CtxStr

        async def prepare(self, field, value, ctx):
            return PreparedInput(model_value=value)

    with pytest.raises(TypeError, match="ctx_str.*CtxStrInputAdapter"):
        PredictRLM(
            InContextSignature,
            sub_lm=MagicMock(),
            adapters=[UnsafeCtxStrAdapter()],
            max_iterations=1,
        )


@pytest.mark.asyncio
async def test_same_name_ctx_str_adapter_replaces_builtin_and_owns_prompt():
    class ReplacementCtxStrInputAdapter(CtxStrInputAdapter):
        async def prepare(self, field, value, ctx):
            return PreparedInput(model_value=f"replacement:{value}")

    rlm = PredictRLM(
        InContextSignature,
        sub_lm=MagicMock(),
        adapters=[ReplacementCtxStrInputAdapter()],
        max_iterations=1,
    )

    ctx = await _prepare_run(rlm, {"criteria": "RULE", "query": "QUESTION"})
    prompt = str(ctx.state["generate_action"].signature.instructions)

    assert type(ctx.input_bindings["criteria"].adapter) is ReplacementCtxStrInputAdapter
    assert "replacement:RULE" in prompt


@pytest.mark.asyncio
async def test_distinct_exact_ctx_str_adapter_conflicts_with_builtin():
    class OtherCtxStrAdapter(InputAdapter[CtxStr]):
        name = "other_ctx_str"
        value_type = CtxStr

        async def prepare(self, field, value, ctx):
            return PreparedInput(model_value=value)

    rlm = PredictRLM(
        InContextSignature,
        sub_lm=MagicMock(),
        adapters=[OtherCtxStrAdapter()],
        max_iterations=1,
    )

    with pytest.raises(ValueError, match="ctx_str, other_ctx_str"):
        await _prepare_run(rlm, {"criteria": "RULE", "query": "QUESTION"})


@pytest.mark.asyncio
async def test_kernel_does_not_build_ctx_str_instructions(monkeypatch):
    monkeypatch.setattr(
        in_context_module,
        "_build_in_context_instructions",
        MagicMock(side_effect=AssertionError("kernel special case called")),
        raising=False,
    )
    rlm = PredictRLM(InContextSignature, sub_lm=MagicMock(), max_iterations=1)

    ctx = await _prepare_run(rlm, {"criteria": "RULE", "query": "QUESTION"})

    assert "RULE" in str(ctx.state["generate_action"].signature.instructions)



@pytest.mark.asyncio
async def test_in_context_delimiters_avoid_prepared_value_collisions():
    nominal_closing_marker = '<END_IN_CONTEXT_INPUT name="criteria">'
    raw_value = f"Keep this exact marker:\n{nominal_closing_marker}\nwithout changing it."
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

    assert ctx.input_bindings["criteria"].prepared.model_value == raw_value
    assert raw_value in instructions
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
    class TransformAdapter(CtxStrInputAdapter):
        name = "ctx_str"
        value_type = CtxStr

        async def prepare(self, field, value, ctx):
            return PreparedInput(model_value=value)

        def _transform_prompt_signature(self, signature, field, prepared, ctx):
            signature.instructions = f"transformed:{prepared.model_value}"
            return signature

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
        adapters=[TransformAdapter()],
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
    for predictor_name in ("generate_action", "extract"):
        assert "transformed:Criteria block" in str(
            ctx.state[predictor_name].signature.instructions
        )


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
async def test_generic_string_input_adapter_does_not_own_ctx_str_value():
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

    assert ctx.input_bindings["criteria"].prepared.model_value == "RAW-RULE"
    assert "\nRAW-RULE\n" in action
    assert "prepared:RAW-RULE" not in action


@pytest.mark.asyncio
async def test_ctx_str_prompt_uses_final_bound_custom_adapter_value():
    class BindingAdapter(CtxStrInputAdapter):

        async def prepare(self, field, value, ctx):
            return PreparedInput(model_value=f"prepared:{value}")

        async def bind(self, field, prepared, ctx, session):
            return BoundInput(model_value=f"bound:{prepared.model_value}")

    rlm = PredictRLM(
        InContextSignature,
        sub_lm=MagicMock(),
        adapters=[BindingAdapter()],
        max_iterations=1,
    )
    input_values = {"criteria": "RULE", "query": "QUESTION"}
    ctx = rlm._new_run_context(input_values)
    await rlm._prepare_runtime_inputs(ctx, input_values)
    ctx.session = MagicMock()
    await rlm._bind_runtime_inputs(ctx)
    rlm._configure_run_predictors(ctx, None)

    action = str(ctx.state["generate_action"].signature.instructions)
    assert "bound:prepared:RULE" in action
    assert "\nprepared:RULE\n" not in action


@pytest.mark.asyncio
async def test_input_adapter_append_prompt_sees_current_prompt_for_action_and_extract():
    seen = []

    class PromptAdapter(PrefixStringInputAdapter):
        def append_prompt(self, prompt, field, prepared, ctx):
            seen.append((prompt, field.name, prepared.model_value, ctx.run_id))
            return f"{prompt}\n\nHOOK:{field.name}:{prepared.model_value}"

    rlm = PredictRLM(
        InContextSignature,
        sub_lm=MagicMock(),
        adapters=[PromptAdapter()],
        max_iterations=1,
    )
    ctx = await _prepare_run(
        rlm,
        {"criteria": "RULE", "query": "QUESTION"},
    )

    action = str(ctx.state["generate_action"].signature.instructions)
    extract = str(ctx.state["extract"].signature.instructions)
    assert len(seen) == 2
    assert [(field, value) for _, field, value, _ in seen] == [
        ("query", "prepared:QUESTION"),
        ("query", "prepared:QUESTION"),
    ]
    assert all(run_id == ctx.run_id for *_, run_id in seen)
    assert "HOOK:query:prepared:QUESTION" in action
    assert "HOOK:query:prepared:QUESTION" in extract


@pytest.mark.asyncio
async def test_prompt_hooks_chain_in_signature_field_order():
    class CriteriaAdapter(CtxStrInputAdapter):

        async def prepare(self, field, value, ctx):
            return PreparedInput(model_value=value)

        def append_prompt(self, prompt, field, prepared, ctx):
            return f"{prompt}\nFIRST"

    class QueryAdapter(InputAdapter[str]):
        name = "value"
        value_type = str
        fallback = True

        async def prepare(self, field, value, ctx):
            return PreparedInput(model_value=value)

        def append_prompt(self, prompt, field, prepared, ctx):
            assert "FIRST" in prompt
            return f"{prompt}\nSECOND"

    rlm = PredictRLM(
        InContextSignature,
        sub_lm=MagicMock(),
        adapters=[QueryAdapter(), CriteriaAdapter()],
        max_iterations=1,
    )
    ctx = await _prepare_run(rlm, {"criteria": "RULE", "query": "QUESTION"})

    for key in ("generate_action", "extract"):
        prompt = str(ctx.state[key].signature.instructions)
        assert prompt.index("FIRST") < prompt.index("SECOND")


@pytest.mark.asyncio
async def test_in_place_prompt_signature_transform_is_run_local():
    transformed_signatures = {}

    class TransformAdapter(CtxStrInputAdapter):
        name = "ctx_str"
        value_type = CtxStr

        async def prepare(self, field, value, ctx):
            return PreparedInput(model_value=f"prepared:{value}")

        def _transform_prompt_signature(self, signature, field, prepared, ctx):
            run_value = prepared.model_value
            signature.instructions = f"run instructions:{run_value}"
            signature.input_fields["query"].json_schema_extra["desc"] = f"run:{run_value}"
            transformed_signatures[run_value] = signature
            return signature

    rlm = PredictRLM(
        InContextSignature,
        sub_lm=MagicMock(),
        adapters=[TransformAdapter()],
        max_iterations=1,
    )
    original_instructions = str(rlm.signature.instructions)
    original_description = rlm.signature.input_fields["query"].json_schema_extra["desc"]

    first, second = await asyncio.gather(
        _prepare_run(rlm, {"criteria": "FIRST", "query": "one"}),
        _prepare_run(rlm, {"criteria": "SECOND", "query": "two"}),
    )
    third = await _prepare_run(rlm, {"criteria": "THIRD", "query": "three"})

    for ctx, own_value, sibling_values in (
        (first, "FIRST", ("SECOND", "THIRD")),
        (second, "SECOND", ("FIRST", "THIRD")),
        (third, "THIRD", ("FIRST", "SECOND")),
    ):
        for predictor_name in ("generate_action", "extract"):
            signature = ctx.state[predictor_name].signature
            prompt = str(signature.instructions)
            assert f"run instructions:prepared:{own_value}" in prompt
            for sibling_value in sibling_values:
                assert f"run instructions:prepared:{sibling_value}" not in prompt

        transformed = transformed_signatures[f"prepared:{own_value}"]
        assert transformed is not rlm.signature
        assert transformed.input_fields["query"].json_schema_extra["desc"] == (
            f"run:prepared:{own_value}"
        )

    assert rlm.signature is InContextSignature
    assert str(rlm.signature.instructions) == original_instructions
    assert rlm.signature.input_fields["query"].json_schema_extra["desc"] == original_description


@pytest.mark.asyncio
async def test_instance_bound_prompt_signature_transform_is_applied():
    adapter = PrefixStringInputAdapter()

    def transform_prompt_signature(self, signature, field, prepared, ctx):
        signature.instructions += f"\ninstance:{field.name}:{prepared.model_value}"
        return signature

    adapter._transform_prompt_signature = MethodType(transform_prompt_signature, adapter)
    rlm = PredictRLM(
        InContextSignature,
        sub_lm=MagicMock(),
        adapters=[adapter],
        max_iterations=1,
    )

    ctx = await _prepare_run(rlm, {"criteria": "RULE", "query": "QUESTION"})

    for predictor_name in ("generate_action", "extract"):
        prompt = str(ctx.state[predictor_name].signature.instructions)
        assert "instance:query:prepared:QUESTION" in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("hook", ["append_prompt", "_transform_prompt_signature"])
async def test_invalid_prompt_hook_return_type_fails_clearly(hook):
    class InvalidAdapter(PrefixStringInputAdapter):
        pass

    setattr(InvalidAdapter, hook, lambda *args: None)
    rlm = PredictRLM(
        InContextSignature,
        sub_lm=MagicMock(),
        adapters=[InvalidAdapter()],
        max_iterations=1,
    )

    with pytest.raises(TypeError, match=hook):
        await _prepare_run(rlm, {"criteria": "RULE", "query": "QUESTION"})


def test_removed_prompt_contributor_magic_is_absent():
    import predict_rlm.runtime as runtime_module

    assert not hasattr(runtime_module, "_PromptContributor")
    assert not hasattr(runtime_module, "_discover_annotation_prompt_contributors")
    assert not hasattr(CtxStr, "_predict_rlm_prompt_contributor")


def test_input_adapter_prompt_hook_defaults_are_noops():
    adapter = PrefixStringInputAdapter()
    prepared = PreparedInput(model_value="value")
    field = FieldDescriptor("query", str)
    ctx = RunContext(MagicMock(), {})

    assert adapter.append_prompt("prompt", field, prepared, ctx) == "prompt"
    assert adapter._transform_prompt_signature(InContextSignature, field, prepared, ctx) is InContextSignature


@pytest.mark.asyncio
async def test_default_prompt_hook_preserves_custom_action_predictor():
    class PlainSignature(dspy.Signature):
        query: str = dspy.InputField()
        answer: str = dspy.OutputField()

    class CustomAction:
        pass

    action = CustomAction()
    rlm = PredictRLM(
        PlainSignature,
        sub_lm=MagicMock(),
        adapters=[PrefixStringInputAdapter()],
        max_iterations=1,
    )
    rlm.generate_action = action

    ctx = await _prepare_run(rlm, {"query": "QUESTION"})

    assert ctx.state["generate_action"] is action


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
