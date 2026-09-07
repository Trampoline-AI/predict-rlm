"""Context-input precedence, composition boundaries, and run isolation."""

import asyncio
from unittest.mock import MagicMock

import dspy
import pytest

from predict_rlm import CtxStr, CtxStrInputAdapter, PredictRLM
from predict_rlm.runtime import BoundInput, InputAdapter, PreparedInput


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
