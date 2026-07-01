"""Tests for CtxStr prompt-injected string inputs."""

from unittest.mock import MagicMock, patch

import dspy
import pytest

from predict_rlm import CtxStr, PredictRLM, Skill
from predict_rlm.in_context import build_in_context_instructions


class InContextSignature(dspy.Signature):
    """Answer the query using the provided criteria."""

    criteria: CtxStr = dspy.InputField(desc="Full rubric to apply")
    query: str = dspy.InputField(desc="Question to answer")
    answer: str = dspy.OutputField(desc="Answer to the query")


def _mock_lm():
    lm = MagicMock()
    lm.history = []
    return lm


def _fake_interpreter_context():
    context = MagicMock()
    context.__enter__.return_value = MagicMock()
    context.__exit__.return_value = False
    return context


def test_in_context_is_pydantic_string_schema():
    field = InContextSignature.input_fields["criteria"]

    assert field.annotation is CtxStr
    assert InContextSignature.model_json_schema()["properties"]["criteria"]["type"] == "string"


def test_build_in_context_instructions_includes_only_marked_fields():
    criteria = "Use every cited fact.\nPrefer concise answers."

    instructions = build_in_context_instructions(
        InContextSignature,
        {
            "criteria": criteria,
            "query": "This ordinary string input should not be injected.",
        },
    )

    assert instructions.startswith("## In-Context Inputs")
    assert "### `criteria`" in instructions
    assert "Full rubric to apply" not in instructions
    assert "Characters:" not in instructions
    assert criteria in instructions
    assert "<BEGIN_IN_CONTEXT_INPUT name=\"criteria\">" in instructions
    assert "<END_IN_CONTEXT_INPUT name=\"criteria\">" in instructions
    assert "This ordinary string input should not be injected." not in instructions


def test_runtime_in_context_inputs_rebuild_action_and_extract_signatures_for_call():
    rlm = PredictRLM(InContextSignature, sub_lm=MagicMock(), max_iterations=1)
    original_action = rlm.generate_action
    original_extract = rlm.extract
    captured = {}
    criteria = "Always mention the controlling rule."

    def finish_run(*_args):
        captured["action"] = str(rlm.generate_action.signature.instructions)
        captured["extract"] = str(rlm.extract.signature.instructions)
        return dspy.Prediction(answer="done")

    with (
        dspy.context(lm=_mock_lm()),
        patch.object(rlm, "_interpreter_context", return_value=_fake_interpreter_context()),
        patch.object(rlm, "_execute_iteration", side_effect=finish_run),
    ):
        result = rlm._forward_traced(
            None,
            criteria=criteria,
            query="What matters?",
        )

    assert result.answer == "done"
    assert criteria in captured["action"]
    assert criteria in captured["extract"]
    assert captured["action"].rstrip().endswith(
        "<END_IN_CONTEXT_INPUT name=\"criteria\">"
    )
    assert rlm.generate_action is original_action
    assert rlm.extract is original_extract


def test_runtime_in_context_instructions_come_after_files_and_skills():
    skill = Skill(name="domain", instructions="Skill block")
    rlm = PredictRLM(
        InContextSignature,
        sub_lm=MagicMock(),
        max_iterations=1,
        skills=[skill],
    )

    action, extract = rlm._build_signatures_with_runtime_instructions(
        file_instructions="## Files\n\nFile block",
        in_context_instructions="## In-Context Inputs\n\nCriteria block",
    )
    action_instructions = str(action.signature.instructions)

    assert action_instructions.index("File block") < action_instructions.index(
        "Skill block"
    )
    assert action_instructions.index("Skill block") < action_instructions.index(
        "Criteria block"
    )
    assert action_instructions.rstrip().endswith("Criteria block")
    assert str(extract.signature.instructions).rstrip().endswith("Criteria block")


def test_in_context_rejects_non_string_runtime_value():
    rlm = PredictRLM(InContextSignature, sub_lm=MagicMock(), max_iterations=1)

    with dspy.context(lm=_mock_lm()):
        with pytest.raises(TypeError, match="expects a string"):
            rlm._forward_traced(None, criteria=123, query="What matters?")


def test_in_context_is_input_only():
    class BadOutput(dspy.Signature):
        prompt: str = dspy.InputField()
        answer: CtxStr = dspy.OutputField()

    with pytest.raises(TypeError, match="CtxStr fields are input-only"):
        PredictRLM(BadOutput, sub_lm=MagicMock(), max_iterations=1)


def test_in_context_rejects_wrapped_input_annotations():
    class BadOptional(dspy.Signature):
        criteria: CtxStr | None = dspy.InputField()
        answer: str = dspy.OutputField()

    with pytest.raises(TypeError, match="annotated directly"):
        PredictRLM(BadOptional, sub_lm=MagicMock(), max_iterations=1)
