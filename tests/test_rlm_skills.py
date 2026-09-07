"""Skill composition executes tools and refuses ambiguous resource ownership."""

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from predict_rlm import PredictRLM
from predict_rlm.rlm_skills import Skill, merge_skills


@pytest.mark.integration
def test_composed_skills_execute_together_in_the_rlm_loop():
    from predict_rlm.backends import JspiBackend

    def double(value: int) -> int:
        return value * 2

    def label(value: int) -> str:
        return f"result:{value}"

    interpreter = JspiBackend(preinstall_packages=False)
    try:
        rlm = PredictRLM(
            "query -> answer",
            interpreter=interpreter,
            skills=[
                Skill(name="arithmetic", tools={"double": double}),
                Skill(name="report", tools={"label": label}),
            ],
            max_iterations=1,
        )
        with dspy.context(
            lm=DummyLM(
                [
                    {
                        "reasoning": "compose both skills",
                        "code": "value = await double(21)\nSUBMIT(answer=await label(value))",
                    }
                ]
            )
        ):
            result = rlm(query="compute")
        assert result.answer == "result:42"
    finally:
        interpreter.shutdown()


def test_duplicate_skill_tool_names_are_rejected():
    def tool():
        return "value"

    with pytest.raises(ValueError, match="Tool name conflict.*shared"):
        merge_skills(
            [
                Skill(name="first", tools={"shared": tool}),
                Skill(name="second", tools={"shared": tool}),
            ]
        )


def test_duplicate_skill_module_names_are_rejected():
    with pytest.raises(ValueError, match="Module name conflict.*shared_mod"):
        merge_skills(
            [
                Skill(name="first", modules={"shared_mod": "/path/a.py"}),
                Skill(name="second", modules={"shared_mod": "/path/b.py"}),
            ]
        )


def test_skill_tool_cannot_silently_replace_user_tool():
    def tool():
        return "value"

    with pytest.raises(ValueError, match="Tool name conflict.*shared"):
        PredictRLM(
            "query -> answer",
            skills=[Skill(name="skill", tools={"shared": tool})],
            tools={"shared": tool},
        )
