"""Capability composition executes tools and refuses ambiguous resource ownership."""

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from predict_rlm import PredictRLM
from predict_rlm.capabilities import Capability, merge_capabilities


@pytest.mark.integration
def test_composed_capabilities_execute_together_in_the_rlm_loop():
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
            capabilities=[
                Capability(name="arithmetic", tools={"double": double}),
                Capability(name="report", tools={"label": label}),
            ],
            max_iterations=1,
        )
        with dspy.context(
            lm=DummyLM(
                [
                    {
                        "plan": "compose both capabilities",
                        "code": "value = await double(21)\nSUBMIT(answer=await label(value))",
                    }
                ],
                adapter=dspy.JSONAdapter(),
            )
        ):
            result = rlm(query="compute")
        assert result.answer == "result:42"
    finally:
        interpreter.shutdown()


def test_duplicate_capability_tool_names_are_rejected():
    def tool():
        return "value"

    with pytest.raises(ValueError, match="Tool name conflict.*shared"):
        merge_capabilities(
            [
                Capability(name="first", tools={"shared": tool}),
                Capability(name="second", tools={"shared": tool}),
            ]
        )


def test_duplicate_capability_module_names_are_rejected():
    with pytest.raises(ValueError, match="Module name conflict.*shared_mod"):
        merge_capabilities(
            [
                Capability(name="first", modules={"shared_mod": "/path/a.py"}),
                Capability(name="second", modules={"shared_mod": "/path/b.py"}),
            ]
        )


def test_capability_tool_cannot_silently_replace_user_tool():
    def tool():
        return "value"

    with pytest.raises(ValueError, match="Tool name conflict.*shared"):
        PredictRLM(
            "query -> answer",
            capabilities=[Capability(name="capability", tools={"shared": tool})],
            tools={"shared": tool},
        )
