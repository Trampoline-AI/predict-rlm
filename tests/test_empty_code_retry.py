"""Invalid action outputs recover through JSON fallback or fail before execution."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import dspy
import pytest
from dspy.primitives.repl_types import REPLHistory
from dspy.utils.exceptions import AdapterParseError

from predict_rlm import PredictRLM
from predict_rlm.predict_rlm import _ValidatingChatAdapter


class ScriptedLM:
    model = "openai/gpt-4o-mini"
    supported_params = {"response_format", "temperature", "max_tokens"}
    supports_response_schema = True

    def __init__(self, fallback):
        self.fallback = fallback
        self.responses = iter(
            [
                "[[ ## reasoning ## ]]\ntry an action\n\n"
                "[[ ## code ## ]]\n\n[[ ## completed ## ]]",
                fallback,
            ]
        )

    def __call__(self, messages=None, **kwargs):
        return [next(self.responses, self.fallback)]

    async def acall(self, messages=None, **kwargs):
        await asyncio.sleep(0)
        return self(messages=messages, **kwargs)


def _adapter_inputs():
    return {
        "lm_kwargs": {},
        "signature": PredictRLM("q -> answer").generate_action.signature,
        "demos": [],
        "inputs": {"variables_info": "", "repl_history": REPLHistory(), "iteration": "1/1"},
    }


def test_empty_chat_action_recovers_through_json_fallback():
    result = _ValidatingChatAdapter()(
        ScriptedLM('{"reasoning": "retry succeeded", "code": "print(1)"}'),
        **_adapter_inputs(),
    )
    assert result[0]["code"] == "print(1)"
    assert result[0]["reasoning"] == "retry succeeded"


@pytest.mark.asyncio
async def test_invalid_json_fallback_exhausts_recovery():
    with pytest.raises(AdapterParseError):
        await asyncio.wait_for(
            _ValidatingChatAdapter().acall(
                ScriptedLM('{"reasoning": "still invalid", "code": null}'),
                **_adapter_inputs(),
            ),
            timeout=3,
        )


@pytest.mark.asyncio
async def test_custom_predictor_empty_code_never_reaches_execution():
    rlm = PredictRLM("q -> answer")
    rlm.generate_action = MagicMock()
    rlm.generate_action.acall = AsyncMock(
        return_value=dspy.Prediction(reasoning="attempt", code="")
    )
    repl = MagicMock()
    repl.aexecute = AsyncMock(side_effect=AssertionError("invalid code was executed"))
    with pytest.raises(RuntimeError, match="invalid code"):
        await rlm._aexecute_iteration(repl, [], REPLHistory(), 0, {}, ["answer"])
