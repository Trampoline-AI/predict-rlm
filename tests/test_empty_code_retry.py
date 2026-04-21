"""Regression coverage for invalid action ``code`` predictions.

Background:
    Some models return action-signature responses with empty or null ``code``
    fields. DSPy's adapter/Prediction boundary can otherwise let those values
    materialize as successful ``Prediction`` objects even though the action
    signature requires a non-empty string.

Contract:
    PredictRLM's validating adapter owns recovery: an empty ChatAdapter parse
    should trigger JSON fallback, and JSON ``null`` should raise before a
    ``Prediction`` is constructed. The RLM loop must not silently coerce invalid
    predictions; if a malformed object reaches ``_aexecute_iteration`` directly
    via a mock/custom adapter, it fails loudly.
"""


from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


class _FakeRepl:
    """Minimal repl with an async ``aexecute`` that just records the code."""

    def __init__(self):
        self.last_code = None

    async def aexecute(self, code, variables=None):
        self.last_code = code
        return "[Success] ok"


def _build_executor():
    """Make a bare PredictRLM instance we can call ``_aexecute_iteration`` on."""
    from predict_rlm.predict_rlm import PredictRLM

    executor = PredictRLM.__new__(PredictRLM)
    executor.max_iterations = 50
    executor.verbose = False
    # mock generate_action.acall so we control pred.code
    executor.generate_action = MagicMock()
    executor._partial_pending_entry = None
    executor._partial_history = None
    executor._process_execution_result = lambda pred, result, history, ofn: {
        "result": result,
        "pred_code": getattr(pred, "code", None),
    }
    return executor


def test_none_code_prediction_fails_loudly_in_rlm_loop():
    """A malformed direct Prediction means the validating adapter was bypassed."""
    executor = _build_executor()

    pred = SimpleNamespace(reasoning=None, code=None)
    executor.generate_action.acall = AsyncMock(return_value=pred)

    repl = _FakeRepl()

    async def _run():
        return await executor._aexecute_iteration(
            repl,
            variables=[],
            history=MagicMock(),
            iteration=0,
            input_args={},
            output_field_names=["answer"],
        )

    with pytest.raises(RuntimeError, match="invalid reasoning"):
        asyncio.run(_run())


def test_empty_string_code_prediction_fails_loudly_in_rlm_loop():
    """Empty code should be rejected by the adapter, not executed by the loop."""
    executor = _build_executor()

    pred = SimpleNamespace(reasoning="…", code="")
    executor.generate_action.acall = AsyncMock(return_value=pred)

    repl = _FakeRepl()

    async def _run():
        return await executor._aexecute_iteration(
            repl,
            variables=[],
            history=MagicMock(),
            iteration=0,
            input_args={},
            output_field_names=["answer"],
        )

    with pytest.raises(RuntimeError, match="invalid code"):
        asyncio.run(_run())


def test_code_field_has_min_length_validator():
    """Source-anchor: the action-signature's ``code`` field must carry
    a ``min_length=1`` constraint so PredictRLM's validating adapter rejects
    empty code and triggers the ChatAdapter → JSONAdapter fallback retry.
    If someone removes the constraint, we go back to accepting empty
    responses silently.
    """
    import dspy

    from predict_rlm._shared import build_rlm_signatures

    class _Base(dspy.Signature):
        q: str = dspy.InputField()
        answer: str = dspy.OutputField()

    action_sig, _ = build_rlm_signatures(
        _Base,
        instructions_template="",
        user_tools={},
        format_tool_docs=lambda _: "",
    )

    code_field = action_sig.model_fields.get("code")
    assert code_field is not None, "code field missing from action sig"

    # Pydantic min_length lives in the field's ``metadata`` list as a
    # ``MinLen(min_length=1)`` constraint. Check it's there — more
    # robust than instantiating the whole Signature (which requires
    # populating several unrelated fields just to trigger validation).
    has_min_length = any(
        getattr(constraint, "min_length", None) == 1
        for constraint in code_field.metadata
    )
    assert has_min_length, (
        f"code field metadata lacks min_length=1 constraint — empty/null "
        f"code responses from the LM will parse silently. Field metadata: "
        f"{code_field.metadata}"
    )


def test_validating_adapter_retries_empty_chat_code_via_json_fallback():
    """An empty parsed ChatAdapter code field must be treated as a parse
    failure so DSPy's JSON fallback gets a chance to recover.
    """
    import dspy

    from predict_rlm._shared import build_rlm_signatures
    from predict_rlm.predict_rlm import _ValidatingChatAdapter

    class _Base(dspy.Signature):
        q: str = dspy.InputField()
        answer: str = dspy.OutputField()

    action_sig, _ = build_rlm_signatures(
        _Base,
        instructions_template="",
        user_tools={},
        format_tool_docs=lambda _: "",
    )

    class _FakeLM:
        model = "openai/gpt-4o-mini"

        def __init__(self):
            self.calls = []

        def __call__(self, messages=None, **kwargs):
            self.calls.append({"messages": messages, "kwargs": kwargs})
            if len(self.calls) == 1:
                return [
                    "[[ ## reasoning ## ]]\n"
                    "try an action\n\n"
                    "[[ ## code ## ]]\n\n"
                    "[[ ## completed ## ]]"
                ]
            return ['{"reasoning": "retry succeeded", "code": "print(1)"}']

    lm = _FakeLM()
    result = _ValidatingChatAdapter()(
        lm,
        lm_kwargs={},
        signature=action_sig,
        demos=[],
        inputs={
            "variables_info": "",
            "repl_history": MagicMock(),
            "iteration": "1/1",
        },
    )

    assert result == [{"reasoning": "retry succeeded", "code": "print(1)"}]
    assert len(lm.calls) == 2


def test_validating_json_adapter_rejects_null_code():
    """JSON ``null`` is syntactically valid but invalid for required
    non-optional signature fields, so the adapter must raise.
    """
    import dspy
    from dspy.utils.exceptions import AdapterParseError

    from predict_rlm._shared import build_rlm_signatures
    from predict_rlm.predict_rlm import _ValidatingJSONAdapter

    class _Base(dspy.Signature):
        q: str = dspy.InputField()
        answer: str = dspy.OutputField()

    action_sig, _ = build_rlm_signatures(
        _Base,
        instructions_template="",
        user_tools={},
        format_tool_docs=lambda _: "",
    )

    with pytest.raises(AdapterParseError, match="cannot be null"):
        _ValidatingJSONAdapter().parse(
            action_sig,
            '{"reasoning": "looks like json", "code": null}',
        )
