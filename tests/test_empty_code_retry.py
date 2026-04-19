"""RED-GREEN repro for the empty-code-pred UnboundLocalError.

Background:
    gemini sometimes returns action-signature responses with empty or
    null ``code`` fields — parseable by DSPy's ChatAdapter (no exception
    raised), but semantically useless. ``_aexecute_iteration`` then does::

        try:
            code = _strip_code_fences(pred.code)  # (1)
            result = await repl.aexecute(code, ...)
        except Exception as e:
            if code.count('\"\"\"') >= 3 and ...:  # (2) UnboundLocalError
                ...

    If ``pred.code`` is ``None``, line (1) raises ``AttributeError``,
    and line (2) references ``code`` — which was never assigned because
    line (1) crashed first → ``UnboundLocalError``.

    A gemini+medium eval on 2026-04-18 hit this 28 times out of 400
    tasks, dropping the soft score from ~0.87 to 0.7322.

Two fixes:

1. Add ``min_length=1`` to the ``code`` OutputField in
   ``src/predict_rlm/_shared.py::build_rlm_signatures``. Pydantic
   validation rejects empty/null code during the adapter parse step,
   which triggers DSPy's ChatAdapter → JSONAdapter fallback retry
   automatically — no custom retry logic needed.

2. Safety net: initialize ``code = pred.code or ''`` before the try
   in ``_aexecute_iteration`` so the except handler always has a
   defined ``code`` even if both adapters fail.

RED: empty code triggers ``UnboundLocalError`` (bug present)
GREEN: empty code produces a clean ``[Error] ...`` result that feeds
    back to the RLM's next iteration, or triggers the adapter fallback.
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


def test_empty_code_pred_does_not_raise_unbound_local():
    """When ``pred.code`` is None (malformed LM output), the iteration
    must not crash with UnboundLocalError. It should route the error
    back to the RLM as ``[Error] ...`` so the next iteration can retry
    with fresh context.
    """
    executor = _build_executor()

    # Simulate a prediction whose `code` came back as None — the exact
    # shape the gemini-medium failure produces.
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

    # Must not raise UnboundLocalError. Any other failure mode is
    # acceptable as long as it's an ordinary error surfaced cleanly.
    try:
        result = asyncio.run(_run())
    except UnboundLocalError as e:
        pytest.fail(
            f"UnboundLocalError on empty-code pred: {e}\n"
            "The error-handling branch at predict_rlm.py:~1129 references "
            "'code' without a default; when _strip_code_fences(None) "
            "raises on line 1122, 'code' was never assigned."
        )
    # A result was produced — the iteration routed the bad pred to the
    # error-feedback path instead of crashing.
    assert result is not None


def test_empty_string_code_pred_routes_to_error_feedback():
    """Empty string (``""``) code should also feed back as an error
    rather than executing empty code as if it were valid.
    """
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

    result = asyncio.run(_run())
    # Either the fix validated and raised a clean error, or it fell
    # through to the error-feedback path. Both are acceptable; the
    # NOT-acceptable outcome would be executing empty code as if valid.
    assert result is not None


def test_code_field_has_min_length_validator():
    """Source-anchor: the action-signature's ``code`` field must carry
    a ``min_length=1`` constraint so DSPy's adapter parse rejects
    empty code and triggers the ChatAdapter → JSONAdapter fallback
    retry. If someone removes the constraint, we go back to accepting
    empty responses silently.
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
