"""Action adapter selection, Chat parse tolerance, and visible JSON fallbacks."""

import asyncio
import logging
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import dspy
import pytest
from dspy.primitives.code_interpreter import FinalOutput
from dspy.primitives.repl_types import REPLHistory
from dspy.utils.dummies import DummyLM

from predict_rlm import PredictRLM, ValidatingChatAdapter, ValidatingJSONAdapter


class ScriptedLM:
    model = "openai/gpt-4o-mini"
    supported_params = {"response_format"}
    supports_response_schema = True

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, messages=None, **kwargs):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return [response]


class SubmitRepl:
    def execute(self, code, variables=None, timeout=None):
        return FinalOutput({"answer": "done"})

    async def aexecute(self, code, variables=None, timeout=None):
        return self.execute(code)


JSON_ACTION = '{"plan": "ok", "code": "print(1)"}'


def _call_action_adapter(adapter, lm):
    return adapter(
        lm,
        {},
        PredictRLM("q -> answer").generate_action.signature,
        [],
        {"variables_info": "", "repl_history": REPLHistory(), "iteration": "1/1"},
    )


class TestChatAdapterFallback:
    def test_transport_errors_are_not_resent_through_json(self):
        lm = ScriptedLM(RuntimeError("429 rate limited"), JSON_ACTION)
        with pytest.raises(RuntimeError, match="429"):
            _call_action_adapter(ValidatingChatAdapter(), lm)
        assert lm.calls == 1

    def test_parse_failure_fallback_is_logged(self, caplog):
        lm = ScriptedLM("no markers at all", JSON_ACTION)
        with caplog.at_level(logging.WARNING, logger="predict_rlm.predict_rlm"):
            result = _call_action_adapter(ValidatingChatAdapter(), lm)
        assert result[0]["code"] == "print(1)"
        assert lm.calls == 2
        assert "re-sending" in caplog.text

    def test_inline_field_marker_parses_without_resending(self):
        lm = ScriptedLM(
            "[[ ## plan ## ]]\nI will print. [[ ## code ## ]]\n"
            "```repl\nprint(1)\n```\n[[ ## completed ## ]]"
        )
        result = _call_action_adapter(ValidatingChatAdapter(), lm)
        assert lm.calls == 1
        assert result[0]["plan"] == "I will print."
        assert "print(1)" in result[0]["code"]

    def test_missing_code_marker_before_repl_block_parses_without_resending(self):
        lm = ScriptedLM(
            "[[ ## plan ## ]]\nI will print.\n```repl\nprint(1)\n```\n[[ ## completed ## ]]"
        )
        result = _call_action_adapter(ValidatingChatAdapter(), lm)
        assert lm.calls == 1
        assert result[0]["plan"] == "I will print."
        assert result[0]["code"] == "```repl\nprint(1)\n```"

    def test_marker_text_inside_well_formed_code_is_preserved(self):
        lm = ScriptedLM(
            "[[ ## plan ## ]]\nok\n\n[[ ## code ## ]]\n"
            "print('see [[ ## code ## ]]')\n\n[[ ## completed ## ]]"
        )
        result = _call_action_adapter(ValidatingChatAdapter(), lm)
        assert result[0]["code"] == "print('see [[ ## code ## ]]')"


class TestAdapterSelection:
    def _active_adapter(self, rlm):
        with rlm._lm_context():
            return dspy.settings.adapter

    def test_default_adapter_is_validating_json(self):
        rlm = PredictRLM("q -> answer", sub_lm=DummyLM([]))
        assert isinstance(self._active_adapter(rlm), ValidatingJSONAdapter)

    def test_context_adapter_is_respected(self):
        rlm = PredictRLM("q -> answer", sub_lm=DummyLM([]))
        chat = ValidatingChatAdapter()
        with dspy.context(adapter=chat):
            assert self._active_adapter(rlm) is chat

    def test_constructor_adapter_overrides_context_adapter(self):
        chat = ValidatingChatAdapter()
        rlm = PredictRLM("q -> answer", sub_lm=DummyLM([]), adapter=chat)
        with dspy.context(adapter=ValidatingJSONAdapter()):
            assert self._active_adapter(rlm) is chat


class TestFallbackTrace:
    def _run(self, *, use_async: bool):
        rlm = PredictRLM("images, query -> answer", sub_lm=MagicMock(), max_iterations=2)
        submit = {"plan": "done", "code": "SUBMIT(answer='done')"}
        lm = DummyLM([submit, submit], adapter=dspy.JSONAdapter())
        with (
            dspy.context(lm=lm, adapter=ValidatingChatAdapter()),
            patch.object(
                rlm,
                "_interpreter_context",
                return_value=nullcontext(SubmitRepl()),
            ),
        ):
            if use_async:
                return asyncio.run(rlm._aforward_traced(None, images=["img"], query="q"))
            return rlm._forward_traced(None, images=["img"], query="q")

    @pytest.mark.parametrize("use_async", [False, True])
    def test_fallbacks_are_counted_per_step_and_run(self, use_async):
        result = self._run(use_async=use_async)
        assert result.answer == "done"
        assert [step.adapter_fallbacks for step in result.trace.steps] == [1]
        assert result.trace.adapter_fallbacks == 1
