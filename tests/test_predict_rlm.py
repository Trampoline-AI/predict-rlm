"""Core predict output, action-loop, and submit confirmation contracts."""

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import dspy
import pytest
from dspy.primitives.code_interpreter import FinalOutput
from dspy.utils.dummies import DummyLM
from pydantic import BaseModel, Field, ValidationError

from predict_rlm import PredictRLM
from predict_rlm.predict_rlm import _models_from_schema
from predict_rlm.trace import PredictCallGroup, TokenUsage


class ImageAnalysisSignature(dspy.Signature):
    """Analyze images and answer the query."""

    images: list[str] = dspy.InputField(desc="Base64 encoded images")
    query: str = dspy.InputField(desc="Question about the images")
    answer: str = dspy.OutputField(desc="Answer to the query")


class FakeSubmitRepl:
    def __init__(self, final_payload: dict[str, Any] | None = None):
        self.final_payload = final_payload or {"answer": "done"}
        self.executed: list[str] = []
        self.defer_count = 0
        self.deferred_submit_count = 0

    def defer_next_submit_finalization(self) -> None:
        self.defer_count += 1
        self._defer_next_submit = True

    def execute(self, code, variables=None, timeout=None):
        self.executed.append(code)
        if code.startswith("SUBMIT"):
            if getattr(self, "_defer_next_submit", False):
                self._defer_next_submit = False
                self.deferred_submit_count += 1
            return FinalOutput(self.final_payload)
        self._defer_next_submit = False
        return "checked output"

    async def aexecute(self, code, variables=None, timeout=None):
        return self.execute(code, variables=variables, timeout=timeout)


class FakeInterpreterContext:
    def __init__(self, repl):
        self.repl = repl

    def __enter__(self):
        return self.repl

    def __exit__(self, *_args):
        return False


@pytest.mark.integration
def test_predict_reconstructs_nested_sandbox_models_and_serializes_items_field():
    from predict_rlm.backends import JspiBackend

    payload = {"items": [{"name": "repair", "priority": "high", "note": None}]}
    interpreter = JspiBackend(preinstall_packages=False)
    rlm = PredictRLM(
        "query -> answer",
        interpreter=interpreter,
        sub_lm=DummyLM([{"items": [payload]}]),
        max_iterations=1,
    )
    code = """
import json
from typing import Literal
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    priority: Literal["high", "low"]
    note: str | None = None

class Order(BaseModel):
    items: list[Item]

extracted = await predict("text: str -> items: list[Order]", text="repair urgently")
SUBMIT(answer=json.dumps({"items": [order.model_dump() for order in extracted.items]}))
"""
    try:
        with dspy.context(lm=DummyLM([{"reasoning": "extract", "code": code}])):
            result = rlm(query="extract repairs")
        assert json.loads(result.answer) == {"items": [payload]}
    finally:
        interpreter.shutdown()


def test_reconstructed_defaulted_collections_remain_non_nullable():
    class Item(BaseModel):
        name: str
        tags: list[str] = Field(default_factory=list)
        note: str | None = None

    Model = _models_from_schema(Item.model_json_schema())["Item"]
    assert Model(name="x").model_dump() == {"name": "x", "tags": [], "note": None}
    with pytest.raises(ValidationError):
        Model(name="x", tags=None)


@pytest.mark.asyncio
async def test_predict_rejects_non_nullable_null_from_custom_predictor():
    rlm = PredictRLM("question -> answer", sub_lm=DummyLM([]))
    # A custom predictor can bypass adapter validation; the tool must still reject null.
    with patch("predict_rlm.predict_rlm.dspy.Predict") as predictor:
        predictor.return_value.acall = AsyncMock(return_value=dspy.Prediction(items=None))
        with pytest.raises(RuntimeError, match="None for non-Optional"):
            await rlm.tools["predict"].func("text: str -> items: list[str]", text="input")


@pytest.mark.asyncio
async def test_predict_requires_an_available_lm():
    rlm = PredictRLM("question -> answer")
    with dspy.context(lm=None):
        with pytest.raises(RuntimeError, match="No LM available"):
            await rlm.tools["predict"].func("question -> answer", question="test")


@pytest.mark.asyncio
async def test_concurrent_predict_calls_isolate_context_lms_and_restore_caller():
    rlm = PredictRLM("question -> answer")
    caller = DummyLM([{"answer": "caller"}])

    async def run(answer):
        with dspy.context(lm=DummyLM([{"answer": answer}])):
            await asyncio.sleep(0)
            return await rlm.tools["predict"].func("question -> answer", question="same")

    with dspy.context(lm=caller):
        results = await asyncio.gather(run("first"), run("second"))
        assert results == [{"answer": "first"}, {"answer": "second"}]
        assert dspy.settings.lm is caller
        assert await rlm.tools["predict"].func("question -> answer", question="same") == {
            "answer": "caller"
        }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_async_loop_executes_actions_recovers_errors_and_submits_output():
    from predict_rlm.backends import JspiBackend

    interpreter = JspiBackend(preinstall_packages=False)
    lm = DummyLM(
        [
            {
                "reasoning": "probe",
                "code": "```repl\nvalue = 6\nprint('before failure')\nraise ValueError('retry')\n```",
            },
            {"reasoning": "recover", "code": "SUBMIT(answer=str(value * 7))"},
        ]
    )
    try:
        rlm = PredictRLM("query -> answer", interpreter=interpreter, max_iterations=2)
        with dspy.context(lm=lm):
            result = await rlm.acall(query="compute")
        assert result.answer == "42"
        assert result.trace.status == "completed"
        assert "before failure" in result.trace.steps[0].untruncated_output
        assert result.trace.steps[0].error is True
    finally:
        interpreter.shutdown()


class TestSubmitConfirmation:
    """Tests for configurable submit confirmation in the main RLM loop."""

    @staticmethod
    def _prediction(code: str, reasoning: str = "thinking") -> dspy.Prediction:
        return dspy.Prediction(reasoning=reasoning, code=code)

    def _run_sync(
        self,
        rlm: PredictRLM,
        actions: list[dspy.Prediction],
        repl: FakeSubmitRepl | None = None,
    ) -> dspy.Prediction:
        repl = repl or FakeSubmitRepl()
        mock_lm = MagicMock()
        mock_lm.history = []
        rlm.generate_action = MagicMock(side_effect=actions)

        with (
            dspy.context(lm=mock_lm),
            patch.object(
                rlm, "_interpreter_context", return_value=FakeInterpreterContext(repl)
            ),
        ):
            return rlm._forward_traced(None, images=["img"], query="Original task")

    async def _run_async(
        self,
        rlm: PredictRLM,
        actions: list[dspy.Prediction],
        repl: FakeSubmitRepl | None = None,
    ) -> dspy.Prediction:
        repl = repl or FakeSubmitRepl()
        mock_lm = MagicMock()
        mock_lm.history = []
        rlm.generate_action = MagicMock()
        rlm.generate_action.acall = AsyncMock(side_effect=actions)

        with (
            dspy.context(lm=mock_lm),
            patch.object(
                rlm, "_interpreter_context", return_value=FakeInterpreterContext(repl)
            ),
        ):
            return await rlm._aforward_traced(None, images=["img"], query="Original task")

    def test_first_submit_prompts_and_second_submit_completes(self):
        seen_contexts = []

        def confirm(context):
            seen_contexts.append(context)
            return "Please verify the answer before final submit."

        rlm = PredictRLM(
            ImageAnalysisSignature,
            sub_lm=MagicMock(),
            max_iterations=3,
            submit_confirmation=confirm,
        )

        repl = FakeSubmitRepl()
        result = self._run_sync(
            rlm,
            [
                self._prediction("SUBMIT(answer='done')", reasoning="first submit"),
                self._prediction("SUBMIT(answer='done')", reasoning="second submit"),
            ],
            repl=repl,
        )

        assert result.answer == "done"
        assert result.trace.status == "completed"
        assert [step.output for step in result.trace.steps] == [
            "Please verify the answer before final submit.",
            "FINAL: {'answer': 'done'}",
        ]
        assert len(seen_contexts) == 1
        context = seen_contexts[0]
        assert context.inputs == {"images": ["img"], "query": "Original task"}
        assert context.submitted_payload == {"answer": "done"}

    def test_non_submit_after_confirmation_clears_pending_confirmation(self):
        prompts = []

        def confirm(context):
            prompts.append(context.iteration)
            return f"confirm attempt {context.iteration}"

        rlm = PredictRLM(
            ImageAnalysisSignature,
            sub_lm=MagicMock(),
            max_iterations=5,
            submit_confirmation=confirm,
        )

        result = self._run_sync(
            rlm,
            [
                self._prediction("SUBMIT(answer='done')"),
                self._prediction("print('checking')"),
                self._prediction("SUBMIT(answer='done')"),
                self._prediction("SUBMIT(answer='done')"),
            ],
        )

        assert result.answer == "done"
        assert prompts == [1, 3]
        assert [step.output for step in result.trace.steps] == [
            "confirm attempt 1",
            "checked output",
            "confirm attempt 3",
            "FINAL: {'answer': 'done'}",
        ]

    @pytest.mark.asyncio
    async def test_async_submit_confirmation_matches_sync_path(self):
        rlm = PredictRLM(
            ImageAnalysisSignature,
            sub_lm=MagicMock(),
            max_iterations=3,
            submit_confirmation=lambda _context: "async confirm",
        )

        result = await self._run_async(
            rlm,
            [
                self._prediction("SUBMIT(answer='done')"),
                self._prediction("SUBMIT(answer='done')"),
            ],
        )

        assert result.answer == "done"
        assert result.trace.status == "completed"
        assert [step.output for step in result.trace.steps] == [
            "async confirm",
            "FINAL: {'answer': 'done'}",
        ]


class TestFatalExecutionErrors:
    def test_sync_sandbox_fatal_error_propagates(self):
        from predict_rlm.backends.base import SandboxFatalError

        mock_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=mock_lm, max_iterations=5)

        mock_repl = MagicMock()
        mock_repl.execute = MagicMock(side_effect=SandboxFatalError("fatal"))

        mock_pred = MagicMock()
        mock_pred.reasoning = "thinking"
        mock_pred.code = "print('hello')"
        rlm.generate_action = MagicMock(return_value=mock_pred)

        with pytest.raises(SandboxFatalError, match="fatal"):
            rlm._execute_iteration(
                repl=mock_repl,
                variables=[],
                history=[],
                iteration=0,
                input_args={},
                output_field_names=["answer"],
            )

    @pytest.mark.asyncio
    async def test_sandbox_fatal_error_propagates(self):
        from predict_rlm.backends.base import SandboxFatalError

        mock_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=mock_lm, max_iterations=5)

        mock_repl = MagicMock()
        mock_repl.aexecute = AsyncMock(side_effect=SandboxFatalError("fatal"))

        mock_pred = MagicMock()
        mock_pred.reasoning = "thinking"
        mock_pred.code = "print('hello')"
        rlm.generate_action = MagicMock()
        rlm.generate_action.acall = AsyncMock(return_value=mock_pred)

        with pytest.raises(SandboxFatalError, match="fatal"):
            await rlm._aexecute_iteration(
                repl=mock_repl,
                variables=[],
                history=[],
                iteration=0,
                input_args={},
                output_field_names=["answer"],
            )


def test_iteration_usage_is_not_charged_again_on_the_next_step():
    rlm = PredictRLM("q -> a", sub_lm=DummyLM([]))
    rlm._last_action_lm_usage = TokenUsage(input_tokens=2000, output_tokens=100, cost=0.012)
    groups = [
        PredictCallGroup(
            signature="x -> y",
            instructions=None,
            model="dummy",
            total_usage=TokenUsage(input_tokens=50, output_tokens=10, cost=0.001),
            calls=[],
        ),
        PredictCallGroup(
            signature="x -> z",
            instructions=None,
            model="dummy",
            total_usage=TokenUsage(input_tokens=30, output_tokens=5, cost=0.0005),
            calls=[],
        ),
    ]
    usage = rlm._build_iteration_usage(groups)
    assert usage.main == TokenUsage(input_tokens=2000, output_tokens=100, cost=0.012)
    assert usage.sub.input_tokens == 80
    assert usage.sub.output_tokens == 15
    assert usage.sub.cost == pytest.approx(0.0015)

    next_usage = rlm._build_iteration_usage([])
    assert next_usage.main == TokenUsage()
    assert next_usage.sub == TokenUsage()
