"""Tests for PredictRLM with predict tool for DSPy signatures."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import dspy
import pytest
from pydantic import BaseModel

from predict_rlm import PredictRLM
from predict_rlm.predict_rlm import _models_from_schema
from predict_rlm.rlm_skills import Skill


def _run(coro):
    """Run async predict call from sync test."""
    import nest_asyncio

    nest_asyncio.apply()
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


class ImageAnalysisSignature(dspy.Signature):
    """Analyze images and answer the query."""

    images: list[str] = dspy.InputField(desc="Base64 encoded images")
    query: str = dspy.InputField(desc="Question about the images")
    answer: str = dspy.OutputField(desc="Answer to the query")


class MockLM:
    """Mock LM that returns predictable responses for testing."""

    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = responses or {}
        self.calls: list[dict] = []

    def __call__(self, messages=None, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        content = messages[-1].get("content", "") if messages else ""
        prompt = content if isinstance(content, str) else str(content)
        for key, response in self.responses.items():
            if key.lower() in prompt.lower():
                return [response]
        return ["Default LM response"]


class TestPredictTool:
    """Tests that PredictRLM predict tool correctly runs DSPy signatures."""

    @pytest.mark.asyncio
    async def test_predict_returns_dict_response(self):
        """predict tool runs DSPy Predict and returns dict output."""
        mock_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=mock_lm, max_iterations=5)

        with patch("predict_rlm.predict_rlm.dspy.Predict") as mock_predict_class:
            mock_predictor = MagicMock()
            mock_prediction = MagicMock()
            values = {"answer": "Paris"}
            mock_prediction.keys.return_value = list(values.keys())
            mock_prediction.__getitem__ = MagicMock(side_effect=lambda k: values[k])
            mock_predictor.acall = AsyncMock(return_value=mock_prediction)
            mock_predict_class.return_value = mock_predictor

            result = await rlm.tools["predict"].func(
                "question -> answer",
                question="What is the capital of France?",
            )

            assert isinstance(result, dict)
            assert result == {"answer": "Paris"}
            # Predict is called with a parsed Signature object
            mock_predict_class.assert_called_once()
            sig = mock_predict_class.call_args[0][0]
            assert hasattr(sig, "input_fields") and "question" in sig.input_fields
            mock_predictor.acall.assert_called_once_with(question="What is the capital of France?")

    @pytest.mark.asyncio
    async def test_predict_with_multiple_outputs(self):
        """predict correctly handles signatures with multiple outputs."""
        mock_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=mock_lm, max_iterations=5)

        with patch("predict_rlm.predict_rlm.dspy.Predict") as mock_predict_class:
            mock_predictor = MagicMock()
            mock_prediction = MagicMock()
            values = {"title": "Test Document", "summary": "A brief summary"}
            mock_prediction.keys.return_value = list(values.keys())
            mock_prediction.__getitem__ = MagicMock(side_effect=lambda k: values[k])
            mock_predictor.acall = AsyncMock(return_value=mock_prediction)
            mock_predict_class.return_value = mock_predictor

            result = await rlm.tools["predict"].func(
                "text -> title, summary",
                text="Some document content",
            )

            assert isinstance(result, dict)
            assert result == {"title": "Test Document", "summary": "A brief summary"}

    @pytest.mark.asyncio
    async def test_predict_with_instructions(self):
        """predict passes instructions to create a Signature with instructions."""
        mock_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=mock_lm, max_iterations=5)

        with patch("predict_rlm.predict_rlm.dspy.Predict") as mock_predict_class:
            with patch("predict_rlm.predict_rlm.dspy.Signature") as mock_sig_class:
                mock_predictor = MagicMock()
                mock_prediction = MagicMock()
                values = {"toxic": True}
                mock_prediction.keys.return_value = list(values.keys())
                mock_prediction.__getitem__ = MagicMock(side_effect=lambda k: values[k])
                mock_predictor.acall = AsyncMock(return_value=mock_prediction)
                mock_predict_class.return_value = mock_predictor
                mock_sig_class.return_value = "mocked_signature"

                result = await rlm.tools["predict"].func(
                    "comment -> toxic: bool",
                    instructions="Mark as toxic if the comment includes insults.",
                    comment="You're an idiot!",
                )

                assert result == {"toxic": True}
                mock_sig_class.assert_called_once_with(
                    "comment -> toxic: bool", "Mark as toxic if the comment includes insults."
                )
                mock_predict_class.assert_called_once_with("mocked_signature")

    @pytest.mark.asyncio
    async def test_predict_uses_sub_lm(self):
        """predict uses the sub_lm when provided."""
        mock_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=mock_lm, max_iterations=5)

        with patch("predict_rlm.predict_rlm.dspy.Predict") as mock_predict_class:
            mock_predictor = MagicMock()
            mock_prediction = MagicMock()
            values = {"answer": "Test answer"}
            mock_prediction.keys.return_value = list(values.keys())
            mock_prediction.__getitem__ = MagicMock(side_effect=lambda k: values[k])
            mock_predictor.acall = AsyncMock(return_value=mock_prediction)
            mock_predict_class.return_value = mock_predictor

            result = await rlm.tools["predict"].func(
                "question -> answer",
                question="Test question",
            )

            assert result == {"answer": "Test answer"}

    @pytest.mark.asyncio
    async def test_predict_error_when_no_lm(self):
        """predict raises error when no LM is available."""
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=None, max_iterations=3)

        with dspy.context(lm=None):
            with pytest.raises(RuntimeError, match="No LM available for predict"):
                await rlm.tools["predict"].func("question -> answer", question="test")

    @pytest.mark.asyncio
    async def test_predict_auto_wraps_images_with_type_hint(self):
        """predict automatically wraps image URLs when field has dspy.Image type hint."""
        mock_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=mock_lm, max_iterations=5)

        with patch("predict_rlm.predict_rlm.dspy.Predict") as mock_predict_class:
            mock_predictor = MagicMock()
            mock_prediction = MagicMock()
            values = {"answer": "Extracted text"}
            mock_prediction.keys.return_value = list(values.keys())
            mock_prediction.__getitem__ = MagicMock(side_effect=lambda k: values[k])
            mock_predictor.acall = AsyncMock(return_value=mock_prediction)
            mock_predict_class.return_value = mock_predictor

            result = await rlm.tools["predict"].func(
                "image: dspy.Image, question -> answer",
                image="https://example.com/image.png",
                question="What text is visible?",
            )

            assert result == {"answer": "Extracted text"}
            # Verify the image was wrapped in dspy.Image
            call_kwargs = mock_predictor.acall.call_args.kwargs
            assert isinstance(call_kwargs["image"], dspy.Image)
            assert call_kwargs["question"] == "What text is visible?"

    @pytest.mark.asyncio
    async def test_predict_auto_wraps_base64_images_with_type_hint(self):
        """predict automatically wraps base64 image data when field has dspy.Image type hint."""
        mock_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=mock_lm, max_iterations=5)

        with patch("predict_rlm.predict_rlm.dspy.Predict") as mock_predict_class:
            mock_predictor = MagicMock()
            mock_prediction = MagicMock()
            values = {"text": "OCR result"}
            mock_prediction.keys.return_value = list(values.keys())
            mock_prediction.__getitem__ = MagicMock(side_effect=lambda k: values[k])
            mock_predictor.acall = AsyncMock(return_value=mock_prediction)
            mock_predict_class.return_value = mock_predictor

            result = await rlm.tools["predict"].func(
                "document: dspy.Image -> text",
                document="data:image/png;base64,abc123...",
            )

            assert result == {"text": "OCR result"}
            call_kwargs = mock_predictor.acall.call_args.kwargs
            assert isinstance(call_kwargs["document"], dspy.Image)

    @pytest.mark.asyncio
    async def test_predict_does_not_wrap_without_type_hint(self):
        """predict does not wrap values for fields without dspy.Image type hint."""
        mock_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=mock_lm, max_iterations=5)

        with patch("predict_rlm.predict_rlm.dspy.Predict") as mock_predict_class:
            with patch("predict_rlm.predict_rlm.dspy.Image") as mock_image_class:
                mock_predictor = MagicMock()
                mock_prediction = MagicMock()
                values = {"answer": "42"}
                mock_prediction.keys.return_value = list(values.keys())
                mock_prediction.__getitem__ = MagicMock(side_effect=lambda k: values[k])
                mock_predictor.acall = AsyncMock(return_value=mock_prediction)
                mock_predict_class.return_value = mock_predictor

                result = await rlm.tools["predict"].func(
                    "question -> answer",
                    question="https://example.com/some-url",
                )

                assert result == {"answer": "42"}
                mock_image_class.assert_not_called()
                mock_predictor.acall.assert_called_once_with(
                    question="https://example.com/some-url",
                )

    @pytest.mark.asyncio
    async def test_predict_uses_context_lm_captured_by_forward(self):
        """predict uses context LM captured during forward() for thread-safe execution."""
        context_lm = MagicMock()
        context_lm.name = "context_lm"
        global_lm = MagicMock()
        global_lm.name = "global_lm"

        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=None, max_iterations=3)

        # Simulate forward() capturing the context LM
        rlm._context_lm = context_lm

        with patch("predict_rlm.predict_rlm.dspy.Predict") as mock_predict_class:
            with patch("predict_rlm.predict_rlm.dspy.context") as mock_context:
                mock_predictor = MagicMock()
                mock_prediction = MagicMock()
                mock_prediction.keys.return_value = ["answer"]
                mock_prediction.answer = "Test"
                mock_predictor.acall = AsyncMock(return_value=mock_prediction)
                mock_predict_class.return_value = mock_predictor

                _ = await rlm.tools["predict"].func(
                    "question -> answer",
                    question="Test?",
                )

                mock_context.assert_called_once_with(lm=context_lm)

    def test_forward_captures_and_clears_context_lm(self):
        """forward() captures context LM before execution and clears it after."""
        context_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=None, max_iterations=1)

        assert rlm._context_lm is None

        with patch.object(dspy.RLM, "forward") as mock_super_forward:
            mock_super_forward.return_value = dspy.Prediction(answer="Test")

            with dspy.context(lm=context_lm):

                def check_context_lm(**kwargs):
                    assert rlm._context_lm is context_lm
                    return dspy.Prediction(answer="Test")

                mock_super_forward.side_effect = check_context_lm

                _ = rlm.forward(images=["img"], query="test?")

        assert rlm._context_lm is None

    @pytest.mark.asyncio
    async def test_predict_auto_wraps_list_of_images_with_type_hint(self):
        """predict automatically wraps list of image URLs when field has list[dspy.Image] type hint."""
        mock_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=mock_lm, max_iterations=5)

        with patch("predict_rlm.predict_rlm.dspy.Predict") as mock_predict_class:
            mock_predictor = MagicMock()
            mock_prediction = MagicMock()
            values = {"answer": "Analyzed 3 images"}
            mock_prediction.keys.return_value = list(values.keys())
            mock_prediction.__getitem__ = MagicMock(side_effect=lambda k: values[k])
            mock_predictor.acall = AsyncMock(return_value=mock_prediction)
            mock_predict_class.return_value = mock_predictor

            result = await rlm.tools["predict"].func(
                "images: list[dspy.Image], question -> answer",
                images=[
                    "https://example.com/img1.png",
                    "https://example.com/img2.png",
                    "https://example.com/img3.png",
                ],
                question="What do these images show?",
            )

            assert result == {"answer": "Analyzed 3 images"}
            # Predictor should receive list of wrapped dspy.Image instances
            mock_predictor.acall.assert_called_once()
            call_kwargs = mock_predictor.acall.call_args.kwargs
            assert len(call_kwargs["images"]) == 3
            assert all(isinstance(img, dspy.Image) for img in call_kwargs["images"])
            assert call_kwargs["question"] == "What do these images show?"


class TestTypeContractEnforcement:
    """Tests for type contract enforcement on predict outputs."""

    @pytest.mark.asyncio
    async def test_none_for_non_optional_list_raises(self):
        """VLM returning None for non-Optional list[X] field raises RuntimeError."""
        mock_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=mock_lm, max_iterations=5)

        with patch("predict_rlm.predict_rlm.dspy.Predict") as mock_predict_class:
            mock_predictor = MagicMock()
            mock_prediction = MagicMock()
            values = {"items": None}
            mock_prediction.keys.return_value = list(values.keys())
            mock_prediction.__getitem__ = MagicMock(side_effect=lambda k: values[k])
            mock_predictor.acall = AsyncMock(return_value=mock_prediction)
            mock_predict_class.return_value = mock_predictor

            with pytest.raises(RuntimeError, match="VLM returned None for non-Optional"):
                await rlm.tools["predict"].func(
                    "text: str -> items: list[str]",
                    text="some input",
                )

    @pytest.mark.asyncio
    async def test_empty_list_passes_through_unchanged(self):
        """Empty list [] is always valid — it means 'nothing found', distinct from None."""
        mock_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=mock_lm, max_iterations=5)

        with patch("predict_rlm.predict_rlm.dspy.Predict") as mock_predict_class:
            mock_predictor = MagicMock()
            mock_prediction = MagicMock()
            values = {"items": []}
            mock_prediction.keys.return_value = list(values.keys())
            mock_prediction.__getitem__ = MagicMock(side_effect=lambda k: values[k])
            mock_predictor.acall = AsyncMock(return_value=mock_prediction)
            mock_predict_class.return_value = mock_predictor

            result = await rlm.tools["predict"].func(
                "text: str -> items: list[str]",
                text="some input",
            )
            assert result["items"] == []

    @pytest.mark.asyncio
    async def test_none_for_optional_str_passes_through(self):
        """None for Optional[str] field passes through without error."""
        mock_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=mock_lm, max_iterations=5)

        with patch("predict_rlm.predict_rlm.dspy.Predict") as mock_predict_class:
            mock_predictor = MagicMock()
            mock_prediction = MagicMock()
            values = {"answer": None}
            mock_prediction.keys.return_value = list(values.keys())
            mock_prediction.__getitem__ = MagicMock(side_effect=lambda k: values[k])
            mock_predictor.acall = AsyncMock(return_value=mock_prediction)
            mock_predict_class.return_value = mock_predictor

            result = await rlm.tools["predict"].func(
                "text: str -> answer: Optional[str]",
                text="some input",
            )
            assert result["answer"] is None

    @pytest.mark.asyncio
    async def test_none_for_non_optional_str_raises(self):
        """VLM returning None for non-Optional str field raises RuntimeError."""
        mock_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=mock_lm, max_iterations=5)

        with patch("predict_rlm.predict_rlm.dspy.Predict") as mock_predict_class:
            mock_predictor = MagicMock()
            mock_prediction = MagicMock()
            values = {"answer": None}
            mock_prediction.keys.return_value = list(values.keys())
            mock_prediction.__getitem__ = MagicMock(side_effect=lambda k: values[k])
            mock_predictor.acall = AsyncMock(return_value=mock_prediction)
            mock_predict_class.return_value = mock_predictor

            with pytest.raises(RuntimeError, match="VLM returned None for non-Optional"):
                await rlm.tools["predict"].func(
                    "text: str -> answer: str",
                    text="some input",
                )


class TestPredictRLMConfiguration:
    """Tests for PredictRLM configuration options."""

    def test_predict_always_exists(self):
        """predict tool is always available (uses context LM if sub_lm not provided)."""
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=None, max_iterations=5)
        assert "predict" in rlm.tools

    def test_user_predict_not_overwritten(self):
        """User-provided predict is not replaced."""
        mock_lm = MockLM()

        def user_predict(signature: str, **kwargs) -> dict:
            return {"answer": "user implementation"}

        rlm = PredictRLM(
            ImageAnalysisSignature,
            sub_lm=mock_lm,
            tools={"predict": user_predict},
            max_iterations=5,
        )

        result = rlm.tools["predict"].func("question -> answer", question="test")
        assert result == {"answer": "user implementation"}
        assert len(mock_lm.calls) == 0

    def test_instructions_reference_predict_not_llm_query(self):
        """PredictRLM instructions mention predict, not llm_query or sub_lm_query."""
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=None, max_iterations=5)
        instructions = str(rlm.generate_action.signature.instructions)

        assert "predict" in instructions
        assert "llm_query" not in instructions
        assert "sub_lm_query" not in instructions

    def test_allowed_domains_passed_to_interpreter(self):
        """PredictRLM passes allowed_domains to interpreter."""
        rlm = PredictRLM(
            ImageAnalysisSignature,
            sub_lm=None,
            max_iterations=5,
            allowed_domains=["api.example.com"],
        )
        assert rlm._allowed_domains == ["api.example.com"]


class TestMainLMParameter:
    """Tests for the lm parameter on PredictRLM."""

    def test_lm_as_dspy_lm_instance(self):
        """Passing a dspy.LM instance stores it directly."""
        mock_lm = MagicMock(spec=dspy.LM)
        rlm = PredictRLM(ImageAnalysisSignature, lm=mock_lm, max_iterations=1)
        assert rlm._lm is mock_lm

    def test_lm_as_string_creates_dspy_lm(self):
        """Passing a model string creates a dspy.LM instance."""
        with patch("predict_rlm.predict_rlm.dspy.LM") as mock_lm_class:
            mock_lm_class.return_value = MagicMock()
            rlm = PredictRLM(ImageAnalysisSignature, lm="openai/gpt-4o", max_iterations=1)
            mock_lm_class.assert_any_call("openai/gpt-4o", cache=False)
            assert rlm._lm is mock_lm_class.return_value

    def test_lm_none_by_default(self):
        """lm defaults to None (uses context LM)."""
        rlm = PredictRLM(ImageAnalysisSignature, max_iterations=1)
        assert rlm._lm is None

    def test_forward_uses_lm_as_context(self):
        """forward() wraps execution in dspy.context(lm=...) when lm is provided."""
        mock_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, lm=mock_lm, max_iterations=1)

        with patch.object(dspy.RLM, "forward") as mock_super_forward:
            mock_super_forward.return_value = dspy.Prediction(answer="Test")

            captured_lm = None

            def capture_context(**kwargs):
                nonlocal captured_lm
                captured_lm = dspy.settings.lm
                return dspy.Prediction(answer="Test")

            mock_super_forward.side_effect = capture_context
            rlm.forward(images=["img"], query="test?")
            assert captured_lm is mock_lm

    def test_forward_without_lm_uses_external_context(self):
        """forward() without lm uses whatever is in dspy.context."""
        external_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, lm=None, max_iterations=1)

        with patch.object(dspy.RLM, "forward") as mock_super_forward:
            mock_super_forward.return_value = dspy.Prediction(answer="Test")

            captured_lm = None

            def capture_context(**kwargs):
                nonlocal captured_lm
                captured_lm = dspy.settings.lm
                return dspy.Prediction(answer="Test")

            mock_super_forward.side_effect = capture_context

            with dspy.context(lm=external_lm):
                rlm.forward(images=["img"], query="test?")

            assert captured_lm is external_lm

    def test_forward_clears_context_lm_after_execution(self):
        """forward() clears _context_lm after execution when lm is provided."""
        mock_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, lm=mock_lm, max_iterations=1)

        with patch.object(dspy.RLM, "forward") as mock_super_forward:
            mock_super_forward.return_value = dspy.Prediction(answer="Test")
            rlm.forward(images=["img"], query="test?")

        assert rlm._context_lm is None

    def test_forward_clears_context_lm_on_error(self):
        """forward() clears _context_lm even if execution raises."""
        mock_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, lm=mock_lm, max_iterations=1)

        with patch.object(dspy.RLM, "forward") as mock_super_forward:
            mock_super_forward.side_effect = RuntimeError("boom")

            with pytest.raises(RuntimeError):
                rlm.forward(images=["img"], query="test?")

        assert rlm._context_lm is None

    def test_lm_and_sub_lm_both_accepted(self):
        """Both lm and sub_lm can be provided together."""
        mock_lm = MagicMock(spec=dspy.LM)
        mock_sub_lm = MagicMock(spec=dspy.LM)
        rlm = PredictRLM(
            ImageAnalysisSignature,
            lm=mock_lm,
            sub_lm=mock_sub_lm,
            max_iterations=1,
        )
        assert rlm._lm is mock_lm
        assert rlm._sub_lm is mock_sub_lm

    @pytest.mark.asyncio
    async def test_aforward_uses_lm_as_context(self):
        """aforward() wraps execution in dspy.context(lm=...) when lm is provided."""
        mock_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, lm=mock_lm, max_iterations=1)

        with patch.object(dspy.RLM, "aforward") as mock_super_aforward:
            captured_lm = None

            async def capture_context(**kwargs):
                nonlocal captured_lm
                captured_lm = dspy.settings.lm
                return dspy.Prediction(answer="Test")

            mock_super_aforward.side_effect = capture_context
            await rlm.aforward(images=["img"], query="test?")
            assert captured_lm is mock_lm

    @pytest.mark.asyncio
    async def test_aforward_clears_context_lm_after_execution(self):
        """aforward() clears _context_lm after execution when lm is provided."""
        mock_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, lm=mock_lm, max_iterations=1)

        with patch.object(dspy.RLM, "aforward") as mock_super_aforward:
            mock_super_aforward.return_value = dspy.Prediction(answer="Test")
            await rlm.aforward(images=["img"], query="test?")

        assert rlm._context_lm is None

    @pytest.mark.asyncio
    async def test_aforward_without_lm_uses_external_context(self):
        """aforward() without lm uses whatever is in dspy.context."""
        external_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, lm=None, max_iterations=1)

        with patch.object(dspy.RLM, "aforward") as mock_super_aforward:
            captured_lm = None

            async def capture_context(**kwargs):
                nonlocal captured_lm
                captured_lm = dspy.settings.lm
                return dspy.Prediction(answer="Test")

            mock_super_aforward.side_effect = capture_context

            with dspy.context(lm=external_lm):
                await rlm.aforward(images=["img"], query="test?")

            assert captured_lm is external_lm


class TestModelsFromSchema:
    """Tests for _models_from_schema function that reconstructs Pydantic models."""

    def test_simple_model_from_schema(self):
        """Simple model with basic fields is reconstructed correctly."""

        class TaskItem(BaseModel):
            category: str
            title: str

        schema = TaskItem.model_json_schema()
        models = _models_from_schema(schema)

        assert "TaskItem" in models
        Model = models["TaskItem"]

        # Verify field names and types
        assert set(Model.model_fields.keys()) == {"category", "title"}

        # Test instantiation
        instance = Model(category="Test", title="My Task")
        assert instance.category == "Test"
        assert instance.title == "My Task"

    def test_model_with_optional_fields(self):
        """Model with optional fields is reconstructed correctly."""
        from typing import Optional

        class Item(BaseModel):
            name: str
            description: Optional[str] = None

        schema = Item.model_json_schema()
        models = _models_from_schema(schema)

        Model = models["Item"]

        # Test with optional field omitted
        instance1 = Model(name="Widget")
        assert instance1.name == "Widget"
        assert instance1.description is None

        # Test with optional field provided
        instance2 = Model(name="Widget", description="A useful widget")
        assert instance2.description == "A useful widget"

    def test_model_with_list_fields(self):
        """Model with list fields is reconstructed correctly."""
        from typing import List

        class Tags(BaseModel):
            items: List[str]
            counts: List[int]

        schema = Tags.model_json_schema()
        models = _models_from_schema(schema)

        Model = models["Tags"]
        instance = Model(items=["a", "b"], counts=[1, 2, 3])
        assert instance.items == ["a", "b"]
        assert instance.counts == [1, 2, 3]

    def test_nested_model_from_schema(self):
        """Nested models with $defs are reconstructed correctly."""

        class Address(BaseModel):
            street: str
            city: str

        class Person(BaseModel):
            name: str
            address: Address

        schema = Person.model_json_schema()
        models = _models_from_schema(schema)

        # Both models should be created
        assert "Person" in models
        assert "Address" in models

        # Test instantiation with nested data
        PersonModel = models["Person"]
        AddressModel = models["Address"]

        addr = AddressModel(street="123 Main St", city="NYC")
        person = PersonModel(name="Alice", address=addr)
        assert person.name == "Alice"
        assert person.address.street == "123 Main St"

    def test_deeply_nested_model(self):
        """Deeply nested models are reconstructed correctly."""

        class Country(BaseModel):
            name: str
            code: str

        class Address(BaseModel):
            street: str
            country: Country

        class Person(BaseModel):
            name: str
            address: Address

        schema = Person.model_json_schema()
        models = _models_from_schema(schema)

        assert "Person" in models
        assert "Address" in models
        assert "Country" in models

    def test_model_with_list_of_nested_models(self):
        """Model with list of nested models is reconstructed correctly."""
        from typing import List

        class LineItem(BaseModel):
            product: str
            quantity: int

        class Order(BaseModel):
            order_id: str
            items: List[LineItem]

        schema = Order.model_json_schema()
        models = _models_from_schema(schema)

        assert "Order" in models
        assert "LineItem" in models

        OrderModel = models["Order"]
        LineItemModel = models["LineItem"]

        items = [
            LineItemModel(product="Widget", quantity=2),
            LineItemModel(product="Gadget", quantity=1),
        ]
        order = OrderModel(order_id="ORD-123", items=items)
        assert len(order.items) == 2
        assert order.items[0].product == "Widget"

    def test_model_with_all_primitive_types(self):
        """Model with all supported primitive types is reconstructed."""

        class AllTypes(BaseModel):
            text: str
            number: int
            decimal: float
            flag: bool

        schema = AllTypes.model_json_schema()
        models = _models_from_schema(schema)

        Model = models["AllTypes"]
        instance = Model(text="hello", number=42, decimal=3.14, flag=True)
        assert instance.text == "hello"
        assert instance.number == 42
        assert instance.decimal == 3.14
        assert instance.flag is True

    def test_enum_to_literal(self):
        """Enum types in JSON schema are converted to Literal."""
        schema = {
            "title": "Priority",
            "properties": {
                "level": {"enum": ["p1", "p2", "p3", "p4"], "type": "string"},
            },
            "required": ["level"],
        }
        models = _models_from_schema(schema)
        Model = models["Priority"]
        instance = Model(level="p1")
        assert instance.level == "p1"

    def test_type_array_shorthand_for_optional(self):
        """Type-array shorthand {"type": ["string", "null"]} → Optional[str]."""
        schema = {
            "title": "Item",
            "properties": {
                "name": {"type": "string"},
                "note": {"type": ["string", "null"]},
            },
            "required": ["name"],
        }
        models = _models_from_schema(schema)
        Model = models["Item"]
        instance = Model(name="test", note=None)
        assert instance.name == "test"
        assert instance.note is None

    def test_schema_without_title_gets_fallback(self):
        """Schema without a title key uses the fallback name."""
        schema = {
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        models = _models_from_schema(schema)
        assert "RootModel" in models

    @pytest.mark.asyncio
    async def test_predict_with_pydantic_schemas(self):
        """predict tool uses pydantic_schemas to create custom_types."""
        mock_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=mock_lm, max_iterations=5)

        # Create a schema that would come from sandbox
        class TaskItem(BaseModel):
            category: str
            title: str

        pydantic_schemas = {"TaskItem": TaskItem.model_json_schema()}

        with patch("predict_rlm.predict_rlm.dspy.Predict") as mock_predict_class:
            with patch("predict_rlm.predict_rlm.dspy.Signature") as mock_sig_class:
                mock_predictor = MagicMock()
                mock_prediction = MagicMock()
                mock_prediction.keys.return_value = ["tasks"]
                mock_prediction.tasks = [{"category": "Test", "title": "Task 1"}]
                mock_predictor.acall = AsyncMock(return_value=mock_prediction)
                mock_predict_class.return_value = mock_predictor
                mock_sig_class.return_value = "mocked_signature"

                _ = await rlm.tools["predict"].func(
                    "text: str -> tasks: list[TaskItem]",
                    pydantic_schemas=pydantic_schemas,
                    text="test input",
                )

                # Verify Signature was called with custom_types
                assert mock_sig_class.call_count == 1
                call_args = mock_sig_class.call_args
                assert "custom_types" in call_args.kwargs
                custom_types = call_args.kwargs["custom_types"]
                assert "TaskItem" in custom_types

    @pytest.mark.asyncio
    async def test_predict_without_pydantic_schemas_no_custom_types(self):
        """predict without pydantic_schemas parses signature without custom_types."""
        mock_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=mock_lm, max_iterations=5)

        with patch("predict_rlm.predict_rlm.dspy.Predict") as mock_predict_class:
            mock_predictor = MagicMock()
            mock_prediction = MagicMock()
            mock_prediction.keys.return_value = ["answer"]
            mock_prediction.answer = "Test"
            mock_predictor.acall = AsyncMock(return_value=mock_prediction)
            mock_predict_class.return_value = mock_predictor

            _ = await rlm.tools["predict"].func(
                "question -> answer",
                question="What is 2+2?",
            )

            # Predict should be called once with a parsed Signature
            mock_predict_class.assert_called_once()
            call_args = mock_predict_class.call_args
            sig = call_args[0][0]
            # Check it's a Signature object with expected fields
            assert hasattr(sig, "input_fields")
            assert hasattr(sig, "output_fields")
            assert "question" in sig.input_fields
            assert "answer" in sig.output_fields

    @pytest.mark.asyncio
    async def test_predict_handles_items_field_without_collision(self):
        """predict returns correct value when output field is named 'items'.

        Regression test: using getattr(prediction, 'items') returns the .items()
        method instead of the field value. The fix uses prediction['items'] via
        __getitem__ which bypasses method lookup.
        """
        mock_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=mock_lm, max_iterations=5)

        with patch("predict_rlm.predict_rlm.dspy.Predict") as mock_predict_class:
            mock_predictor = MagicMock()
            mock_prediction = MagicMock()
            mock_prediction.keys.return_value = ["items"]
            # Set up __getitem__ to return the actual value
            expected_items = [{"title": "Task 1"}, {"title": "Task 2"}]
            mock_prediction.__getitem__ = MagicMock(return_value=expected_items)
            mock_predictor.acall = AsyncMock(return_value=mock_prediction)
            mock_predict_class.return_value = mock_predictor

            result = await rlm.tools["predict"].func(
                "page: dspy.Image -> items: list[dict]",
                page="https://example.com/page.png",
            )

            assert isinstance(result, dict)
            assert "items" in result
            # Should return the list, not {} or [] from method collision
            assert result["items"] == expected_items
            mock_prediction.__getitem__.assert_called_once_with("items")


class TestAnnotationHelpers:
    """Tests for _unwrap_optional, _image_field_info, _allows_none, _is_list_output.

    These are inner functions of _create_predict_tool, so we test them
    indirectly through predict() behavior.
    """

    @pytest.mark.asyncio
    async def test_optional_image_none_passes_through(self):
        """Optional[dspy.Image] field with None value passes through without wrapping."""
        mock_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=mock_lm, max_iterations=5)

        with patch("predict_rlm.predict_rlm.dspy.Predict") as mock_predict_class:
            mock_predictor = MagicMock()
            mock_prediction = MagicMock()
            values = {"answer": "no image"}
            mock_prediction.keys.return_value = list(values.keys())
            mock_prediction.__getitem__ = MagicMock(side_effect=lambda k: values[k])
            mock_predictor.acall = AsyncMock(return_value=mock_prediction)
            mock_predict_class.return_value = mock_predictor

            result = await rlm.tools["predict"].func(
                "image: Optional[dspy.Image], question -> answer",
                image=None,
                question="Any image?",
            )

            assert result == {"answer": "no image"}
            call_kwargs = mock_predictor.acall.call_args.kwargs
            assert call_kwargs["image"] is None

    @pytest.mark.asyncio
    async def test_optional_list_image_wraps_correctly(self):
        """Optional[list[dspy.Image]] wraps URLs as dspy.Image when list is provided."""
        mock_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=mock_lm, max_iterations=5)

        with patch("predict_rlm.predict_rlm.dspy.Predict") as mock_predict_class:
            mock_predictor = MagicMock()
            mock_prediction = MagicMock()
            values = {"answer": "found images"}
            mock_prediction.keys.return_value = list(values.keys())
            mock_prediction.__getitem__ = MagicMock(side_effect=lambda k: values[k])
            mock_predictor.acall = AsyncMock(return_value=mock_prediction)
            mock_predict_class.return_value = mock_predictor

            result = await rlm.tools["predict"].func(
                "images: Optional[list[dspy.Image]], question -> answer",
                images=["https://example.com/a.png", "https://example.com/b.png"],
                question="Describe these",
            )

            assert result == {"answer": "found images"}
            call_kwargs = mock_predictor.acall.call_args.kwargs
            assert len(call_kwargs["images"]) == 2
            assert all(isinstance(img, dspy.Image) for img in call_kwargs["images"])

    @pytest.mark.asyncio
    async def test_none_for_optional_list_output_passes_through(self):
        """None for Optional[list[str]] output passes through (allows_none is True)."""
        mock_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=mock_lm, max_iterations=5)

        with patch("predict_rlm.predict_rlm.dspy.Predict") as mock_predict_class:
            mock_predictor = MagicMock()
            mock_prediction = MagicMock()
            values = {"items": None}
            mock_prediction.keys.return_value = list(values.keys())
            mock_prediction.__getitem__ = MagicMock(side_effect=lambda k: values[k])
            mock_predictor.acall = AsyncMock(return_value=mock_prediction)
            mock_predict_class.return_value = mock_predictor

            result = await rlm.tools["predict"].func(
                "text: str -> items: Optional[list[str]]",
                text="some input",
            )
            assert result["items"] is None

    @pytest.mark.asyncio
    async def test_is_list_output_detects_list_type(self):
        """list[str] (non-Optional) output field: None raises RuntimeError (is_list + not allows_none)."""
        mock_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=mock_lm, max_iterations=5)

        with patch("predict_rlm.predict_rlm.dspy.Predict") as mock_predict_class:
            mock_predictor = MagicMock()
            mock_prediction = MagicMock()
            values = {"tags": None}
            mock_prediction.keys.return_value = list(values.keys())
            mock_prediction.__getitem__ = MagicMock(side_effect=lambda k: values[k])
            mock_predictor.acall = AsyncMock(return_value=mock_prediction)
            mock_predict_class.return_value = mock_predictor

            with pytest.raises(RuntimeError, match="VLM returned None for non-Optional"):
                await rlm.tools["predict"].func(
                    "text: str -> tags: list[str]",
                    text="some input",
                )

    @pytest.mark.asyncio
    async def test_non_optional_non_list_str_allows_none_false(self):
        """Plain str output (not Optional): None raises RuntimeError."""
        mock_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=mock_lm, max_iterations=5)

        with patch("predict_rlm.predict_rlm.dspy.Predict") as mock_predict_class:
            mock_predictor = MagicMock()
            mock_prediction = MagicMock()
            values = {"name": None}
            mock_prediction.keys.return_value = list(values.keys())
            mock_prediction.__getitem__ = MagicMock(side_effect=lambda k: values[k])
            mock_predictor.acall = AsyncMock(return_value=mock_prediction)
            mock_predict_class.return_value = mock_predictor

            with pytest.raises(RuntimeError, match="VLM returned None for non-Optional"):
                await rlm.tools["predict"].func(
                    "text: str -> name: str",
                    text="some input",
                )


class TestSchemaTitleInjection:
    """Tests for schema title injection when pydantic_schemas lack a 'title' key."""

    @pytest.mark.asyncio
    async def test_schema_without_title_injects_key_name(self):
        """When pydantic_schemas has a schema missing 'title', the key name is injected."""
        mock_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=mock_lm, max_iterations=5)

        schema_without_title = {
            "properties": {
                "description": {"type": "string"},
                "amount": {"type": "number"},
            },
            "required": ["description", "amount"],
        }

        with patch("predict_rlm.predict_rlm.dspy.Predict") as mock_predict_class:
            with patch("predict_rlm.predict_rlm.dspy.Signature") as mock_sig_class:
                mock_predictor = MagicMock()
                mock_prediction = MagicMock()
                values = {"items": []}
                mock_prediction.keys.return_value = list(values.keys())
                mock_prediction.__getitem__ = MagicMock(side_effect=lambda k: values[k])
                mock_predictor.acall = AsyncMock(return_value=mock_prediction)
                mock_predict_class.return_value = mock_predictor
                mock_sig_class.return_value = "mocked_signature"

                await rlm.tools["predict"].func(
                    "text: str -> items: list[LineItem]",
                    pydantic_schemas={"LineItem": schema_without_title},
                    text="test",
                )

                call_args = mock_sig_class.call_args
                custom_types = call_args.kwargs["custom_types"]
                assert "LineItem" in custom_types

    @pytest.mark.asyncio
    async def test_schema_with_title_preserved(self):
        """When pydantic_schemas already has 'title', it is preserved (no overwrite)."""
        mock_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=mock_lm, max_iterations=5)

        schema_with_title = {
            "title": "LineItem",
            "properties": {
                "description": {"type": "string"},
            },
            "required": ["description"],
        }

        with patch("predict_rlm.predict_rlm.dspy.Predict") as mock_predict_class:
            with patch("predict_rlm.predict_rlm.dspy.Signature") as mock_sig_class:
                mock_predictor = MagicMock()
                mock_prediction = MagicMock()
                values = {"items": []}
                mock_prediction.keys.return_value = list(values.keys())
                mock_prediction.__getitem__ = MagicMock(side_effect=lambda k: values[k])
                mock_predictor.acall = AsyncMock(return_value=mock_prediction)
                mock_predict_class.return_value = mock_predictor
                mock_sig_class.return_value = "mocked_signature"

                await rlm.tools["predict"].func(
                    "text: str -> items: list[LineItem]",
                    pydantic_schemas={"LineItem": schema_with_title},
                    text="test",
                )

                call_args = mock_sig_class.call_args
                custom_types = call_args.kwargs["custom_types"]
                assert "LineItem" in custom_types


class TestUnresolvedTypesFallback:
    """Tests for the fallback when signature has custom types that can't be resolved."""

    @pytest.mark.asyncio
    async def test_unresolved_custom_type_falls_back_to_string_signature(self, caplog):
        """Unresolvable custom type in signature falls back to string signature with warning."""
        mock_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=mock_lm, max_iterations=5)

        with patch("predict_rlm.predict_rlm.dspy.Predict") as mock_predict_class:
            mock_predictor = MagicMock()
            mock_prediction = MagicMock()
            values = {"items": "raw string fallback"}
            mock_prediction.keys.return_value = list(values.keys())
            mock_prediction.__getitem__ = MagicMock(side_effect=lambda k: values[k])
            mock_predictor.acall = AsyncMock(return_value=mock_prediction)
            mock_predict_class.return_value = mock_predictor

            # Patch dspy.Signature to raise on unknown type, simulating a parse failure
            original_sig = dspy.Signature

            def sig_side_effect(*args, **kwargs):
                if kwargs.get("custom_types"):
                    return original_sig(*args, **kwargs)
                sig_str = args[0] if args else ""
                if "UnknownModel" in sig_str:
                    raise ValueError("Unknown name 'UnknownModel'")
                return original_sig(*args, **kwargs)

            with patch(
                "predict_rlm.predict_rlm.dspy.Signature", side_effect=sig_side_effect
            ):
                with caplog.at_level(logging.WARNING, logger="predict_rlm.predict_rlm"):
                    result = await rlm.tools["predict"].func(
                        "text: str -> items: list[UnknownModel]",
                        text="some input",
                    )

                assert result == {"items": "raw string fallback"}
                # Verify dspy.Predict was called with a string (the fallback)
                sig_arg = mock_predict_class.call_args[0][0]
                assert isinstance(sig_arg, str)
                # Verify warning was logged about the fallback
                assert any("UnknownModel" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_non_unknown_name_error_still_falls_back(self):
        """Signature parse error without 'Unknown name' still falls back to string."""
        mock_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=mock_lm, max_iterations=5)

        with patch("predict_rlm.predict_rlm.dspy.Predict") as mock_predict_class:
            mock_predictor = MagicMock()
            mock_prediction = MagicMock()
            values = {"answer": "fallback"}
            mock_prediction.keys.return_value = list(values.keys())
            mock_prediction.__getitem__ = MagicMock(side_effect=lambda k: values[k])
            mock_predictor.acall = AsyncMock(return_value=mock_prediction)
            mock_predict_class.return_value = mock_predictor

            with patch(
                "predict_rlm.predict_rlm.dspy.Signature",
                side_effect=Exception("some parse error"),
            ):
                result = await rlm.tools["predict"].func(
                    "question -> answer",
                    question="test",
                )

                assert result == {"answer": "fallback"}
                sig_arg = mock_predict_class.call_args[0][0]
                assert isinstance(sig_arg, str)


class TestAexecuteIteration:
    """Tests for _aexecute_iteration: async vs sync interpreter dispatch."""

    @pytest.mark.asyncio
    async def test_uses_aexecute_when_available(self):
        """_aexecute_iteration calls repl.aexecute() when it has the method."""
        mock_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=mock_lm, max_iterations=5)

        mock_repl = MagicMock()
        mock_repl.aexecute = AsyncMock(return_value="output from aexecute")

        mock_pred = MagicMock()
        mock_pred.reasoning = "thinking"
        mock_pred.code = "print('hello')"

        rlm.generate_action = MagicMock()
        rlm.generate_action.acall = AsyncMock(return_value=mock_pred)

        mock_result = MagicMock()
        with patch.object(rlm, "_process_execution_result", return_value=mock_result) as mock_process:
            with patch("dspy.predict.rlm._strip_code_fences", return_value="print('hello')"):
                result = await rlm._aexecute_iteration(
                    repl=mock_repl,
                    variables=[],
                    history=[],
                    iteration=0,
                    input_args={},
                    output_field_names=["answer"],
                )

        mock_repl.aexecute.assert_called_once_with("print('hello')", variables={})
        assert result is mock_result
        mock_process.assert_called_once_with(
            mock_pred, "output from aexecute", [], ["answer"]
        )

    @pytest.mark.asyncio
    async def test_falls_back_to_execute_when_no_aexecute(self):
        """_aexecute_iteration falls back to repl.execute() when aexecute is absent."""
        mock_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=mock_lm, max_iterations=5)

        mock_repl = MagicMock(spec=[])  # empty spec = no attributes
        mock_repl.execute = MagicMock(return_value="output from execute")

        mock_pred = MagicMock()
        mock_pred.reasoning = "thinking"
        mock_pred.code = "print('hi')"

        rlm.generate_action = MagicMock()
        rlm.generate_action.acall = AsyncMock(return_value=mock_pred)

        mock_result = MagicMock()
        with patch.object(rlm, "_process_execution_result", return_value=mock_result):
            with patch("dspy.predict.rlm._strip_code_fences", return_value="print('hi')"):
                result = await rlm._aexecute_iteration(
                    repl=mock_repl,
                    variables=[],
                    history=[],
                    iteration=0,
                    input_args={},
                    output_field_names=["answer"],
                )

        mock_repl.execute.assert_called_once_with("print('hi')", variables={})
        assert result is mock_result

    @pytest.mark.asyncio
    async def test_catches_execution_exception(self):
        """_aexecute_iteration catches exceptions from repl and formats as error."""
        mock_lm = MagicMock()
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=mock_lm, max_iterations=5)

        mock_repl = MagicMock()
        mock_repl.aexecute = AsyncMock(side_effect=RuntimeError("sandbox crashed"))

        mock_pred = MagicMock()
        mock_pred.reasoning = "thinking"
        mock_pred.code = "bad_code()"

        rlm.generate_action = MagicMock()
        rlm.generate_action.acall = AsyncMock(return_value=mock_pred)

        mock_result = MagicMock()
        with patch.object(rlm, "_process_execution_result", return_value=mock_result) as mock_process:
            with patch("dspy.predict.rlm._strip_code_fences", return_value="bad_code()"):
                await rlm._aexecute_iteration(
                    repl=mock_repl,
                    variables=[],
                    history=[],
                    iteration=0,
                    input_args={},
                    output_field_names=["answer"],
                )

        # The error should be formatted as "[Error] ..."
        error_arg = mock_process.call_args[0][1]
        assert "[Error]" in error_arg
        assert "sandbox crashed" in error_arg


class TestSkillsMergeIntoInit:
    """Tests for skills merging into PredictRLM.__init__."""

    def test_skills_merge_instructions(self):
        """Skills instructions are merged into _skill_instructions."""
        skill = Skill(
            name="test-skill",
            instructions="Use the frobnicator for all extraction.",
        )
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=None, max_iterations=1, skills=[skill])
        assert "frobnicator" in rlm._skill_instructions
        assert "test-skill" in rlm._skill_instructions

    def test_skills_merge_packages(self):
        """Skills packages are merged into _skill_packages."""
        skill = Skill(
            name="pkg-skill",
            packages=["pdfplumber", "pillow"],
        )
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=None, max_iterations=1, skills=[skill])
        assert "pdfplumber" in rlm._skill_packages
        assert "pillow" in rlm._skill_packages

    def test_skills_merge_modules(self):
        """Skills modules are merged into _skill_modules."""
        skill = Skill(
            name="mod-skill",
            modules={"helpers": "/path/to/helpers.py"},
        )
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=None, max_iterations=1, skills=[skill])
        assert rlm._skill_modules == {"helpers": "/path/to/helpers.py"}

    def test_skills_merge_tools(self):
        """Skills tools are accessible on the RLM alongside predict."""
        def my_tool(x: str) -> str:
            """A custom tool."""
            return x

        skill = Skill(
            name="tool-skill",
            tools={"my_tool": my_tool},
        )
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=None, max_iterations=1, skills=[skill])
        assert "my_tool" in rlm.tools
        assert "predict" in rlm.tools

    def test_skill_tool_conflicts_with_user_tool_raises(self):
        """Tool name conflict between a skill and the tools parameter raises ValueError."""
        def my_tool(x: str) -> str:
            """A tool."""
            return x

        skill = Skill(
            name="conflict-skill",
            tools={"my_tool": my_tool},
        )
        with pytest.raises(ValueError, match="Tool name conflict.*my_tool"):
            PredictRLM(
                ImageAnalysisSignature,
                sub_lm=None,
                max_iterations=1,
                skills=[skill],
                tools={"my_tool": my_tool},
            )

    def test_multiple_skills_merge(self):
        """Multiple skills have their instructions, packages, and tools merged."""
        def tool_a() -> str:
            """Tool A."""
            return "a"

        def tool_b() -> str:
            """Tool B."""
            return "b"

        skill_a = Skill(
            name="skill-a",
            instructions="Use approach A.",
            packages=["pkg-a"],
            tools={"tool_a": tool_a},
        )
        skill_b = Skill(
            name="skill-b",
            instructions="Use approach B.",
            packages=["pkg-b", "pkg-a"],  # duplicate pkg-a
            tools={"tool_b": tool_b},
        )
        rlm = PredictRLM(
            ImageAnalysisSignature,
            sub_lm=None,
            max_iterations=1,
            skills=[skill_a, skill_b],
        )
        assert "approach A" in rlm._skill_instructions
        assert "approach B" in rlm._skill_instructions
        assert "pkg-a" in rlm._skill_packages
        assert "pkg-b" in rlm._skill_packages
        # Deduplicated packages
        assert rlm._skill_packages.count("pkg-a") == 1
        assert "tool_a" in rlm.tools
        assert "tool_b" in rlm.tools
        assert "predict" in rlm.tools

    def test_no_skills_leaves_defaults_empty(self):
        """Without skills, skill fields are empty."""
        rlm = PredictRLM(ImageAnalysisSignature, sub_lm=None, max_iterations=1)
        assert rlm._skill_instructions == ""
        assert rlm._skill_packages == []
        assert rlm._skill_modules == {}


class TestModelsFromSchemaEdgeCases:
    """Tests for _models_from_schema edge cases: unknown $ref, anyOf with all nulls."""

    def test_unknown_ref_falls_back_to_dict(self):
        """$ref to a name not in $defs falls back to dict type."""
        schema = {
            "title": "Container",
            "properties": {
                "data": {"$ref": "#/$defs/MissingModel"},
            },
            "required": ["data"],
        }
        models = _models_from_schema(schema)
        Model = models["Container"]
        # Should accept a dict since the ref fell back to dict type
        instance = Model(data={"key": "value"})
        assert instance.data == {"key": "value"}

    def test_anyof_all_null_falls_back_to_optional_str(self):
        """anyOf with only null types falls back to Optional[str]."""
        schema = {
            "title": "Weird",
            "properties": {
                "field": {"anyOf": [{"type": "null"}]},
            },
            "required": ["field"],
        }
        models = _models_from_schema(schema)
        Model = models["Weird"]
        instance = Model(field=None)
        assert instance.field is None
