"""ImageAnalyzer — RLM service for analyzing images with natural language queries.

Usage::

    from predict_rlm import File

    images = [File(path="photo1.png"), File(path="photo2.jpg")]
    analyzer = ImageAnalyzer(sub_lm="openai/gpt-5.1")
    prediction = await analyzer.aforward(images=images, query="What do you see?")
    # prediction.answer — str with the analysis
"""

import dspy

from predict_rlm import File, PredictRLM

from .signature import AnalyzeImages


class ImageAnalyzer(dspy.Module):
    """DSPy Module that wraps AnalyzeImages + PredictRLM."""

    def __init__(
        self,
        sub_lm: dspy.LM | str | None = None,
        max_iterations: int = 30,
        verbose: bool = False,
        debug: bool = False,
    ):
        self.sub_lm = sub_lm
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.debug = debug

    async def aforward(self, images: list[File], query: str):
        """Analyze images and return the prediction.

        Returns a dspy.Prediction with:
        - answer: str with the analysis
        """
        predictor = PredictRLM(
            AnalyzeImages,
            sub_lm=self.sub_lm,
            max_iterations=self.max_iterations,
            verbose=self.verbose,
            debug=self.debug,
        )
        return await predictor.acall(images=images, query=query)
