"""DocumentAnalyzer — example RLM service for document analysis.

Usage::

    from predict_rlm import File

    documents = [File(path="report.pdf"), File(path="appendix.pdf")]
    analyzer = DocumentAnalyzer(sub_lm="openai/gpt-5.1")
    result = await analyzer.aforward(
        documents=documents,
        criteria="Extract key dates, entities, and a summary.",
    )
    print(result.report)
"""

import dspy

from predict_rlm import File, PredictRLM
from predict_rlm.skills import pdf as pdf_skill

from .schema import DocumentAnalysis
from .signature import AnalyzeDocuments


class DocumentAnalyzer(dspy.Module):
    """DSPy Module that wraps AnalyzeDocuments + PredictRLM."""

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

    async def aforward(
        self, documents: list[File], criteria: str
    ) -> DocumentAnalysis:
        signature = AnalyzeDocuments.with_instructions(
            AnalyzeDocuments.instructions + "\n\n# Task\n\n" + criteria.strip()
        )
        predictor = PredictRLM(
            signature,
            sub_lm=self.sub_lm,
            skills=[pdf_skill],
            max_iterations=self.max_iterations,
            verbose=self.verbose,
            debug=self.debug,
        )
        result = await predictor.acall(documents=documents)
        return result.analysis
