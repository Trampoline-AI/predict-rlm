"""DocumentAnalyzer — example RLM service for document analysis.

Usage::

    import dspy
    from dspy_codex_lm import CodexLM
    from predict_rlm import File

    documents = [File(path="report.pdf"), File(path="appendix.pdf")]
    analyzer = DocumentAnalyzer(sub_lm=CodexLM("gpt-5.6-terra"))
    with dspy.context(lm=CodexLM("gpt-5.6-terra")):
        prediction = await analyzer.aforward(
            documents=documents,
            criteria="Extract key dates, entities, and a summary.",
        )
    print(prediction.result.report)
"""

from pathlib import Path

import dspy

from predict_rlm import File, PredictRLM
from predict_rlm.capabilities import docx as docx_capability
from predict_rlm.capabilities import pdf as pdf_capability

from .signature import AnalyzeDocuments


class DocumentAnalyzer(dspy.Module):
    """DSPy Module that wraps AnalyzeDocuments + PredictRLM."""

    def __init__(
        self,
        sub_lm: dspy.LM | str | None = None,
        max_iterations: int = 30,
        verbose: bool = True,
        debug: bool = False,
    ):
        self.sub_lm = sub_lm
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.debug = debug

    async def aforward(self, documents: list[File], criteria: str):
        signature = AnalyzeDocuments.with_instructions(
            AnalyzeDocuments.instructions + "\n\n# Task\n\n" + criteria.strip()
        )
        predictor = PredictRLM(
            signature,
            sub_lm=self.sub_lm,
            capabilities=[pdf_capability, docx_capability],
            max_iterations=self.max_iterations,
            verbose=self.verbose,
            debug=self.debug,
            sandbox_backend="sbx",
            run_export_root=Path(__file__).resolve().parent / ".run",
        )
        return await predictor.acall(documents=documents)
