"""DocumentRedactor — RLM service for redacting sensitive content from PDFs.

Usage::

    import dspy
    from dspy_codex_lm import CodexLM
    from predict_rlm import File

    documents = [File(path="contract.pdf"), File(path="appendix.pdf")]
    redactor = DocumentRedactor(sub_lm=CodexLM("gpt-5.6-terra"))
    with dspy.context(lm=CodexLM("gpt-5.6-terra")):
        prediction = await redactor.aforward(
            documents=documents,
            criteria="Redact all personal names, phone numbers, and email addresses.",
        )
    # prediction.result — RedactionResult with counts and targets
    # prediction.redacted_documents — list[File] with redacted PDFs
"""

from pathlib import Path

import dspy

from predict_rlm import File, PredictRLM

from .signature import RedactDocuments
from .skills import pdf_skill, redaction_skill


class DocumentRedactor(dspy.Module):
    """DSPy Module that wraps RedactDocuments + PredictRLM."""

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
        """Run redaction and return the prediction.

        Returns a dspy.Prediction with:
        - result: RedactionResult with counts and targets
        - redacted_documents: list[File] with redacted PDFs
        """
        signature = RedactDocuments.with_instructions(
            RedactDocuments.instructions
            + "\n\n# Redaction Criteria\n\n"
            + criteria.strip()
        )
        predictor = PredictRLM(
            signature,
            sub_lm=self.sub_lm,
            skills=[pdf_skill, redaction_skill],
            max_iterations=self.max_iterations,
            verbose=self.verbose,
            debug=self.debug,
            run_export_root=Path(__file__).resolve().parent / ".run",
        )
        return await predictor.acall(documents=documents)
