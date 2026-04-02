"""InvoiceProcessor — RLM service for extracting data from PDF invoices.

Usage::

    from predict_rlm import File

    invoices = [File(path="invoice1.pdf"), File(path="invoice2.pdf")]
    processor = InvoiceProcessor(sub_lm="openai/gpt-5.1")
    prediction = await processor.aforward(invoices=invoices)
    # prediction.result — InvoiceExtractionResult with structured invoice data
    # prediction.workbook — File with the Excel workbook
"""

import dspy

from predict_rlm import File, PredictRLM
from predict_rlm.skills import pdf as pdf_skill
from predict_rlm.skills import spreadsheet as spreadsheet_skill

from .signature import ProcessInvoices


class InvoiceProcessor(dspy.Module):
    """DSPy Module that wraps ProcessInvoices + PredictRLM."""

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

    async def aforward(self, invoices: list[File]):
        """Run invoice processing and return the prediction.

        Returns a dspy.Prediction with:
        - result: InvoiceExtractionResult with invoice details and totals
        - workbook: File with the Excel workbook
        """
        predictor = PredictRLM(
            ProcessInvoices,
            sub_lm=self.sub_lm,
            skills=[pdf_skill, spreadsheet_skill],
            max_iterations=self.max_iterations,
            verbose=self.verbose,
            debug=self.debug,
        )
        return await predictor.acall(invoices=invoices)
