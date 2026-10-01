"""ContractComparator — RLM service for comparing PDF contracts.

Usage::

    import dspy
    from dspy_codex_lm import CodexLM
    from predict_rlm import File

    contracts = [File(path="contract-v1.pdf"), File(path="contract-v2.pdf")]
    comparator = ContractComparator(sub_lm=CodexLM("gpt-5.4-mini"))
    with dspy.context(lm=CodexLM("gpt-5.5")):
        result = await comparator.aforward(contracts=contracts)
    # result — ComparisonResult with report, section diffs, and key differences
"""

from pathlib import Path

import dspy

from predict_rlm import File, PredictRLM
from predict_rlm.skills import pdf as pdf_skill

from .signature import CompareContracts


class ContractComparator(dspy.Module):
    """DSPy Module that wraps CompareContracts + PredictRLM."""

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

    async def aforward(self, contracts: list[File]):
        """Run comparison and return the ComparisonResult."""
        predictor = PredictRLM(
            CompareContracts,
            sub_lm=self.sub_lm,
            skills=[pdf_skill],
            max_iterations=self.max_iterations,
            verbose=self.verbose,
            debug=self.debug,
            run_export_root=Path(__file__).resolve().parent / ".run",
        )
        prediction = await predictor.acall(contracts=contracts)
        return prediction.result
