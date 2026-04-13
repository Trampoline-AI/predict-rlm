import dspy

from predict_rlm import File, PredictRLM
from .signature import ManipulateSpreadsheet

from .skills import libreoffice_spreadsheet_skill


class SpreadsheetRLM(dspy.Module):
    def __init__(
        self,
        lm: dspy.LM | str | None = None,
        sub_lm: dspy.LM | str | None = None,
        max_iterations: int = 30,
        verbose: bool = True,
        debug: bool = False,
    ):
        self.lm = lm
        self.sub_lm = sub_lm
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.debug = debug

    async def aforward(self, spreadsheet: File, instruction: str) -> File:
        predictor = PredictRLM(
            ManipulateSpreadsheet,
            lm=self.lm,
            sub_lm=self.sub_lm,
            skills=[libreoffice_spreadsheet_skill],
            max_iterations=self.max_iterations,
            verbose=self.verbose,
            debug=self.debug,
        )
        result = await predictor.acall(
            input_spreadsheet=spreadsheet,
            instruction=instruction,
        )
        return result.output_spreadsheet
