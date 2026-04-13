import dspy

from predict_rlm import File


class ManipulateSpreadsheet(dspy.Signature):
    """Manipulate a spreadsheet according to the given instruction.

    1. Load the input spreadsheet from /sandbox/input/input_spreadsheet/.
    2. Inspect its structure (sheets, columns, data types, row count).
    3. Follow the instruction literally. Write Python code using openpyxl.
    4. Save to /sandbox/output/output_spreadsheet/ with the same filename.
    5. Verify the output before submitting.
    """

    input_spreadsheet: File = dspy.InputField(desc="The input .xlsx spreadsheet file to manipulate, mounted at /sandbox/input/input_spreadsheet/")
    instruction: str = dspy.InputField(desc="Natural language description of the spreadsheet manipulation to perform")

    output_spreadsheet: File = dspy.OutputField(desc="The modified .xlsx spreadsheet file, saved to /sandbox/output/output_spreadsheet/")
