import dspy

from predict_rlm import File


class ManipulateSpreadsheet(dspy.Signature):
    """Manipulate a spreadsheet according to the given instruction.

    1. **Find the input spreadsheet** at /sandbox/input/input_spreadsheet/.
       List that directory to get the exact filename, then load it with openpyxl.

    2. **Inspect the spreadsheet** to understand its structure: sheet names,
       column headers, data types, and row count.

    3. **Understand the instruction** and determine what manipulation is
       needed — formulas, filtering, formatting, conditional formatting, etc.

    4. **Write Python code** using openpyxl to perform the manipulation.
       Load the workbook, apply the changes, and save the modified file
       to /sandbox/output/output_spreadsheet/ with the same filename.

    5. **Verify** the output file exists and spot-check that the changes
       look correct.
    """

    input_spreadsheet: File = dspy.InputField(desc="The input .xlsx spreadsheet file to manipulate, mounted at /sandbox/input/input_spreadsheet/")
    instruction: str = dspy.InputField(desc="Natural language description of the spreadsheet manipulation to perform")

    output_spreadsheet: File = dspy.OutputField(desc="The modified .xlsx spreadsheet file, saved to /sandbox/output/output_spreadsheet/")
