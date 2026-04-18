import dspy

from predict_rlm import File


class ManipulateSpreadsheet(dspy.Signature):
    """Manipulate spreadsheets through Python code.

    Generate Python code that loads the input spreadsheet, follows the
    instruction, and saves the modified spreadsheet to the output location.

    1. Load the input spreadsheet from /sandbox/input/input_spreadsheet/.
    2. Inspect its structure (sheets, columns, data types, row count).
    3. You MUST use Python with openpyxl. Compute all values in Python and
       write the final numeric / string / boolean / date results directly
       to cells. Do NOT write VBA macros, Python source, Excel formulas as
       literal text, or explanatory prose into cells.
    4. Save to /sandbox/output/output_spreadsheet/ with the same filename.
    5. Verify the output before submitting.
    """

    input_spreadsheet: File = dspy.InputField(
        desc="The input .xlsx spreadsheet file to manipulate, "
        "mounted at /sandbox/input/input_spreadsheet/."
    )
    instruction: str = dspy.InputField(desc="The manipulation to perform.")

    output_spreadsheet: File = dspy.OutputField(
        desc="The modified .xlsx spreadsheet file, "
        "saved to /sandbox/output/output_spreadsheet/."
    )
