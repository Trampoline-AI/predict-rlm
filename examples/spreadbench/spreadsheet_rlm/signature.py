import dspy

from predict_rlm import File


class ManipulateSpreadsheet(dspy.Signature):
    """Manipulate a spreadsheet according to the given instruction.

    1. **Find the input spreadsheet** at /sandbox/input/input_spreadsheet/.
       List that directory to get the exact filename, then load it with openpyxl.

    2. **Inspect the spreadsheet** to understand its structure: sheet names,
       column headers, data types, and row count. Print them so you can
       reference them later.

    3. **Understand the instruction** and determine what manipulation is
       needed — formulas, filtering, formatting, conditional formatting, etc.
       **Follow instructions literally.** If the instruction says to write to
       column C, write to column C — even if column C doesn't exist yet. Do not
       reinterpret the target location, column, or sheet based on what seems
       more logical from the data. Create missing columns, sheets, or ranges as
       needed. The instruction defines *what* to do; the data defines *how*.

    4. **Write Python code** using openpyxl to perform the manipulation.
       When writing formulas, keep them simple (SUM, IF, VLOOKUP on
       data cells). When the logic is complex, compute the result in
       Python and write the value directly instead. Save the modified
       file to /sandbox/output/output_spreadsheet/ with the **same filename**.

    5. **Preserve sheet names exactly.** Never rename sheets unless the
       instruction explicitly asks you to. The output must have the same
       sheet names as the input.

    6. **Never write VBA code or explanations into cells.** If the
       instruction mentions VBA, implement the equivalent transformation
       directly in Python — do not paste macro text into the spreadsheet.

    7. **Verify before submitting.** Reload the saved file with
       openpyxl and print the value of the first target cell you
       modified. If it is None, the formula failed or you wrote to the
       wrong cell. Rewrite using a Python-computed value instead,
       then save and check again.

    5. **Verify** the output file exists and spot-check that the changes
       look correct. Fix any issues before submitting.

    **IMPORTANT: When you call `SUBMIT()`, it must be the only thing in
    that turn — no other code, no file writes, no tool calls.**
    Use one turn to verify everything is correct, then a separate turn
    to submit.
    """

    input_spreadsheet: File = dspy.InputField(desc="The input .xlsx spreadsheet file to manipulate, mounted at /sandbox/input/input_spreadsheet/")
    instruction: str = dspy.InputField(desc="Natural language description of the spreadsheet manipulation to perform")

    output_spreadsheet: File = dspy.OutputField(desc="The modified .xlsx spreadsheet file, saved to /sandbox/output/output_spreadsheet/")
