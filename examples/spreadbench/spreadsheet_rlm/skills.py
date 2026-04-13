"""Spreadsheet skill for LibreOffice-compatible workbook manipulation."""

import os
from pathlib import Path
from typing import Annotated

from predict_rlm import Skill, SyncedFile

from .recalculate import recalculate as _recalc_impl


def recalculate(file_path: Annotated[Path, SyncedFile()]) -> str:
    """Recalculate all formulas in an xlsx file and cache the results in place.

    Runs a two-stage pipeline: the Python `formulas` library first (fast,
    in-process), then LibreOffice headless as a fallback for any formulas
    the library couldn't evaluate. The winning candidate is whichever
    resolved the most formula cells — the untouched original is always
    a candidate too, so the call is strictly additive and never downgrades
    a file. After this call, re-open the file with
    `load_workbook(file_path, data_only=True)` to read the cached values.

    Args:
        file_path: Path to the .xlsx file to recalculate.

    Returns:
        "ok" on success (optionally annotated with the source that won and
        the resolved/total counts), or an error message starting with
        "Error:".
    """
    path = os.path.abspath(file_path)
    if not os.path.isfile(path):
        return f"Error: file not found: {path}"
    try:
        result = _recalc_impl(path)
    except Exception as e:
        return f"Error: {e}"

    status = (
        f"ok (source={result.source}, "
        f"resolved={result.resolved}/{result.total_formulas})"
    )
    if result.errors:
        status += f" [warnings: {'; '.join(result.errors)}]"
    return status


libreoffice_spreadsheet_skill = Skill(
    name="spreadsheet-libreoffice",
    instructions="""Use openpyxl and pandas to manipulate .xlsx spreadsheet files.

Output files are evaluated with **LibreOffice Calc**. This constrains which formulas work.

# Formulas vs Python

Use Excel formulas for per-cell calculations (sums, lookups, conditionals). Use Python to filter, sort, deduplicate, or populate sheets — write values row by row with openpyxl.

```python
# Per-cell formulas (good — stay live in LibreOffice)
ws["F25"] = "=SUM(F3:F24)"
ws["G10"] = "=IF(G8>0, G6/G8, 0)"
ws["B10"] = "=VLOOKUP(TRIM(A10), Sheet2!A:C, 3, FALSE)"

# Filtering/sorting (do in Python, write values directly)
for r in range(2, src.max_row + 1):
    if src.cell(r, 4).value > threshold:
        for c in range(1, src.max_column + 1):
            dest.cell(out_row, c).value = src.cell(r, c).value
        out_row += 1
```

Prefer Python-computed literal values over complex multi-step formulas for small, static datasets — unless the instruction explicitly asks for formulas.

# LibreOffice Formula Compatibility

## Forbidden (spill/dynamic-array — results not persisted)
`FILTER`, `SORT`, `UNIQUE`, `SORTBY`, `SEQUENCE`, `RANDARRAY`

Write every cell individually. If a formula should fill a range, loop and write to each cell.

## Forbidden (Excel 365-only — evaluate to `#NAME?`)
`LET`, `LAMBDA`, `MAP`, `REDUCE`, `SCAN`, `MAKEARRAY`, `BYROW`, `BYCOL`,
`TEXTSPLIT`, `TEXTBEFORE`, `TEXTAFTER`, `CHOOSECOLS`, `CHOOSEROWS`,
`TAKE`, `DROP`, `EXPAND`, `WRAPCOLS`, `WRAPROWS`, `TOCOL`, `TOROW`,
`VSTACK`, `HSTACK`

## Require `_xlfn.` prefix in LibreOffice
`IFNA`, `IFS`, `SWITCH`, `TEXTJOIN`, `AGGREGATE`, `MAXIFS`, `MINIFS`, `CONCAT`
Write as `=_xlfn.IFNA(...)` etc.

## Safe single-cell functions
`SUM`, `SUMIF`, `SUMIFS`, `SUMPRODUCT`, `AVERAGE`, `AVERAGEIF`, `MIN`, `MAX`,
`ROUND`, `ABS`, `IF`, `IFS`, `AND`, `OR`, `NOT`, `IFERROR`,
`VLOOKUP`, `HLOOKUP`, `XLOOKUP`, `INDEX`, `MATCH`, `OFFSET`, `INDIRECT`,
`COUNT`, `COUNTA`, `COUNTIF`, `COUNTIFS`, `COUNTBLANK`,
`LEFT`, `RIGHT`, `MID`, `LEN`, `TRIM`, `TEXT`, `VALUE`, `FIND`, `SUBSTITUTE`,
`DATE`, `YEAR`, `MONTH`, `DAY`, `TODAY`, `ROW`, `COLUMN`, `LARGE`, `SMALL`, `RANK`

# Respecting Existing Files

Treat the layout, fonts, colors, naming conventions, and data ordering of provided workbooks as authoritative. Mimic existing patterns. Before writing any value, read nearby cells to learn the convention (casing, abbreviations, placeholders) and match it exactly.

Never create explanation sheets, VBA code sheets, or "Answer" sheets. Implement the transformation directly.

# Verifying Formulas

After saving a workbook with formulas, call `recalculate(output_path)` and
then reload with `data_only=True` to confirm key target cells are not `None`.

`#NAME?` errors mean you used an unsupported function — replace with Python logic or a compatible formula.

# Common Pitfalls

- **Circular refs**: never write a formula that references its own cell. Read the value into Python first, then write back as a literal.
- **Division by zero**: guard divisors with `IFERROR` or check in Python.
- **`{=FORMULA}` braces**: never write literal curly braces — openpyxl treats them as text, not array formula markers.
- **Merged cells**: only the top-left cell has a value; others are `None`. Unmerge before restructuring.
- **`data_only=True`**: use when reading computed values from formula cells. Never save a workbook opened this way (replaces formulas with cached values).
- **Full-column refs**: prefer bounded ranges matching actual data extent. `$A:$A` hangs the sandbox formula verifier.
- **`ws.delete_rows()` / `ws.insert_rows()`**: O(n²) in WASM — hangs on large sheets. Read rows into a list, transform in Python, write back.
- **SUMIF with array math**: `SUMIF` does not support computed arrays as sum_range. Use `SUMPRODUCT` instead.
- **SUM on text-formatted numbers**: returns 0. Coerce with `VALUE()` inside `SUMPRODUCT`, or compute in Python.

# openpyxl Essentials

- 1-indexed: `cell(row=1, column=1)` is `A1`.
- Formulas are stored as text — openpyxl does not evaluate them.
- `read_only=True` / `write_only=True` for large files.
""",
    packages=["openpyxl", "pandas", "formulas"],
    tools={"recalculate": recalculate},
)
