"""Spreadsheet skill for LibreOffice-compatible workbook manipulation."""

import os
from pathlib import Path
from typing import Annotated

from predict_rlm import Skill, SyncedFile

from ..tools.recalculate import recalculate as _recalc_impl
from ..tools.render import render_to_data_uri as _render_impl


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


def render(
    file_path: Annotated[Path, SyncedFile(writeback=False)],
    cell_range: str | None = None,
    sheet_name: str | None = None,
) -> str:
    """Render an xlsx file to a PNG for visual inspection.

    Converts the workbook to PDF via LibreOffice, then rasterizes the first
    page to PNG via pdftoppm. Returns the rendered image as a
    ``data:image/png;base64,...`` URI which you can pass directly to
    ``predict(... rendered: dspy.Image ...)`` for sub-LM visual verification.

    Args:
        file_path: Path to the .xlsx file to render.
        cell_range: Optional cell range to restrict the render to, e.g.
            ``"J3:N5"`` or ``"Sheet1!J3:N5"``. Use this when the area you
            want to inspect is not on the first printed page of a
            whole-sheet export (wide or tall sheets). When omitted, the
            whole sheet's first page is rendered.
        sheet_name: Sheet to set the print area on when *cell_range* is
            given. Ignored if *cell_range* already has a ``Sheet!range``
            prefix. Defaults to the workbook's active sheet.

    Returns:
        A ``data:image/png;base64,...`` URI on success, or a string starting
        with ``Error:`` if rendering failed.
    """
    path = os.path.abspath(file_path)
    if not os.path.isfile(path):
        return f"Error: file not found: {path}"
    try:
        return _render_impl(path, cell_range=cell_range, sheet_name=sheet_name)
    except Exception as e:
        return f"Error: {e}"

# Adapted from OpenAI's Apache-2.0 curated spreadsheet skill.
# See ../THIRD_PARTY_NOTICES.md for provenance and license terms.
libreoffice_spreadsheet_skill = Skill(
    name="spreadsheet",
    instructions="""# Spreadsheet Skill

## Workflow
1. Prefer `openpyxl` for `.xlsx` editing and formatting. Use `pandas` for analysis and CSV/TSV workflows.
2. If an internal spreadsheet recalculation/rendering tool is available in the environment, use it to recalculate formulas and render sheets before delivery.
3. Use formulas for derived values instead of hardcoding results.
4. If layout matters, render the output and inspect it visually with your sub-LM.
5. Save outputs, keep filenames stable, and clean up intermediate files.

## Primary tooling
- Use `openpyxl` for creating/editing `.xlsx` files and preserving formatting.
- Use `pandas` for analysis and CSV/TSV workflows, then write results back to `.xlsx` or `.csv`.
- Use `openpyxl.chart` for native Excel charts when needed.
- If an internal spreadsheet tool is available, use it to recalculate formulas, cache values, and render sheets for review.
- For sheets over ~1000 rows, read via `iter_rows(values_only=True)` with `read_only=True`; never iterate `ws.cell(r, c).value` in Python loops (prohibitively slow on WASM sandboxes, and still slow on native openpyxl for very large sheets).

## Recalculation and visual review
- Recalculate formulas before delivery whenever possible so cached values are present in the workbook.
- `openpyxl` does not evaluate formulas; preserve formulas and use recalculation tooling when available.
- A `recalculate(path)` tool is available — call it on your output before submitting, then reload with `data_only=True` to confirm cells resolved.

## Rendering and visual checks
- Render the output sheet to a PNG when layout matters — call `render(path)` for the whole active sheet, or `render(path, cell_range="J3:N5")` / `render(path, cell_range="Sheet1!J3:N5")` to focus on a specific region. Use the cell_range form when your target area is on a wide/tall sheet and wouldn't fit on the first printed page.
- Inspect the rendered image with your sub-LM: `await predict("rendered: dspy.Image, task_instruction: str -> matches_instruction: bool, issues: list[str]", rendered=img_uri, task_instruction=task_instruction)` — pass `img_uri` as a `data:image/png;base64,...` URI.
- Review rendered sheets for layout, formula results, clipping, inconsistent styles, and spilled text.

## Formula requirements
- Use formulas for derived values rather than hardcoding results.
- Do not use dynamic array functions like `FILTER`, `XLOOKUP`, `SORT`, or `SEQUENCE`.
- Keep formulas simple and legible; use helper cells for complex logic.
- Avoid volatile functions like `INDIRECT` and `OFFSET` unless required.
- Prefer cell references over magic numbers (for example, `=H6*(1+$B$3)` instead of `=H6*1.04`).
- Use absolute (`$B$4`) or relative (`B4`) references carefully so copied formulas behave correctly.
- If you need literal text that starts with `=`, prefix it with a single quote.
- Guard against `#REF!`, `#DIV/0!`, `#VALUE!`, `#N/A`, and `#NAME?` errors.
- Check for off-by-one mistakes, circular references, and incorrect ranges.
- Prefix these functions with `_xlfn.` (e.g. `=_xlfn.IFNA(...)`) or they evaluate to `#NAME?`: `IFNA`, `IFS`, `SWITCH`, `TEXTJOIN`, `AGGREGATE`, `MAXIFS`, `MINIFS`, `CONCAT`.

## Citation requirements
- Cite sources inside the spreadsheet using plain-text URLs.
- For financial models, cite model inputs in cell comments.
- For tabular data sourced externally, add a source column when each row represents a separate item.

## Formatting requirements (existing formatted spreadsheets)
- Render and inspect a provided spreadsheet before modifying it when possible.
- Preserve existing formatting and style exactly.
- Match styles for any newly filled cells that were previously blank.
- Never overwrite established formatting unless the user explicitly asks for a redesign.

## Formatting requirements (new or unstyled spreadsheets)
- Use appropriate number and date formats.
- Dates should render as dates, not plain numbers.
- Percentages should usually default to one decimal place unless the data calls for something else.
- Currencies should use the appropriate currency format.
- Headers should be visually distinct from raw inputs and derived cells.
- Use fill colors, borders, spacing, and merged cells sparingly and intentionally.
- Set row heights and column widths so content is readable without excessive whitespace.
- Do not apply borders around every filled cell.
- Group related calculations and make totals simple sums of the cells above them.
- Add whitespace to separate sections.
- Ensure text does not spill into adjacent cells.
- Avoid unsupported spreadsheet data-table features such as `=TABLE`.

## Color conventions (if no style guidance)
- Blue: user input
- Black: formulas and derived values
- Green: linked or imported values
- Gray: static constants
- Orange: review or caution
- Light red: error or flag
- Purple: control or logic
- Teal: visualization anchors and KPI highlights

## Finance-specific requirements
- Format zeros as `-`.
- Negative numbers should be red and in parentheses.
- Format multiples as `5.2x`.
- Always specify units in headers (for example, `Revenue ($mm)`).
- Cite sources for all raw inputs in cell comments.
- For new financial models with no user-specified style, use blue text for hardcoded inputs, black for formulas, green for internal workbook links, red for external links, and yellow fill for key assumptions that need attention.

## Investment banking layouts
If the spreadsheet is an IB-style model (LBO, DCF, 3-statement, valuation):
- Totals should sum the range directly above.
- Hide gridlines and use horizontal borders above totals across relevant columns.
- Section headers should be merged cells with dark fill and white text.
- Column labels for numeric data should be right-aligned; row labels should be left-aligned.
- Indent submetrics under their parent line items.
""",
    packages=["openpyxl", "pandas", "formulas"],
    tools={"recalculate": recalculate, "render": render},
)
