"""Regression test for the full-column-reference hang in ``recalculate()``.

Context:
    The SpreadsheetBench task ``34365`` asks the model to sum annual
    values from column B based on dates in column A and a TRUE flag in
    column C, output per-row in column G. When the mercury-2 model was
    run on it, the RLM produced an output xlsx with ~18k formulas of
    the form::

        G{n}: =SUMIFS(B:B, A:A, ">="&DATE(YEAR(A{n}),1,1),
                      A:A, "<="&DATE(YEAR(A{n}),12,31),
                      C:C, TRUE)

    Every formula uses **full-column references** (``B:B``, ``A:A``,
    ``C:C``). When ``formulas.ExcelModel().loads(path).finish()`` builds
    the dependency graph, it materialises a ``Ranges`` node per
    referenced column (each covering ~1M cells) × ~18k formulas, then
    iterates them via a pure-Python NumPy object-mode ufunc loop under
    the GIL. The observed wall-time was well beyond an hour before we
    killed the run.

    This test exercises ``recalculate()`` on the captured output file
    in a **subprocess** with a hard ``timeout=60`` so the test itself
    cannot hang pytest — Python threads are uncancellable, so the only
    reliable way to test "does this call complete in bounded time" is
    to run it in a child process we can actually kill.

Expected states:

* **RED (current)**: the subprocess times out; ``subprocess.run``
  raises ``TimeoutExpired``; the test fails with a clear "hung >60s"
  message.
* **GREEN (post-fix)**: the subprocess returns in well under the
  budget; ``RecalcResult.source`` is ``"baseline"`` or ``"libreoffice"``
  (whichever the pipeline picks once ``formulas`` is bypassed).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_EXAMPLE_DIR = _TESTS_DIR.parent
_REPO_ROOT = _EXAMPLE_DIR.parent.parent

FIXTURE = _TESTS_DIR / "fixtures" / "hang_huge_range_34365.xlsx"

# Hard wall-clock budget for the subprocess. Anything above this is a
# hang for our purposes — a correctly pipelined recalc on this file
# (formulas stage time-boxed in a child process + LibreOffice fallback)
# completes in a handful of seconds, so 60s gives a huge safety margin
# without making the test painful when RED.
_HARD_TIMEOUT_SEC = 60


def test_recalculate_does_not_hang_on_full_column_refs(tmp_path: Path):
    assert FIXTURE.is_file(), f"fixture missing: {FIXTURE}"

    src = tmp_path / "hang.xlsx"
    shutil.copy2(FIXTURE, src)

    child_src = str(_EXAMPLE_DIR)
    script = (
        "import sys; "
        f"sys.path.insert(0, {child_src!r}); "
        "import json; "
        "from spreadsheet_rlm.recalculate import recalculate; "
        f"r = recalculate({str(src)!r}); "
        "print(json.dumps({'source': r.source, 'resolved': r.resolved, "
        "'total': r.total_formulas, 'errors': r.errors}))"
    )

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=_HARD_TIMEOUT_SEC,
            cwd=str(_REPO_ROOT),
        )
    except subprocess.TimeoutExpired as e:
        elapsed = time.perf_counter() - t0
        pytest.fail(
            f"recalculate() hung >{_HARD_TIMEOUT_SEC}s on "
            f"hang_huge_range_34365.xlsx (elapsed={elapsed:.1f}s). "
            f"The source xlsx has ~18k formulas with full-column "
            f"references (e.g. SUMIFS(B:B, A:A, ..., C:C, ...)) "
            f"which makes the `formulas` library's dependency-graph "
            f"build O(formulas × columns × 1M cells) via a pure-Python "
            f"ufunc loop. Partial stderr: {e.stderr[-500:] if e.stderr else ''}"
        )

    elapsed = time.perf_counter() - t0
    assert result.returncode == 0, (
        f"recalc subprocess failed (rc={result.returncode}, "
        f"elapsed={elapsed:.1f}s)\n"
        f"stdout: {result.stdout}\n"
        f"stderr (tail): {result.stderr[-500:]}"
    )

    payload = json.loads(result.stdout.strip())
    source = payload["source"]
    assert source in ("baseline", "libreoffice"), (
        f"expected the pipeline to time-box/bypass the `formulas` library and "
        f"land on baseline or libreoffice, got source={source!r}"
    )
    assert any("timed out after" in err for err in payload["errors"]), payload
