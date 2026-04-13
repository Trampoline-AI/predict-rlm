"""SpreadsheetBench dataset loader.

Owns the ``SpreadsheetTask`` shape used by the eval/optimize pipelines
and the filename-discovery logic that stitches per-task test cases
together across the two on-disk naming conventions
(``{idx}_{id}_{init|input}.xlsx`` + ``{answer|golden}`` and the
unnumbered ``initial.xlsx`` / ``golden.xlsx`` pair).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

_EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _EXAMPLE_DIR / "data"


@dataclass(frozen=True)
class SpreadsheetTask:
    """A single benchmark task with its discovered test cases."""

    task_id: str
    instruction: str
    instruction_type: str
    answer_position: str
    spreadsheet_dir: str
    # (idx, input_path, answer_path) per test case — tuple for hashability
    test_cases: tuple[tuple[int, str, str | None], ...]


def discover_test_cases(
    spreadsheet_dir: str | Path,
    task_id: str,
) -> list[tuple[int, str, str | None]]:
    """Discover input/answer file pairs for a task.

    Datasets use inconsistent naming across versions:

    * ``sample_data_200``: ``{idx}_{id}_input.xlsx`` / ``{idx}_{id}_answer.xlsx``
      (3 cases).
    * ``spreadsheetbench_verified_400``: ``{idx}_{id}_init.xlsx`` /
      ``{idx}_{id}_golden.xlsx`` (1 case).
    * 5 tasks in ``verified_400`` (13284, 32023, 32789, 56274, 58109) use
      unnumbered ``initial.xlsx`` / ``golden.xlsx`` — treated as case 1.
    * 1 task (42930) has a golden file with a wrong id in its filename
      (upstream bug) — handled by a fallback search.

    Returns a list of ``(idx, input_path, answer_path)`` tuples sorted
    by idx. ``answer_path`` is ``None`` when only the input file exists.
    """
    spreadsheet_dir = str(spreadsheet_dir)
    if not os.path.isdir(spreadsheet_dir):
        return []

    files = os.listdir(spreadsheet_dir)
    cases: list[tuple[int, str, str | None]] = []

    input_pattern = re.compile(
        rf"^(\d+)_{re.escape(task_id)}_(init|input)\.xlsx$"
    )
    for f in files:
        m = input_pattern.match(f)
        if not m:
            continue
        idx = int(m.group(1))
        input_path = os.path.join(spreadsheet_dir, f)

        answer_path: str | None = None
        for suffix in ("answer", "golden"):
            candidate = os.path.join(
                spreadsheet_dir, f"{idx}_{task_id}_{suffix}.xlsx"
            )
            if os.path.exists(candidate):
                answer_path = candidate
                break

        if answer_path is None:
            for af in files:
                if af == f:
                    continue
                if not af.startswith(f"{idx}_") or not af.endswith(".xlsx"):
                    continue
                if "_golden." in af or "_answer." in af:
                    answer_path = os.path.join(spreadsheet_dir, af)
                    break

        cases.append((idx, input_path, answer_path))

    if not cases:
        init = os.path.join(spreadsheet_dir, "initial.xlsx")
        golden = os.path.join(spreadsheet_dir, "golden.xlsx")
        if os.path.exists(init):
            cases.append((1, init, golden if os.path.exists(golden) else None))

    cases.sort(key=lambda x: x[0])
    return cases


def load_dataset(
    dataset_name: str,
    max_cases_per_task: int = 0,
) -> list[SpreadsheetTask]:
    """Load ``data/{dataset_name}/dataset.json`` and discover test cases.

    Args:
        dataset_name: Folder name under ``examples/spreadbench/data/``.
            With the current layout, ``"testset"`` maps to the verified
            400-set (symlinked) and ``"trainset"`` to the 512-entry
            train split.
        max_cases_per_task: Cap the number of test cases per task. Pass
            ``0`` to keep all cases (the default — eval uses every case,
            GEPA training typically uses only one for speed).
    """
    dataset_path = DATA_DIR / dataset_name
    dataset_json = dataset_path / "dataset.json"
    if not dataset_json.is_file():
        raise FileNotFoundError(
            f"dataset.json not found at {dataset_json}. "
            f"Run `make dataset` first."
        )
    raw = json.loads(dataset_json.read_text())

    tasks: list[SpreadsheetTask] = []
    for entry in raw:
        task_id = str(entry["id"])
        spreadsheet_dir = str(dataset_path / entry["spreadsheet_path"])
        cases = discover_test_cases(spreadsheet_dir, task_id)
        if not cases:
            continue
        if max_cases_per_task:
            cases = cases[:max_cases_per_task]
        tasks.append(
            SpreadsheetTask(
                task_id=task_id,
                instruction=entry["instruction"],
                instruction_type=entry["instruction_type"],
                answer_position=entry["answer_position"],
                spreadsheet_dir=spreadsheet_dir,
                test_cases=tuple(cases),
            )
        )
    return tasks
