from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_fake_three_task_smoke_outputs_expected_score_distinctions() -> None:
    script = Path(__file__).resolve().parent.parent / "scripts" / "smoke_three_tasks.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--json"],
        check=True,
        text=True,
        capture_output=True,
    )

    rows = json.loads(completed.stdout)
    assert [row["task_id"] for row in rows] == [
        "fake-all-pass",
        "fake-partial-pass",
        "fake-all-fail",
    ]
    assert [(row["soft_score"], row["hard_score"], row["passed"], row["total"]) for row in rows] == [
        (1.0, 1.0, 3, 3),
        (2 / 3, 0.0, 2, 3),
        (0.0, 0.0, 0, 3),
    ]
