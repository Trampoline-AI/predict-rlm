#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

EXAMPLE_DIR = Path(__file__).resolve().parent.parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from terminal_bench_rlm.scoring import score_details  # noqa: E402
from terminal_bench_rlm.tools.tbench_agent import (  # noqa: E402
    TerminalBenchRLMBaseAgent,
)


def fake_smoke_rows() -> list[dict[str, Any]]:
    _exercise_agent_adapter()
    fake_results = [
        (
            "fake-all-pass",
            SimpleNamespace(
                is_resolved=True,
                parser_results={
                    "test_a": "passed",
                    "test_b": "passed",
                    "test_c": "passed",
                },
            ),
        ),
        (
            "fake-partial-pass",
            SimpleNamespace(
                is_resolved=False,
                parser_results={
                    "test_a": "passed",
                    "test_b": "passed",
                    "test_c": "failed",
                },
            ),
        ),
        (
            "fake-all-fail",
            SimpleNamespace(
                is_resolved=False,
                parser_results={
                    "test_a": "failed",
                    "test_b": "failed",
                    "test_c": "failed",
                },
            ),
        ),
    ]
    rows = []
    for task_id, result in fake_results:
        details = score_details(result)
        rows.append(
            {
                "task_id": task_id,
                "soft_score": details["soft_score"],
                "hard_score": details["hard_score"],
                "passed": details["passed"],
                "total": details["total"],
            }
        )
    return rows


def _exercise_agent_adapter() -> None:
    name = TerminalBenchRLMBaseAgent.name()
    if name != "predict-rlm":
        raise RuntimeError(f"unexpected Terminal-Bench agent name: {name}")


def _print_table(rows: list[dict[str, Any]]) -> None:
    print("task_id soft_score hard_score passed total")
    for row in rows:
        print(
            f"{row['task_id']} "
            f"{row['soft_score']:.3f} "
            f"{row['hard_score']:.3f} "
            f"{row['passed']} "
            f"{row['total']}"
        )


def _real_smoke_unavailable() -> str:
    return (
        "Real Terminal-Bench smoke is environment-specific. "
        "Install terminal_bench and run its Harbor/Daytona harness with the "
        "terminal_bench_rlm.tools.tbench_agent.DaytonaRemotePredictRLMAgent adapter "
        "against three selected task IDs. The default mode is a hermetic fake smoke "
        "to avoid LLM calls and external harness work."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a quick Terminal-Bench scoring smoke.")
    parser.add_argument(
        "--real",
        action="store_true",
        help="Fail with guidance for wiring a real Terminal-Bench harness smoke.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON rows.")
    args = parser.parse_args(argv)

    if args.real:
        print(_real_smoke_unavailable(), file=sys.stderr)
        return 2

    rows = fake_smoke_rows()
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        _print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
