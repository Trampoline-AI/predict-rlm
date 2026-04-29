from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from rlm_gepa import AgentSpec, OptimizeConfig
from spreadsheet_rlm.agent.signature import ManipulateSpreadsheet
from spreadsheet_rlm.agent.skills import recalculate as recalculate_tool
from spreadsheet_rlm.agent.skills import render


@dataclass
class SpreadsheetGepaConfig(OptimizeConfig):
    """Configuration for optimizing the SpreadsheetRLM skill with RLM-GEPA."""

    train_dataset: str = "trainset"
    val_ratio: float = 0.20
    val_limit: int | None = None
    cases_per_task: int = 1


def default_config() -> SpreadsheetGepaConfig:
    return SpreadsheetGepaConfig(
        executor_lm="openai/gpt-5.4-mini",
        executor_sub_lm="openai/gpt-5.4-mini",
        proposer_lm="anthropic/claude-sonnet-4-6",
        proposer_sub_lm="openai/gpt-5.4-mini",
    )


def _spreadsheet_tool_signatures() -> str:
    return "\n\n".join(_format_tool(tool) for tool in (recalculate_tool, render))


def _format_tool(tool: Any) -> str:
    return f"{tool.__name__}{inspect.signature(tool)}\n{inspect.getdoc(tool) or ''}"


def _spreadsheet_target_signature() -> str:
    return inspect.getdoc(ManipulateSpreadsheet) or "ManipulateSpreadsheet"


SPREADSHEET_SPEC = AgentSpec(
    agent_type=(
        "a spreadsheet-manipulation agent that writes Python against openpyxl "
        "in a Pyodide/WASM sandbox, with host-side tools for LibreOffice "
        "recalculation and workbook rendering"
    ),
    use_cases=[
        "investment-banking modeling (DCF, cashflow, 3-statement, LBO)",
        "filling structured forms (tax, compliance, HR onboarding)",
        "project-management tracking (status rollups, milestone sheets)",
        "mundane data wrangling (dedup, reformat, reshape, join)",
    ],
    runtime_grounding_examples={
        "library symbols": ["openpyxl `cell.value`, `iter_rows`, `MergedCell`, `ArrayFormula`"],
        "tool contracts": ["`recalculate(path)` and `render(path, cell_range, sheet_name)`"],
        "sandbox facts": ["Pyodide/WASM memory and wall-clock limits"],
        "spreadsheet compatibility": ["LibreOffice formula support and Excel private prefixes"],
    },
    tool_signatures=_spreadsheet_tool_signatures(),
    target_signature=_spreadsheet_target_signature(),
    scoring_description=(
        "Each task has one or more workbook cases. The evaluator compares target "
        "cells in the produced workbook against a reference answer. Per-case score "
        "is matched_cells / total_cells, and task score is the mean of case scores. "
        "Feedback includes per-case pass/fail, cell mismatches, and crash reasons."
    ),
    domain_conventions_note=(
        "Spreadsheet domain conventions are valid only when grounded in runtime "
        "behavior that transfers beyond a specific benchmark workbook."
    ),
)
