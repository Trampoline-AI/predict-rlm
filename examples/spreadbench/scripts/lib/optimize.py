"""GEPA-based multi-component optimization of the SpreadsheetRLM prompts.

Evolves both the ``ManipulateSpreadsheet`` signature docstring AND the
``libreoffice_spreadsheet_skill`` instructions simultaneously using
GEPA's evolutionary loop. Pulls heavily from the patterns established in
the sibling ``predict-rlm.spreadbench/SpreadsheetBench/optimize_signature.py``,
extended to multi-component evolution and adapted to use this repo's
shared ``lib/`` helpers (dataset loading, scoring, LM config, recalc
pipeline).

Two GEPA components evolved per run:

* ``signature_docstring``  — high-level workflow prose from
  ``ManipulateSpreadsheet.__doc__``
* ``skill_instructions``   — domain rules from
  ``libreoffice_spreadsheet_skill.instructions``

By default the GEPA module selector is ``round_robin`` so each round
touches one of the two; the ``all`` strategy is also supported via
``OptimizeConfig.module_selector``.

The "SURGICAL RLM proposer" pattern from the sibling repo is preserved:
when ``--rlm_proposer`` is passed, instead of GEPA's stock single-shot
reflection LM the proposer is itself a ``PredictRLM`` that loads the
recent task traces from a sandbox-mounted JSON file, iteratively
analyzes them with Python, and writes new instructions under an
explicit "surgical edit" constraint that prevents val-set overfitting.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pickle
import random
import re
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import dspy
import gepa
import nest_asyncio
from gepa import EvaluationBatch
from spreadsheet_rlm.recalculate import recalculate
from spreadsheet_rlm.signature import ManipulateSpreadsheet
from spreadsheet_rlm.skills import libreoffice_spreadsheet_skill
from tqdm import tqdm

from predict_rlm import File, PredictRLM, Skill

from .dataset import SpreadsheetTask, load_dataset
from .evaluation import (
    LMCost,
    _build_instruction,
    make_dynamic_signature,
    make_dynamic_skill,
    parse_answer_position,
    summarize_lm_cost,
)
from .lm_config import compute_lm_cost, get_lm_config, get_sub_lm_config
from .scoring import score_workbooks

nest_asyncio.apply()

log = logging.getLogger("spreadsheet_rlm.optimize")


# ---------------------------------------------------------------------------
# Component identifiers
# ---------------------------------------------------------------------------

COMPONENT_SIGNATURE = "signature_docstring"
COMPONENT_SKILL = "skill_instructions"

ALL_COMPONENTS = (COMPONENT_SIGNATURE, COMPONENT_SKILL)


# ---------------------------------------------------------------------------
# Dynamic component constructors
# ---------------------------------------------------------------------------

def seed_candidate() -> dict[str, str]:
    """Return the GEPA seed candidate: skill instructions only.

    Signature evolution is intentionally out of scope — prior runs found
    no meaningful lift from evolving the task-level workflow docstring,
    and letting GEPA touch it was a source of wasted proposer budget +
    skill-shaped output contaminating the signature slot. The signature
    is frozen at the pristine ``ManipulateSpreadsheet`` class docstring
    for the whole run; every proposer call only edits the skill.
    """
    return {
        COMPONENT_SKILL: libreoffice_spreadsheet_skill.instructions,
    }


# ---------------------------------------------------------------------------
# RLM proposer signature (verbatim SURGICAL prompt + multi-component marker)
# ---------------------------------------------------------------------------


class ImproveInstructions(dspy.Signature):
    """Analyze execution traces and propose improved SKILL INSTRUCTIONS for a
    general-purpose spreadsheet agent. The goal is NOT to game any particular
    benchmark — it is to make the agent a master of its tools and its sandbox
    environment over ARBITRARY Excel workbooks.

    The skill you are editing will be reused for (at minimum):
      - investment-banking modeling (DCF, cashflow, 3-statement, LBO)
      - filling structured forms (tax, compliance, HR onboarding)
      - project-management tracking (status rollups, milestone sheets)
      - mundane data wrangling (dedup, reformat, reshape, join)

    Every change you propose must help across this whole surface. Rules that
    only pay off on the specific eval in the traces file must be REJECTED,
    even if they look like they would lift the score on that eval.

    # The traces file
    /sandbox/input/traces_file/ is a JSON file. Each record is one task
    invocation and has exactly these three fields:
      - "Inputs"            : the natural-language instruction and the
                              instruction type.
      - "Generated Outputs" : the full RLM execution trace (reasoning,
                              code, sandbox stdout). BEST and WORST
                              cases are banner-framed inside this field.
      - "Feedback"          : the evaluator signal for THIS record —
                              task-level score, per-case pass/fail,
                              cell-level expected-vs-got mismatches,
                              and crash reasons (timeouts, uncaught
                              exceptions, max-iterations-without-SUBMIT).
                              This is the primary place to read what
                              went wrong.

    # What counts as a GOOD change

    A rule is GOOD only if it passes ALL FOUR structural tests:

      (a) NAMES A SPECIFIC API, TOOL, OR ENVIRONMENT FACT that exists
          regardless of any particular benchmark. Examples of what counts:
          an openpyxl symbol (`cell.value`, `data_only=True`, `iter_rows`,
          `MergedCell`, `ArrayFormula`), a LibreOffice behavior (formula
          evaluation, dynamic-array support, prefix handling), a sandbox
          limit (WASM memory, timeouts, missing C extensions), or a
          concrete tool name (`recalculate`, `render`). "Check the data
          carefully" does NOT count.

      (b) HAS A CONCRETE NON-BENCHMARK USE CASE. Before writing the rule,
          name a task from {IB modeling, form-filling, PM tracking, data
          wrangling} where obeying the rule prevents a real bug. If you
          cannot name one, the rule is benchmark-specific — reject it.

      (c) STATES A PRINCIPLE, NOT A LITERAL TRACE TOKEN. Rules must be
          written in terms of cell properties, API behavior, and data
          semantics — NOT in terms of specific labels, sheet names,
          ranges, values, or phrases that appear in the traces. If your
          rule mentions a literal string, a specific sheet name, or a
          specific cell range that you copied from a trace, you are
          overfitting AND contaminating the proposer's own prompt with
          benchmark data. Lift the rule up one level of abstraction
          until it says "strings of type X" or "cells in role Y" instead
          of the literal token.

      (d) MAY INCLUDE A SHORT CODE EXAMPLE. When a rule is hard to
          apply from prose alone, add a compact code example (preferably
          2-8 lines) showing the exact API or sequencing pattern.
          The snippet must use only generic symbols (`cell`, `ws`,
          `recalculate`, etc.) and must not copy literal ranges,
          sheet names, function arguments, or values from any single
          trace token.

    ## GOOD rule examples (principles, not trace tokens)

      1. "The agent's OUTPUT CHANNEL is the workbook itself. `cell.value`
          must hold EITHER a scalar result (number, string, bool,
          datetime) OR a formula string that evaluates to the result
          (e.g. `cell.value = '=Sheet1!B12 * 1.05'`). It must NOT hold
          explanatory prose, a question, a code tutorial, or a pseudo-
          code sketch. The test: would a downstream human or formula
          reading this cell receive a usable value? If no, it is not a
          valid write."
          — openpyxl semantic: `cell.value` is the typed slot —
          — use case: every IB model (formula cells are load-bearing),
            every form (fields hold values), every PM sheet (status
            cells hold enums), every data-wrangling output —

      2. "After calling the `recalculate` tool, reopen the workbook with
          `openpyxl.load_workbook(path, data_only=True)` before reading
          formula results. `data_only` must be set at LOAD time; toggling
          it on an already-loaded workbook does nothing. Without it,
          openpyxl returns the formula string, not the computed value."
          — openpyxl semantic: `data_only` is a load-time flag —
          — use case: any DCF or LBO that chains through computed cells,
            any form whose totals come from formula-driven subtotals —

      3. "String equality on `cell.value` is case-sensitive and whitespace-
          sensitive. When MATCHING an instruction-supplied label against
          existing cell text, normalize both sides
          (`.strip().casefold()`). When WRITING labels, preserve the
          exact casing and spacing of the source text — normalization is
          a READ-SIDE operation only."
          — Python string semantics; openpyxl returns raw strings —
          — use case: form-filling (field labels), PM sheets (status
            column headers), IB line-item lookups by name —

      4. "Distinguish DATE VALUES from DATE LABELS by their ROLE, not
          their type. A date VALUE is operated on arithmetically
          (transaction date, milestone date, report date) — keep it as
          a `datetime` object through the pipeline and assign it back
          to `cell.value` so Excel's number format renders it
          correctly. A date LABEL is a header or lookup key that
          happens to describe a period (fiscal-period tags, quarter
          column headers, period markers) — these are conventionally
          strings in real models and must STAY strings, because
          downstream INDEX/MATCH and XLOOKUP use them as exact-match
          keys."
          — openpyxl semantic: `cell.value` accepts `datetime` or `str` —
          — use case: cashflow dating (values) vs. P&L column headers
            (labels); project milestone dates (values) vs. Gantt
            period columns (labels) —

      5. "Merged cells: only the TOP-LEFT anchor of a merged range
          holds the value. The other cells in the range are `MergedCell`
          instances whose `.value` is `None`. To read a merged value by
          its display position, walk `worksheet.merged_cells.ranges`
          and resolve to the anchor. To write into a merged range,
          either write to the anchor or call `unmerge_cells` first."
          — openpyxl semantic: `MergedCell` is a distinct class —
          — use case: IB models use merged title rows; forms use merged
            section headers; PM dashboards use merged summary bands —

      6. "When a cell contains a dynamic-array formula, its `.value` is
          an `openpyxl.worksheet.formula.ArrayFormula` object, not a
          string. Read `.text` to get the formula source. The computed
          result only materializes after a `recalculate` pass followed
          by a `data_only=True` reopen of the workbook."
          — openpyxl semantic: `ArrayFormula` is a specific class —
          — use case: IB sensitivity tables (dynamic-array formulas),
            modern form templates with live-filtered drop-downs —

      7. "LibreOffice rejects the private prefixes that Excel silently
          prepends to newer built-in functions. When writing a formula
          that will be evaluated by `recalculate`, strip these prefixes
          so the function name is plain. Discover the specific set of
          affected functions EMPIRICALLY from the traces you see — do
          not memorize a static list, because the set depends on
          openpyxl's version and the workbook's origin."
          — LibreOffice/Excel compatibility fact —
          — use case: any IB model using modern Excel functions; any
            form originally authored in a recent Excel version —

      8. "When you INTEND to fill a rectangular region (a column of
          per-row results, a row of per-column results, a full 2D
          block), verify your write actually covered it. The pitfall:
          `ws[range_str]` returns a tuple whose shape is fixed by the
          range string, but if you iterate in parallel with a Python
          source that has fewer entries, you silently leave the
          trailing cells as `None`. Compute `intended_count` BEFORE
          the write loop starts — from instruction-text tokens (e.g.
          'fill rows 2 through 25' → 24) or, for data-conditional
          writes ('bump positive values in column D'), from a
          read-only pre-scan snapshot of the data (e.g.
          `positive_rows = [r for r in df if r.amount > 0];
          intended_count = len(positive_rows)`). Once set,
          `intended_count` MUST be a constant from the moment the
          write loop starts until the assertion fires: the write loop
          may READ it by name but MUST NOT reassign it, increment it,
          recompute it, or shadow it. At save time:
          `if len(written_cells) != intended_count: raise
          ValueError(f'intent mismatch: committed {intended_count},
          wrote {len(written_cells)}')`. A raised exception is
          visible in the trace; a silent partial write is not. The
          anti-patterns that defeat this rule are all variations of
          making `intended_count` rise in lockstep with the write
          loop: `intended_count += 1` inside the loop, `intended_count
          = len(written_cells)` at save time, or `intended_count = 0`
          as a dummy followed by mid-loop reassignment. Each makes
          the assertion tautological. `intended_count` represents
          what the agent PLANNED to write, not what it actually
          wrote. For a single cell or subset ('bump this one value'),
          your frozen `intended_count` is 1 or whatever your subset
           has — don't inflate it to match an external region hint,
           and don't grow it to match your write loop."
           — openpyxl semantic: `ws[range_str]` returns a fixed-shape
             tuple regardless of your data length —
           — use case: form-filling (every field you intended to touch
             must actually be touched); PM rollups (every milestone row
             in the cohort you intended to cover); data-wrangling
             outputs where your intent is a full rectangle —

      9. "Before writing filtered outputs, compute the write target set once,
          then iterate only that snapshot."
          Example:
          ```python
          target_rows = [r for r in rows if r.status == "needs_update"]
          for row in target_rows:
              ws[f"D{row.idx}"] = transform(row.value)
          ```
          — openpyxl and Python flow semantics —
          — use case: data-wrangling and PM rollups, where target cohorts
            are fixed before the write loop to avoid partial updates.

    Notice that NONE of these rules mention a specific label, sheet
    name, or value from the traces. They describe the TYPE of thing the
    agent is dealing with, not the particular instance.

    ## What counts as a BAD change

    Any rule that fails any of tests (a), (b), (c), (d). Common failure modes:

      1. "Always format amounts with specific decimal places and a
         specific currency symbol." — fails (b): IB models use parens
         for negatives, some forms require ISO currency codes, PM
         sheets may want unformatted numbers. Style rule, not a
         runtime fact.

      2. "When the instruction asks for a rank, use a specific
         tie-breaking rule by default." — fails (b): rank semantics
         are task-specific; no universal default.

      3. "Coerce scalar types to specific string representations to
         match a downstream scoring contract." — fails (a): Excel has
         typed cell values — numbers, booleans, dates — and real
         consumers (forms with validation, formulas, charts, downstream
         scripts) depend on those types. Forcing a particular string
         serialization of a scalar is a workaround for one consumer's
         choice and will BREAK the others.

      4. "Sort descending by default when the instruction is
         ambiguous." — fails (b): a guess dressed up as a rule; IB
         models are typically sorted by fiscal period, not magnitude.

      5. "If unsure about the answer cell, leave it blank rather than
         risk a mismatch." — fails everything: teaches the agent to
         stop trying. Pure benchmark gaming; destroys utility on every
         real use case.

      6. "If the instruction mentions <a specific literal from the
         trace>, output the value in sheet <a specific sheet from the
         trace>." — fails (c): literal copy of the trace into the rule.
         Even if the rule appears to generalize, it has contaminated
         the proposer's own prompt with benchmark data. The principle
         you are actually trying to capture — "sheet targets are
         specified in the instruction text, parse them out" — is fine;
         the literal tokens are not.

    Note: domain formatting conventions (e.g. "in IB models, hard-coded
    inputs are blue and formulas are black") ARE good rules when you
    (i) scope them to a domain, (ii) justify them with a specific cell
    property (`cell.font.color`), and (iii) would apply them even on a
    task that never appears in this traces file.

    # Workflow

    1. Load /sandbox/input/traces_file/.
    2. Bucket records by score. Read the worst failing traces carefully.
    3. For each failure, identify what the agent did at the code/cell
       level that was wrong — in terms of API behavior, tool usage, or
       sandbox environment — NOT in terms of "it got the wrong answer".
       Some failing traces will appear as CRASHED instead of
       cell-mismatched — these are cases where the agent hit a timeout,
       raised an unhandled exception, or exhausted max_iterations
       without SUBMIT. The crash reason (e.g. "RLM SyntaxError:
       unterminated triple-quoted string literal", "RLM timeout at
       300s", "RLM max iterations reached (50) without SUBMIT") is the
       first-class corrective signal: a rule that prevents the crash
       pattern is as valuable as one that prevents a cell-level
       mismatch. Read the partial trajectory up to the crash point to
       understand what the agent was trying to do.
    4. Compare with the best successful traces to confirm the failure
       is not just random sampling noise.
    5. For each candidate improvement, apply the four structural tests:
        (a) does it name a specific API/tool/env fact?
        (b) does it have a non-benchmark use case?
        (c) does it state a principle, not a literal trace token?
        (d) does it avoid trace-specific snippets, or use a short
            abstract snippet only when it materially improves
            transfer?
    6. Principle-vs-token check: scan the candidate rule for any label,
       sheet name, range, or specific value that was lifted from a
       trace. Replace each with an abstract role ("a column labeled
       with an identifier", "the target sheet", "the instruction
       range"). If the rule loses all meaning after abstraction, it
       was memorization — drop it.
    7. Counterfactual check: name TWO SPECIFIC non-benchmark workbook
       shapes on which obeying this rule would change the agent's
       behavior, and the two shapes MUST come from DIFFERENT domains
       across {IB modeling, form-filling, PM tracking, data
       wrangling}. A DCF and an LBO are the SAME domain (both IB); a
       DCF and a compliance form are DIFFERENT domains. If you cannot
       name two cross-domain counterfactuals, the rule does not
       generalize — drop it. One-domain rules are benchmark-adjacent
       memorization dressed up as a principle.
    8. Keep improvements SURGICAL. Prefer targeted additions over full
       rewrites. Preserve rules that are already working. Emit the
       FULL revised instructions text in `new_instructions` — the old
       content you want to keep, plus your changes — not a diff.
    9. For each rule you kept or added, emit one audit line in
       `generalization_check` of the form:
         `<rule> | api: <specific api/env fact> | use case: <concrete
          non-benchmark task> | principle: <abstract principle,
          naming no trace token> | counterfactual_1: <workbook shape,
          domain A> | counterfactual_2: <workbook shape, domain B,
          B ≠ A>`
       The two counterfactuals MUST come from different domains
       across {IB modeling, form-filling, PM tracking, data
       wrangling}. Any rule whose audit line cannot be fully populated
       — or whose two counterfactuals come from the same domain —
       is memorization; drop it from `new_instructions` too.
    """

    current_instructions: str = dspy.InputField(
        desc="The current skill instructions text being improved"
    )
    traces_file: File = dspy.InputField(
        desc="JSON file with task execution traces, mounted at /sandbox/input/traces_file/"
    )
    new_instructions: str = dspy.OutputField(
        desc="The full revised instructions text for the indicated component. "
        "Emit the entire skill/docstring you want the agent to use, not a diff. "
        "Prefer concise code examples to clarify behavior where prose is "
        "ambiguous, while keeping snippets abstract and trace-agnostic."
    )
    generalization_check: list[str] = dspy.OutputField(
        desc="One line per rule kept or added, in the format: "
        "'<rule> | api: <specific api/env fact> | use case: <concrete "
        "non-benchmark task> | principle: <abstract principle, naming no "
        "trace token> | counterfactual_1: <workbook shape, domain A> | "
        "counterfactual_2: <workbook shape, domain B, B ≠ A>'. "
        "Rules may include a brief code example in the <rule> field when it"
        " materially improves transfer."
        "The two counterfactuals MUST come from different domains across "
        "{IB modeling, form-filling, PM tracking, data wrangling} — a DCF "
        "and an LBO are the SAME domain. Any rule whose slots cannot be "
        "filled, or whose counterfactuals share a domain, must be dropped "
        "from new_instructions as well."
    )


# ---------------------------------------------------------------------------
# Train/val split
# ---------------------------------------------------------------------------


def split_train_val(
    tasks: list[SpreadsheetTask],
    val_ratio: float,
    seed: int,
) -> tuple[list[SpreadsheetTask], list[SpreadsheetTask]]:
    """Seeded ``val_ratio`` partition of *tasks* into (train, val).

    Reproducible across machines and Python versions: same seed +
    same input list = same partition. Result lists are sorted by
    task_id for log stability.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    rng = random.Random(seed)
    indices = list(range(len(tasks)))
    rng.shuffle(indices)
    val_size = int(round(len(tasks) * val_ratio))
    val_indices = set(indices[:val_size])
    train = [t for i, t in enumerate(tasks) if i not in val_indices]
    val = [t for i, t in enumerate(tasks) if i in val_indices]
    return train, val


def write_split_log(
    run_dir: Path,
    seed: int,
    val_ratio: float,
    train: list[SpreadsheetTask],
    val: list[SpreadsheetTask],
) -> None:
    """Persist the resolved train/val task ids to ``{run_dir}/split.json``."""
    payload = {
        "seed": seed,
        "val_ratio": val_ratio,
        "train_size": len(train),
        "val_size": len(val),
        "train_ids": [t.task_id for t in train],
        "val_ids": [t.task_id for t in val],
    }
    (run_dir / "split.json").write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Config + report dataclasses
# ---------------------------------------------------------------------------


@dataclass
class OptimizeConfig:
    """Knobs for a single optimize run."""

    # Task LM (used to evaluate candidates)
    lm: str = "openai/gpt-5.4"
    sub_lm: str = "openai/gpt-5.1"
    reasoning_effort: str | None = "low"

    # Reflection / proposer LM
    reflection_lm: str = "openai/gpt-5.4"
    # gpt-5.4 at medium effort is the default proposer: strong enough to
    # reason about replicated failure patterns across traces, cheap
    # enough to stay well inside GEPA's "executor cost dominates"
    # economics.
    reflection_reasoning_effort: str | None = "medium"

    # RLM proposer's own sub LM (for its predict() calls). Independent
    # of ``sub_lm`` (which is the executor's sub LM). Used only when
    # ``rlm_proposer`` is True. Defaulting to gpt-5.4 medium matches
    # the reflection LM — classification / clustering over trace
    # excerpts benefits from reasoning capacity the executor sub LM
    # doesn't need.
    proposer_sub_lm: str = "openai/gpt-5.4"
    proposer_sub_lm_reasoning_effort: str | None = "medium"

    # Datasets
    train_dataset: str = "trainset"
    val_ratio: float = 0.20
    cases_per_task: int = 1

    # GEPA budget + sampling
    max_metric_calls: int = 2000
    minibatch_size: int = 50
    val_limit: int | None = None  # None = full held-out val set (no cap)
    seed: int = 42

    # Component selection strategy
    module_selector: str = "round_robin"  # "round_robin" or "all"

    # Candidate selection strategy for GEPA's parent picker.
    # "pareto" → sample from Pareto frontier (diversity-oriented)
    # "current_best" → always pick best-by-val-aggregate (pure exploit)
    # "epsilon_greedy" → 90% best, 10% uniform random (explore-then-exploit)
    candidate_selection_strategy: str = "pareto"

    # RLM proposer toggle
    rlm_proposer: bool = False
    proposer_max_iterations: int = 20
    # When True, the RLM proposer uses the generic skill-evolution template
    # from lib/optimize_gen.py (built from SPREADSHEET_SPEC) instead of the
    # hardcoded ImproveInstructions class. Discovery-first framing: 4 rule
    # shapes (corrective/strategic/directional/novel), signal-processing
    # workflow (denoise/average/amplify), 3-bucket trace pass (bottom/top/
    # middle). Requires rlm_proposer=True.
    use_optimize_gen: bool = False

    # Proposer timeout (seconds) for each SURGICAL RLM proposer call.
    # The proposer is LLM-facing and can occasionally stall without
    # returning; this upper bound prevents runs from hanging indefinitely.
    proposer_timeout: int = 600

    # RLM eval budget (per task case)
    concurrency: int = 30
    max_iterations: int = 50
    task_timeout: int = 300
    # Default off: DSPy's LM cache silently serves prior responses with
    # zeroed ``response.usage``, which makes per-run cost accounting look
    # broken (0 tokens logged even though the code path ran). GEPA
    # already checkpoints candidate scores in gepa_state.bin, so the
    # per-LM-call cache adds no value for resumes — it only masks real
    # token consumption. Opt in via ``--with_cache`` when developing.
    cache: bool = False
    # When True, stream per-iteration reasoning/code/output logs from the
    # eval-side PredictRLM (`dspy.predict.rlm` INFO level) to stdout so
    # you can watch progress live. Off by default because with concurrency
    # ≥ 2, logs from many cases interleave unreadably. The same information
    # is always persisted to task_traces/evaluate_NNNN.jsonl regardless of
    # this flag, so you can postmortem after the fact either way.
    verbose_rlm: bool = False

    # Run directory (resume if it already exists)
    run_dir: Path | None = None


@dataclass
class OptimizeReport:
    config: OptimizeConfig
    run_dir: str
    best_idx: int
    best_val_score: float
    total_candidates: int
    total_metric_calls: int
    duration_seconds: float
    best_candidate: dict[str, str]
    val_aggregate_scores: list[float]
    costs: list[LMCost]

    @property
    def rollout_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.costs if c.role in {"main", "sub"})

    @property
    def optimization_cost_usd(self) -> float:
        return self.total_cost_usd - self.rollout_cost_usd

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.costs)

    def to_dict(self) -> dict:
        return {
            "config": {
                "lm": self.config.lm,
                "sub_lm": self.config.sub_lm,
                "reasoning_effort": self.config.reasoning_effort,
                "reflection_lm": self.config.reflection_lm,
                "reflection_reasoning_effort": self.config.reflection_reasoning_effort,
                "train_dataset": self.config.train_dataset,
                "val_ratio": self.config.val_ratio,
                "cases_per_task": self.config.cases_per_task,
                "max_metric_calls": self.config.max_metric_calls,
                "minibatch_size": self.config.minibatch_size,
                "val_limit": self.config.val_limit,
                "seed": self.config.seed,
                "module_selector": self.config.module_selector,
                "rlm_proposer": self.config.rlm_proposer,
                "proposer_max_iterations": self.config.proposer_max_iterations,
                "proposer_timeout": self.config.proposer_timeout,
                "use_optimize_gen": self.config.use_optimize_gen,
                "concurrency": self.config.concurrency,
                "max_iterations": self.config.max_iterations,
                "task_timeout": self.config.task_timeout,
            },
            "run_dir": self.run_dir,
            "best_idx": self.best_idx,
            "best_val_score": self.best_val_score,
            "total_candidates": self.total_candidates,
            "total_metric_calls": self.total_metric_calls,
            "duration_seconds": self.duration_seconds,
            "best_candidate": self.best_candidate,
            "val_aggregate_scores": self.val_aggregate_scores,
            "costs": [c.to_dict() for c in self.costs],
            "rollout_cost_usd": self.rollout_cost_usd,
            "optimization_cost_usd": self.optimization_cost_usd,
            "total_cost_usd": self.total_cost_usd,
        }


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class SpreadsheetAdapter:
    """GEPAAdapter for evolving sig docstring + skill instructions in one run.

    The adapter implements the three Protocol methods required by GEPA:
    :meth:`evaluate`, :meth:`make_reflective_dataset`, and
    :attr:`propose_new_texts`. The first two always execute regardless
    of which proposer is in use; ``propose_new_texts`` is only set when
    the SURGICAL RLM proposer is enabled (via ``proposer_lm`` being
    non-None), in which case GEPA bypasses its default reflection LM
    flow and calls our custom dispatcher.

    On every ``evaluate`` call the adapter constructs a fresh dynamic
    signature (from ``candidate[COMPONENT_SIGNATURE]``) and a fresh
    dynamic skill (from ``candidate[COMPONENT_SKILL]``), so each
    candidate sees a fully-rebuilt RLM pipeline.
    """

    def __init__(
        self,
        lm: dspy.LM,
        sub_lm: dspy.LM,
        max_iterations: int,
        concurrency: int,
        task_timeout: int,
        proposer_lm: dspy.LM | None = None,
        proposer_sub_lm: dspy.LM | None = None,
        proposer_max_iterations: int = 20,
        proposer_trace_dir: str | None = None,
        proposer_timeout: int = 600,
        reflection_lm_instance: dspy.LM | None = None,
        cost_snapshot_path: Path | None = None,
        use_optimize_gen: bool = False,
        verbose_rlm: bool = False,
    ):
        self.lm = lm
        self.sub_lm = sub_lm
        self.max_iterations = max_iterations
        self.concurrency = concurrency
        self.task_timeout = task_timeout
        self.proposer_lm = proposer_lm
        self.proposer_sub_lm = proposer_sub_lm or sub_lm
        self.proposer_max_iterations = proposer_max_iterations
        self.proposer_timeout = max(1, int(proposer_timeout))
        self.proposer_trace_dir = proposer_trace_dir
        self.use_optimize_gen = use_optimize_gen
        self.verbose_rlm = verbose_rlm
        self._proposer_call_count = 0
        # Per-kind evaluate counters, 0-indexed. The two kinds are
        # distinguished by ``capture_traces``: True = minibatch (GEPA
        # wants traces for the reflective dataset), False = full valset
        # (just a promotion/baseline score). Separate counters mean the
        # filenames ``minibatch_NNNN.jsonl`` and ``valset_NNNN.jsonl``
        # are independently numbered — ``valset_0000`` is the seed's
        # baseline, ``minibatch_0000`` is iter 0's first MB, etc.
        self._valset_count = -1
        self._minibatch_count = -1
        # Directory for per-case RunTrace dumps. If only ``cost_snapshot_path``
        # is available, infer the sibling ``task_traces`` directory.
        self.task_trace_dir: Path | None = (
            cost_snapshot_path.parent / "task_traces" if cost_snapshot_path else None
        )
        # Handle to the reflection LM instance so we can summarise its cost
        # even when --rlm_proposer is off (in which case proposer_lm is None
        # but GEPA still uses reflection_lm_instance for its default
        # reflection-LM flow). When --rlm_proposer is on, this is the same
        # object as proposer_lm.
        self.reflection_lm_instance = reflection_lm_instance
        # Append-only JSONL event log. Every completed evaluate() call
        # and every proposer_call writes one JSONL line per (role, model)
        # reporting the per-invocation token/cost usage pulled directly
        # from each PredictRLM's `prediction.trace.usage` (see
        # predict_rlm.trace). No accumulator state, no read-modify-write.
        self.cost_log_path = cost_snapshot_path  # kept kwarg name for callers

        self._resume_from_artifacts()

        # GEPA's adapter Protocol: setting propose_new_texts on the instance
        # opts in to our custom proposer; leaving it None falls back to GEPA's
        # default reflection-LM flow.
        if proposer_lm is not None:
            self.propose_new_texts = self._rlm_propose_new_texts
        else:
            self.propose_new_texts = None

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _max_index_from_proposer_traces(path: str | None) -> int:
        if not path:
            return -1
        trace_dir = Path(path)
        if not trace_dir.is_dir():
            return -1

        pattern = re.compile(r"^proposer_(\d+)_.*\.json$")
        max_idx = -1
        for file in trace_dir.iterdir():
            match = pattern.match(file.name)
            if not match:
                continue
            idx = SpreadsheetAdapter._safe_int(match.group(1))
            if idx is not None:
                max_idx = max(max_idx, idx)
        return max_idx

    @staticmethod
    def _max_indices_from_cost_log(path: Path | None) -> tuple[int, int, int]:
        proposer_max = -1
        minibatch_max = -1
        valset_max = -1

        if path is None or not path.is_file():
            return proposer_max, minibatch_max, valset_max

        try:
            with path.open("r", encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        row = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, dict):
                        continue

                    event = row.get("event")
                    if event in {"proposer_call", "proposer_error"}:
                        idx = SpreadsheetAdapter._safe_int(row.get("proposer_call_idx"))
                        if idx is not None:
                            proposer_max = max(proposer_max, idx)
                        continue
                    if event == "minibatch":
                        idx = SpreadsheetAdapter._safe_int(row.get("evaluate_idx"))
                        if idx is not None:
                            minibatch_max = max(minibatch_max, idx)
                        continue
                    if event == "valset":
                        idx = SpreadsheetAdapter._safe_int(row.get("evaluate_idx"))
                        if idx is not None:
                            valset_max = max(valset_max, idx)
        except Exception:
            return proposer_max, minibatch_max, valset_max

        return proposer_max, minibatch_max, valset_max

    @staticmethod
    def _max_indices_from_task_traces(path: Path | None) -> tuple[int, int]:
        minibatch_max = -1
        valset_max = -1
        if path is None or not path.is_dir():
            return minibatch_max, valset_max

        mb_pat = re.compile(r"^minibatch_(\d+)\.jsonl$")
        vs_pat = re.compile(r"^valset_(\d+)\.jsonl$")

        for file in path.iterdir():
            if not file.is_file():
                continue
            match = mb_pat.match(file.name)
            if match is not None:
                idx = SpreadsheetAdapter._safe_int(match.group(1))
                if idx is not None:
                    minibatch_max = max(minibatch_max, idx)
                continue
            match = vs_pat.match(file.name)
            if match is not None:
                idx = SpreadsheetAdapter._safe_int(match.group(1))
                if idx is not None:
                    valset_max = max(valset_max, idx)
        return minibatch_max, valset_max

    def _resume_from_artifacts(self) -> None:
        proposer_max = max(
            self._max_index_from_proposer_traces(self.proposer_trace_dir),
            self._max_indices_from_cost_log(self.cost_log_path)[0],
        )
        trace_minibatch_max, trace_valset_max = self._max_indices_from_task_traces(
            self.task_trace_dir
        )
        _, cost_minibatch_max, cost_valset_max = self._max_indices_from_cost_log(
            self.cost_log_path
        )

        self._proposer_call_count = max(self._proposer_call_count, proposer_max)
        self._minibatch_count = max(
            self._minibatch_count, trace_minibatch_max, cost_minibatch_max
        )
        self._valset_count = max(
            self._valset_count, trace_valset_max, cost_valset_max
        )

    # -- cost logging -------------------------------------------------------

    def _write_trace_event(
        self,
        event: str,
        *,
        main_model: str,
        main_usage: Any,
        sub_model: str | None,
        sub_usage: Any,
        main_role: str,
        sub_role: str,
        **extra: Any,
    ) -> None:
        """Append one JSONL row per nonzero (role, model) from a RunTrace.

        ``main_usage`` / ``sub_usage`` are ``predict_rlm.trace.TokenUsage``
        objects (attributes: input_tokens, output_tokens, cost). The shape
        mirrors :class:`predict_rlm.trace.RunTrace.usage`; we just flatten
        it into per-role rows with an event label.
        """
        if self.cost_log_path is None:
            return
        try:
            ts = datetime.now().isoformat()
            rows: list[dict] = []
            if main_usage is not None and (
                getattr(main_usage, "input_tokens", 0)
                or getattr(main_usage, "output_tokens", 0)
                or getattr(main_usage, "cache_hits", 0)
            ):
                rows.append(
                    {
                        "ts": ts,
                        "event": event,
                        "role": main_role,
                        "model": main_model,
                        "input_tokens": int(main_usage.input_tokens),
                        "output_tokens": int(main_usage.output_tokens),
                        "cost_usd": float(main_usage.cost),
                        "cache_hits": int(getattr(main_usage, "cache_hits", 0)),
                        **extra,
                    }
                )
            if sub_usage is not None and (
                getattr(sub_usage, "input_tokens", 0)
                or getattr(sub_usage, "output_tokens", 0)
                or getattr(sub_usage, "cache_hits", 0)
            ):
                rows.append(
                    {
                        "ts": ts,
                        "event": event,
                        "role": sub_role,
                        "model": sub_model,
                        "input_tokens": int(sub_usage.input_tokens),
                        "output_tokens": int(sub_usage.output_tokens),
                        "cost_usd": float(sub_usage.cost),
                        "cache_hits": int(getattr(sub_usage, "cache_hits", 0)),
                        **extra,
                    }
                )
            if not rows:
                return
            self.cost_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cost_log_path.open("a") as f:
                for row in rows:
                    f.write(json.dumps(row, default=str) + "\n")
        except Exception as e:
            log.debug("event log append failed: %s", e)

    def _persist_proposer_error(
        self,
        *,
        call_idx: int,
        component_name: str,
        current_text: str,
        exc: BaseException,
    ) -> None:
        """Write an ERROR trace file + cost_log event when a proposer raises.

        On the success path we dump a full ``proposer_NNNN_<component>.json``
        with new_instructions, audit lines, and the RunTrace. On error we
        want an analogous artifact so the 'missing call_idx' mystery never
        recurs: write ``proposer_NNNN_<component>_ERROR.json`` with the
        exception + traceback and the partial RunTrace that PredictRLM
        attached to ``exc.trace``. Emit a ``proposer_error`` row in
        ``cost_log.jsonl`` so the per-iter activity log surfaces the failed
        attempt with token/cost accounting when available.
        """
        import traceback

        from predict_rlm.trace import extract_trace_from_exc

        trace_obj = extract_trace_from_exc(exc)
        run_trace_json: Any = None
        main_usage = None
        sub_usage = None
        main_model = ""
        sub_model: str | None = None
        if trace_obj is not None:
            try:
                run_trace_json = json.loads(trace_obj.to_exportable_json(indent=0))
            except Exception:
                run_trace_json = None
            main_usage = getattr(getattr(trace_obj, "usage", None), "main", None)
            sub_usage = getattr(getattr(trace_obj, "usage", None), "sub", None)
            main_model = str(getattr(trace_obj, "model", ""))
            sub_model = getattr(trace_obj, "sub_model", None)

        # Emit the cost_log event first — cheap and gives us at-a-glance
        # visibility even if the file write fails for some reason.
        self._write_trace_event(
            "proposer_error",
            main_model=main_model,
            main_usage=main_usage,
            sub_model=sub_model,
            sub_usage=sub_usage,
            main_role="proposer",
            sub_role="proposer_sub",
            component=component_name,
            proposer_call_idx=call_idx,
            error=str(exc),
        )

        if not self.proposer_trace_dir:
            return
        try:
            os.makedirs(self.proposer_trace_dir, exist_ok=True)
            payload = {
                "call_idx": call_idx,
                "component": component_name,
                "status": "error",
                "error": str(exc),
                "error_type": type(exc).__name__,
                "traceback": "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
                "current_instructions": current_text,
                "run_trace": run_trace_json,
            }
            out_path = os.path.join(
                self.proposer_trace_dir,
                f"proposer_{call_idx:04d}_{component_name}_ERROR.json",
            )
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2, default=str)
        except Exception as e:
            log.debug("proposer error-trace persist failed: %s", e)

    @staticmethod
    def _sum_traces(traces: list[Any]) -> tuple[Any, Any, str, str | None]:
        """Sum usage across multiple RunTraces. Returns (main, sub, main_model, sub_model).

        Models are taken from the first non-empty trace — they should all
        match within a single evaluate() call since the adapter uses the
        same ``self.lm`` / ``self.sub_lm`` for every task.
        """
        from predict_rlm.trace import TokenUsage

        main = TokenUsage()
        sub = TokenUsage()
        main_model = ""
        sub_model: str | None = None
        for t in traces:
            if t is None:
                continue
            if not main_model:
                main_model = str(getattr(t, "model", ""))
                sub_model = getattr(t, "sub_model", None)
            main += t.usage.main
            if t.usage.sub is not None:
                sub += t.usage.sub
        return main, sub if sub.input_tokens or sub.output_tokens else None, main_model, sub_model

    def _write_case_trace_row(
        self,
        trace_file: Any,
        task: SpreadsheetTask,
        case_result: dict,
    ) -> None:
        """Append one JSONL row to the open per-evaluate task-trace file.

        Called eagerly as each case finishes (via ``asyncio.as_completed``)
        so the file grows in real time instead of being written in one
        shot at the end. Tail the file to watch eval progress live;
        partial reads work because each line is a complete JSON object.
        """
        rt = case_result.get("run_trace")
        row: dict[str, Any] = {
            "task_id": task.task_id,
            "case_idx": case_result.get("idx"),
            "score": case_result.get("score"),
            "passed": case_result.get("passed"),
            "message": case_result.get("message"),
            "instruction_type": task.instruction_type,
        }
        if rt is not None:
            try:
                row["trace"] = json.loads(rt.to_exportable_json(indent=0))
            except Exception:
                row["trace"] = None
        else:
            row["trace"] = None
        try:
            trace_file.write(json.dumps(row, default=str) + "\n")
            trace_file.flush()
        except Exception as e:
            log.debug("case trace row write failed: %s", e)

    def _dump_task_traces(
        self,
        evaluate_idx: int,
        batch: list[SpreadsheetTask],
        task_results: list[tuple[float, list[dict]]],
    ) -> None:
        """Persist one JSONL row per case with the full RunTrace payload.

        Writes ``{self.task_trace_dir}/evaluate_{evaluate_idx:04d}.jsonl``.
        Each row is self-contained: task_id, case idx, score/passed, and
        the RunTrace serialized via ``to_exportable_json`` (which shrinks
        dspy.Image base64 payloads to compact placeholders). Best-effort:
        any failure is swallowed so the eval loop can't be broken.
        """
        if self.task_trace_dir is None:
            return
        try:
            self.task_trace_dir.mkdir(parents=True, exist_ok=True)
            out_path = self.task_trace_dir / f"evaluate_{evaluate_idx:04d}.jsonl"
            with out_path.open("w") as f:
                for task, (_score, case_results) in zip(batch, task_results):
                    for c in case_results:
                        rt = c.get("run_trace")
                        row: dict[str, Any] = {
                            "task_id": task.task_id,
                            "case_idx": c.get("idx"),
                            "score": c.get("score"),
                            "passed": c.get("passed"),
                            "message": c.get("message"),
                            "instruction_type": task.instruction_type,
                        }
                        if rt is not None:
                            # to_exportable_json returns a string; re-parse
                            # so the JSONL row remains one self-contained object.
                            try:
                                row["trace"] = json.loads(rt.to_exportable_json(indent=0))
                            except Exception:
                                row["trace"] = None
                        else:
                            row["trace"] = None
                        f.write(json.dumps(row, default=str) + "\n")
        except Exception as e:
            log.debug("task trace dump failed: %s", e)

    # -- evaluate -----------------------------------------------------------

    def evaluate(
        self,
        batch: list[SpreadsheetTask],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        return asyncio.get_event_loop().run_until_complete(
            self._evaluate_async(batch, candidate, capture_traces)
        )

    async def _evaluate_async(
        self,
        batch: list[SpreadsheetTask],
        candidate: dict[str, str],
        capture_traces: bool,
    ) -> EvaluationBatch:
        # Signature is frozen at the pristine class docstring — GEPA
        # only ever evolves the skill now, so candidate may not carry
        # COMPONENT_SIGNATURE. Fall back to the original docstring.
        sig_text = candidate.get(
            COMPONENT_SIGNATURE, ManipulateSpreadsheet.__doc__ or ""
        )
        sig_cls = make_dynamic_signature(sig_text)
        skill = make_dynamic_skill(candidate[COMPONENT_SKILL])

        tmp_dir = tempfile.mkdtemp(prefix="gepa_eval_")
        rlm_sem = asyncio.Semaphore(self.concurrency)
        pbar = tqdm(
            total=len(batch),
            desc=f"  eval ({len(batch)} tasks)",
            leave=False,
        )

        # Open the per-evaluate task-trace file eagerly so each case can
        # append its row as it finishes — not all at once at the end.
        # On a 100-case val eval with concurrency=30, this means you can
        # tail the file and watch progress in real time instead of
        # waiting 5-10 min for the final write.
        #
        # Filename prefix distinguishes minibatch (GEPA's reflective
        # dataset source, capture_traces=True) from valset (full
        # promotion/baseline eval, capture_traces=False). Each has its
        # own 0-indexed counter so ``valset_0000`` = seed VAL,
        # ``minibatch_0000`` = iter 0's MB, and so on.
        eval_kind = "minibatch" if capture_traces else "valset"
        if eval_kind == "minibatch":
            eval_idx = self._minibatch_count + 1
        else:
            eval_idx = self._valset_count + 1
        trace_file = None
        trace_lock = asyncio.Lock()
        if self.task_trace_dir is not None:
            try:
                self.task_trace_dir.mkdir(parents=True, exist_ok=True)
                trace_path = self.task_trace_dir / f"{eval_kind}_{eval_idx:04d}.jsonl"
                trace_file = trace_path.open("w")
            except Exception as e:
                log.debug("failed to open task_traces file: %s", e)
                trace_file = None

        try:
            async def _process_task(task: SpreadsheetTask):
                case_coros = [
                    self._process_case(
                        task, idx, input_path, answer_path,
                        sig_cls, skill, rlm_sem, tmp_dir, capture_traces,
                    )
                    for idx, input_path, answer_path in task.test_cases
                ]
                case_results = []
                for coro in asyncio.as_completed(case_coros):
                    c = await coro
                    case_results.append(c)
                    # Append this case's trace row as soon as it's ready.
                    if trace_file is not None:
                        async with trace_lock:
                            self._write_case_trace_row(trace_file, task, c)
                # Re-sort by case idx so downstream readers see canonical order
                case_results.sort(key=lambda c: c.get("idx", 0))
                score = (
                    sum(c["score"] for c in case_results) / len(case_results)
                    if case_results
                    else 0.0
                )
                pbar.set_postfix_str(f"{task.task_id}={score:.0%}")
                pbar.update(1)
                return score, case_results

            task_results = await asyncio.gather(
                *(_process_task(t) for t in batch)
            )
        finally:
            pbar.close()
            shutil.rmtree(tmp_dir, ignore_errors=True)
            if trace_file is not None:
                try:
                    trace_file.close()
                except Exception:
                    pass

        scores: list[float] = []
        outputs: list[dict] = []
        trajectories: list[dict] | None = [] if capture_traces else None

        for task, (score, case_results) in zip(batch, task_results):
            scores.append(score)
            outputs.append({"task_id": task.task_id, "score": score})
            if capture_traces:
                num_passed = sum(1 for c in case_results if c["passed"])
                trajectories.append(
                    {
                        "task_id": task.task_id,
                        "instruction": task.instruction,
                        "instruction_type": task.instruction_type,
                        "num_passed": num_passed,
                        "num_total": len(case_results),
                        "cases": case_results,
                    }
                )

        # Aggregate per-case RunTrace.usage into one evaluate event.
        # Each PredictRLM call attaches a trace with its own token/cost
        # totals; we just sum across the cases that actually ran.
        all_traces: list[Any] = []
        for _, case_results in task_results:
            for c in case_results:
                rt = c.get("run_trace")
                if rt is not None:
                    all_traces.append(rt)
        main_usage, sub_usage, main_model, sub_model = self._sum_traces(all_traces)
        if eval_kind == "minibatch":
            self._minibatch_count += 1
        else:
            self._valset_count += 1
        self._write_trace_event(
            eval_kind,
            main_model=main_model,
            main_usage=main_usage,
            sub_model=sub_model,
            sub_usage=sub_usage,
            main_role="main",
            sub_role="sub",
            evaluate_idx=eval_idx,
            kind=eval_kind,
            cases=sum(len(cr) for _, cr in task_results),
            traces_captured=len(all_traces),
        )

        # Note: task_traces/{kind}_NNNN.jsonl was written incrementally
        # by ``_write_case_trace_row`` as each case finished — no final
        # flush step needed here.

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    @staticmethod
    def _recover_partial_trace(predictor: "PredictRLM | None") -> list[dict]:
        """Read whatever REPL steps completed before a crash.

        PredictRLM snapshots its history after every iteration and captures
        reasoning+code before each execution attempt, so a timeout or
        exception still leaves a visible trail of what the agent did.
        """
        if predictor is None:
            return []
        out: list[dict] = []
        hist = getattr(predictor, "_partial_history", None)
        if hist is not None and hasattr(hist, "entries"):
            for entry in hist.entries:
                out.append(
                    {
                        "reasoning": getattr(entry, "reasoning", "") or "",
                        "code": getattr(entry, "code", "") or "",
                        "output": getattr(entry, "output", "") or "",
                    }
                )
        pending = getattr(predictor, "_partial_pending_entry", None)
        if pending is not None:
            out.append(
                {
                    "reasoning": getattr(pending, "reasoning", "") or "",
                    "code": getattr(pending, "code", "") or "",
                    "output": "(execution did not complete)",
                }
            )
        return out

    async def _process_case(
        self,
        task: SpreadsheetTask,
        idx: int,
        input_path: str,
        answer_path: str | None,
        sig_cls: type[dspy.Signature],
        skill: Skill,
        rlm_sem: asyncio.Semaphore,
        tmp_dir: str,
        capture_traces: bool,
    ) -> dict:
        """Pipeline a single test case through RLM → recalc → score."""
        output_path = os.path.join(tmp_dir, f"{idx}_{task.task_id}_output.xlsx")
        trace: list = []

        answer_sheet, answer_range = await asyncio.to_thread(
            parse_answer_position, task.answer_position, input_path
        )
        formatted_instruction = _build_instruction(
            task.instruction, answer_range, answer_sheet, task.instruction_type
        )

        async with rlm_sem:
            predictor: "PredictRLM | None" = None
            run_trace: Any = None
            try:
                predictor = PredictRLM(
                    sig_cls,
                    lm=self.lm,
                    sub_lm=self.sub_lm,
                    skills=[skill],
                    max_iterations=self.max_iterations,
                    verbose=self.verbose_rlm,
                    debug=False,
                )
                result = await asyncio.wait_for(
                    predictor.acall(
                        input_spreadsheet=File(path=input_path),
                        instruction=formatted_instruction,
                    ),
                    timeout=self.task_timeout,
                )
                run_trace = getattr(result, "trace", None)
                if capture_traces:
                    trace = getattr(result, "trajectory", []) or []
                if not (
                    result
                    and result.output_spreadsheet
                    and result.output_spreadsheet.path
                    and os.path.exists(result.output_spreadsheet.path)
                ):
                    # Fall back to the partial-trajectory capture if
                    # capture_traces was False so the proposer still sees
                    # what the agent did in its REPL steps.
                    if not trace:
                        trace = self._recover_partial_trace(predictor)
                    if len(trace) >= self.max_iterations:
                        msg = (
                            f"RLM max iterations reached "
                            f"({self.max_iterations}) without SUBMIT"
                        )
                    else:
                        msg = "RLM returned without producing output_spreadsheet"
                    return {
                        "idx": idx,
                        "score": 0.0,
                        "passed": False,
                        "message": msg,
                        "trace": trace,
                        "run_trace": run_trace,
                    }
                shutil.copy2(result.output_spreadsheet.path, output_path)
            except asyncio.TimeoutError as e:
                from predict_rlm.trace import extract_trace_from_exc

                return {
                    "idx": idx,
                    "score": 0.0,
                    "passed": False,
                    "message": f"RLM timeout at {self.task_timeout}s",
                    "trace": self._recover_partial_trace(predictor),
                    "run_trace": extract_trace_from_exc(e),
                }
            except Exception as e:
                from predict_rlm.trace import extract_trace_from_exc

                return {
                    "idx": idx,
                    "score": 0.0,
                    "passed": False,
                    "message": f"RLM {type(e).__name__}: {e}",
                    "trace": self._recover_partial_trace(predictor),
                    "run_trace": extract_trace_from_exc(e),
                }

        # Host-side recalc — the same two-stage pipeline used by eval.py.
        try:
            await asyncio.to_thread(recalculate, output_path)
        except Exception:
            pass

        if answer_path is None:
            return {
                "idx": idx,
                "score": 0.0,
                "passed": False,
                "message": "Answer file not found",
                "trace": trace,
                "run_trace": run_trace,
            }

        try:
            ratio, msg = await asyncio.to_thread(
                score_workbooks,
                answer_path,
                output_path,
                task.instruction_type,
                task.answer_position,
            )
            return {
                "idx": idx,
                "score": ratio,
                "passed": ratio == 1.0,
                "message": msg,
                "trace": trace,
                "run_trace": run_trace,
            }
        except Exception as e:
            return {
                "idx": idx,
                "score": 0.0,
                "passed": False,
                "message": f"Comparison error: {e}",
                "trace": trace,
                "run_trace": run_trace,
            }

    # -- make_reflective_dataset --------------------------------------------

    @staticmethod
    def _render_trace(trace: Any, tag: str) -> str:
        """Render a case's trace for the proposer.

        Accepts either:
          - a ``RunTrace`` (preferred) — rich render with per-step timing,
            ``tool_calls`` (name, args, result / error), and ``predict_calls``
            (signature, model, aggregated usage). This is the same trace
            representation we persist in ``task_traces/evaluate_NNNN.jsonl``,
            so the proposer sees exactly what observability sees.
          - a ``list[dict]`` trajectory (legacy) — plain reasoning / code /
            output per step. Used when ``run_trace`` is missing (e.g. a
            partial trajectory recovered from a crash before ``RunTrace``
            could be built).
        """
        from predict_rlm.trace import RunTrace

        if isinstance(trace, RunTrace):
            return SpreadsheetAdapter._render_run_trace(trace, tag)
        if not trace:
            return f"({tag}: no trace captured)"
        # Legacy trajectory path.
        parts = []
        n = len(trace)
        for i, step in enumerate(trace):
            label = f"{tag}: step {i + 1} of {n}"
            ban = f"** {label} **"
            rule = "*" * len(ban)
            parts.append(f"{rule}\n{ban}\n{rule}")
            reasoning = step.get("reasoning", "")
            code = step.get("code", "")
            output = step.get("output", "")
            if reasoning:
                parts.append(f"REASONING:\n{reasoning}")
            if code:
                parts.append(f"CODE:\n{code}")
            if output:
                parts.append(f"OUTPUT:\n{output}")
        return "\n\n".join(parts)

    @staticmethod
    def _render_run_trace(rt: Any, tag: str) -> str:
        steps = list(getattr(rt, "steps", []) or [])
        if not steps:
            return f"({tag}: no steps captured; status={rt.status})"
        parts: list[str] = []
        n = len(steps)
        for i, step in enumerate(steps):
            label = (
                f"{tag}: step {i + 1} of {n} — {step.duration_ms:,}ms"
                + (" (ERROR)" if step.error else "")
            )
            ban = f"** {label} **"
            rule = "*" * len(ban)
            parts.append(f"{rule}\n{ban}\n{rule}")
            if step.reasoning:
                parts.append(f"REASONING:\n{step.reasoning}")
            if step.code:
                parts.append(f"CODE:\n{step.code}")
            tool_lines = SpreadsheetAdapter._format_tool_calls(step.tool_calls)
            if tool_lines:
                parts.append("TOOL CALLS:\n" + tool_lines)
            predict_lines = SpreadsheetAdapter._format_predict_calls(step.predict_calls)
            if predict_lines:
                parts.append("PREDICT CALLS:\n" + predict_lines)
            if step.output:
                parts.append(f"OUTPUT:\n{step.output}")
        return "\n\n".join(parts)

    @staticmethod
    def _format_tool_calls(tool_calls: list[Any]) -> str:
        if not tool_calls:
            return ""
        lines: list[str] = []
        for tc in tool_calls:
            kwargs = getattr(tc, "kwargs", {}) or {}
            args = getattr(tc, "args", []) or []
            arg_strs = [repr(a) for a in args] + [
                f"{k}={v!r}" for k, v in kwargs.items()
            ]
            sig = f"{tc.name}({', '.join(arg_strs)})"
            err = getattr(tc, "error", None)
            if err:
                outcome = f"ERROR: {err}"
            else:
                result_str = str(tc.result or "")
                if len(result_str) > 200:
                    result_str = result_str[:200] + f"... (truncated, {len(str(tc.result))} chars)"
                outcome = f"→ {result_str}"
            lines.append(f"  {sig} {outcome} ({tc.duration_ms:,}ms)")
        return "\n".join(lines)

    @staticmethod
    def _format_predict_calls(groups: list[Any]) -> str:
        if not groups:
            return ""
        lines: list[str] = []
        for g in groups:
            tu = g.total_usage
            n = len(g.calls)
            lines.append(
                f"  [{n}×] sig={g.signature!r} model={g.model} "
                f"total in={tu.input_tokens:,} out={tu.output_tokens:,} "
                f"cost=${tu.cost:.4f}"
            )
        return "\n".join(lines)

    @staticmethod
    def _case_summary(c: dict) -> str:
        base = (
            f"case {c['idx']}: score={c['score']:.2f} "
            f"({'PASS' if c['passed'] else 'FAIL'})"
        )
        rt = c.get("run_trace")
        if rt is None:
            return base
        status = getattr(rt, "status", None)
        iters = getattr(rt, "iterations", None)
        max_iters = getattr(rt, "max_iterations", None)
        dur_ms = getattr(rt, "duration_ms", 0) or 0
        usage = getattr(rt, "usage", None)
        extras: list[str] = []
        if status:
            extras.append(f"status={status}")
        if iters is not None and max_iters:
            extras.append(f"{iters}/{max_iters} iters")
        elif iters is not None:
            extras.append(f"{iters} iters")
        if dur_ms:
            extras.append(f"{dur_ms/1000:.1f}s total ({dur_ms:,}ms)")
        if usage is not None:
            main = getattr(usage, "main", None)
            if main is not None and (main.input_tokens or main.output_tokens):
                extras.append(
                    f"main={main.input_tokens:,}/{main.output_tokens:,} tokens"
                )
            sub = getattr(usage, "sub", None)
            if sub is not None and (sub.input_tokens or sub.output_tokens):
                extras.append(
                    f"sub={sub.input_tokens:,}/{sub.output_tokens:,} tokens"
                )
        if not extras:
            return base
        return base + " | " + " ".join(extras)

    @staticmethod
    def _banner(text: str) -> str:
        line = f"== {text} =="
        rule = "=" * len(line)
        return f"{rule}\n{line}\n{rule}"

    @staticmethod
    def _component_focus(component_name: str) -> str:
        """Build the per-component framing passed as `component_focus`.

        This is orthogonal to the per-item `Feedback` field — it tells
        the proposer which of several components it's currently editing
        and what failure modes that component can actually fix. Returns
        an empty string for single-component skills (the generic
        template stands on its own).
        """
        common: list[str] = []
        if component_name == COMPONENT_SIGNATURE:
            specific = [
                "## You Are Editing The Signature Docstring",
                "The signature docstring controls the high-level workflow the "
                "RLM follows: how it loads the input, how it inspects the "
                "spreadsheet, when it writes versus computes, when it "
                "verifies, when it submits. It does NOT control domain rules "
                "about which Excel formulas work in LibreOffice — those live "
                "in the skill instructions, which is a separate component.",
                "",
                "## What To Look For",
                "Compare the BEST and WORST execution traces. Look for "
                "workflow-level patterns:",
                "- Did the RLM skip an inspection step that would have "
                "  revealed the right column/sheet/range?",
                "- Did it submit before verifying that target cells were "
                "  populated?",
                "- Did it call recalculate() after writing formulas, or did "
                "  it submit while target cells were still None?",
                "- Did it follow the instruction literally, or did it "
                "  reinterpret the target location based on what looked "
                "  reasonable?",
                "- Did it preserve the original sheet names and structure?",
                "",
                "Update the workflow steps to prevent the procedural failures "
                "you see in WORST traces while preserving what worked in BEST.",
            ]
        elif component_name == COMPONENT_SKILL:
            specific = [
                "## You Are Editing The Skill Instructions",
                "The skill instructions are the LO-compatible spreadsheet "
                "skill's domain-knowledge prose. It tells the RLM which "
                "Excel formulas evaluate cleanly in LibreOffice, which ones "
                "silently fail, common openpyxl pitfalls, and when to use "
                "Python-computed literal values versus formulas. It does NOT "
                "control the high-level workflow — that lives in the "
                "signature docstring, which is a separate component.",
                "",
                "## What To Look For",
                "Compare the BEST and WORST execution traces. Look for "
                "domain-rule failures:",
                "- Did the RLM write formulas that evaluate to None? "
                "  (spill functions, dynamic-array tricks, CSE-required "
                "  patterns.)",
                "- Did it use Excel-365-only functions that produce "
                "  #NAME? in LibreOffice?",
                "- Did it use the wrong data type (number vs string, "
                "  datetime vs time-only)?",
                "- Did it hit openpyxl gotchas (merged cells, "
                "  delete_rows hangs, full-column refs)?",
                "- Did it use the right normalization (TRIM in lookups, "
                "  matching the source column's casing/abbreviations)?",
                "",
                "Update the domain rules to warn against the failure modes "
                "you see in WORST traces. Keep examples concrete.",
            ]
        else:
            specific = []
        return "\n".join(common + specific)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        result: dict[str, list[dict]] = {}
        for component_name in components_to_update:
            result[component_name] = self._build_records_for_component(
                eval_batch, component_name
            )
        return result

    def _build_records_for_component(
        self,
        eval_batch: EvaluationBatch,
        component_name: str,
    ) -> list[dict]:
        records: list[dict] = []
        for trajectory, score in zip(eval_batch.trajectories, eval_batch.scores):
            cases = trajectory["cases"]
            cases_sorted = sorted(cases, key=lambda c: c["score"])

            inputs_summary = (
                f"Instruction type: {trajectory['instruction_type']}\n\n"
                f"Instruction:\n{trajectory['instruction']}"
            )

            # Per-item Feedback: task-level score + per-case pass/fail,
            # mismatch diffs, and crash reasons. This is the evaluator
            # signal for THIS record — per GEPA's reflective-dataset
            # contract — not a global meta-prompt.
            feedback_lines = [
                f"Task score: {score:.3f} (mean of case scores)",
                "",
                "Per-case results:",
            ]
            for c in cases_sorted:
                header = f"  {self._case_summary(c)}"
                detail = (c.get("message") or "").strip()
                if detail:
                    detail_indented = "\n".join(
                        f"    {line}" for line in detail.splitlines()
                    )
                    feedback_lines.append(f"{header}\n{detail_indented}")
                else:
                    feedback_lines.append(header)
            feedback = "\n".join(feedback_lines)

            # Partition cases for the WORST / BEST renders. Crashed cases
            # (timeouts, exceptions, max-iterations-without-SUBMIT) show up
            # with score 0.0 and whatever partial trajectory PredictRLM
            # captured before the failure. They always belong in the WORST
            # bucket so the proposer can write corrective rules against
            # real crash reasons, not just cell-level mismatches.
            def _is_crash(case: dict) -> bool:
                msg = (case.get("message") or "").lower()
                return (
                    case["score"] == 0.0
                    and (
                        msg.startswith("rlm timeout")
                        or msg.startswith("rlm max iterations")
                        or msg.startswith("rlm returned without producing")
                        or (msg.startswith("rlm ") and ":" in msg)
                    )
                )

            # Prefer the RunTrace (rich: tool_calls, predict_calls, status,
            # per-step timing, token usage) — same representation persisted
            # under task_traces/evaluate_NNNN.jsonl for observability. Fall
            # back to the flat trajectory dict list when run_trace is missing
            # (partial trajectory recovered from a crash before the trace
            # could be built).
            def _case_trace(case: dict) -> Any:
                rt = case.get("run_trace")
                if rt is not None:
                    return rt
                return case.get("trace") or []

            crashed_cases = [c for c in cases_sorted if _is_crash(c)]
            scored_cases = [
                c for c in cases_sorted
                if (c.get("run_trace") is not None or c.get("trace"))
                and not _is_crash(c)
            ]

            trace_parts: list[str] = []

            # WORST slot: crashes first (rarest and most informative),
            # then the lowest-scored completed cases.
            for c in crashed_cases[:2]:
                idx = c["idx"]
                header = (
                    f"CRASHED — {self._case_summary(c)}\n"
                    f"Reason: {c['message']}"
                )
                trace_parts.append(
                    f"{self._banner(header)}\n\n"
                    f"{self._render_trace(_case_trace(c), f'CRASHED CASE {idx}')}"
                )

            slots_left = max(0, 2 - len(trace_parts))
            for c in scored_cases[:slots_left]:
                idx = c["idx"]
                header = (
                    f"WORST — {self._case_summary(c)}\n"
                    f"Cell mismatches:\n{c['message']}"
                )
                trace_parts.append(
                    f"{self._banner(header)}\n\n"
                    f"{self._render_trace(_case_trace(c), f'WORST CASE {idx}')}"
                )

            # BEST slot: highest-scoring completed case, when it actually
            # beats the worst. Crashes are never "best".
            if scored_cases and scored_cases[-1]["score"] > cases_sorted[0]["score"]:
                c = scored_cases[-1]
                idx = c["idx"]
                header = f"BEST — {self._case_summary(c)}"
                if c["passed"]:
                    header += "\nAll cells matched."
                else:
                    header += f"\nCell mismatches:\n{c['message']}"
                trace_parts.append(
                    f"{self._banner(header)}\n\n"
                    f"{self._render_trace(_case_trace(c), f'BEST CASE {idx}')}"
                )

            traces_text = (
                "\n\n\n".join(trace_parts) if trace_parts else "(no traces)"
            )

            records.append(
                {
                    "Inputs": inputs_summary,
                    "Generated Outputs": traces_text,
                    "Feedback": feedback,
                }
            )
        return records

    # -- propose_new_texts (RLM proposer) -----------------------------------

    def _rlm_propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """Multi-component dispatch: one PredictRLM call per component."""
        new_texts: dict[str, str] = {}
        for component_name in components_to_update:
            if component_name not in candidate:
                continue
            current_text = candidate[component_name]
            records = list(reflective_dataset.get(component_name, []))
            new_text = self._propose_one_component(
                component_name, current_text, records
            )
            new_texts[component_name] = new_text
        return new_texts

    def _propose_one_component(
        self,
        component_name: str,
        current_text: str,
        records: list[Mapping[str, Any]],
    ) -> str:
        """Run the SURGICAL RLM proposer on a single component's records."""
        self._proposer_call_count += 1
        call_idx = self._proposer_call_count

        serializable = [
            {
                "Inputs": r.get("Inputs", ""),
                "Generated Outputs": r.get("Generated Outputs", ""),
                "Feedback": r.get("Feedback", ""),
            }
            for r in records
        ]

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix="_traces.json",
            delete=False,
            prefix=f"gepa_proposer_{component_name}_",
        ) as f:
            json.dump(serializable, f, indent=2)
            traces_path = f.name

        if self.use_optimize_gen:
            from lib.optimize_gen import (
                SPREADSHEET_SPEC,
                build_proposer_signature,
            )
            proposer_sig = build_proposer_signature(SPREADSHEET_SPEC)
        else:
            proposer_sig = ImproveInstructions

        try:
            predictor = PredictRLM(
                proposer_sig,
                lm=self.proposer_lm,
                sub_lm=self.proposer_sub_lm,
                skills=[],
                max_iterations=self.proposer_max_iterations,
                verbose=True,
                debug=False,
            )

            # Stream proposer iterations to stdout via tqdm.write so they
            # don't get clobbered by the GEPA progress bar.
            rlm_logger = logging.getLogger("dspy.predict.rlm")
            old_level = rlm_logger.level
            old_propagate = rlm_logger.propagate
            rlm_logger.setLevel(logging.DEBUG)
            rlm_logger.propagate = False

            class _StreamWriter(logging.Handler):
                def emit(self, record):
                    try:
                        tqdm.write(self.format(record))
                    except Exception:
                        pass

            stream_handler = _StreamWriter()
            stream_handler.setFormatter(
                logging.Formatter(f"[PROPOSER {component_name}] %(message)s")
            )
            rlm_logger.addHandler(stream_handler)

            tqdm.write("\n" + "=" * 80)
            tqdm.write(f"RLM PROPOSER STARTING ({component_name})")
            tqdm.write("=" * 80)

            acall_kwargs: dict[str, Any] = {
                "current_instructions": current_text,
                "traces_file": File(path=traces_path),
            }
            if (
                self.use_optimize_gen
                and "component_focus" in proposer_sig.input_fields
            ):
                acall_kwargs["component_focus"] = self._component_focus(
                    component_name
                )

            try:
                proposer_timeout = max(1, int(getattr(self, "proposer_timeout", 600)))
                try:
                    loop = asyncio.get_event_loop()
                    result = loop.run_until_complete(
                        asyncio.wait_for(
                            predictor.acall(**acall_kwargs),
                            timeout=proposer_timeout,
                        )
                    )
                finally:
                    rlm_logger.removeHandler(stream_handler)
                    rlm_logger.setLevel(old_level)
                    rlm_logger.propagate = old_propagate
            except BaseException as exc:
                # Proposer runs error out for real reasons (sandbox timeouts,
                # gemini rate limits, empty outputs). Persist what we have —
                # PredictRLM attaches its partial RunTrace via ``exc.trace``
                # on error — and surface the call in cost_log so the iter
                # log isn't silently missing these attempts. Then re-raise
                # so GEPA records the failed proposal as a 0-score minibatch.
                self._persist_proposer_error(
                    call_idx=call_idx,
                    component_name=component_name,
                    current_text=current_text,
                    exc=exc,
                )
                raise

            tqdm.write("=" * 80)
            tqdm.write(
                f"RLM PROPOSER DONE ({component_name}, "
                f"{len(getattr(result, 'trajectory', []))} iterations)"
            )
            tqdm.write("=" * 80 + "\n")

            new_text = getattr(result, "new_instructions", None)
            if not new_text or not isinstance(new_text, str) or not new_text.strip():
                raise RuntimeError(
                    f"RLM proposer returned empty new_instructions "
                    f"for {component_name}"
                )

            # Log one proposer_call event per component using the RunTrace
            # the proposer's PredictRLM attached to its Prediction.
            proposer_trace = getattr(result, "trace", None)
            if proposer_trace is not None:
                self._write_trace_event(
                    "proposer_call",
                    main_model=str(getattr(proposer_trace, "model", "")),
                    main_usage=proposer_trace.usage.main,
                    sub_model=getattr(proposer_trace, "sub_model", None),
                    sub_usage=proposer_trace.usage.sub,
                    main_role="proposer",
                    sub_role="proposer_sub",
                    component=component_name,
                    proposer_call_idx=call_idx,
                    iterations=proposer_trace.iterations,
                )

            if self.proposer_trace_dir:
                try:
                    os.makedirs(self.proposer_trace_dir, exist_ok=True)
                    # Attach the full RunTrace (sanitized) so postmortem
                    # readers see the proposer's per-iteration reasoning,
                    # code, tool calls, and token usage — not just the
                    # flat trajectory list.
                    run_trace_json: Any = None
                    if proposer_trace is not None:
                        try:
                            run_trace_json = json.loads(
                                proposer_trace.to_exportable_json(indent=0)
                            )
                        except Exception:
                            run_trace_json = None
                    payload = {
                        "call_idx": call_idx,
                        "component": component_name,
                        "current_instructions": current_text,
                        "new_instructions": new_text,
                        "generalization_check": getattr(
                            result, "generalization_check", None
                        ),
                        "reflective_dataset": serializable,
                        "rlm_trajectory": getattr(result, "trajectory", []),
                        "run_trace": run_trace_json,
                    }
                    out_path = os.path.join(
                        self.proposer_trace_dir,
                        f"proposer_{call_idx:04d}_{component_name}.json",
                    )
                    with open(out_path, "w") as f:
                        json.dump(payload, f, indent=2, default=str)
                except Exception as e:
                    tqdm.write(f"[PROPOSER {component_name}] persist failed: {e}")

            return new_text
        finally:
            try:
                os.unlink(traces_path)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def _resolve_run_dir(config: OptimizeConfig) -> Path:
    if config.run_dir is not None:
        return Path(config.run_dir)
    example_dir = Path(__file__).resolve().parent.parent.parent
    runs_dir = example_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return runs_dir / f"optimize_{timestamp}"


def _extract_reflection_lm_text(value: Any, *, join_lists: bool = False) -> str | None:
    """Extract assistant text from the response shapes DSPy/LiteLLM can return."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, Mapping):
        for key in ("output_text", "text", "completion"):
            if key in value:
                text = _extract_reflection_lm_text(value[key], join_lists=True)
                if text is not None:
                    return text

        for key in ("message", "delta", "response"):
            if key in value:
                text = _extract_reflection_lm_text(value[key])
                if text is not None:
                    return text

        for key, nested_join in (
            ("content", True),
            ("output", True),
            ("choices", False),
        ):
            if key in value:
                text = _extract_reflection_lm_text(
                    value[key], join_lists=nested_join
                )
                if text is not None:
                    return text
        return None

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        texts: list[str] = []
        for item in value:
            text = _extract_reflection_lm_text(item, join_lists=True)
            if text:
                texts.append(text)
        if texts:
            return "\n".join(texts) if join_lists else texts[0]
        return "" if not value else None

    for attr in ("output_text", "text", "content", "message"):
        if hasattr(value, attr):
            text = _extract_reflection_lm_text(getattr(value, attr), join_lists=True)
            if text is not None:
                return text

    return None


def _coerce_reflection_lm_text(value: Any) -> str:
    text = _extract_reflection_lm_text(value)
    if text is not None:
        return text

    detail = type(value).__name__
    if isinstance(value, Mapping):
        detail += f" keys={sorted(str(k) for k in value.keys())}"
    raise TypeError(f"Reflection LM returned non-text response ({detail})")


def run_optimization(config: OptimizeConfig) -> OptimizeReport:
    """Run GEPA multi-component optimization described by *config*.

    Builds the task LM, sub-LM, reflection LM, dataset split, and a
    :class:`SpreadsheetAdapter`, then hands off to ``gepa.optimize``.
    Returns an :class:`OptimizeReport` with the best candidate (both
    components), aggregated cost summary, and run metadata. The full
    GEPA state is persisted to ``{run_dir}/gepa_state.bin`` so the run
    is resumable by re-launching with the same ``config.run_dir``.
    """
    run_dir = _resolve_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # --- Dataset + train/val split ----------------------------------------
    print(f"Loading dataset: {config.train_dataset}")
    full_train = load_dataset(
        config.train_dataset, max_cases_per_task=config.cases_per_task
    )
    train, val = split_train_val(full_train, config.val_ratio, config.seed)
    write_split_log(run_dir, config.seed, config.val_ratio, train, val)
    print(
        f"Split (seed={config.seed}, val_ratio={config.val_ratio}): "
        f"{len(train)} train / {len(val)} val "
        f"(of {len(full_train)} total)"
    )

    if config.val_limit and len(val) > config.val_limit:
        rng = random.Random(config.seed)
        rng.shuffle(val)
        val = val[: config.val_limit]
        print(f"  capped val to {len(val)} tasks via val_limit={config.val_limit}")

    # --- Task LMs ----------------------------------------------------------
    lm = dspy.LM(**get_lm_config(config.lm, config.reasoning_effort), cache=config.cache)
    sub_lm = dspy.LM(**get_sub_lm_config(config.sub_lm), cache=config.cache)
    task_effort = config.reasoning_effort if config.reasoning_effort else "none"
    print(f"Task LM:    {config.lm}  (reasoning_effort={task_effort})")
    print(f"Task sub:   {config.sub_lm}")

    # --- Reflection / proposer LM ------------------------------------------
    reflection_lm_kwargs: dict = {"max_tokens": 32768, "num_retries": 5}
    if config.reflection_reasoning_effort:
        reflection_lm_kwargs["reasoning_effort"] = config.reflection_reasoning_effort
    reflection_lm_instance = dspy.LM(
        config.reflection_lm, **reflection_lm_kwargs, cache=config.cache
    )

    def reflection_lm_call(prompt: str) -> str:
        return _coerce_reflection_lm_text(reflection_lm_instance(prompt))

    reflect_effort = (
        config.reflection_reasoning_effort
        if config.reflection_reasoning_effort
        else "none"
    )
    print(f"Reflection: {config.reflection_lm}  (reasoning_effort={reflect_effort})")

    # --- Adapter ------------------------------------------------------------
    proposer_trace_dir = (
        str(run_dir / "proposer_traces") if config.rlm_proposer else None
    )
    proposer_sub_lm: dspy.LM | None = None
    if config.rlm_proposer:
        proposer_sub_lm = dspy.LM(
            **get_lm_config(
                config.proposer_sub_lm,
                reasoning_effort=config.proposer_sub_lm_reasoning_effort,
            ),
            cache=config.cache,
        )
        proposer_sub_effort = (
            config.proposer_sub_lm_reasoning_effort
            if config.proposer_sub_lm_reasoning_effort
            else "none"
        )
        print(
            f"Proposer sub: {config.proposer_sub_lm}  "
            f"(reasoning_effort={proposer_sub_effort})"
        )

    adapter = SpreadsheetAdapter(
        lm=lm,
        sub_lm=sub_lm,
        max_iterations=config.max_iterations,
        concurrency=config.concurrency,
        task_timeout=config.task_timeout,
        proposer_lm=reflection_lm_instance if config.rlm_proposer else None,
        proposer_sub_lm=proposer_sub_lm,
        proposer_max_iterations=config.proposer_max_iterations,
        proposer_timeout=config.proposer_timeout,
        proposer_trace_dir=proposer_trace_dir,
        reflection_lm_instance=reflection_lm_instance,
        cost_snapshot_path=run_dir / "cost_log.jsonl",
        use_optimize_gen=config.use_optimize_gen,
        verbose_rlm=config.verbose_rlm,
    )
    # Point the adapter at the per-evaluate task-trace directory.
    # One JSONL file per evaluate() call, one row per case, each row
    # carrying the full RunTrace via to_exportable_json().
    adapter.task_trace_dir = run_dir / "task_traces"
    # Anchor the session boundary so downstream log readers can tell
    # per-session subtotals apart. No trace exists at startup (no LM
    # calls yet), so we write a sentinel row directly.
    if adapter.cost_log_path is not None:
        try:
            adapter.cost_log_path.parent.mkdir(parents=True, exist_ok=True)
            with adapter.cost_log_path.open("a") as _f:
                _f.write(json.dumps({
                    "ts": datetime.now().isoformat(),
                    "event": "startup",
                    "role": None,
                    "model": None,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                }) + "\n")
        except Exception as e:
            log.debug("startup marker write failed: %s", e)
    if config.rlm_proposer:
        proposer_label = (
            f"RLM {'GENERIC (optimize_gen)' if config.use_optimize_gen else 'SURGICAL'} "
            f"({config.reflection_lm}, max_iters={config.proposer_max_iterations})"
        )
    else:
        proposer_label = "GEPA default reflection LM"
    print(f"Proposer:  {proposer_label}")
    if proposer_trace_dir:
        print(f"  proposer traces: {proposer_trace_dir}")

    # --- Seed candidate -----------------------------------------------------
    seed = seed_candidate()
    print("Seed candidate:")
    for key in seed:
        print(f"  {key}: {len(seed[key])} chars")
    frozen_sig = ManipulateSpreadsheet.__doc__ or ""
    if COMPONENT_SIGNATURE not in seed:
        print(
            f"  (signature_docstring frozen at {len(frozen_sig)} chars — "
            f"skill-only evolution)"
        )

    # --- GEPA optimize ------------------------------------------------------
    print(
        f"\nStarting GEPA: max_metric_calls={config.max_metric_calls}, "
        f"minibatch_size={config.minibatch_size}, "
        f"module_selector={config.module_selector}, "
        f"concurrency={config.concurrency}"
    )
    t0 = time.time()
    result = gepa.optimize(
        seed_candidate=seed,
        trainset=train,
        valset=val,
        adapter=adapter,
        reflection_lm=reflection_lm_call,
        reflection_minibatch_size=config.minibatch_size,
        max_metric_calls=config.max_metric_calls,
        perfect_score=1.0,
        skip_perfect_score=True,
        candidate_selection_strategy=config.candidate_selection_strategy,
        module_selector=config.module_selector,
        use_merge=True,
        max_merge_invocations=10,
        run_dir=str(run_dir),
        seed=config.seed,
        use_cloudpickle=True,
        display_progress_bar=True,
    )
    elapsed = time.time() - t0

    # --- Costs --------------------------------------------------------------
    costs = [
        summarize_lm_cost("main", lm),
        summarize_lm_cost("sub", sub_lm),
        summarize_lm_cost(
            "proposer" if config.rlm_proposer else "reflection",
            reflection_lm_instance,
        ),
    ]
    if proposer_sub_lm is not None:
        costs.append(summarize_lm_cost("proposer_sub", proposer_sub_lm))

    best_idx = result.best_idx
    best_candidate = result.candidates[best_idx]
    best_score = result.val_aggregate_scores[best_idx]

    # --- Persist artefacts -------------------------------------------------
    # Signature is frozen (skill-only evolution); write the pristine
    # class docstring for historical continuity with prior runs that
    # evolved both components.
    frozen_sig = ManipulateSpreadsheet.__doc__ or ""
    (run_dir / "best_signature_docstring.txt").write_text(
        best_candidate.get(COMPONENT_SIGNATURE, frozen_sig)
    )
    (run_dir / "best_skill_instructions.txt").write_text(
        best_candidate[COMPONENT_SKILL]
    )
    (run_dir / "all_candidates.json").write_text(
        json.dumps(
            [
                {
                    "idx": i,
                    "score": result.val_aggregate_scores[i],
                    COMPONENT_SIGNATURE: c.get(COMPONENT_SIGNATURE, frozen_sig),
                    COMPONENT_SKILL: c[COMPONENT_SKILL],
                }
                for i, c in enumerate(result.candidates)
            ],
            indent=2,
        )
    )

    report = OptimizeReport(
        config=config,
        run_dir=str(run_dir),
        best_idx=best_idx,
        best_val_score=best_score,
        total_candidates=len(result.candidates),
        total_metric_calls=result.total_metric_calls,
        duration_seconds=elapsed,
        best_candidate=best_candidate,
        val_aggregate_scores=list(result.val_aggregate_scores),
        costs=costs,
    )

    return report


def extract_best_candidate(run_dir: str | Path) -> dict[str, str]:
    """Pull the best-by-mean candidate (both components) from a GEPA run dir.

    Mirrors :func:`evaluation.extract_best_prompt` but returns the full
    multi-component candidate dict instead of just the signature
    docstring. The ``candidate`` dict contains both
    ``signature_docstring`` and ``skill_instructions`` keys; callers
    that only care about the signature can ``[COMPONENT_SIGNATURE]``
    into the result.
    """
    state_path = Path(run_dir) / "gepa_state.bin"
    with state_path.open("rb") as f:
        state = pickle.load(f)
    subs = state["prog_candidate_val_subscores"]
    cands = state["program_candidates"]
    best_idx = max(
        range(len(cands)),
        key=lambda i: (sum(subs[i].values()) / len(subs[i])) if subs[i] else 0.0,
    )
    return cands[best_idx]


__all__ = [
    "ALL_COMPONENTS",
    "COMPONENT_SIGNATURE",
    "COMPONENT_SKILL",
    "ImproveInstructions",
    "OptimizeConfig",
    "OptimizeReport",
    "SpreadsheetAdapter",
    "compute_lm_cost",  # re-exported for the CLI
    "extract_best_candidate",
    "make_dynamic_skill",
    "run_optimization",
    "seed_candidate",
    "split_train_val",
    "write_split_log",
]
