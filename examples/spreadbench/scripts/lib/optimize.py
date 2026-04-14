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
    make_dynamic_signature,
    make_dynamic_skill,
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
    """Return the GEPA seed candidate: current sig docstring + skill instructions."""
    return {
        COMPONENT_SIGNATURE: ManipulateSpreadsheet.__doc__ or "",
        COMPONENT_SKILL: libreoffice_spreadsheet_skill.instructions,
    }


# ---------------------------------------------------------------------------
# RLM proposer signature (verbatim SURGICAL prompt + multi-component marker)
# ---------------------------------------------------------------------------


class ImproveInstructions(dspy.Signature):
    """Analyze execution traces from a spreadsheet-manipulation assistant and propose improved instructions for one of two components.

    The `current_kind` field tells you which component you are editing:
      * `signature_docstring`  — high-level workflow (load → inspect → write → save → verify)
      * `skill_instructions`   — domain rules (LibreOffice formula compatibility, openpyxl pitfalls, formula vs Python guidance)

    You have a JSON file at /sandbox/input/traces_file/ containing recent task executions. Each record has:
    - "Inputs": the natural-language task instruction and type
    - "Generated Outputs": the full RLM execution trace (reasoning, code, and sandbox output per iteration)
    - "Feedback": per-case scoring (cell-match ratio 0-1) with expected vs got cell mismatches

    Workflow:
    1. Load the JSON file.
    2. Count successes and failures; inspect the score distribution.
    3. Read the WORST failing traces carefully — what did the RLM do wrong at the code/cell level?
    4. Compare with the BEST successful traces — what did they do right?
    5. Identify concrete, actionable failure patterns (not abstract principles).
    6. Write improved instructions for the `current_kind` component that specifically address those failures.

    Keep improvements SURGICAL: prefer small targeted changes over full rewrites. Preserve rules that are already working. Every sentence should tell the assistant what to do or warn what not to do.

    Output the new instructions as plain text in the new_instructions field.
    """

    current_kind: str = dspy.InputField(
        desc="Which component you are improving: 'signature_docstring' or 'skill_instructions'"
    )
    current_instructions: str = dspy.InputField(
        desc="The current instructions text for the component being improved"
    )
    traces_file: File = dspy.InputField(
        desc="JSON file with task execution traces, mounted at /sandbox/input/traces_file/"
    )
    new_instructions: str = dspy.OutputField(
        desc="The improved instructions text for the indicated component"
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
    reflection_lm: str = "anthropic/claude-opus-4-6"
    # Default unset (no extended thinking). On Claude 4.6 any non-empty
    # value maps to adaptive thinking via LiteLLM, which meaningfully
    # raises proposer cost. The sibling repo's validated run used None.
    reflection_reasoning_effort: str | None = None

    # Datasets
    train_dataset: str = "trainset"
    val_ratio: float = 0.20
    cases_per_task: int = 1

    # GEPA budget + sampling
    max_metric_calls: int = 2000
    minibatch_size: int = 20
    val_limit: int | None = 50  # cap val tasks during inner loop
    seed: int = 42

    # Component selection strategy
    module_selector: str = "round_robin"  # "round_robin" or "all"

    # RLM proposer toggle
    rlm_proposer: bool = False
    proposer_max_iterations: int = 20

    # RLM eval budget (per task case)
    concurrency: int = 30
    max_iterations: int = 50
    task_timeout: int = 300
    cache: bool = True

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
        reflection_lm_instance: dspy.LM | None = None,
        cost_snapshot_path: Path | None = None,
    ):
        self.lm = lm
        self.sub_lm = sub_lm
        self.max_iterations = max_iterations
        self.concurrency = concurrency
        self.task_timeout = task_timeout
        self.proposer_lm = proposer_lm
        self.proposer_sub_lm = proposer_sub_lm or sub_lm
        self.proposer_max_iterations = proposer_max_iterations
        self.proposer_trace_dir = proposer_trace_dir
        self._proposer_call_count = 0
        # Handle to the reflection LM instance so we can summarise its cost
        # even when --rlm_proposer is off (in which case proposer_lm is None
        # but GEPA still uses reflection_lm_instance for its default
        # reflection-LM flow). When --rlm_proposer is on, this is the same
        # object as proposer_lm.
        self.reflection_lm_instance = reflection_lm_instance
        # When set, the adapter dumps a fresh LiteLLM-authoritative cost
        # snapshot to this file after every evaluate() and every proposer
        # call. The file is overwritten each time — latest state only.
        self.cost_snapshot_path = cost_snapshot_path

        # GEPA's adapter Protocol: setting propose_new_texts on the instance
        # opts in to our custom proposer; leaving it None falls back to GEPA's
        # default reflection-LM flow.
        if proposer_lm is not None:
            self.propose_new_texts = self._rlm_propose_new_texts
        else:
            self.propose_new_texts = None

    # -- cost snapshotting --------------------------------------------------

    def _dump_costs(self) -> None:
        """Write current LiteLLM-authoritative costs to ``cost_snapshot_path``.

        Best-effort: any failure is logged at debug level and swallowed so
        the snapshot code path cannot break the eval loop. Walks
        ``lm.history`` via :func:`summarize_lm_cost`, which honours the
        per-model price overrides (mercury-2) and reports real cache-hit
        costs from LiteLLM. The file is overwritten each call — readers
        get the latest snapshot by cat'ing it.
        """
        if self.cost_snapshot_path is None:
            return
        try:
            costs = [
                summarize_lm_cost("main", self.lm),
                summarize_lm_cost("sub", self.sub_lm),
            ]
            if self.reflection_lm_instance is not None:
                role = "proposer" if self.proposer_lm is not None else "reflection"
                costs.append(summarize_lm_cost(role, self.reflection_lm_instance))
            payload = {
                "timestamp": datetime.now().isoformat(),
                "proposer_calls_so_far": self._proposer_call_count,
                "total_cost_usd": sum(c.cost_usd for c in costs),
                "costs": [c.to_dict() for c in costs],
            }
            self.cost_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            self.cost_snapshot_path.write_text(
                json.dumps(payload, indent=2, default=str)
            )
        except Exception as e:
            log.debug("cost snapshot write failed: %s", e)

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
        sig_cls = make_dynamic_signature(candidate[COMPONENT_SIGNATURE])
        skill = make_dynamic_skill(candidate[COMPONENT_SKILL])

        tmp_dir = tempfile.mkdtemp(prefix="gepa_eval_")
        rlm_sem = asyncio.Semaphore(self.concurrency)
        pbar = tqdm(
            total=len(batch),
            desc=f"  eval ({len(batch)} tasks)",
            leave=False,
        )

        try:
            async def _process_task(task: SpreadsheetTask):
                case_coros = [
                    self._process_case(
                        task, idx, input_path, answer_path,
                        sig_cls, skill, rlm_sem, tmp_dir, capture_traces,
                    )
                    for idx, input_path, answer_path in task.test_cases
                ]
                case_results = await asyncio.gather(*case_coros)
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

        # Snapshot LiteLLM-authoritative costs after each evaluate() call.
        # Best-effort — any failure is swallowed by _dump_costs so the
        # eval loop can't be broken by file I/O problems.
        self._dump_costs()

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

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

        async with rlm_sem:
            try:
                predictor = PredictRLM(
                    sig_cls,
                    lm=self.lm,
                    sub_lm=self.sub_lm,
                    skills=[skill],
                    max_iterations=self.max_iterations,
                    verbose=False,
                    debug=False,
                )
                result = await asyncio.wait_for(
                    predictor.acall(
                        input_spreadsheet=File(path=input_path),
                        instruction=task.instruction,
                    ),
                    timeout=self.task_timeout,
                )
                if capture_traces:
                    trace = getattr(result, "trajectory", []) or []
                if not (
                    result
                    and result.output_spreadsheet
                    and result.output_spreadsheet.path
                    and os.path.exists(result.output_spreadsheet.path)
                ):
                    return {
                        "idx": idx,
                        "score": 0.0,
                        "passed": False,
                        "message": "No output produced",
                        "trace": trace,
                    }
                shutil.copy2(result.output_spreadsheet.path, output_path)
            except asyncio.TimeoutError:
                return {
                    "idx": idx,
                    "score": 0.0,
                    "passed": False,
                    "message": f"RLM timeout ({self.task_timeout}s)",
                    "trace": [],
                }
            except Exception as e:
                return {
                    "idx": idx,
                    "score": 0.0,
                    "passed": False,
                    "message": f"RLM error: {e}",
                    "trace": [],
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
            }
        except Exception as e:
            return {
                "idx": idx,
                "score": 0.0,
                "passed": False,
                "message": f"Comparison error: {e}",
                "trace": trace,
            }

    # -- make_reflective_dataset --------------------------------------------

    @staticmethod
    def _render_trace(trace: list[dict], tag: str) -> str:
        if not trace:
            return f"({tag}: no trace captured)"
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
    def _case_summary(c: dict) -> str:
        return (
            f"case {c['idx']}: score={c['score']:.2f} "
            f"({'PASS' if c['passed'] else 'FAIL'})"
        )

    @staticmethod
    def _banner(text: str) -> str:
        line = f"== {text} =="
        rule = "=" * len(line)
        return f"{rule}\n{line}\n{rule}"

    @staticmethod
    def _component_feedback(component_name: str) -> str:
        """Build the per-component Feedback block.

        The Inputs and Generated Outputs are identical between the two
        components (same eval traces); only Feedback shifts emphasis to
        steer the proposer toward the failure modes that the component
        in question can actually fix.
        """
        common = [
            "## How Scoring Works",
            "The RLM receives a spreadsheet + instruction and must produce a "
            "modified spreadsheet. Scoring compares each target cell in the "
            "output against the ground truth answer:",
            "  score = matched_cells / total_cells  (0.0 to 1.0)",
            "A score of 1.0 means every target cell matches exactly.",
            "",
        ]
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
        feedback = self._component_feedback(component_name)
        for trajectory, score in zip(eval_batch.trajectories, eval_batch.scores):
            cases = trajectory["cases"]
            cases_sorted = sorted(cases, key=lambda c: c["score"])

            case_score_lines = "\n".join(
                f"  {self._case_summary(c)}" for c in cases_sorted
            )
            inputs_summary = (
                f"Task score: {score:.3f} (mean cell-match ratio)\n"
                f"Instruction type: {trajectory['instruction_type']}\n"
                f"Per-case scores:\n{case_score_lines}\n\n"
                f"Instruction:\n{trajectory['instruction']}"
            )

            cases_with_traces = [c for c in cases_sorted if c.get("trace")]
            trace_parts: list[str] = []

            for c in cases_with_traces[:2]:
                idx = c["idx"]
                header = (
                    f"WORST — {self._case_summary(c)}\n"
                    f"Cell mismatches:\n{c['message']}"
                )
                trace_parts.append(
                    f"{self._banner(header)}\n\n"
                    f"{self._render_trace(c['trace'], f'WORST CASE {idx}')}"
                )

            if (
                cases_with_traces
                and cases_with_traces[-1]["score"] > cases_with_traces[0]["score"]
            ):
                c = cases_with_traces[-1]
                idx = c["idx"]
                header = f"BEST — {self._case_summary(c)}"
                if c["passed"]:
                    header += "\nAll cells matched."
                else:
                    header += f"\nCell mismatches:\n{c['message']}"
                trace_parts.append(
                    f"{self._banner(header)}\n\n"
                    f"{self._render_trace(c['trace'], f'BEST CASE {idx}')}"
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
        # Snapshot costs after the proposer has finished — the reflection LM
        # history now includes the new calls, so the dump captures them
        # promptly rather than waiting for the next evaluate() call.
        self._dump_costs()
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

        try:
            predictor = PredictRLM(
                ImproveInstructions,
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

            try:
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(
                    predictor.acall(
                        current_kind=component_name,
                        current_instructions=current_text,
                        traces_file=File(path=traces_path),
                    )
                )
            finally:
                rlm_logger.removeHandler(stream_handler)
                rlm_logger.setLevel(old_level)
                rlm_logger.propagate = old_propagate

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

            if self.proposer_trace_dir:
                try:
                    os.makedirs(self.proposer_trace_dir, exist_ok=True)
                    payload = {
                        "call_idx": call_idx,
                        "component": component_name,
                        "current_instructions": current_text,
                        "new_instructions": new_text,
                        "reflective_dataset": serializable,
                        "rlm_trajectory": getattr(result, "trajectory", []),
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
        return reflection_lm_instance(prompt)[0]

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
            **get_sub_lm_config(config.sub_lm),
            cache=config.cache,
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
        proposer_trace_dir=proposer_trace_dir,
        reflection_lm_instance=reflection_lm_instance,
        cost_snapshot_path=run_dir / "costs_live.json",
    )
    # Write an initial zero-cost snapshot so the file exists from the start
    # and readers have something to cat even before the first evaluate().
    adapter._dump_costs()
    print(
        "Proposer:  "
        + (
            f"RLM SURGICAL ({config.reflection_lm}, max_iters={config.proposer_max_iterations})"
            if config.rlm_proposer
            else "GEPA default reflection LM"
        )
    )
    if proposer_trace_dir:
        print(f"  proposer traces: {proposer_trace_dir}")

    # --- Seed candidate -----------------------------------------------------
    seed = seed_candidate()
    print(
        f"Seed signature_docstring: {len(seed[COMPONENT_SIGNATURE])} chars\n"
        f"Seed skill_instructions:  {len(seed[COMPONENT_SKILL])} chars"
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
        candidate_selection_strategy="pareto",
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
    (run_dir / "best_signature_docstring.txt").write_text(
        best_candidate[COMPONENT_SIGNATURE]
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
                    COMPONENT_SIGNATURE: c[COMPONENT_SIGNATURE],
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
