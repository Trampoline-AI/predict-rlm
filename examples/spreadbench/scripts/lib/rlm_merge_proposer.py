"""RlmMergeProposer: subclass of gepa.proposer.merge.MergeProposer that
replaces stock mechanical merges with RLM synthesis on two parents +
common ancestor + a paired execution-trace file.

Design reference: docs/rlm_merge_proposer_proposal.md (v11).

# What this is doing conceptually

A merge takes TWO sibling candidates A and B that evolved from a common
ancestor along different paths (winning different val tasks) and asks
an LM to synthesize their strengths into one combined skill. Think
git three-way merge: the ancestor is the base, A and B are branches,
and the synthesis produces a new candidate that inherits from both.

For that to be possible, the candidate pool needs ≥2 sibling
candidates (neither descended from the other) sharing a common
ancestor. On fresh runs, this structure doesn't exist until at least
iter 2-3 AFTER some non-linear acceptance (i.e. GEPA accepting a
child from a DIFFERENT parent than the most recent one). Early
iterations where only {seed, cand_1} exist WILL return pair_skipped
— that's expected, not a failure mode. See
``merge_selection.pick_merge_triplet`` docstring for the full set of
eligibility rules.

# Key architectural choices

  - Accept/reject decision is made INSIDE propose() (engine.py:288 refuses
    to drain merges_due on subsample-score rejects, so we can't let the
    engine decide). The returned CandidateProposal always represents an
    ACCEPTED merge; rejections are silent None returns with our own
    counter adjustments.
  - Pair selection replaces stock
    ``sample_and_attempt_merge_programs_by_common_predictors`` entirely.
    The stock helper's per-component filter blocks single-component
    divergent pairs (the case we care about).
  - Trace capture uses ``adapter.evaluate(kind="merge_trace_capture",
    capture_traces=True)`` to get per-parent trajectories on a shared
    train minibatch. Must manually increment ``state.total_num_evals``
    after each capture (mirrors reflective_mutation.py:112).
  - The paired trace file uses ``adapter.make_reflective_dataset``
    rich records ("Generated Outputs" + "Feedback") per parent, NOT
    summary stubs (num_passed/num_total). The merge RLM needs the
    same rich trace rendering the reflective proposer sees to
    synthesize anything meaningful. See
    ``_write_paired_trace_file`` and its regression test at
    ``tests/test_rlm_merge_proposer.py::test_paired_trace_file_contains_rich_reflective_fields``.
  - ``max_rlm_merge_attempts`` is an ATTEMPT cap (cost-aware), distinct
    from GEPA's ``max_merge_invocations`` which counts ACCEPTED merges
    only.
  - All writes to ``state.full_program_trace[-1]`` use the
    ``rlm_merge_*`` namespace prefix so reflective mutation can't
    overwrite them if it fires in the same iteration after we return
    None.
"""

from __future__ import annotations

import json
import os
import random
import re
from collections.abc import Sequence
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from gepa.core.adapter import RolloutOutput
from gepa.core.data_loader import DataId, DataLoader
from gepa.core.state import GEPAState, ObjectiveScores
from gepa.gepa_utils import find_dominator_programs
from gepa.logging.logger import LoggerProtocol
from gepa.proposer.base import CandidateProposal
from gepa.proposer.merge import MergeProposer

from predict_rlm import File

from .merge_selection import pick_merge_triplet

COMPONENT_SKILL = "skill_instructions"

# Statuses for rlm_merge_status field on state.full_program_trace[-1].
# See docs/rlm_merge_proposer_proposal.md v11 for semantics.
VALID_STATUSES = frozenset({
    "attempt_cap_exhausted",
    "pair_skipped",
    "preflight_failed",
    "structural_rejected",
    "duplicate",
    "length_rejected",
    "subsample_rejected",
    "accepted",
    "error",
})

LENGTH_BUDGET_RATIO = 1.10


class RlmMergeProposer(MergeProposer):
    """Custom merge proposer: subclasses MergeProposer for scheduling and
    bookkeeping (``use_merge``, ``merges_due``, ``merges_performed``,
    ``select_eval_subsample_for_merged_program``), overrides
    ``propose(state)`` wholesale.
    """

    def __init__(
        self,
        *,
        logger: LoggerProtocol,
        valset: DataLoader[DataId, Any],
        evaluator: Callable[
            [list, dict[str, str]],
            tuple[list[RolloutOutput], list[float], Sequence[ObjectiveScores] | None],
        ],
        adapter: Any,  # SpreadsheetAdapter — typed loosely to avoid circular import
        trainset: DataLoader[DataId, Any],
        use_merge: bool,
        max_merge_invocations: int,
        max_rlm_merge_attempts: int,
        min_each: int = 3,
        merge_minibatch_size: int = 50,
        val_overlap_floor: int = 5,
        component_name: str = COMPONENT_SKILL,
        rlm_merge_state_path: Path | None = None,
        rng: random.Random | None = None,
    ):
        # Keep val_overlap_floor consistent with the divergence gate.
        val_overlap_floor = max(val_overlap_floor, 2 * min_each)
        super().__init__(
            logger=logger,
            valset=valset,
            evaluator=evaluator,
            use_merge=use_merge,
            max_merge_invocations=max_merge_invocations,
            val_overlap_floor=val_overlap_floor,
            rng=rng,
        )
        self.adapter = adapter
        self.trainset = trainset
        self.max_rlm_merge_attempts = max_rlm_merge_attempts
        self._min_each = min_each
        self.merge_minibatch_size = merge_minibatch_size
        self.component_name = component_name
        self.rlm_merge_state_path = rlm_merge_state_path
        self.rlm_merge_attempts_used = 0

        # Trainset size guard (v11 fix: fail at init, not per-attempt).
        all_train_ids = list(self.trainset.all_ids())
        if self.merge_minibatch_size > len(all_train_ids):
            raise ValueError(
                f"merge_minibatch_size={self.merge_minibatch_size} exceeds "
                f"trainset size {len(all_train_ids)}"
            )

        # Load persisted state (attempt counter + merges_performed) if resume.
        self._load_rlm_merge_state()

    # -- persistence --------------------------------------------------------

    def _load_rlm_merge_state(self) -> None:
        """Load rlm_merge_attempts_used + merges_performed from disk on
        resume. Does NOT load merges_due — engine owns that counter and
        mutates it after propose() returns."""
        if not self.rlm_merge_state_path or not self.rlm_merge_state_path.is_file():
            return
        try:
            data = json.loads(self.rlm_merge_state_path.read_text())
        except Exception:
            return
        self.rlm_merge_attempts_used = int(
            data.get("rlm_merge_attempts_used", 0)
        )
        merges_performed = data.get("merges_performed", [])
        if isinstance(merges_performed, list):
            for entry in merges_performed:
                if isinstance(entry, list) and len(entry) == 3:
                    tup = (int(entry[0]), int(entry[1]), int(entry[2]))
                    if tup not in self.merges_performed[0]:
                        self.merges_performed[0].append(tup)

    def _flush_rlm_merge_state(self) -> None:
        """Synchronous flush of rlm_merge_state.json. Called from the outer
        ``finally`` around propose() so every mutation is persisted before
        returning."""
        if not self.rlm_merge_state_path:
            return
        try:
            self.rlm_merge_state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "rlm_merge_attempts_used": self.rlm_merge_attempts_used,
                "merges_performed": [list(t) for t in self.merges_performed[0]],
                "flushed_at": datetime.now().isoformat(),
            }
            self.rlm_merge_state_path.write_text(json.dumps(payload, indent=2))
        except Exception as e:
            self.logger.log(f"rlm_merge_state flush failed: {e}")

    # -- status helper ------------------------------------------------------

    def _record_merge_status(
        self,
        state: GEPAState,
        status: str,
        attempt_idx: int | None,
        **fields: Any,
    ) -> None:
        """Write namespaced rlm_merge_* fields on state.full_program_trace[-1].

        Every return path in propose() must call this exactly once so
        postmortem can disambiguate outcomes from a single key.
        """
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid merge status {status!r}")
        trace_dict = state.full_program_trace[-1]
        trace_dict["rlm_merge_status"] = status
        trace_dict["rlm_merge_attempt_idx"] = attempt_idx
        if status in {"pair_skipped", "attempt_cap_exhausted"}:
            trace_dict["rlm_merge_considered"] = True
        for k, v in fields.items():
            if not k.startswith("rlm_merge_"):
                raise ValueError(
                    f"namespace field {k!r} must start with 'rlm_merge_'"
                )
            trace_dict[k] = v

    # -- structural [NEW] audit check --------------------------------------

    @staticmethod
    def _new_audit_task_ids(audit_lines: list[str]) -> list[tuple[int, str | None]]:
        """For each [NEW] line in ``audit_lines``, return (index, cited_task_id).
        ``cited_task_id`` is None if the line has no recognizable task_id.

        Recognizes any of:
          - 'citing task_id=<id>'
          - 'task_id: <id>'
          - 'task_id=<id>'
          - '<id>' inside explicit quotes following 'task_id' keyword
        """
        pat = re.compile(
            r"task[_ ]id\s*[:=]\s*([A-Za-z0-9_\-./]+)", re.IGNORECASE
        )
        out: list[tuple[int, str | None]] = []
        for i, line in enumerate(audit_lines):
            if not isinstance(line, str):
                continue
            stripped = line.lstrip()
            if not stripped.startswith("[NEW]"):
                continue
            m = pat.search(line)
            out.append((i, m.group(1) if m else None))
        return out

    def _structural_new_audit_check(
        self,
        audit_lines: list[str],
        paired_task_ids: set[str],
    ) -> tuple[bool, str]:
        """Returns (ok, reason). ok=True if every [NEW] line cites a task_id
        present in ``paired_task_ids``. reason describes the first failure.
        """
        checks = self._new_audit_task_ids(audit_lines)
        for idx, tid in checks:
            if tid is None:
                return False, f"[NEW] line {idx} missing task_id citation"
            if tid not in paired_task_ids:
                return False, (
                    f"[NEW] line {idx} cites task_id={tid!r} not in "
                    f"paired trace (size={len(paired_task_ids)})"
                )
        return True, ""

    # -- propose ------------------------------------------------------------

    def propose(
        self,
        state: GEPAState[RolloutOutput, DataId],
    ) -> CandidateProposal[DataId] | None:
        """Full override of MergeProposer.propose. See flow steps in
        docs/rlm_merge_proposer_proposal.md §"propose(state) flow".
        """
        # Outer try/finally guarantees rlm_merge_state.json flushes on
        # every exit path (accept, any reject, error, pair_skipped).
        try:
            return self._propose_inner(state)
        finally:
            self._flush_rlm_merge_state()

    def _propose_inner(
        self,
        state: GEPAState[RolloutOutput, DataId],
    ) -> CandidateProposal[DataId] | None:
        i = state.i + 1
        state.full_program_trace[-1]["invoked_merge"] = True

        # Stock guard: only attempt when scheduled and last iter accepted.
        if not (
            self.use_merge
            and self.last_iter_found_new_program
            and self.merges_due > 0
        ):
            self.logger.log(f"Iteration {i}: No merge candidates scheduled")
            return None

        # 1) Attempt-cap gate
        if self.rlm_merge_attempts_used >= self.max_rlm_merge_attempts:
            self.merges_due -= 1
            self._record_merge_status(
                state, "attempt_cap_exhausted", attempt_idx=None
            )
            self.logger.log(
                f"Iteration {i}: RLM merge attempt cap exhausted "
                f"({self.rlm_merge_attempts_used}/{self.max_rlm_merge_attempts})"
            )
            return None

        # 3) Pareto-dominator narrowing (matches merge.py:291-296)
        pareto_front_programs = state.get_pareto_front_mapping()
        tracked_scores: Sequence[float] = getattr(
            state, "per_program_tracked_scores", state.program_full_scores_val_set
        )
        merge_candidates = find_dominator_programs(
            pareto_front_programs, list(tracked_scores)
        )

        # 4) Pick triplet
        triplet = pick_merge_triplet(
            merge_candidates=merge_candidates,
            program_candidates=state.program_candidates,
            parent_program_for_candidate=state.parent_program_for_candidate,
            prog_candidate_val_subscores=state.prog_candidate_val_subscores,
            tracked_scores=list(tracked_scores),
            merges_performed=list(self.merges_performed[0]),
            rng=self.rng,
            component_name=self.component_name,
            min_each=self._min_each,
        )
        if triplet is None:
            self.merges_due -= 1
            self._record_merge_status(state, "pair_skipped", attempt_idx=None)
            self.logger.log(f"Iteration {i}: No divergent pairs found")
            return None

        id1, id2, ancestor = triplet

        # 5) Record attempt BEFORE trace capture (dedup anchor)
        self.merges_performed[0].append((id1, id2, ancestor))

        # 6) Consume attempt
        self.rlm_merge_attempts_used += 1
        attempt_idx = self.rlm_merge_attempts_used - 1

        # Initialize error-path variables before the try block so the
        # except handler never touches undefined names (v11 fix).
        trace_task_ids: list[str] = []
        merge_call_idx: int | None = None

        # Resolve candidate texts for this pair + ancestor.
        instructions_a = state.program_candidates[id1][self.component_name]
        instructions_b = state.program_candidates[id2][self.component_name]
        instructions_ancestor = state.program_candidates[ancestor][
            self.component_name
        ]

        try:
            # 7) Trace capture on shared train minibatch
            all_train_ids = list(self.trainset.all_ids())
            task_data_ids = self.rng.sample(
                all_train_ids,
                k=min(self.merge_minibatch_size, len(all_train_ids)),
            )
            batch = self.trainset.fetch(task_data_ids)

            eval_a = self.adapter.evaluate(
                batch,
                state.program_candidates[id1],
                capture_traces=True,
                kind="merge_trace_capture",
            )
            state.total_num_evals += len(task_data_ids)

            eval_b = self.adapter.evaluate(
                batch,
                state.program_candidates[id2],
                capture_traces=True,
                kind="merge_trace_capture",
            )
            state.total_num_evals += len(task_data_ids)

            # Task IDs (benchmark-side) for preflight, paired-trace records,
            # and [NEW] audit citation. Trajectory order matches batch order.
            trace_task_ids = [
                traj.get("task_id", str(traj))
                for traj in (eval_a.trajectories or [])
            ]
            # Alignment assertion (v11 safety).
            trace_task_ids_b = [
                traj.get("task_id", str(traj))
                for traj in (eval_b.trajectories or [])
            ]
            if trace_task_ids != trace_task_ids_b:
                raise RuntimeError(
                    "task_id misalignment between paired trace captures"
                )

            # 8) Preflight: require ≥1 A-win AND ≥1 B-win on the train
            # minibatch (divergence must be visible in the trace the RLM
            # will see, not just on val).
            eps = 1e-6
            a_wins = sum(
                1 for sa, sb in zip(eval_a.scores, eval_b.scores)
                if sa > sb + eps
            )
            b_wins = sum(
                1 for sa, sb in zip(eval_a.scores, eval_b.scores)
                if sb > sa + eps
            )
            if a_wins < 1 or b_wins < 1:
                self.merges_due -= 1
                self._record_merge_status(
                    state, "preflight_failed", attempt_idx,
                    rlm_merge_train_task_ids=trace_task_ids,
                    rlm_merge_candidate_pair=(id1, id2),
                    rlm_merge_preflight_a_wins=a_wins,
                    rlm_merge_preflight_b_wins=b_wins,
                )
                self.logger.log(
                    f"Iteration {i}: RLM merge preflight failed "
                    f"(a_wins={a_wins}, b_wins={b_wins})"
                )
                return None

            # 9) Build paired-trace file — use adapter.make_reflective_dataset
            # to get the same rich "Generated Outputs" / "Feedback" rendering
            # the reflective proposer sees for each parent, then zip by index.
            ra = self.adapter.make_reflective_dataset(
                state.program_candidates[id1],
                eval_a,
                [self.component_name],
            ).get(self.component_name, [])
            rb = self.adapter.make_reflective_dataset(
                state.program_candidates[id2],
                eval_b,
                [self.component_name],
            ).get(self.component_name, [])
            paired_trace_path = self._write_paired_trace_file(
                id1=id1, id2=id2, ancestor=ancestor,
                attempt_idx=attempt_idx,
                task_data_ids=task_data_ids,
                trace_task_ids=trace_task_ids,
                reflective_a=list(ra),
                reflective_b=list(rb),
                eval_a=eval_a,
                eval_b=eval_b,
                eps=eps,
            )

            # 10) Reserve call_idx BEFORE dispatch, call synthesis
            merge_call_idx = self.adapter._reserve_merge_proposer_call_idx()
            new_instructions, audit_lines = self.adapter._rlm_propose_merge_texts(
                call_idx=merge_call_idx,
                id1=id1, id2=id2, ancestor=ancestor,
                current_instructions_a=instructions_a,
                current_instructions_b=instructions_b,
                common_ancestor_instructions=instructions_ancestor,
                paired_traces_file=File(path=paired_trace_path),
                trace_task_ids=trace_task_ids,
            )

            # 11) Post-process new_instructions
            # 11a) Structural [NEW] audit check
            paired_task_id_set = set(trace_task_ids)
            ok, reason = self._structural_new_audit_check(
                list(audit_lines), paired_task_id_set
            )
            if not ok:
                self.merges_due -= 1
                self._record_merge_status(
                    state, "structural_rejected", attempt_idx,
                    rlm_merge_candidate_pair=(id1, id2),
                    rlm_merge_reject_reason=reason,
                )
                self.logger.log(
                    f"Iteration {i}: RLM merge structural reject: {reason}"
                )
                return None

            # 11b) Exact-duplicate check
            if new_instructions in (
                instructions_a, instructions_b, instructions_ancestor
            ):
                self.merges_due -= 1
                self._record_merge_status(
                    state, "duplicate", attempt_idx,
                    rlm_merge_candidate_pair=(id1, id2),
                )
                self.logger.log(
                    f"Iteration {i}: RLM merge produced exact duplicate"
                )
                return None

            # 11c) Length check
            max_len = LENGTH_BUDGET_RATIO * max(
                len(instructions_a), len(instructions_b)
            )
            if len(new_instructions) > max_len:
                self.merges_due -= 1
                self._record_merge_status(
                    state, "length_rejected", attempt_idx,
                    rlm_merge_candidate_pair=(id1, id2),
                    rlm_merge_new_len=len(new_instructions),
                    rlm_merge_max_len=int(max_len),
                )
                self.logger.log(
                    f"Iteration {i}: RLM merge length reject "
                    f"({len(new_instructions)} > {int(max_len)})"
                )
                return None

            # 12) Subsample eval on shared val subsample
            # (select_eval_subsample_for_merged_program). Use
            # merge_minibatch_size (matching the trace-capture train
            # minibatch) instead of the stock default of 5: with n=5 the
            # McNemar flip stats aren't meaningful and the accept/reject
            # decision is noisy. Matching reflective minibatch sizing
            # gives merge the same signal-to-noise as reflective
            # iterations.
            subsample_ids = self.select_eval_subsample_for_merged_program(
                state.prog_candidate_val_subscores[id1],
                state.prog_candidate_val_subscores[id2],
                num_subsample_ids=self.merge_minibatch_size,
            )
            if not subsample_ids:
                self.merges_due -= 1
                self._record_merge_status(
                    state, "subsample_rejected", attempt_idx,
                    rlm_merge_candidate_pair=(id1, id2),
                    rlm_merge_reject_reason="empty subsample",
                )
                self.logger.log(
                    f"Iteration {i}: RLM merge subsample empty"
                )
                return None

            id1_sub_scores = [
                state.prog_candidate_val_subscores[id1][k]
                for k in subsample_ids
            ]
            id2_sub_scores = [
                state.prog_candidate_val_subscores[id2][k]
                for k in subsample_ids
            ]

            # Build the new program (candidate dict) with the merged
            # component text and unchanged other components.
            new_program: dict[str, str] = deepcopy(
                state.program_candidates[ancestor]
            )
            new_program[self.component_name] = new_instructions

            new_sub_scores, actual_evals_count = state.cached_evaluate(
                new_program, subsample_ids, self.valset.fetch, self.evaluator
            )
            state.total_num_evals += actual_evals_count

            new_sum = sum(new_sub_scores)
            parent_sums = [sum(id1_sub_scores), sum(id2_sub_scores)]
            max_parent_sum = max(parent_sums)

            # 13) Internal accept/reject
            if new_sum >= max_parent_sum:
                # ACCEPTED: mirror stock merge proposer trace fields
                # so resume-time recomputation from
                # parent_program_for_candidate still identifies this as
                # a merge (candidate will have 2 parents).
                state.full_program_trace[-1]["merged"] = True
                state.full_program_trace[-1]["merged_entities"] = (
                    id1, id2, ancestor,
                )
                state.full_program_trace[-1]["merge_proposer"] = "rlm"
                # Mirror stock MergeProposer.propose:343-345 by persisting
                # the full per-task score arrays (not just sums). Makes
                # downstream analysis (hard-pass counts, paired flips,
                # McNemar p-values) possible — same rendering the
                # reflective iteration table uses.
                state.full_program_trace[-1]["id1_subsample_scores"] = list(id1_sub_scores)
                state.full_program_trace[-1]["id2_subsample_scores"] = list(id2_sub_scores)
                state.full_program_trace[-1]["new_program_subsample_scores"] = list(new_sub_scores)
                self._record_merge_status(
                    state, "accepted", attempt_idx,
                    rlm_merge_train_task_ids=trace_task_ids,
                    rlm_merge_subsample_ids=list(subsample_ids),
                    rlm_merge_candidate_pair=(id1, id2),
                    rlm_merge_new_sum=new_sum,
                    rlm_merge_parent_sums=parent_sums,
                )
                return CandidateProposal(
                    candidate=new_program,
                    parent_program_ids=[id1, id2],
                    subsample_indices=subsample_ids,
                    subsample_scores_before=parent_sums,
                    subsample_scores_after=new_sub_scores,
                    tag="merge",
                    metadata={
                        "proposer": "rlm",
                        "ancestor": ancestor,
                        "merge_call_idx": merge_call_idx,
                        "attempt_idx": attempt_idx,
                    },
                )

            # Subsample-score reject — persist per-task arrays for
            # postmortem flip-stats analysis (same as accept branch).
            state.full_program_trace[-1]["id1_subsample_scores"] = list(id1_sub_scores)
            state.full_program_trace[-1]["id2_subsample_scores"] = list(id2_sub_scores)
            state.full_program_trace[-1]["new_program_subsample_scores"] = list(new_sub_scores)
            self.merges_due -= 1
            self._record_merge_status(
                state, "subsample_rejected", attempt_idx,
                rlm_merge_train_task_ids=trace_task_ids,
                rlm_merge_subsample_ids=list(subsample_ids),
                rlm_merge_candidate_pair=(id1, id2),
                rlm_merge_new_sum=new_sum,
                rlm_merge_parent_sums=parent_sums,
            )
            self.logger.log(
                f"Iteration {i}: RLM merge subsample reject: "
                f"new_sum={new_sum:.4f} < max_parent_sum={max_parent_sum:.4f}"
            )
            return None

        except Exception as exc:
            self._record_merge_status(
                state, "error", attempt_idx,
                rlm_merge_candidate_pair=(id1, id2),
                rlm_merge_error_type=type(exc).__name__,
            )
            try:
                self.adapter._persist_merge_proposer_error(
                    call_idx=merge_call_idx,
                    attempt_idx=attempt_idx,
                    id1=id1, id2=id2, ancestor=ancestor,
                    instructions_a=instructions_a,
                    instructions_b=instructions_b,
                    instructions_ancestor=instructions_ancestor,
                    trace_task_ids=trace_task_ids,
                    exc=exc,
                )
            except Exception as persist_exc:
                self.logger.log(
                    f"merge error persist itself failed: {persist_exc}"
                )
            self.merges_due -= 1
            self.logger.log(
                f"Iteration {i}: RLM merge error: {type(exc).__name__}: {exc}"
            )
            return None

    # -- paired trace file --------------------------------------------------

    def _write_paired_trace_file(
        self,
        *,
        id1: int,
        id2: int,
        ancestor: int,
        attempt_idx: int,
        task_data_ids: list,
        trace_task_ids: list[str],
        reflective_a: list,
        reflective_b: list,
        eval_a: Any,
        eval_b: Any,
        eps: float,
    ) -> str:
        """Write a JSONL file with paired trajectories, one record per task.
        Returns the absolute filesystem path (passed to
        predict_rlm.File(path=...) for sandbox mount).

        Uses adapter.make_reflective_dataset rich records (Inputs /
        Generated Outputs / Feedback) for each parent — same rendering
        the reflective proposer sees — so the merge RLM has actual
        reasoning/code/feedback to synthesize against, not just summary
        stubs.

        Record schema:
          {
            "data_id": <loader id>,
            "task_id": <benchmark id, used for [NEW] citations>,
            "inputs": <Inputs text, same shape as reflective proposer>,
            "parent_a": {
                "generated_outputs": <full trajectory rendering>,
                "feedback": <per-case scores + mismatch diffs + crash reasons>,
                "score": float
            },
            "parent_b": {...},  # symmetric
            "winner": "a" | "b" | "tie"
          }
        """
        trace_dir: Path = (
            Path(self.adapter.proposer_trace_dir)
            if self.adapter.proposer_trace_dir
            else Path.cwd()
        )
        trace_dir.mkdir(parents=True, exist_ok=True)
        fname = (
            f"merge_attempt_{attempt_idx:04d}_from_cand_{id1}_and_cand_{id2}_paired.jsonl"
        )
        path = trace_dir / fname

        with path.open("w") as f:
            for idx, data_id in enumerate(task_data_ids):
                task_id = trace_task_ids[idx] if idx < len(trace_task_ids) else str(data_id)
                rec_a = reflective_a[idx] if idx < len(reflective_a) else {}
                rec_b = reflective_b[idx] if idx < len(reflective_b) else {}
                score_a = eval_a.scores[idx] if idx < len(eval_a.scores) else 0.0
                score_b = eval_b.scores[idx] if idx < len(eval_b.scores) else 0.0
                if score_a > score_b + eps:
                    winner = "a"
                elif score_b > score_a + eps:
                    winner = "b"
                else:
                    winner = "tie"
                # The rich "Inputs" text from make_reflective_dataset is
                # canonical across parents (same benchmark task); use
                # parent_a's by default.
                inputs = rec_a.get("Inputs") or rec_b.get("Inputs") or ""
                rec = {
                    "data_id": data_id,
                    "task_id": task_id,
                    "inputs": inputs,
                    "parent_a": {
                        "generated_outputs": rec_a.get("Generated Outputs", ""),
                        "feedback": rec_a.get("Feedback", ""),
                        "score": score_a,
                    },
                    "parent_b": {
                        "generated_outputs": rec_b.get("Generated Outputs", ""),
                        "feedback": rec_b.get("Feedback", ""),
                        "score": score_b,
                    },
                    "winner": winner,
                }
                f.write(json.dumps(rec, default=str) + "\n")

        return str(path)
