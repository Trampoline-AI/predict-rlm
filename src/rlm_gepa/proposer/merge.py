from __future__ import annotations

import json
import random
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
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

from ..runtime.utils import atomic_write_json
from .selection import pick_patch_merge_pair

VALID_STATUSES = frozenset(
    {
        "attempt_cap_exhausted",
        "pair_skipped",
        "preflight_failed",
        "subsample_rejected",
        "accepted",
        "error",
    }
)
@dataclass(frozen=True)
class PatchDisagreementEvidence:
    records: list[dict[str, Any]]
    paired_trace_path: str
    trace_task_ids: list[str]
    sampled_train_ids: list[Any]
    base_wins_count: int
    patch_source_wins_count: int
    selected_base_wins_count: int
    selected_patch_source_wins_count: int


class RlmMergeProposer(MergeProposer):
    def __init__(
        self,
        *,
        logger: LoggerProtocol,
        valset: DataLoader[DataId, Any],
        evaluator: Callable[
            [list[Any], dict[str, str]],
            tuple[list[RolloutOutput], list[float], Sequence[ObjectiveScores] | None],
        ],
        adapter: Any,
        trainset: DataLoader[DataId, Any],
        use_merge: bool,
        max_merge_invocations: int,
        max_rlm_merge_attempts: int,
        min_each: int = 3,
        merge_minibatch_size: int = 50,
        val_overlap_floor: int = 5,
        component_name: str = "skill_instructions",
        rlm_merge_state_path: Path | None = None,
        rng: random.Random | None = None,
    ):
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

        all_train_ids = list(self.trainset.all_ids())
        if self.merge_minibatch_size > len(all_train_ids):
            raise ValueError(
                f"merge_minibatch_size={self.merge_minibatch_size} exceeds "
                f"trainset size {len(all_train_ids)}"
            )

        self._load_rlm_merge_state()

    def propose(self, state: GEPAState[RolloutOutput, DataId]) -> CandidateProposal[DataId] | None:
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

        if not (
            self.use_merge
            and self.last_iter_found_new_program
            and self.merges_due > 0
        ):
            self.logger.log(f"Iteration {i}: No merge candidates scheduled")
            return None

        if self.rlm_merge_attempts_used >= self.max_rlm_merge_attempts:
            self.merges_due -= 1
            self._record_merge_status(state, "attempt_cap_exhausted", attempt_idx=None)
            self.logger.log(
                f"Iteration {i}: RLM merge attempt cap exhausted "
                f"({self.rlm_merge_attempts_used}/{self.max_rlm_merge_attempts})"
            )
            return None

        pareto_front_programs = state.get_pareto_front_mapping()
        tracked_scores: Sequence[float] = getattr(
            state,
            "per_program_tracked_scores",
            state.program_full_scores_val_set,
        )
        merge_candidates = find_dominator_programs(pareto_front_programs, list(tracked_scores))
        return self._propose_patch_merge(
            state=state,
            iteration=i,
            merge_candidates=merge_candidates,
            tracked_scores=tracked_scores,
        )

    def _propose_patch_merge(
        self,
        *,
        state: GEPAState[RolloutOutput, DataId],
        iteration: int,
        merge_candidates: Sequence[int],
        tracked_scores: Sequence[float],
    ) -> CandidateProposal[DataId] | None:
        pair = pick_patch_merge_pair(
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
        if pair is None:
            self.merges_due -= 1
            self._record_merge_status(state, "pair_skipped", attempt_idx=None)
            self.logger.log(f"Iteration {iteration}: No patch merge pairs found")
            return None

        id1 = pair.parent_a_id
        id2 = pair.parent_b_id
        base_parent_id = pair.base_parent_id
        patch_source_parent_id = pair.patch_source_parent_id
        ancestor = pair.ancestor
        self.merges_performed[0].append((id1, id2, ancestor))
        self.rlm_merge_attempts_used += 1
        attempt_idx = self.rlm_merge_attempts_used - 1
        state.full_program_trace[-1]["rlm_merge_triplet"] = (id1, id2, ancestor)
        state.full_program_trace[-1]["rlm_merge_mode"] = "patch"
        state.full_program_trace[-1]["rlm_merge_base_parent"] = base_parent_id
        state.full_program_trace[-1]["rlm_merge_patch_source_parent"] = patch_source_parent_id
        state.full_program_trace[-1]["rlm_merge_common_ancestors"] = list(pair.common_ancestors)
        state.full_program_trace[-1]["rlm_merge_oracle_score"] = pair.oracle_score
        state.full_program_trace[-1]["rlm_merge_oracle_gain"] = pair.oracle_gain

        merge_call_idx: int | None = None
        trace_task_ids: list[str] = []
        base_instructions = state.program_candidates[base_parent_id][self.component_name]
        patch_source_instructions = state.program_candidates[patch_source_parent_id][
            self.component_name
        ]

        try:
            evidence = self._build_patch_disagreement_evidence(
                state=state,
                iteration=iteration,
                attempt_idx=attempt_idx,
                base_parent_id=base_parent_id,
                patch_source_parent_id=patch_source_parent_id,
            )
            trace_task_ids = evidence.trace_task_ids
            if (
                evidence.base_wins_count < self._min_each
                or evidence.patch_source_wins_count < self._min_each
                or evidence.selected_base_wins_count < self._min_each
                or evidence.selected_patch_source_wins_count < self._min_each
            ):
                self.merges_due -= 1
                self._record_merge_status(
                    state,
                    "preflight_failed",
                    attempt_idx,
                    rlm_merge_train_task_ids=trace_task_ids,
                    rlm_merge_candidate_pair=(id1, id2),
                    rlm_merge_ancestor=ancestor,
                    rlm_merge_base_parent=base_parent_id,
                    rlm_merge_patch_source_parent=patch_source_parent_id,
                    rlm_merge_preflight_base_wins=evidence.base_wins_count,
                    rlm_merge_preflight_patch_source_wins=evidence.patch_source_wins_count,
                    rlm_merge_selected_base_wins=evidence.selected_base_wins_count,
                    rlm_merge_selected_patch_source_wins=evidence.selected_patch_source_wins_count,
                    rlm_merge_paired_disagreement_traces_file=evidence.paired_trace_path,
                    rlm_merge_reject_reason=(
                        "insufficient patch disagreement evidence: "
                        f"base_wins={evidence.base_wins_count}, "
                        f"patch_source_wins={evidence.patch_source_wins_count}, "
                        f"selected_base_wins={evidence.selected_base_wins_count}, "
                        "selected_patch_source_wins="
                        f"{evidence.selected_patch_source_wins_count}, "
                        f"min_each={self._min_each}"
                    ),
                )
                self.logger.log(
                    f"Iteration {iteration}: RLM patch preflight failed "
                    f"(base_wins={evidence.base_wins_count}, "
                    f"patch_source_wins={evidence.patch_source_wins_count})"
                )
                return None

            merge_call_idx = self.adapter._reserve_merge_proposer_call_idx()
            new_instructions, patch_output = self.adapter._rlm_propose_patch_merge_texts(
                call_idx=merge_call_idx,
                attempt_idx=attempt_idx,
                base_parent_id=base_parent_id,
                base_parent_instructions=base_instructions,
                patch_source_parent_id=patch_source_parent_id,
                patch_source_parent_instructions=patch_source_instructions,
                paired_disagreement_traces_file=File(path=evidence.paired_trace_path),
                trace_task_ids=trace_task_ids,
            )

            subsample_ids = [record["data_id"] for record in evidence.records]
            if not subsample_ids:
                self.merges_due -= 1
                self._record_merge_status(
                    state,
                    "subsample_rejected",
                    attempt_idx,
                    rlm_merge_candidate_pair=(id1, id2),
                    rlm_merge_ancestor=ancestor,
                    rlm_merge_base_parent=base_parent_id,
                    rlm_merge_patch_source_parent=patch_source_parent_id,
                    rlm_merge_preflight_base_wins=evidence.base_wins_count,
                    rlm_merge_preflight_patch_source_wins=evidence.patch_source_wins_count,
                    rlm_merge_selected_base_wins=evidence.selected_base_wins_count,
                    rlm_merge_selected_patch_source_wins=evidence.selected_patch_source_wins_count,
                    rlm_merge_reject_reason="empty disagreement gate",
                )
                self.logger.log(f"Iteration {iteration}: RLM patch merge disagreement gate empty")
                return None

            base_sub_scores = [record["score_base"] for record in evidence.records]
            patch_source_sub_scores = [record["score_patch_source"] for record in evidence.records]
            new_program: dict[str, str] = deepcopy(state.program_candidates[base_parent_id])
            new_program[self.component_name] = new_instructions
            child_idx = len(state.program_candidates)
            with self.adapter.progress_label(
                f"Iteration {iteration} Patch Merge Child #{child_idx} Disagreement Gate"
            ):
                new_sub_scores, actual_evals_count = state.cached_evaluate(
                    new_program,
                    subsample_ids,
                    self.trainset.fetch,
                    self.evaluator,
                )
            state.total_num_evals += actual_evals_count
            new_sum = sum(new_sub_scores)
            parent_sums = [sum(base_sub_scores), sum(patch_source_sub_scores)]
            max_parent_sum = max(parent_sums)
            state.full_program_trace[-1]["id1_subsample_scores"] = list(base_sub_scores)
            state.full_program_trace[-1]["id2_subsample_scores"] = list(patch_source_sub_scores)
            state.full_program_trace[-1]["new_program_subsample_scores"] = list(new_sub_scores)
            state.full_program_trace[-1]["rlm_merge_patch_output"] = patch_output

            if new_sum > max_parent_sum:
                state.full_program_trace[-1]["merged"] = True
                state.full_program_trace[-1]["merged_entities"] = (
                    base_parent_id,
                    patch_source_parent_id,
                    ancestor,
                )
                state.full_program_trace[-1]["merge_proposer"] = "rlm"
                self._record_merge_status(
                    state,
                    "accepted",
                    attempt_idx,
                    rlm_merge_train_task_ids=trace_task_ids,
                    rlm_merge_subsample_ids=list(subsample_ids),
                    rlm_merge_candidate_pair=(id1, id2),
                    rlm_merge_ancestor=ancestor,
                    rlm_merge_base_parent=base_parent_id,
                    rlm_merge_patch_source_parent=patch_source_parent_id,
                    rlm_merge_preflight_base_wins=evidence.base_wins_count,
                    rlm_merge_preflight_patch_source_wins=evidence.patch_source_wins_count,
                    rlm_merge_selected_base_wins=evidence.selected_base_wins_count,
                    rlm_merge_selected_patch_source_wins=evidence.selected_patch_source_wins_count,
                    rlm_merge_new_sum=new_sum,
                    rlm_merge_parent_sums=parent_sums,
                )
                self.adapter.queue_valset_progress_label(
                    f"Candidate #{child_idx} Valset (Patch Merge #{base_parent_id}+{patch_source_parent_id})"
                )
                return CandidateProposal(
                    candidate=new_program,
                    parent_program_ids=[base_parent_id, patch_source_parent_id],
                    subsample_indices=subsample_ids,
                    subsample_scores_before=parent_sums,
                    subsample_scores_after=new_sub_scores,
                    tag="merge",
                    metadata={
                        "proposer": "rlm",
                        "mode": "patch",
                        "ancestor": ancestor,
                        "base_parent_id": base_parent_id,
                        "patch_source_parent_id": patch_source_parent_id,
                        "merge_call_idx": merge_call_idx,
                        "attempt_idx": attempt_idx,
                    },
                )

            self.merges_due -= 1
            self._record_merge_status(
                state,
                "subsample_rejected",
                attempt_idx,
                rlm_merge_train_task_ids=trace_task_ids,
                rlm_merge_subsample_ids=list(subsample_ids),
                rlm_merge_candidate_pair=(id1, id2),
                rlm_merge_ancestor=ancestor,
                rlm_merge_base_parent=base_parent_id,
                rlm_merge_patch_source_parent=patch_source_parent_id,
                rlm_merge_preflight_base_wins=evidence.base_wins_count,
                rlm_merge_preflight_patch_source_wins=evidence.patch_source_wins_count,
                rlm_merge_selected_base_wins=evidence.selected_base_wins_count,
                rlm_merge_selected_patch_source_wins=evidence.selected_patch_source_wins_count,
                rlm_merge_new_sum=new_sum,
                rlm_merge_parent_sums=parent_sums,
            )
            self.logger.log(
                f"Iteration {iteration}: RLM patch merge disagreement gate reject: "
                f"new_sum={new_sum:.4f} <= max_parent_sum={max_parent_sum:.4f}"
            )
            return None
        except Exception as exc:
            self._record_merge_status(
                state,
                "error",
                attempt_idx,
                rlm_merge_candidate_pair=(id1, id2),
                rlm_merge_ancestor=ancestor,
                rlm_merge_base_parent=base_parent_id,
                rlm_merge_patch_source_parent=patch_source_parent_id,
                rlm_merge_error_type=type(exc).__name__,
            )
            self.merges_due -= 1
            self.logger.log(
                f"Iteration {iteration}: RLM patch merge error: {type(exc).__name__}: {exc}"
            )
            return None

    def _build_patch_disagreement_evidence(
        self,
        *,
        state: GEPAState[RolloutOutput, DataId],
        iteration: int,
        attempt_idx: int,
        base_parent_id: int,
        patch_source_parent_id: int,
        eps: float = 1e-6,
    ) -> PatchDisagreementEvidence:
        all_train_ids = list(self.trainset.all_ids())
        sample_size = min(2 * self.merge_minibatch_size, len(all_train_ids))
        task_data_ids = self.rng.sample(all_train_ids, k=sample_size)
        batch = self.trainset.fetch(task_data_ids)

        with self.adapter.progress_label(
            f"Iteration {iteration} Patch Base Parent #{base_parent_id} Trace"
        ):
            eval_base = self.adapter.evaluate(
                batch,
                state.program_candidates[base_parent_id],
                capture_traces=True,
                kind="merge_trace_capture",
            )
        state.total_num_evals += len(task_data_ids)
        with self.adapter.progress_label(
            f"Iteration {iteration} Patch Source Parent #{patch_source_parent_id} Trace"
        ):
            eval_source = self.adapter.evaluate(
                batch,
                state.program_candidates[patch_source_parent_id],
                capture_traces=True,
                kind="merge_trace_capture",
            )
        state.total_num_evals += len(task_data_ids)

        trace_task_ids = [
            str(traj.get("task_id") or traj.get("example_id") or traj)
            for traj in (eval_base.trajectories or [])
        ]
        source_trace_task_ids = [
            str(traj.get("task_id") or traj.get("example_id") or traj)
            for traj in (eval_source.trajectories or [])
        ]
        if trace_task_ids != source_trace_task_ids:
            raise RuntimeError("task_id misalignment between paired patch trace captures")

        reflective_base = self.adapter.make_reflective_dataset(
            state.program_candidates[base_parent_id],
            eval_base,
            [self.component_name],
        ).get(self.component_name, [])
        reflective_source = self.adapter.make_reflective_dataset(
            state.program_candidates[patch_source_parent_id],
            eval_source,
            [self.component_name],
        ).get(self.component_name, [])

        records: list[dict[str, Any]] = []
        both_success_records: list[dict[str, Any]] = []
        for index, data_id in enumerate(task_data_ids):
            score_base = eval_base.scores[index] if index < len(eval_base.scores) else 0.0
            score_source = eval_source.scores[index] if index < len(eval_source.scores) else 0.0
            abs_delta = abs(score_base - score_source)
            if score_base > score_source + eps:
                winner = "base"
                evidence_role = "base_win"
            elif score_source > score_base + eps:
                winner = "patch_source"
                evidence_role = "patch_source_win"
            elif score_base >= 1.0 - eps and score_source >= 1.0 - eps:
                winner = "both_success"
                evidence_role = "both_success_guardrail"
            else:
                continue

            rec_base = reflective_base[index] if index < len(reflective_base) else {}
            rec_source = reflective_source[index] if index < len(reflective_source) else {}
            task_id = trace_task_ids[index] if index < len(trace_task_ids) else str(data_id)
            record = {
                "schema_version": 1,
                "data_id": data_id,
                "task_id": task_id,
                "base_parent_id": base_parent_id,
                "patch_source_parent_id": patch_source_parent_id,
                "score_base": score_base,
                "score_patch_source": score_source,
                "winner": winner,
                "abs_delta": abs_delta,
                "evidence_role": evidence_role,
                "inputs": rec_base.get("Inputs") or rec_source.get("Inputs") or "",
                "base_parent": {
                    "generated_outputs": rec_base.get("Generated Outputs", ""),
                    "feedback": rec_base.get("Feedback", ""),
                    "score": score_base,
                },
                "patch_source_parent": {
                    "generated_outputs": rec_source.get("Generated Outputs", ""),
                    "feedback": rec_source.get("Feedback", ""),
                    "score": score_source,
                },
            }
            if winner == "both_success":
                both_success_records.append(record)
            else:
                records.append(record)

        records.sort(key=lambda row: row["abs_delta"], reverse=True)
        base_records = [record for record in records if record["winner"] == "base"]
        source_records = [record for record in records if record["winner"] == "patch_source"]
        selected = self._balance_patch_disagreement_records(
            base_records,
            source_records,
            both_success_records,
        )
        trace_path = self._write_patch_disagreement_trace_file(
            base_parent_id=base_parent_id,
            patch_source_parent_id=patch_source_parent_id,
            attempt_idx=attempt_idx,
            records=selected,
        )
        selected_base_wins = sum(1 for record in selected if record["winner"] == "base")
        selected_patch_source_wins = sum(
            1 for record in selected if record["winner"] == "patch_source"
        )
        return PatchDisagreementEvidence(
            records=selected,
            paired_trace_path=trace_path,
            trace_task_ids=[str(record["task_id"]) for record in selected],
            sampled_train_ids=task_data_ids,
            base_wins_count=len(base_records),
            patch_source_wins_count=len(source_records),
            selected_base_wins_count=selected_base_wins,
            selected_patch_source_wins_count=selected_patch_source_wins,
        )

    def _balance_patch_disagreement_records(
        self,
        base_records: list[dict[str, Any]],
        source_records: list[dict[str, Any]],
        both_success_records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        max_even = self.merge_minibatch_size // 2
        take_each = min(len(base_records), len(source_records), max_even)
        selected = base_records[:take_each] + source_records[:take_each]

        if self.merge_minibatch_size % 2 and len(selected) < self.merge_minibatch_size:
            next_base = base_records[take_each] if take_each < len(base_records) else None
            next_source = source_records[take_each] if take_each < len(source_records) else None
            if next_base is not None and next_source is not None:
                if next_base["abs_delta"] >= next_source["abs_delta"]:
                    selected.append(next_base)
                else:
                    selected.append(next_source)
            elif next_base is not None:
                selected.append(next_base)
            elif next_source is not None:
                selected.append(next_source)

        fill_count = min(self.merge_minibatch_size - len(selected), len(both_success_records))
        if fill_count > 0:
            selected.extend(self.rng.sample(both_success_records, k=fill_count))

        return sorted(selected, key=lambda row: row["abs_delta"], reverse=True)

    def _write_patch_disagreement_trace_file(
        self,
        *,
        base_parent_id: int,
        patch_source_parent_id: int,
        attempt_idx: int,
        records: list[dict[str, Any]],
    ) -> str:
        trace_dir = Path(self.adapter.proposer_trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        event_id = (
            f"{self.adapter.run_id}_patch_merge_attempt_{attempt_idx:04d}_"
            f"base_{base_parent_id}_source_{patch_source_parent_id}"
        )
        path = (
            trace_dir
            / f"{event_id}_patch_disagreement_cand_{base_parent_id}_and_cand_{patch_source_parent_id}.jsonl"
        )
        with path.open("x", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, default=str) + "\n")
        return str(path)

    def _load_rlm_merge_state(self) -> None:
        if self.rlm_merge_state_path is None or not self.rlm_merge_state_path.is_file():
            return
        data = json.loads(self.rlm_merge_state_path.read_text())
        self.rlm_merge_attempts_used = int(data.get("rlm_merge_attempts_used", 0))
        merges_performed = data.get("merges_performed", [])
        if isinstance(merges_performed, list):
            for entry in merges_performed:
                if isinstance(entry, list | tuple) and len(entry) == 3:
                    triplet = (int(entry[0]), int(entry[1]), int(entry[2]))
                    if triplet not in self.merges_performed[0]:
                        self.merges_performed[0].append(triplet)

    def _flush_rlm_merge_state(self) -> None:
        if self.rlm_merge_state_path is None:
            return
        atomic_write_json(
            self.rlm_merge_state_path,
            {
                "schema_version": 1,
                "rlm_merge_attempts_used": self.rlm_merge_attempts_used,
                "merges_performed": [list(triplet) for triplet in self.merges_performed[0]],
                "flushed_at": datetime.now().isoformat(),
            },
        )

    def _record_merge_status(
        self,
        state: GEPAState[RolloutOutput, DataId],
        status: str,
        attempt_idx: int | None,
        **fields: Any,
    ) -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid merge status {status!r}")
        trace_dict = state.full_program_trace[-1]
        trace_dict["rlm_merge_status"] = status
        trace_dict["rlm_merge_attempt_idx"] = attempt_idx
        if status in {"pair_skipped", "attempt_cap_exhausted"}:
            trace_dict["rlm_merge_considered"] = True
        for key, value in fields.items():
            if not key.startswith("rlm_merge_"):
                raise ValueError(f"namespace field {key!r} must start with 'rlm_merge_'")
            trace_dict[key] = value
