from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
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

from ..runtime.utils import atomic_write_json
from .selection import pick_merge_triplet

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
        self.merges_performed[0].append((id1, id2, ancestor))
        self.rlm_merge_attempts_used += 1
        attempt_idx = self.rlm_merge_attempts_used - 1
        state.full_program_trace[-1]["rlm_merge_triplet"] = (id1, id2, ancestor)

        trace_task_ids: list[str] = []
        merge_call_idx: int | None = None
        instructions_a = state.program_candidates[id1][self.component_name]
        instructions_b = state.program_candidates[id2][self.component_name]
        instructions_ancestor = state.program_candidates[ancestor][self.component_name]

        try:
            all_train_ids = list(self.trainset.all_ids())
            task_data_ids = self.rng.sample(
                all_train_ids,
                k=min(self.merge_minibatch_size, len(all_train_ids)),
            )
            batch = self.trainset.fetch(task_data_ids)
            with self.adapter.progress_label(f"Iteration {i} Merge Parent #{id1} Trace"):
                eval_a = self.adapter.evaluate(
                    batch,
                    state.program_candidates[id1],
                    capture_traces=True,
                    kind="merge_trace_capture",
                )
            state.total_num_evals += len(task_data_ids)
            with self.adapter.progress_label(f"Iteration {i} Merge Parent #{id2} Trace"):
                eval_b = self.adapter.evaluate(
                    batch,
                    state.program_candidates[id2],
                    capture_traces=True,
                    kind="merge_trace_capture",
                )
            state.total_num_evals += len(task_data_ids)

            trace_task_ids = [
                str(traj.get("task_id") or traj.get("example_id") or traj)
                for traj in (eval_a.trajectories or [])
            ]
            trace_task_ids_b = [
                str(traj.get("task_id") or traj.get("example_id") or traj)
                for traj in (eval_b.trajectories or [])
            ]
            if trace_task_ids != trace_task_ids_b:
                raise RuntimeError("task_id misalignment between paired trace captures")

            eps = 1e-6
            a_wins = sum(1 for score_a, score_b in zip(eval_a.scores, eval_b.scores, strict=False) if score_a > score_b + eps)
            b_wins = sum(1 for score_a, score_b in zip(eval_a.scores, eval_b.scores, strict=False) if score_b > score_a + eps)
            if a_wins < 1 or b_wins < 1:
                self.merges_due -= 1
                self._record_merge_status(
                    state,
                    "preflight_failed",
                    attempt_idx,
                    rlm_merge_train_task_ids=trace_task_ids,
                    rlm_merge_candidate_pair=(id1, id2),
                    rlm_merge_ancestor=ancestor,
                    rlm_merge_preflight_a_wins=a_wins,
                    rlm_merge_preflight_b_wins=b_wins,
                )
                self.logger.log(
                    f"Iteration {i}: RLM merge preflight failed "
                    f"(a_wins={a_wins}, b_wins={b_wins})"
                )
                return None

            reflective_a = self.adapter.make_reflective_dataset(
                state.program_candidates[id1],
                eval_a,
                [self.component_name],
            ).get(self.component_name, [])
            reflective_b = self.adapter.make_reflective_dataset(
                state.program_candidates[id2],
                eval_b,
                [self.component_name],
            ).get(self.component_name, [])
            paired_trace_path = self._write_paired_trace_file(
                id1=id1,
                id2=id2,
                ancestor=ancestor,
                attempt_idx=attempt_idx,
                task_data_ids=task_data_ids,
                trace_task_ids=trace_task_ids,
                reflective_a=list(reflective_a),
                reflective_b=list(reflective_b),
                eval_a=eval_a,
                eval_b=eval_b,
                eps=eps,
            )

            merge_call_idx = self.adapter._reserve_merge_proposer_call_idx()
            new_instructions, _audit_lines = self.adapter._rlm_propose_merge_texts(
                call_idx=merge_call_idx,
                attempt_idx=attempt_idx,
                id1=id1,
                id2=id2,
                ancestor=ancestor,
                current_instructions_a=instructions_a,
                current_instructions_b=instructions_b,
                common_ancestor_instructions=instructions_ancestor,
                paired_traces_file=File(path=paired_trace_path),
                trace_task_ids=trace_task_ids,
            )

            subsample_ids = self.select_eval_subsample_for_merged_program(
                state.prog_candidate_val_subscores[id1],
                state.prog_candidate_val_subscores[id2],
                num_subsample_ids=self.merge_minibatch_size,
            )
            if not subsample_ids:
                self.merges_due -= 1
                self._record_merge_status(
                    state,
                    "subsample_rejected",
                    attempt_idx,
                    rlm_merge_candidate_pair=(id1, id2),
                    rlm_merge_ancestor=ancestor,
                    rlm_merge_reject_reason="empty subsample",
                )
                self.logger.log(f"Iteration {i}: RLM merge subsample empty")
                return None

            id1_sub_scores = [state.prog_candidate_val_subscores[id1][key] for key in subsample_ids]
            id2_sub_scores = [state.prog_candidate_val_subscores[id2][key] for key in subsample_ids]
            new_program: dict[str, str] = deepcopy(state.program_candidates[ancestor])
            new_program[self.component_name] = new_instructions
            child_idx = len(state.program_candidates)
            with self.adapter.progress_label(f"Iteration {i} Merge Child #{child_idx} Subsample"):
                new_sub_scores, actual_evals_count = state.cached_evaluate(
                    new_program,
                    subsample_ids,
                    self.valset.fetch,
                    self.evaluator,
                )
            state.total_num_evals += actual_evals_count
            new_sum = sum(new_sub_scores)
            parent_sums = [sum(id1_sub_scores), sum(id2_sub_scores)]
            max_parent_sum = max(parent_sums)
            state.full_program_trace[-1]["id1_subsample_scores"] = list(id1_sub_scores)
            state.full_program_trace[-1]["id2_subsample_scores"] = list(id2_sub_scores)
            state.full_program_trace[-1]["new_program_subsample_scores"] = list(new_sub_scores)

            if new_sum > max_parent_sum:
                state.full_program_trace[-1]["merged"] = True
                state.full_program_trace[-1]["merged_entities"] = (id1, id2, ancestor)
                state.full_program_trace[-1]["merge_proposer"] = "rlm"
                self._record_merge_status(
                    state,
                    "accepted",
                    attempt_idx,
                    rlm_merge_train_task_ids=trace_task_ids,
                    rlm_merge_subsample_ids=list(subsample_ids),
                    rlm_merge_candidate_pair=(id1, id2),
                    rlm_merge_ancestor=ancestor,
                    rlm_merge_new_sum=new_sum,
                    rlm_merge_parent_sums=parent_sums,
                )
                self.adapter.queue_valset_progress_label(
                    f"Candidate #{child_idx} Valset (Merge #{id1}+#{id2})"
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

            self.merges_due -= 1
            self._record_merge_status(
                state,
                "subsample_rejected",
                attempt_idx,
                rlm_merge_train_task_ids=trace_task_ids,
                rlm_merge_subsample_ids=list(subsample_ids),
                rlm_merge_candidate_pair=(id1, id2),
                rlm_merge_ancestor=ancestor,
                rlm_merge_new_sum=new_sum,
                rlm_merge_parent_sums=parent_sums,
            )
            self.logger.log(
                f"Iteration {i}: RLM merge subsample reject: "
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
                rlm_merge_error_type=type(exc).__name__,
            )
            try:
                self.adapter._persist_merge_proposer_error(
                    call_idx=merge_call_idx,
                    attempt_idx=attempt_idx,
                    id1=id1,
                    id2=id2,
                    ancestor=ancestor,
                    instructions_a=instructions_a,
                    instructions_b=instructions_b,
                    instructions_ancestor=instructions_ancestor,
                    trace_task_ids=trace_task_ids,
                    exc=exc,
                )
            except Exception as persist_exc:
                self.logger.log(f"merge error persist itself failed: {persist_exc}")
            self.merges_due -= 1
            self.logger.log(f"Iteration {i}: RLM merge error: {type(exc).__name__}: {exc}")
            return None

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

    def _write_paired_trace_file(
        self,
        *,
        id1: int,
        id2: int,
        ancestor: int,
        attempt_idx: int,
        task_data_ids: list[Any],
        trace_task_ids: list[str],
        reflective_a: list[Mapping[str, Any]],
        reflective_b: list[Mapping[str, Any]],
        eval_a: Any,
        eval_b: Any,
        eps: float,
    ) -> str:
        trace_dir = Path(self.adapter.proposer_trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        event_id = (
            f"{self.adapter.run_id}_merge_attempt_{attempt_idx:04d}_"
            f"cand_{id1}_cand_{id2}_ancestor_{ancestor}"
        )
        path = trace_dir / f"{event_id}_merge_from_cand_{id1}_and_cand_{id2}_paired.jsonl"
        with path.open("x", encoding="utf-8") as f:
            for index, data_id in enumerate(task_data_ids):
                task_id = trace_task_ids[index] if index < len(trace_task_ids) else str(data_id)
                rec_a = reflective_a[index] if index < len(reflective_a) else {}
                rec_b = reflective_b[index] if index < len(reflective_b) else {}
                score_a = eval_a.scores[index] if index < len(eval_a.scores) else 0.0
                score_b = eval_b.scores[index] if index < len(eval_b.scores) else 0.0
                if score_a > score_b + eps:
                    winner = "a"
                elif score_b > score_a + eps:
                    winner = "b"
                else:
                    winner = "tie"
                row = {
                    "schema_version": 1,
                    "data_id": data_id,
                    "task_id": task_id,
                    "inputs": rec_a.get("Inputs") or rec_b.get("Inputs") or "",
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
                f.write(json.dumps(row, default=str) + "\n")
        return str(path)
