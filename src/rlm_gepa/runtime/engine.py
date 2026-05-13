from __future__ import annotations

import traceback
from dataclasses import asdict
from typing import Any

from gepa.core.engine import GEPAEngine
from gepa.core.state import GEPAState, ValsetEvaluation, initialize_gepa_state

from .acceptance import should_accept_reflective_candidate

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


class RLMGepaEngine(GEPAEngine):
    def run(self) -> GEPAState:
        progress_bar = None
        if self.display_progress_bar:
            if tqdm is None:
                raise ImportError("tqdm must be installed when display_progress_bar is enabled")

            total_calls: int | None = None
            stop_cb = self.stop_callback
            if stop_cb is not None:
                max_calls_attr = getattr(stop_cb, "max_metric_calls", None)
                if isinstance(max_calls_attr, int):
                    total_calls = max_calls_attr
                else:
                    stoppers = getattr(stop_cb, "stoppers", None)
                    if stoppers is not None:
                        for stopper in stoppers:
                            stopper_max = getattr(stopper, "max_metric_calls", None)
                            if isinstance(stopper_max, int):
                                total_calls = stopper_max
                                break

            if total_calls is not None:
                progress_bar = tqdm(total=total_calls, desc="GEPA Optimization", unit="rollouts")
            else:
                progress_bar = tqdm(desc="GEPA Optimization", unit="rollouts")
            progress_bar.update(0)

        valset = self.valset
        if valset is None:
            raise ValueError("valset must be provided to GEPAEngine.run()")

        def valset_evaluator(program: dict[str, str]) -> ValsetEvaluation:
            all_ids = list(valset.all_ids())
            outputs, scores, objective_scores = self.evaluator(valset.fetch(all_ids), program)
            outputs_dict = dict(zip(all_ids, outputs, strict=False))
            scores_dict = dict(zip(all_ids, scores, strict=False))
            objective_scores_dict = (
                dict(zip(all_ids, objective_scores, strict=False))
                if objective_scores is not None
                else None
            )
            return ValsetEvaluation(
                outputs_by_val_id=outputs_dict,
                scores_by_val_id=scores_dict,
                objective_scores_by_val_id=objective_scores_dict,
            )

        state = initialize_gepa_state(
            run_dir=self.run_dir,
            logger=self.logger,
            seed_candidate=self.seed_candidate,
            valset_evaluator=valset_evaluator,
            track_best_outputs=self.track_best_outputs,
            frontier_type=self.frontier_type,
            evaluation_cache=self._initial_evaluation_cache,
        )

        base_val_avg, base_val_coverage = state.get_program_average_val_subset(0)
        self.experiment_tracker.log_metrics(
            {
                "base_program_full_valset_score": base_val_avg,
                "base_program_val_coverage": base_val_coverage,
                "iteration": state.i + 1,
                "total_metric_calls": state.total_num_evals,
            },
            step=state.i + 1,
        )

        self.logger.log(
            f"Iteration {state.i + 1}: Base program full valset score: {base_val_avg} "
            f"over {base_val_coverage} / {len(valset)} examples"
        )

        if self.merge_proposer is not None:
            self.merge_proposer.last_iter_found_new_program = False

        last_pbar_val = 0
        while not self._should_stop(state):
            if self.display_progress_bar and progress_bar is not None:
                delta = state.total_num_evals - last_pbar_val
                progress_bar.update(delta)
                last_pbar_val = state.total_num_evals

            assert state.is_consistent()
            try:
                state.save(self.run_dir, use_cloudpickle=self.use_cloudpickle)
                state.i += 1
                state.full_program_trace.append({"i": state.i})

                if self.merge_proposer is not None and self.merge_proposer.use_merge:
                    if (
                        self.merge_proposer.merges_due > 0
                        and self.merge_proposer.last_iter_found_new_program
                    ):
                        proposal = self.merge_proposer.propose(state)
                        self.merge_proposer.last_iter_found_new_program = False

                        if proposal is not None and proposal.tag == "merge":
                            parent_sums = proposal.subsample_scores_before or [
                                float("-inf"),
                                float("-inf"),
                            ]
                            new_sum = sum(proposal.subsample_scores_after or [])

                            if new_sum >= max(parent_sums):
                                self._run_full_eval_and_add(
                                    new_program=proposal.candidate,
                                    state=state,
                                    parent_program_idx=proposal.parent_program_ids,
                                )
                                self.merge_proposer.merges_due -= 1
                                self.merge_proposer.total_merges_tested += 1
                                continue
                            self.logger.log(
                                f"Iteration {state.i + 1}: New program subsample score {new_sum} "
                                f"is worse than both parents {parent_sums}, skipping merge"
                            )
                            continue

                    self.merge_proposer.last_iter_found_new_program = False

                proposal = self.reflective_proposer.propose(state)
                if proposal is None:
                    self.logger.log(
                        f"Iteration {state.i + 1}: Reflective mutation did not propose a new candidate"
                    )
                    continue

                before_scores = proposal.subsample_scores_before or []
                after_scores = proposal.subsample_scores_after or []
                decision = should_accept_reflective_candidate(
                    before_scores=before_scores,
                    after_scores=after_scores,
                    perfect_score=self.perfect_score,
                )
                state.full_program_trace[-1]["acceptance_decision"] = asdict(decision)
                if not decision.accepted:
                    self.logger.log(
                        _format_acceptance_log(
                            iteration=state.i + 1,
                            decision=decision,
                            accepted=False,
                        )
                    )
                    continue

                self.logger.log(
                    _format_acceptance_log(
                        iteration=state.i + 1,
                        decision=decision,
                        accepted=True,
                    )
                )

                self._run_full_eval_and_add(
                    new_program=proposal.candidate,
                    state=state,
                    parent_program_idx=proposal.parent_program_ids,
                )

                if self.merge_proposer is not None:
                    self.merge_proposer.last_iter_found_new_program = True
                    if self.merge_proposer.total_merges_tested < self.merge_proposer.max_merge_invocations:
                        self.merge_proposer.merges_due += 1

            except Exception as e:
                self.logger.log(f"Iteration {state.i + 1}: Exception during optimization: {e}")
                self.logger.log(traceback.format_exc())
                if self.raise_on_exception:
                    raise e
                continue

        if self.display_progress_bar and progress_bar is not None:
            progress_bar.close()

        state.save(self.run_dir, use_cloudpickle=self.use_cloudpickle)
        return state


def _format_acceptance_log(*, iteration: int, decision: Any, accepted: bool) -> str:
    status = "accepted" if accepted else "skipping"
    return (
        f"Iteration {iteration}: Reflective candidate {status}: reason={decision.reason}, "
        f"dense_delta={decision.dense_delta:.4f}, "
        f"hard_wins={decision.hard_wins}, hard_losses={decision.hard_losses}, "
        f"hard_flip_p={decision.hard_flip_p_value:.3f}"
    )
