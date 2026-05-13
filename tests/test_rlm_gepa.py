from __future__ import annotations

import argparse
import asyncio
import json
import os
import pickle
import random
from pathlib import Path
from types import SimpleNamespace

import dspy
import pytest

import rlm_gepa.cli as cli_module
from predict_rlm.telemetry import JsonlTelemetrySink, TelemetryContext, classify_failure
from predict_rlm.trace import (
    IterationStep,
    LMFinishMetadata,
    LMUsage,
    PredictCallDetail,
    PredictCallGroup,
    RunTrace,
    TokenUsage,
    ToolCall,
)
from rlm_gepa import (
    AgentSpec,
    EvaluationContext,
    OptimizeConfig,
    RLMGepaProject,
    agent_spec_from_rlm,
    build_merge_signature,
    build_patch_merge_signature,
    build_proposer_for_rlm,
    build_proposer_signature,
    check_optimization,
    run_optimization,
)
from rlm_gepa.cli import apply_optimize_args, run_project_cli
from rlm_gepa.proposer.merge import VALID_STATUSES, RlmMergeProposer
from rlm_gepa.proposer.rlm import SelectedCapability
from rlm_gepa.proposer.selection import (
    PatchMergePair,
    pick_patch_merge_pair,
)
from rlm_gepa.reporting import stats as stats_report
from rlm_gepa.reporting.cost import CostRow, aggregate_costs_from_log, append_cost_rows
from rlm_gepa.reporting.plots import resolve_plot_output_paths
from rlm_gepa.reporting.stats import (
    candidate_rows,
    cost_rows,
    eval_cost_rows,
    eval_task_rows,
    iteration_rows,
    merge_rows,
    render_stats,
    render_table,
)
from rlm_gepa.runtime.acceptance import should_accept_reflective_candidate
from rlm_gepa.runtime.adapter import RLMGepaAdapter, _row_failure_metadata
from rlm_gepa.schema import RLMGepaExampleResult, validate_project
from rlm_gepa.service import (
    _coerce_reflection_lm_text,
    _ProgressCandidateSelector,
    prepare_run_dir,
)


class _DummyLM:
    model = "dummy/model"


class _Logger:
    def log(self, *_args, **_kwargs):
        pass


def _spec() -> AgentSpec:
    return AgentSpec(
        agent_type="test agent",
        use_cases=["case a", "case b"],
        runtime_grounding_examples={
            "tools": ["tool()"],
            "env": ["sandbox timeout"],
            "spec": ["protocol behavior"],
        },
        tool_signatures="tool() -> str",
        target_signature="input: str -> output: str",
        scoring_description="score is exact match",
    )


def _spec_without_agent_type() -> AgentSpec:
    return AgentSpec(
        use_cases=["case a", "case b"],
        runtime_grounding_examples={
            "tools": ["tool()"],
            "env": ["sandbox timeout"],
            "spec": ["protocol behavior"],
        },
        tool_signatures="tool() -> str",
        target_signature="input: str -> output: str",
        scoring_description="score is exact match",
    )


class _Project(RLMGepaProject):
    project_name = "test-project"
    components = ("skill_instructions",)
    agent_spec = _spec()

    def seed_candidate(self) -> dict[str, str]:
        return {"skill_instructions": "seed rules"}

    def load_trainset(self):
        return ["train"]

    def load_valset(self):
        return ["val"]

    async def evaluate_example(self, candidate, example, context):  # pragma: no cover
        raise NotImplementedError


def test_reflective_candidate_accepts_bounded_dense_loss_with_hard_flip_signal():
    decision = should_accept_reflective_candidate(
        before_scores=[0.99, 0.99, 0.99, 0.50],
        after_scores=[1.00, 1.00, 1.00, 0.45],
    )

    assert decision.accepted
    assert decision.reason == "hard_flip_signal"
    assert decision.dense_delta < 0.0
    assert decision.hard_wins == 3
    assert decision.hard_losses == 0
    assert decision.hard_flip_p_value <= 0.40


def test_reflective_candidate_rejects_bounded_dense_loss_without_significant_hard_flips():
    decision = should_accept_reflective_candidate(
        before_scores=[0.99, 0.80],
        after_scores=[1.00, 0.77],
    )

    assert not decision.accepted
    assert decision.reason == "not_improved"
    assert decision.hard_wins == 1
    assert decision.hard_losses == 0
    assert decision.hard_flip_p_value > 0.40


def test_reflective_candidate_reports_two_sided_hard_flip_p_value_for_ties():
    decision = should_accept_reflective_candidate(
        before_scores=[0.99, 0.99, 0.99, 0.99, 1.00, 1.00, 1.00, 1.00],
        after_scores=[1.00, 1.00, 1.00, 1.00, 0.99, 0.99, 0.99, 0.99],
    )

    assert decision.hard_wins == 4
    assert decision.hard_losses == 4
    assert decision.hard_flip_p_value == pytest.approx(1.0)


def test_reflective_candidate_accepts_two_sided_hard_flip_signal_under_default_threshold():
    decision = should_accept_reflective_candidate(
        before_scores=[0.99, 0.99, 0.99, 0.99, 1.00],
        after_scores=[1.00, 1.00, 1.00, 1.00, 0.94],
    )

    assert decision.accepted
    assert decision.reason == "hard_flip_signal"
    assert decision.dense_delta >= -0.01
    assert decision.hard_wins == 4
    assert decision.hard_losses == 1
    assert decision.hard_flip_p_value == pytest.approx(0.375)


def test_build_signatures_render_agent_spec():
    spec = _spec()
    proposer = build_proposer_signature(spec)
    merge = build_merge_signature(spec)

    assert "test agent" in proposer.instructions
    assert "{{" not in proposer.instructions
    assert "paired_disagreement_traces_file" in merge.input_fields


def test_agent_spec_agent_type_is_optional():
    proposer = build_proposer_signature(_spec_without_agent_type())
    merge = build_merge_signature(_spec_without_agent_type())

    assert "target RLM" in proposer.instructions
    assert "target RLM" in merge.instructions
    assert "for ." not in proposer.instructions
    assert "{{" not in proposer.instructions


def test_agent_spec_from_rlm_can_omit_agent_type():
    class DemoSignature(dspy.Signature):
        """Answer questions."""

        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    def lookup(query: str) -> str:
        """Look up a fact."""
        return query

    rlm = SimpleNamespace(signature=DemoSignature, tools=[lookup])

    spec = agent_spec_from_rlm(
        rlm,
        use_cases=["case a", "case b"],
        runtime_grounding_examples={
            "tools": ["lookup(query)"],
            "env": ["sandbox timeout"],
            "spec": ["question answering"],
        },
        scoring_description="score is exact match",
    )

    assert spec.agent_type == ""
    assert "DemoSignature" in spec.target_signature
    assert "lookup" in spec.tool_signatures


def test_proposer_signature_allows_multiple_async_predict_passes_for_concrete_edits():
    proposer = build_proposer_signature(_spec())
    instructions = " ".join(proposer.instructions.split())

    assert "one or more `predict()` calls" in instructions
    assert "asyncio.gather" in instructions
    assert "structured brief containing root causes and edit decisions" in instructions
    assert "must not draft the substantive wording from scratch" in instructions


def test_patch_merge_signature_uses_base_and_patch_source_without_ancestor():
    patch = build_patch_merge_signature(_spec())

    assert "base_parent_id" in patch.input_fields
    assert "base_parent_instructions" in patch.input_fields
    assert "patch_source_parent_id" in patch.input_fields
    assert "patch_source_parent_instructions" in patch.input_fields
    assert "paired_disagreement_traces_file" in patch.input_fields
    assert "common_ancestor_instructions" not in patch.input_fields
    assert "common ancestor" not in patch.instructions.lower()


def test_patch_merge_signature_exposes_selected_capability_contract():
    patch = build_patch_merge_signature(_spec())

    assert "selected_capability" in patch.output_fields
    assert "imported_from_other" in patch.output_fields
    assert "rejected_from_other" in patch.output_fields
    assert "new_instructions" in patch.output_fields
    assert set(SelectedCapability.model_fields) == {
        "name",
        "evidence_task_ids",
        "trigger",
        "action",
        "non_application_boundary",
        "preservation_note",
    }


def test_merge_signature_is_evidence_backed_patch_contract():
    merge = build_merge_signature(_spec())

    assert set(merge.input_fields) == set(build_patch_merge_signature(_spec()).input_fields)
    assert "base_parent_id" in merge.input_fields
    assert "paired_disagreement_traces_file" in merge.input_fields
    assert "common_ancestor_instructions" not in merge.input_fields
    assert "synthesize" not in merge.instructions.lower()


def test_validate_project_accepts_minimal_project():
    result = validate_project(_Project())
    assert result.seed_candidate == {"skill_instructions": "seed rules"}
    assert list(result.trainset) == ["train"]
    assert list(result.valset) == ["val"]


def test_validate_project_rejects_seed_key_mismatch():
    class BadProject(_Project):
        def seed_candidate(self) -> dict[str, str]:
            return {"other": "text"}

    with pytest.raises(ValueError, match="exactly the declared component keys"):
        validate_project(BadProject())


def test_pick_patch_merge_pair_rejects_when_oracle_does_not_beat_current_best():
    parents = [[None], [0], [0]]
    candidates = [
        {"skill_instructions": "seed"},
        {"skill_instructions": "base"},
        {"skill_instructions": "patch"},
    ]
    scores_a = {f"t{i}": 1.0 if i % 2 else 0.0 for i in range(6)}
    scores_b = {f"t{i}": 0.0 if i % 2 else 1.0 for i in range(6)}

    pair = pick_patch_merge_pair(
        merge_candidates=[1, 2],
        program_candidates=candidates,
        parent_program_for_candidate=parents,
        prog_candidate_val_subscores=[{}, scores_a, scores_b],
        tracked_scores=[0.0, 1.0, 0.5],
        merges_performed=[],
        rng=random.Random(0),
        component_name="skill_instructions",
        min_each=2,
    )

    assert pair is None


def test_pick_patch_merge_pair_rejects_when_one_parent_lacks_unique_wins():
    parents = [[None], [0], [0]]
    candidates = [
        {"skill_instructions": "seed"},
        {"skill_instructions": "base"},
        {"skill_instructions": "patch"},
    ]
    scores_a = {f"t{i}": 1.0 for i in range(6)}
    scores_b = {f"t{i}": 0.0 for i in range(6)}

    pair = pick_patch_merge_pair(
        merge_candidates=[1, 2],
        program_candidates=candidates,
        parent_program_for_candidate=parents,
        prog_candidate_val_subscores=[{}, scores_a, scores_b],
        tracked_scores=[0.0, 0.4, 0.3],
        merges_performed=[],
        rng=random.Random(0),
        component_name="skill_instructions",
        min_each=2,
    )

    assert pair is None


def test_pick_patch_merge_pair_dedups_sorted_pair_across_ancestors():
    parents = [[None], [None], [0, 1], [0, 1]]
    candidates = [
        {"skill_instructions": "seed a"},
        {"skill_instructions": "seed b"},
        {"skill_instructions": "base"},
        {"skill_instructions": "patch"},
    ]
    scores_a = {f"t{i}": 1.0 if i % 2 else 0.0 for i in range(6)}
    scores_b = {f"t{i}": 0.0 if i % 2 else 1.0 for i in range(6)}

    pair = pick_patch_merge_pair(
        merge_candidates=[2, 3],
        program_candidates=candidates,
        parent_program_for_candidate=parents,
        prog_candidate_val_subscores=[{}, {}, scores_a, scores_b],
        tracked_scores=[0.1, 0.2, 0.4, 0.3],
        merges_performed=[(2, 3, 0)],
        rng=random.Random(0),
        component_name="skill_instructions",
        min_each=2,
    )

    assert pair is None


def test_pick_patch_merge_pair_weighted_sampling_is_not_pure_argmax():
    class FakeRng:
        def __init__(self):
            self.population = []
            self.weights = []

        def choices(self, population, weights, k):
            assert k == 1
            self.population = list(population)
            self.weights = list(weights)
            return [self.population[1]]

    parents = [[None], [0], [0], [0]]
    candidates = [
        {"skill_instructions": "seed"},
        {"skill_instructions": "parent one"},
        {"skill_instructions": "parent two"},
        {"skill_instructions": "parent three"},
    ]
    scores_1 = {"t0": 1.0, "t1": 1.0, "t2": 0.0, "t3": 0.0}
    scores_2 = {"t0": 0.0, "t1": 0.0, "t2": 1.0, "t3": 1.0}
    scores_3 = {"t0": 0.3, "t1": 0.3, "t2": 0.9, "t3": 0.9}
    rng = FakeRng()

    pair = pick_patch_merge_pair(
        merge_candidates=[1, 2, 3],
        program_candidates=candidates,
        parent_program_for_candidate=parents,
        prog_candidate_val_subscores=[{}, scores_1, scores_2, scores_3],
        tracked_scores=[0.0, 0.4, 0.4, 0.4],
        merges_performed=[],
        rng=rng,
        component_name="skill_instructions",
        min_each=2,
    )

    assert pair == rng.population[1]
    assert len(rng.population) > 1
    weights_by_pair = {
        (item.parent_a_id, item.parent_b_id): weight
        for item, weight in zip(rng.population, rng.weights, strict=True)
    }
    assert weights_by_pair[(1, 2)] > weights_by_pair[(1, 3)]
    assert (pair.parent_a_id, pair.parent_b_id) != (1, 2)


def test_pick_patch_merge_pair_ties_pair_weights_deterministically():
    class NoChoiceRng:
        def choices(self, *_args, **_kwargs):
            raise AssertionError("equal-weight selector should not call rng.choices")

    parents = [[None], [0], [0], [0]]
    candidates = [
        {"skill_instructions": "seed"},
        {"skill_instructions": "parent one"},
        {"skill_instructions": "parent two"},
        {"skill_instructions": "parent three"},
    ]
    scores_1 = {"t0": 1.0, "t1": 1.0, "t2": 0.0, "t3": 0.0}
    scores_2 = {"t0": 0.0, "t1": 0.0, "t2": 1.0, "t3": 1.0}
    scores_3 = dict(scores_2)

    pair = pick_patch_merge_pair(
        merge_candidates=[3, 2, 1],
        program_candidates=candidates,
        parent_program_for_candidate=parents,
        prog_candidate_val_subscores=[{}, scores_1, scores_2, scores_3],
        tracked_scores=[0.0, 0.4, 0.4, 0.4],
        merges_performed=[],
        rng=NoChoiceRng(),
        component_name="skill_instructions",
        min_each=2,
    )

    assert pair is not None
    assert (pair.parent_a_id, pair.parent_b_id) == (1, 2)


def test_pick_patch_merge_pair_chooses_higher_tracked_parent_as_base():
    parents = [[None], [0], [0]]
    candidates = [
        {"skill_instructions": "seed"},
        {"skill_instructions": "stronger"},
        {"skill_instructions": "patch source"},
    ]
    scores_a = {f"t{i}": 1.0 if i % 2 else 0.0 for i in range(6)}
    scores_b = {f"t{i}": 0.0 if i % 2 else 1.0 for i in range(6)}

    pair = pick_patch_merge_pair(
        merge_candidates=[1, 2],
        program_candidates=candidates,
        parent_program_for_candidate=parents,
        prog_candidate_val_subscores=[{}, scores_a, scores_b],
        tracked_scores=[0.0, 0.6, 0.5],
        merges_performed=[],
        rng=random.Random(0),
        component_name="skill_instructions",
        min_each=2,
    )

    assert pair is not None
    assert pair.base_parent_id == 1
    assert pair.patch_source_parent_id == 2


def test_pick_patch_merge_pair_ties_base_parent_deterministically():
    parents = [[None], [0], [0]]
    candidates = [
        {"skill_instructions": "seed"},
        {"skill_instructions": "lower id"},
        {"skill_instructions": "higher id"},
    ]
    scores_a = {f"t{i}": 1.0 if i % 2 else 0.0 for i in range(6)}
    scores_b = {f"t{i}": 0.0 if i % 2 else 1.0 for i in range(6)}

    pair = pick_patch_merge_pair(
        merge_candidates=[2, 1],
        program_candidates=candidates,
        parent_program_for_candidate=parents,
        prog_candidate_val_subscores=[{}, scores_a, scores_b],
        tracked_scores=[0.0, 0.5, 0.5],
        merges_performed=[],
        rng=random.Random(0),
        component_name="skill_instructions",
        min_each=2,
    )

    assert pair is not None
    assert pair.base_parent_id == 1
    assert pair.patch_source_parent_id == 2


def test_rlm_merge_proposer_uses_patch_selector_by_default(tmp_path: Path, monkeypatch):
    from gepa.core.data_loader import ensure_loader

    import rlm_gepa.proposer.merge as merge_module

    calls = {"patch": 0}

    def fake_find_dominators(*_args):
        return [1, 2]

    def fake_patch_selector(**_kwargs):
        calls["patch"] += 1
        return None

    def evaluator(_inputs, _candidate):
        return [], [], None

    monkeypatch.setattr(merge_module, "find_dominator_programs", fake_find_dominators)
    monkeypatch.setattr(merge_module, "pick_patch_merge_pair", fake_patch_selector)

    state = SimpleNamespace(
        i=0,
        full_program_trace=[{}],
        program_candidates=[
            {"skill_instructions": "ancestor"},
            {"skill_instructions": "parent a"},
            {"skill_instructions": "parent b"},
        ],
        parent_program_for_candidate=[[None], [0], [0]],
        prog_candidate_val_subscores=[{}, {"v1": 1.0}, {"v1": 0.0}],
        program_full_scores_val_set=[0.0, 0.5, 0.4],
        per_program_tracked_scores=[0.0, 0.5, 0.4],
        total_num_evals=0,
    )
    state.get_pareto_front_mapping = lambda: {}
    proposer = RlmMergeProposer(
        logger=_Logger(),
        valset=ensure_loader(["val"]),
        evaluator=evaluator,
        adapter=SimpleNamespace(),
        trainset=ensure_loader(["train"]),
        use_merge=True,
        max_merge_invocations=1,
        max_rlm_merge_attempts=5,
        min_each=1,
        merge_minibatch_size=1,
        rlm_merge_state_path=tmp_path / "state.json",
        rng=random.Random(0),
    )
    proposer.last_iter_found_new_program = True
    proposer.merges_due = 1

    assert proposer.propose(state) is None
    assert calls == {"patch": 1}


def _make_merge_helper(tmp_path: Path) -> RlmMergeProposer:
    proposer = RlmMergeProposer.__new__(RlmMergeProposer)
    proposer.rlm_merge_state_path = tmp_path / "rlm_merge_state.json"
    proposer.rlm_merge_attempts_used = 0
    proposer.merges_performed = ([], [])
    return proposer


def test_rlm_merge_status_helpers_validate_status_and_namespace(tmp_path: Path):
    proposer = _make_merge_helper(tmp_path)
    state = SimpleNamespace(full_program_trace=[{"i": 0}])

    proposer._record_merge_status(
        state,
        "accepted",
        attempt_idx=0,
        rlm_merge_candidate_pair=(1, 2),
    )

    assert state.full_program_trace[-1]["rlm_merge_status"] == "accepted"
    assert state.full_program_trace[-1]["rlm_merge_candidate_pair"] == (1, 2)
    assert VALID_STATUSES == frozenset(
        {
            "attempt_cap_exhausted",
            "pair_skipped",
            "preflight_failed",
            "subsample_rejected",
            "accepted",
            "error",
        }
    )

    with pytest.raises(ValueError, match="invalid merge status"):
        proposer._record_merge_status(state, "not_real", attempt_idx=None)
    with pytest.raises(ValueError, match="must start with 'rlm_merge_'"):
        proposer._record_merge_status(state, "accepted", attempt_idx=0, bad_field="value")
class _FirstKRng(random.Random):
    def sample(self, population, k):
        return list(population)[:k]


class _PatchEvidenceAdapter:
    proposer_trace_dir: Path
    run_id = "run_test"

    def __init__(self, tmp_path: Path, base_scores: list[float], source_scores: list[float]):
        self.proposer_trace_dir = tmp_path
        self.base_scores = base_scores
        self.source_scores = source_scores
        self.evaluate_calls = 0
        self.progress_labels: list[str] = []

    def progress_label(self, label):
        self.progress_labels.append(label)

        class NoopContext:
            def __enter__(self):
                return None

            def __exit__(self, *_args):
                return False

        return NoopContext()

    def evaluate(self, batch, _candidate, *, capture_traces, kind):
        scores = self.base_scores if self.evaluate_calls == 0 else self.source_scores
        self.evaluate_calls += 1
        selected_scores = scores[: len(batch)]
        trajectories = [
            {
                "task_id": str(item).replace(" ", "_"),
                "record": {
                    "Inputs": f"input for {item}",
                    "Generated Outputs": f"trace for {item}",
                    "Feedback": f"score {score}",
                },
            }
            for item, score in zip(batch, selected_scores, strict=False)
        ]
        return SimpleNamespace(scores=selected_scores, trajectories=trajectories)

    def make_reflective_dataset(self, _candidate, eval_batch, components):
        records = [trajectory["record"] for trajectory in eval_batch.trajectories]
        return {component: records for component in components}

    def _reserve_merge_proposer_call_idx(self):
        return 1

    def queue_valset_progress_label(self, _label):
        pass


def _patch_evidence_state():
    return SimpleNamespace(
        i=0,
        full_program_trace=[{}],
        program_candidates=[
            {"skill_instructions": "ancestor"},
            {"skill_instructions": "base"},
            {"skill_instructions": "source"},
        ],
        total_num_evals=0,
    )


def _make_patch_evidence_proposer(
    tmp_path: Path,
    *,
    base_scores: list[float],
    source_scores: list[float],
    merge_minibatch_size: int = 4,
    min_each: int = 1,
) -> RlmMergeProposer:
    from gepa.core.data_loader import ensure_loader

    def evaluator(_inputs, _candidate):
        return [], [], None

    proposer = RlmMergeProposer(
        logger=_Logger(),
        valset=ensure_loader(["val"]),
        evaluator=evaluator,
        adapter=_PatchEvidenceAdapter(tmp_path, base_scores, source_scores),
        trainset=ensure_loader([f"train {index}" for index in range(len(base_scores))]),
        use_merge=True,
        max_merge_invocations=1,
        max_rlm_merge_attempts=5,
        min_each=min_each,
        merge_minibatch_size=merge_minibatch_size,
        rlm_merge_state_path=tmp_path / "state.json",
        rng=_FirstKRng(),
    )
    return proposer


def test_patch_evidence_oversamples_two_minibatches_before_selecting_records(tmp_path: Path):
    proposer = _make_patch_evidence_proposer(
        tmp_path,
        base_scores=[1.0, 0.9, 0.8, 0.7, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        source_scores=[0.1, 0.2, 0.3, 0.4, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        merge_minibatch_size=4,
    )

    evidence = proposer._build_patch_disagreement_evidence(
        state=_patch_evidence_state(),
        iteration=1,
        attempt_idx=0,
        base_parent_id=1,
        patch_source_parent_id=2,
    )

    assert len(evidence.sampled_train_ids) == 8
    assert len(evidence.records) == 4


def test_patch_evidence_progress_labels_use_zero_indexed_iteration(tmp_path: Path):
    proposer = _make_patch_evidence_proposer(
        tmp_path,
        base_scores=[1.0, 0.0],
        source_scores=[0.0, 1.0],
        merge_minibatch_size=2,
    )

    proposer._build_patch_disagreement_evidence(
        state=_patch_evidence_state(),
        iteration=4,
        attempt_idx=0,
        base_parent_id=1,
        patch_source_parent_id=2,
    )

    assert proposer.adapter.progress_labels == [
        "Iteration 4 Patch Base Parent #1 Trace",
        "Iteration 4 Patch Source Parent #2 Trace",
    ]


def test_patch_evidence_prefers_larger_disagreements_and_caps_records(tmp_path: Path):
    proposer = _make_patch_evidence_proposer(
        tmp_path,
        base_scores=[1.0, 0.9, 0.6, 0.1, 0.3, 0.5],
        source_scores=[0.1, 0.2, 0.4, 0.9, 0.9, 0.6],
        merge_minibatch_size=4,
    )

    evidence = proposer._build_patch_disagreement_evidence(
        state=_patch_evidence_state(),
        iteration=1,
        attempt_idx=0,
        base_parent_id=1,
        patch_source_parent_id=2,
    )

    assert len(evidence.records) == 4
    assert [record["abs_delta"] for record in evidence.records] == sorted(
        [record["abs_delta"] for record in evidence.records],
        reverse=True,
    )
    assert {record["task_id"] for record in evidence.records} == {
        "train_0",
        "train_1",
        "train_3",
        "train_4",
    }


def test_patch_evidence_balances_base_and_patch_source_win_directions(tmp_path: Path):
    proposer = _make_patch_evidence_proposer(
        tmp_path,
        base_scores=[1.0, 0.9, 0.8, 0.7, 0.1, 0.2],
        source_scores=[0.1, 0.1, 0.1, 0.1, 0.3, 0.3],
        merge_minibatch_size=4,
    )

    evidence = proposer._build_patch_disagreement_evidence(
        state=_patch_evidence_state(),
        iteration=1,
        attempt_idx=0,
        base_parent_id=1,
        patch_source_parent_id=2,
    )

    winners = [record["winner"] for record in evidence.records]
    assert winners.count("base") == 2
    assert winners.count("patch_source") == 2


def test_patch_evidence_tops_up_with_both_success_records(tmp_path: Path):
    proposer = _make_patch_evidence_proposer(
        tmp_path,
        base_scores=[1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        source_scores=[0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        merge_minibatch_size=6,
    )

    evidence = proposer._build_patch_disagreement_evidence(
        state=_patch_evidence_state(),
        iteration=1,
        attempt_idx=0,
        base_parent_id=1,
        patch_source_parent_id=2,
    )

    winners = [record["winner"] for record in evidence.records]
    assert winners.count("base") == 1
    assert winners.count("patch_source") == 1
    assert winners.count("both_success") == 4
    assert len(evidence.records) == 6


def test_patch_merge_preflight_fails_without_balanced_disagreement_evidence(
    tmp_path: Path,
    monkeypatch,
):
    from gepa.core.data_loader import ensure_loader

    import rlm_gepa.proposer.merge as merge_module

    class FakeAdapter(_PatchEvidenceAdapter):
        def __init__(self):
            super().__init__(tmp_path, base_scores=[1.0, 1.0], source_scores=[0.0, 0.0])
            self.patch_calls = 0

        def _rlm_propose_patch_merge_texts(self, **_kwargs):
            self.patch_calls += 1
            return "should not be called", {}

    def evaluator(_inputs, _candidate):
        return [], [], None

    adapter = FakeAdapter()
    monkeypatch.setattr(merge_module, "find_dominator_programs", lambda *_args: [1, 2])
    monkeypatch.setattr(
        merge_module,
        "pick_patch_merge_pair",
        lambda **_kwargs: PatchMergePair(
            parent_a_id=1,
            parent_b_id=2,
            base_parent_id=1,
            patch_source_parent_id=2,
            ancestor=0,
            common_ancestors=(0,),
            oracle_score=1.0,
            oracle_gain=0.5,
            base_wins=("v1",),
            patch_source_wins=("v2",),
            weight=0.5,
        ),
    )
    proposer = RlmMergeProposer(
        logger=_Logger(),
        valset=ensure_loader(["v1", "v2"]),
        evaluator=evaluator,
        adapter=adapter,
        trainset=ensure_loader(["train_1", "train_2"]),
        use_merge=True,
        max_merge_invocations=1,
        max_rlm_merge_attempts=5,
        min_each=1,
        merge_minibatch_size=2,
        rlm_merge_state_path=tmp_path / "state.json",
        rng=_FirstKRng(),
    )
    proposer.last_iter_found_new_program = True
    proposer.merges_due = 1
    state = SimpleNamespace(
        i=0,
        full_program_trace=[{}],
        program_candidates=[
            {"skill_instructions": "ancestor"},
            {"skill_instructions": "base"},
            {"skill_instructions": "source"},
        ],
        parent_program_for_candidate=[[None], [0], [0]],
        prog_candidate_val_subscores=[{}, {"v1": 1.0, "v2": 0.0}, {"v1": 0.0, "v2": 1.0}],
        program_full_scores_val_set=[0.0, 0.5, 0.5],
        per_program_tracked_scores=[0.0, 0.5, 0.5],
        total_num_evals=0,
    )
    state.get_pareto_front_mapping = lambda: {}

    proposal = proposer.propose(state)

    assert proposal is None
    assert adapter.patch_calls == 0
    assert state.full_program_trace[-1]["rlm_merge_status"] == "preflight_failed"
    assert state.full_program_trace[-1]["rlm_merge_reject_reason"].startswith(
        "insufficient patch disagreement evidence"
    )
    assert state.full_program_trace[-1]["rlm_merge_preflight_base_wins"] == 2
    assert state.full_program_trace[-1]["rlm_merge_preflight_patch_source_wins"] == 0


def test_patch_merge_preflight_fails_when_prompt_cap_cannot_carry_min_each(
    tmp_path: Path,
    monkeypatch,
):
    from gepa.core.data_loader import ensure_loader

    import rlm_gepa.proposer.merge as merge_module

    class FakeAdapter(_PatchEvidenceAdapter):
        def __init__(self):
            super().__init__(
                tmp_path,
                base_scores=[1.0, 1.0, 0.0, 0.0],
                source_scores=[0.0, 0.0, 1.0, 1.0],
            )
            self.patch_calls = 0

        def _rlm_propose_patch_merge_texts(self, **_kwargs):
            self.patch_calls += 1
            return "should not be called", {}

    def evaluator(_inputs, _candidate):
        return [], [], None

    adapter = FakeAdapter()
    monkeypatch.setattr(merge_module, "find_dominator_programs", lambda *_args: [1, 2])
    monkeypatch.setattr(
        merge_module,
        "pick_patch_merge_pair",
        lambda **_kwargs: PatchMergePair(
            parent_a_id=1,
            parent_b_id=2,
            base_parent_id=1,
            patch_source_parent_id=2,
            ancestor=0,
            common_ancestors=(0,),
            oracle_score=1.0,
            oracle_gain=0.5,
            base_wins=("v1", "v2"),
            patch_source_wins=("v3", "v4"),
            weight=0.5,
        ),
    )
    proposer = RlmMergeProposer(
        logger=_Logger(),
        valset=ensure_loader(["v1", "v2", "v3", "v4"]),
        evaluator=evaluator,
        adapter=adapter,
        trainset=ensure_loader(["train_1", "train_2", "train_3", "train_4"]),
        use_merge=True,
        max_merge_invocations=1,
        max_rlm_merge_attempts=5,
        min_each=2,
        merge_minibatch_size=2,
        rlm_merge_state_path=tmp_path / "state.json",
        rng=_FirstKRng(),
    )
    proposer.last_iter_found_new_program = True
    proposer.merges_due = 1
    state = SimpleNamespace(
        i=0,
        full_program_trace=[{}],
        program_candidates=[
            {"skill_instructions": "ancestor"},
            {"skill_instructions": "base"},
            {"skill_instructions": "source"},
        ],
        parent_program_for_candidate=[[None], [0], [0]],
        prog_candidate_val_subscores=[
            {},
            {"v1": 1.0, "v2": 1.0, "v3": 0.0, "v4": 0.0},
            {"v1": 0.0, "v2": 0.0, "v3": 1.0, "v4": 1.0},
        ],
        program_full_scores_val_set=[0.0, 0.5, 0.5],
        per_program_tracked_scores=[0.0, 0.5, 0.5],
        total_num_evals=0,
    )
    state.get_pareto_front_mapping = lambda: {}

    proposal = proposer.propose(state)

    assert proposal is None
    assert adapter.patch_calls == 0
    assert state.full_program_trace[-1]["rlm_merge_status"] == "preflight_failed"
    assert state.full_program_trace[-1]["rlm_merge_preflight_base_wins"] == 2
    assert state.full_program_trace[-1]["rlm_merge_preflight_patch_source_wins"] == 2
    assert state.full_program_trace[-1]["rlm_merge_selected_base_wins"] == 1
    assert state.full_program_trace[-1]["rlm_merge_selected_patch_source_wins"] == 1


def test_patch_disagreement_trace_jsonl_contains_patch_schema(tmp_path: Path):
    proposer = _make_patch_evidence_proposer(
        tmp_path,
        base_scores=[1.0, 0.0],
        source_scores=[0.0, 1.0],
        merge_minibatch_size=2,
    )

    evidence = proposer._build_patch_disagreement_evidence(
        state=_patch_evidence_state(),
        iteration=1,
        attempt_idx=0,
        base_parent_id=1,
        patch_source_parent_id=2,
    )
    records = [json.loads(line) for line in Path(evidence.paired_trace_path).read_text().splitlines()]

    assert len(records) == 2
    assert records[0]["schema_version"] == 1
    assert records[0]["winner"] in {"base", "patch_source"}
    assert records[0]["evidence_role"] in {"base_win", "patch_source_win"}
    assert records[0]["abs_delta"] == pytest.approx(1.0)
    assert records[0]["base_parent_id"] == 1
    assert records[0]["patch_source_parent_id"] == 2
    assert records[0]["base_parent"]["generated_outputs"]
    assert records[0]["patch_source_parent"]["feedback"]


def test_patch_mode_proposer_wires_base_source_fields_without_ancestor(
    tmp_path: Path,
    monkeypatch,
):
    from gepa.core.data_loader import ensure_loader

    import rlm_gepa.proposer.merge as merge_module

    class FakeAdapter(_PatchEvidenceAdapter):
        def __init__(self):
            super().__init__(tmp_path, base_scores=[1.0, 0.0], source_scores=[0.0, 1.0])
            self.patch_kwargs = None

        def _reserve_merge_proposer_call_idx(self):
            return 9

        def _rlm_propose_patch_merge_texts(self, **kwargs):
            self.patch_kwargs = kwargs
            return "patched instructions", {"patch_summary": "imported one clause"}

        def queue_valset_progress_label(self, _label):
            pass

    def evaluator(_inputs, _candidate):
        return [], [], None

    adapter = FakeAdapter()
    monkeypatch.setattr(merge_module, "find_dominator_programs", lambda *_args: [1, 2])
    monkeypatch.setattr(
        merge_module,
        "pick_patch_merge_pair",
        lambda **_kwargs: PatchMergePair(
            parent_a_id=1,
            parent_b_id=2,
            base_parent_id=1,
            patch_source_parent_id=2,
            ancestor=0,
            common_ancestors=(0,),
            oracle_score=1.0,
            oracle_gain=0.5,
            base_wins=("v1",),
            patch_source_wins=("v2",),
            weight=0.5,
        ),
    )
    proposer = RlmMergeProposer(
        logger=_Logger(),
        valset=ensure_loader(["v1", "v2"]),
        evaluator=evaluator,
        adapter=adapter,
        trainset=ensure_loader(["train_1", "train_2"]),
        use_merge=True,
        max_merge_invocations=1,
        max_rlm_merge_attempts=5,
        min_each=1,
        merge_minibatch_size=2,
        rlm_merge_state_path=tmp_path / "state.json",
        rng=_FirstKRng(),
    )
    proposer.last_iter_found_new_program = True
    proposer.merges_due = 1
    captured_child = {}
    state = SimpleNamespace(
        i=0,
        full_program_trace=[{}],
        program_candidates=[
            {"skill_instructions": "ancestor", "other": "ancestor kept"},
            {"skill_instructions": "base", "other": "base kept"},
            {"skill_instructions": "source", "other": "source ignored"},
        ],
        parent_program_for_candidate=[[None], [0], [0]],
        prog_candidate_val_subscores=[{}, {"v1": 1.0, "v2": 0.0}, {"v1": 0.0, "v2": 1.0}],
        program_full_scores_val_set=[0.0, 0.5, 0.5],
        per_program_tracked_scores=[0.0, 0.5, 0.5],
        total_num_evals=0,
    )
    state.get_pareto_front_mapping = lambda: {}

    def cached_evaluate(candidate, ids, fetch, _evaluator):
        captured_child.update(candidate)
        assert list(ids) == [0, 1]
        assert fetch(list(ids)) == ["train_1", "train_2"]
        return [0.0, 0.0], 2

    state.cached_evaluate = cached_evaluate

    proposal = proposer.propose(state)

    assert proposal is None
    assert adapter.patch_kwargs is not None
    assert adapter.patch_kwargs["base_parent_id"] == 1
    assert adapter.patch_kwargs["base_parent_instructions"] == "base"
    assert adapter.patch_kwargs["patch_source_parent_id"] == 2
    assert adapter.patch_kwargs["patch_source_parent_instructions"] == "source"
    assert "paired_disagreement_traces_file" in adapter.patch_kwargs
    assert "common_ancestor_instructions" not in adapter.patch_kwargs
    assert captured_child == {"skill_instructions": "patched instructions", "other": "base kept"}


def test_reflection_lm_text_normalization_accepts_common_payloads():
    response = {
        "id": "resp_123",
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "<proposal>new instructions</proposal>"}
                ],
            }
        ],
    }

    assert _coerce_reflection_lm_text([{"text": "new skill instructions"}]) == (
        "new skill instructions"
    )
    assert _coerce_reflection_lm_text(response) == "<proposal>new instructions</proposal>"
    assert _coerce_reflection_lm_text({"choices": [{"message": {"content": "chat"}}]}) == "chat"
    with pytest.raises(TypeError, match="non-text response"):
        _coerce_reflection_lm_text({"usage": {"input_tokens": 10}})


def test_progress_candidate_selector_uses_zero_indexed_iteration():
    class Selector:
        def select_candidate_idx(self, _state):
            return 4

    class Adapter:
        def __init__(self):
            self.context = None

        def set_reflective_progress_context(self, **kwargs):
            self.context = kwargs

    adapter = Adapter()
    selector = _ProgressCandidateSelector(Selector(), adapter)

    assert selector.select_candidate_idx(SimpleNamespace(i=5, program_candidates=[{}, {}, {}])) == 4
    assert adapter.context == {"iteration": 5, "parent_idx": 4, "child_idx": 3}


def _make_merge_proposer(tmp_path: Path, state_payload: dict | None = None) -> RlmMergeProposer:
    from gepa.core.data_loader import ensure_loader

    state_path = tmp_path / "rlm_merge_state.json"
    if state_payload is not None:
        state_path.write_text(json.dumps(state_payload))

    def evaluator(_inputs, _candidate):
        return [], [], None

    return RlmMergeProposer(
        logger=_Logger(),
        valset=ensure_loader(["val"]),
        evaluator=evaluator,
        adapter=SimpleNamespace(),
        trainset=ensure_loader(["train"]),
        use_merge=True,
        max_merge_invocations=1,
        max_rlm_merge_attempts=5,
        min_each=1,
        merge_minibatch_size=1,
        rlm_merge_state_path=state_path,
        rng=random.Random(0),
    )


def test_rlm_merge_proposer_loads_sidecar_state(tmp_path: Path):
    proposer = _make_merge_proposer(
        tmp_path,
        {
            "schema_version": 1,
            "rlm_merge_attempts_used": 2,
            "merges_performed": [[1, 2, 0], [3, 4, 1]],
        },
    )

    assert proposer.rlm_merge_attempts_used == 2
    assert proposer.merges_performed[0] == [(1, 2, 0), (3, 4, 1)]


def test_rlm_merge_proposer_flushes_sidecar_on_propose_exit(tmp_path: Path):
    proposer = _make_merge_proposer(tmp_path)
    proposer.use_merge = False
    proposer.rlm_merge_attempts_used = 1
    proposer.merges_performed[0].append((1, 2, 0))
    state = SimpleNamespace(i=0, full_program_trace=[{}])

    assert proposer.propose(state) is None

    data = json.loads((tmp_path / "rlm_merge_state.json").read_text())
    assert data["schema_version"] == 1
    assert data["rlm_merge_attempts_used"] == 1
    assert data["merges_performed"] == [[1, 2, 0]]
    assert data["flushed_at"]
    assert state.full_program_trace[-1]["invoked_merge"] is True


def test_plot_output_paths_default_to_run_dir(tmp_path: Path):
    score_path, lineage_path = resolve_plot_output_paths(tmp_path)

    assert score_path == tmp_path / "plots" / "score_vs_rollouts.png"
    assert lineage_path == tmp_path / "plots" / "candidate_lineage.png"


def test_plot_output_paths_accept_directory_or_prefix(tmp_path: Path):
    score_path, lineage_path = resolve_plot_output_paths(tmp_path, tmp_path / "plots")

    assert score_path == tmp_path / "plots" / "score_vs_rollouts.png"
    assert lineage_path == tmp_path / "plots" / "candidate_lineage.png"

    score_path, lineage_path = resolve_plot_output_paths(tmp_path, tmp_path / "summary.png")

    assert score_path == tmp_path / "summary_score_vs_rollouts.png"
    assert lineage_path == tmp_path / "summary_candidate_lineage.png"


def test_cost_aggregation_raw_and_logical(tmp_path: Path):
    path = tmp_path / "cost_log.jsonl"
    row = CostRow(
        event_id="event_1",
        operation_id="op_1",
        attempt_id="attempt_1",
        event="minibatch",
        role="executor",
        model="dummy",
        calls=2,
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.1,
    )
    append_cost_rows(path, [row, row])

    raw = aggregate_costs_from_log(path)
    logical = aggregate_costs_from_log(path, logical=True)

    assert raw[0].calls == 4
    assert raw[0].cost_usd == pytest.approx(0.2)
    assert logical[0].calls == 2
    assert logical[0].cost_usd == pytest.approx(0.1)


def test_logical_cost_keeps_resumed_operations_with_reused_local_counters(tmp_path: Path):
    path = tmp_path / "cost_log.jsonl"
    original = CostRow(
        event_id="run_a_eval_minibatch_attempt_0000",
        operation_id="eval_minibatch_0000",
        attempt_id="attempt_0000",
        event="minibatch",
        role="executor",
        model="dummy",
        calls=2,
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.1,
    )
    resumed = CostRow(
        event_id="run_a_resume_b_eval_minibatch_attempt_0000",
        operation_id="eval_minibatch_0000",
        attempt_id="attempt_0000",
        event="minibatch",
        role="executor",
        model="dummy",
        calls=3,
        input_tokens=20,
        output_tokens=8,
        cost_usd=0.2,
    )
    append_cost_rows(path, [original, original, resumed])

    logical = aggregate_costs_from_log(path, logical=True)

    assert logical[0].calls == 5
    assert logical[0].cost_usd == pytest.approx(0.3)


def test_logical_cost_does_not_collapse_legacy_rows_without_operation_ids(tmp_path: Path):
    path = tmp_path / "cost_log.jsonl"
    append_cost_rows(
        path,
        [
            {
                "event": "valset",
                "role": "executor",
                "model": "dummy",
                "calls": 2,
                "input_tokens": 10,
                "output_tokens": 5,
                "cost_usd": 0.1,
                "evaluate_idx": 0,
            },
            {
                "event": "valset",
                "role": "executor",
                "model": "dummy",
                "calls": 3,
                "input_tokens": 20,
                "output_tokens": 8,
                "cost_usd": 0.2,
                "evaluate_idx": 1,
            },
        ],
    )

    logical = aggregate_costs_from_log(path, logical=True)

    assert logical[0].calls == 5
    assert logical[0].cost_usd == pytest.approx(0.3)


def test_merge_iteration_rows_use_best_actual_parent_instead_of_oracle(tmp_path: Path):
    state = {
        "full_program_trace": [
            {
                "i": 12,
                "rlm_merge_candidate_pair": (2, 5),
                "rlm_merge_status": "accepted",
                "new_program_idx": 6,
                "id1_subsample_scores": [1.0, 0.8, 0.0, 0.0],
                "id2_subsample_scores": [0.0, 0.0, 1.0, 0.0],
                "new_program_subsample_scores": [0.0, 0.8, 0.0, 0.0],
            }
        ],
    }
    with (tmp_path / "gepa_state.bin").open("wb") as f:
        pickle.dump(state, f)

    rows = iteration_rows(tmp_path)

    assert rows[0]["iter"] == "12 [2, 5]"
    assert rows[0]["soft: par → child"] == "0.450 → 0.200 -0.250"
    assert rows[0]["hard: par → child"] == "0.250 → 0.000 -0.250; 1 → 0"
    assert rows[0]["flips"] == "+0/-1 -1"
    assert rows[0]["p"] == "1.00"

    merge_stats = merge_rows(tmp_path)
    assert merge_stats[0]["soft: best(par) -> merge"] == "0.450 → 0.200 -0.250"
    assert merge_stats[0]["hard: best(par) -> merge"] == "0.250 → 0.000 -0.250; 1 → 0"
    assert merge_stats[0]["flips"] == "+0/-1 -1"
    assert merge_stats[0]["p"] == "1.00"
    assert "score Δ" not in merge_stats[0]


def test_stats_hard_flip_p_values_use_two_sided_exact_for_ties(tmp_path: Path):
    parent_scores = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    child_scores = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    state = {
        "full_program_trace": [
            {
                "i": 0,
                "selected_program_candidate": 0,
                "new_program_idx": 1,
                "subsample_scores": parent_scores,
                "new_subsample_scores": child_scores,
            },
            {
                "i": 1,
                "rlm_merge_candidate_pair": (0, 1),
                "id1_subsample_scores": parent_scores,
                "id2_subsample_scores": [0.0] * len(parent_scores),
                "new_program_subsample_scores": child_scores,
            },
        ],
    }
    with (tmp_path / "gepa_state.bin").open("wb") as f:
        pickle.dump(state, f)

    rows = iteration_rows(tmp_path)
    merge_stats = merge_rows(tmp_path)

    assert rows[0]["flips"] == "+4/-4 +0"
    assert rows[0]["p"] == "1.00"
    assert merge_stats[0]["flips"] == "+4/-4 +0"
    assert merge_stats[0]["p"] == "1.00"


def test_merge_rows_report_accepted_child_full_val_against_both_parents(tmp_path: Path):
    state = {
        "prog_candidate_val_subscores": [
            {"a": 0.0, "b": 1.0, "c": 0.0},
            {"a": 1.0, "b": 1.0, "c": 0.0},
            {"a": 0.0, "b": 1.0, "c": 1.0},
            {"a": 1.0, "b": 1.0, "c": 1.0},
        ],
        "full_program_trace": [
            {
                "i": 8,
                "rlm_merge_candidate_pair": (1, 2),
                "rlm_merge_ancestor": 0,
                "rlm_merge_status": "accepted",
                "rlm_merge_base_parent": 1,
                "rlm_merge_patch_source_parent": 2,
                "new_program_idx": 3,
                "id1_subsample_scores": [1.0, 0.0],
                "id2_subsample_scores": [0.0, 1.0],
                "new_program_subsample_scores": [1.0, 1.0],
            }
        ],
    }
    with (tmp_path / "gepa_state.bin").open("wb") as f:
        pickle.dump(state, f)

    rows = merge_rows(tmp_path)

    assert [key for key in rows[0] if not key.startswith("_")] == [
        "iter",
        "pair@anc",
        "soft: best(par) -> merge",
        "hard: best(par) -> merge",
        "flips",
        "p",
        "outcome",
    ]
    assert "pre" not in rows[0]
    assert "n" not in rows[0]
    assert "val Δ" not in rows[0]
    assert "score Δ" not in rows[0]
    assert "status" not in rows[0]
    assert rows[0]["soft: best(par) -> merge"] == "0.500 → 1.000 +0.500"
    assert rows[0]["hard: best(par) -> merge"] == "0.500 → 1.000 +0.500; 1 → 2"
    assert rows[0]["flips"] == "+1/-0 +1"
    assert rows[0]["p"] == "1.00"
    assert rows[0]["outcome"] == "accepted"
    assert rows[0]["_muted_prefix"] == {
        "soft: best(par) -> merge": "0.500 → 1.000",
        "hard: best(par) -> merge": "0.500 → 1.000",
        "flips": "+1/-0",
    }
    assert rows[0]["_detail"] == (
        "→ cand 3; full val "
        "vs 1: 0.667→1.000 +0.333, hard 2→3/3, flips +1/-0; "
        "vs 2: 0.667→1.000 +0.333, hard 2→3/3, flips +1/-0"
    )


def test_merge_rows_do_not_use_normal_mutation_child_for_rejected_merge_val(tmp_path: Path):
    state = {
        "prog_candidate_val_subscores": [
            {"a": 1.0, "b": 0.0, "c": 0.0},
            {"a": 1.0, "b": 1.0, "c": 0.0},
            {"a": 0.0, "b": 1.0, "c": 0.0},
            {"a": 1.0, "b": 1.0, "c": 1.0},
        ],
        "full_program_trace": [
            {
                "i": 9,
                "rlm_merge_candidate_pair": (1, 2),
                "rlm_merge_ancestor": 0,
                "rlm_merge_status": "subsample_rejected",
                "rlm_merge_new_program_idx": None,
                "new_program_idx": 3,
                "rlm_merge_reject_reason": "not better than best parent",
                "id1_subsample_scores": [1.0, 0.0],
                "id2_subsample_scores": [0.0, 1.0],
                "new_program_subsample_scores": [0.0, 1.0],
            }
        ],
    }
    with (tmp_path / "gepa_state.bin").open("wb") as f:
        pickle.dump(state, f)

    rows = merge_rows(tmp_path)

    assert "val Δ" not in rows[0]
    assert "score Δ" not in rows[0]
    assert "status" not in rows[0]
    assert rows[0]["soft: best(par) -> merge"] == "0.500 → 0.500 +0.000"
    assert rows[0]["hard: best(par) -> merge"] == "0.500 → 0.500 +0.000; 1 → 1"
    assert rows[0]["flips"] == "+1/-1 +0"
    assert rows[0]["p"] == "1.00"
    assert rows[0]["outcome"] == "rejected"
    assert rows[0]["_muted_prefix"] == {
        "soft: best(par) -> merge": "0.500 → 0.500",
        "hard: best(par) -> merge": "0.500 → 0.500",
        "flips": "+1/-1",
    }
    assert rows[0]["_detail"] == "not better than best parent"


def test_merge_rows_use_explicit_merge_child_for_accepted_full_val(tmp_path: Path):
    state = {
        "prog_candidate_val_subscores": [
            {"a": 0.0, "b": 1.0, "c": 0.0},
            {"a": 1.0, "b": 1.0, "c": 0.0},
            {"a": 0.0, "b": 1.0, "c": 1.0},
            {"a": 0.0, "b": 0.0, "c": 0.0},
            {"a": 1.0, "b": 1.0, "c": 1.0},
        ],
        "full_program_trace": [
            {
                "i": 10,
                "rlm_merge_candidate_pair": (1, 2),
                "rlm_merge_ancestor": 0,
                "rlm_merge_status": "accepted",
                "rlm_merge_new_program_idx": 4,
                "new_program_idx": 3,
                "id1_subsample_scores": [1.0, 0.0],
                "id2_subsample_scores": [0.0, 1.0],
                "new_program_subsample_scores": [1.0, 1.0],
            }
        ],
    }
    with (tmp_path / "gepa_state.bin").open("wb") as f:
        pickle.dump(state, f)

    rows = merge_rows(tmp_path)

    assert "val Δ" not in rows[0]
    assert "score Δ" not in rows[0]
    assert "status" not in rows[0]
    assert rows[0]["soft: best(par) -> merge"] == "0.500 → 1.000 +0.500"
    assert rows[0]["hard: best(par) -> merge"] == "0.500 → 1.000 +0.500; 1 → 2"
    assert rows[0]["flips"] == "+1/-0 +1"
    assert rows[0]["p"] == "1.00"
    assert rows[0]["outcome"] == "accepted"
    assert rows[0]["_muted_prefix"] == {
        "soft: best(par) -> merge": "0.500 → 1.000",
        "hard: best(par) -> merge": "0.500 → 1.000",
        "flips": "+1/-0",
    }
    assert rows[0]["_detail"].startswith("→ cand 4; full val vs 1:")


def test_merge_rows_compact_outcomes_for_terminal_width(tmp_path: Path):
    state = {
        "full_program_trace": [
            {"i": 1, "rlm_merge_status": "pair_skipped", "rlm_merge_candidate_pair": (0, 1)},
            {
                "i": 2,
                "rlm_merge_status": "no_merge_candidate",
                "rlm_merge_candidate_pair": (0, 1),
            },
            {
                "i": 3,
                "rlm_merge_status": "subsample_rejected",
                "rlm_merge_candidate_pair": (0, 1),
            },
            {
                "i": 4,
                "rlm_merge_status": "preflight_failed",
                "rlm_merge_candidate_pair": (0, 1),
            },
            {"i": 5, "rlm_merge_status": "accepted", "rlm_merge_candidate_pair": (0, 1)},
            {"i": 6, "rlm_merge_status": "custom_status", "rlm_merge_candidate_pair": (0, 1)},
        ]
    }
    with (tmp_path / "gepa_state.bin").open("wb") as f:
        pickle.dump(state, f)

    rows = merge_rows(tmp_path)

    assert [row["outcome"] for row in rows] == [
        "skipped",
        "skipped",
        "rejected",
        "rejected",
        "accepted",
        "custom_status",
    ]


def test_reporting_tables_from_artifacts(tmp_path: Path):
    state = {
        "i": 1,
        "total_num_evals": 4,
        "program_candidates": [{"skill_instructions": "seed"}, {"skill_instructions": "new"}],
        "program_full_scores_val_set": [0.5, 0.75],
        "prog_candidate_val_subscores": [
            {"a": 0.0, "b": 1.0},
            {"a": 1.0, "b": 1.0},
        ],
        "parent_program_for_candidate": [[None], [0]],
        "full_program_trace": [
            {
                "i": 0,
                "selected_program_candidate": 0,
                "subsample_scores": {"a": 0.0, "b": 1.0},
                "new_subsample_scores": {"a": 1.0, "b": 1.0},
                "new_program_idx": 1,
            },
            {
                "i": 1,
                "rlm_merge_candidate_pair": (0, 1),
                "rlm_merge_ancestor": 0,
                "rlm_merge_attempt_idx": 0,
                "rlm_merge_status": "subsample_rejected",
                "rlm_merge_preflight_a_wins": 3,
                "rlm_merge_preflight_b_wins": 2,
                "rlm_merge_reject_reason": "not better than best parent",
                "id1_subsample_scores": [0.0, 1.0],
                "id2_subsample_scores": [1.0, 0.0],
                "new_program_subsample_scores": [1.0, 0.0],
            },
        ],
    }
    with (tmp_path / "gepa_state.bin").open("wb") as f:
        pickle.dump(state, f)
    (tmp_path / "run_metadata.json").write_text(
        json.dumps(
            {
                "resolved_config": {
                    "executor_reasoning_effort": "low",
                    "executor_sub_lm_reasoning_effort": "none",
                    "proposer_reasoning_effort": "medium",
                    "proposer_sub_lm_reasoning_effort": "medium",
                }
            }
        )
    )
    append_cost_rows(
        tmp_path / "cost_log.jsonl",
        [
            CostRow(
                event_id="e",
                operation_id="op",
                attempt_id="a",
                event="valset",
                role="executor",
                model="dummy",
                calls=1,
                input_tokens=10,
                output_tokens=2,
                cost_usd=0.01,
            ),
            CostRow(
                event_id="e",
                operation_id="op",
                attempt_id="a",
                event="valset",
                role="executor",
                model="dummy",
                calls=1,
                input_tokens=10,
                output_tokens=2,
                cost_usd=0.01,
            ),
        ],
    )

    rows = iteration_rows(tmp_path)
    assert rows[0]["outcome"] == "→ cand 1"
    assert rows[0]["soft: par → child"] == "0.500 → 1.000 +0.500"
    assert rows[0]["hard: par → child"] == "0.500 → 1.000 +0.500; 1 → 2"
    assert rows[0]["flips"] == "+1/-0 +1"
    assert rows[0]["p"] == "1.00"
    assert rows[0]["iter"] == "0 [0]"
    assert rows[0]["_highlight"] is True
    iteration_terminal = render_table(rows)
    iteration_plain_lines = [
        stats_report.re.sub(r"\033\[[0-9;]*m", "", line) for line in iteration_terminal.splitlines()
    ]
    iteration_header_line = next(line for line in iteration_plain_lines if "hard: par" in line)
    iteration_hard_header = iteration_header_line.strip("│").split("│")[2]
    assert iteration_hard_header.rstrip().endswith("/2")
    assert "1 → 2 /2" not in iteration_terminal
    assert rows[1]["iter"] == "1 [0, 1]"
    merges = merge_rows(tmp_path)
    assert {
        key: value
        for key, value in merges[0].items()
        if key not in {"_merge_hard_denominator", "_terminal_header_aliases", "_terminal_header_suffixes"}
    } == {
        "iter": "1",
        "pair@anc": "0+1@0",
        "soft: best(par) -> merge": "0.500 → 0.500 +0.000",
        "hard: best(par) -> merge": "0.500 → 0.500 +0.000; 1 → 1",
        "flips": "+1/-1 +0",
        "p": "1.00",
        "outcome": "rejected",
        "_detail": "not better than best parent",
        "_muted_prefix": {
            "soft: best(par) -> merge": "0.500 → 0.500",
            "hard: best(par) -> merge": "0.500 → 0.500",
            "flips": "+1/-1",
        },
    }
    assert rows[0]["_terminal_header_suffixes"]["hard: par → child"] == "/2"
    assert merges[0]["_terminal_header_suffixes"]["hard: best(par) -> merge"] == "/2"
    candidates = candidate_rows(tmp_path)
    assert candidates[0]["cand [par]"] == "0 [seed]"
    assert candidates[0]["hard"] == "0.500 (1/2)"
    assert candidates[0]["Δ-seed"] == "-"
    assert candidates[1]["cand [par]"] == "1 [0]"
    assert candidates[1]["hard"] == "1.000 (2/2)"
    assert candidates[1]["Δ-seed"] == "+0.500"
    assert candidates[1]["_highlight"] is True
    costs = cost_rows(tmp_path)
    assert costs[0]["scope"] == "executor"
    assert costs[0]["model"] == ""
    assert costs[0]["calls"] == ""
    assert costs[0]["_category"] is True
    assert costs[1]["scope"] == "  - main"
    assert costs[1]["model"] == "dummy-low"
    assert costs[1]["total_cost"] == "$0.02"
    assert costs[1]["repeat_cost"] == "$0.01"
    assert costs[1]["effective_cost"] == "$0.01"
    assert costs[2]["scope"] == "  - sub"
    assert costs[2]["model"] == "-"
    assert costs[2]["calls"] == "-"
    assert costs[3]["_spacer"] is True
    assert costs[-1]["scope"] == "TOTAL"
    assert costs[-1]["model"] == ""
    assert costs[-1]["calls"] == "2"
    assert costs[-1]["total_cost"] == "$0.02"
    assert costs[-1]["repeat_cost"] == "$0.01"
    assert costs[-1]["effective_cost"] == "$0.01"
    rendered = render_stats(tmp_path, output_format="markdown")
    assert "iterations:" in rendered
    assert "merges:" in rendered
    assert "| iter" in rendered
    assert "| soft: par → child" in rendered
    assert "| hard: par → child" in rendered
    assert "| soft: best(par) -> merge" in rendered
    assert "| hard: best(par) -> merge" in rendered
    assert "| pair@anc" in rendered
    assert "rejected" in rendered
    assert "merge details:" in rendered
    assert "iter 1 0+1@0: not better than best parent" in rendered
    assert "| cand [par]" in rendered
    assert "| Δ-seed" in rendered
    assert "| ----" in rendered
    assert "**1 [0]**" in rendered
    terminal = render_stats(tmp_path)
    assert "┌" in terminal
    assert "\033[3m" in terminal
    assert "\033[38;5;248m" in terminal
    assert "\033[38;5;178m0.500 → 1.000\033[0m\033[1;38;5;220m +0.500" in terminal
    assert "\033[38;5;178m0.500 → 1.000\033[0m\033[1;38;5;220m +0.500; 1 → 2" in terminal
    assert "\033[38;5;178m+1/-0\033[0m\033[1;38;5;220m +1" in terminal
    assert "\033[38;5;248m+1/-1\033[0m +0" in terminal
    assert "\033[38;5;248m0.500 → 0.500\033[0m +0.000" in terminal
    assert "\033[38;5;248m0.500 → 0.500\033[0m +0.000; 1 → 1" in terminal
    assert "\033[1;38;5;220m" in terminal
    assert "**1**" not in terminal
    assert "costs:" in terminal
    assert "total" in terminal
    assert "repeat" in terminal
    assert "eff" in terminal
    assert "costs (raw spend: all logged LM calls):" not in terminal
    assert "costs (deduped spend: stable operation ids only; legacy rows counted raw):" not in terminal


def test_cost_rows_group_patch_merge_roles(tmp_path: Path):
    (tmp_path / "run_metadata.json").write_text(
        json.dumps(
            {
                "resolved_config": {
                    "proposer_reasoning_effort": "medium",
                    "proposer_sub_lm_reasoning_effort": "low",
                }
            }
        )
    )
    append_cost_rows(
        tmp_path / "cost_log.jsonl",
        [
            CostRow(
                event_id="e",
                operation_id="op",
                attempt_id="a",
                event="patch_merge",
                role="patch_merge_proposer",
                model="dummy-patch",
                calls=1,
                input_tokens=10,
                output_tokens=2,
                cost_usd=0.01,
            ),
            CostRow(
                event_id="e-sub",
                operation_id="op-sub",
                attempt_id="a-sub",
                event="patch_merge",
                role="patch_merge_proposer_sub_lm",
                model="dummy-patch-sub",
                calls=1,
                input_tokens=11,
                output_tokens=3,
                cost_usd=0.02,
            ),
        ],
    )

    rows = cost_rows(tmp_path)

    assert any(row.get("scope") == "patch-merge" and row.get("_category") for row in rows)
    assert any(
        row.get("scope") == "  - main" and row.get("model") == "dummy-patch-medium"
        for row in rows
    )
    assert any(
        row.get("scope") == "  - sub" and row.get("model") == "dummy-patch-sub-low"
        for row in rows
    )
    assert not any(row.get("scope") == "other" for row in rows)


def test_highlighted_terminal_rows_use_dim_gold_for_muted_prefixes():
    rendered = render_table(
        [
            {
                "metric": "0.100 → 0.200 +0.100",
                "_highlight": True,
                "_muted_prefix": {"metric": "0.100 → 0.200"},
            }
        ]
    )

    assert "\033[38;5;178m.100 → .200" in rendered
    assert "\033[38;5;248m.100 → .200" not in rendered


def test_terminal_cost_table_wraps_scope_and_model_to_terminal_width(monkeypatch):
    monkeypatch.setattr(
        stats_report.shutil,
        "get_terminal_size",
        lambda fallback=(120, 24): os.terminal_size((82, 24)),
    )
    rows = [
        {
            "scope": "  - patch_merge_proposer_sub_lm",
            "model": "openai/gpt-5.4-mini-medium",
            "calls": "1,234",
            "prompt_tok": "12,345,678",
            "completion_tok": "123,456",
            "total_cost": "$123.45",
            "repeat_cost": "$0.00",
            "effective_cost": "$123.45",
        }
    ]

    rendered = render_table(rows)
    plain_lines = [stats_report.re.sub(r"\033\[[0-9;]*m", "", line) for line in rendered.splitlines()]

    assert max(len(line) for line in plain_lines) <= 82
    assert "in_tok" in rendered
    assert "out_tok" in rendered
    assert "total_cost" not in rendered
    assert "  - patch" in rendered
    assert "│     _merg" in rendered
    assert "poser" in rendered


def test_terminal_merge_table_wraps_headers_and_status_to_terminal_width(monkeypatch):
    def plain_lines(rendered: str) -> list[str]:
        return [stats_report.re.sub(r"\033\[[0-9;]*m", "", line) for line in rendered.splitlines()]

    def cell_lines(rendered: str, column_index: int) -> list[str]:
        lines = []
        for line in plain_lines(rendered):
            if line.startswith("│"):
                cells = line.strip("│").split("│")
                lines.append(cells[column_index].strip())
        return lines

    def raw_cell_lines(rendered: str, column_index: int) -> list[str]:
        lines = []
        for line in plain_lines(rendered):
            if line.startswith("│"):
                cells = line.strip("│").split("│")
                lines.append(cells[column_index])
        return lines

    def body_cell_rows(rendered: str) -> list[list[str]]:
        lines = plain_lines(rendered)
        body_start = next(index for index, line in enumerate(lines) if line.startswith("├")) + 1
        return [line.strip("│").split("│") for line in lines[body_start:] if line.startswith("│")]

    monkeypatch.setattr(
        stats_report.shutil,
        "get_terminal_size",
        lambda fallback=(120, 24): os.terminal_size((90, 24)),
    )
    rows = [
        {
            "iter": "12345 [123, 456]",
            "pair@anc": "123+456@789",
            "soft: best(par) -> merge": "0.123 → 0.987 +0.864",
            "hard: best(par) -> merge": "0.111 → 0.999 +0.888; 1 → 9",
            "flips": "+8/-0 +8",
            "p": "0.01",
            "outcome": "rejected",
            "_terminal_header_aliases": {
                "soft: best(par) -> merge": "soft\nbest(par) -> merge",
                "hard: best(par) -> merge": "hard\nbest(par) -> merge",
            },
            "_terminal_header_suffixes": {
                "hard: best(par) -> merge": "/10",
            },
            "_muted_prefix": {
                "soft: best(par) -> merge": "0.123 → 0.987",
                "hard: best(par) -> merge": "0.111 → 0.999",
                "flips": "+8/-0",
            },
        }
    ]

    rendered = render_table(rows)

    assert "soft: best(par) -> merge" not in rendered
    assert "hard: best(par) -> merge" not in rendered
    assert "soft" in rendered
    assert "best(par) -> merge" in rendered
    assert stats_report.re.search(r"best\(par\) -> merge\s+/10", "\n".join(plain_lines(rendered)))
    assert "par→merge" not in rendered
    assert "1 → 9 /10" not in rendered
    assert "rejected" in rendered

    monkeypatch.setattr(
        stats_report.shutil,
        "get_terminal_size",
        lambda fallback=(120, 24): os.terminal_size((92, 24)),
    )
    moderate = render_table(
        [
            {
                "iter": "12",
                "pair@anc": "3+4@2",
                "soft: best(par) -> merge": "0.123 → 0.987 +0.864",
                "hard: best(par) -> merge": "0.111 → 0.999 +0.888; 1 → 9",
                "flips": "+8/-0 +8",
                "p": "0.01",
                "outcome": "accepted",
                "_terminal_header_aliases": {
                    "soft: best(par) -> merge": "soft\nbest(par) -> merge",
                    "hard: best(par) -> merge": "hard\nbest(par) -> merge",
                },
                "_terminal_header_suffixes": {
                    "hard: best(par) -> merge": "/10",
                },
                "_muted_prefix": {
                    "soft: best(par) -> merge": "0.123 → 0.987",
                    "hard: best(par) -> merge": "0.111 → 0.999",
                    "flips": "+8/-0",
                },
            }
        ]
    )
    moderate_lines = plain_lines(moderate)
    pair_lines = cell_lines(moderate, 1)
    hard_lines = raw_cell_lines(moderate, 3)
    moderate_body_cells = body_cell_rows(moderate)

    assert moderate_lines
    assert pair_lines[:2] == ["pair", "@anc"]
    assert "@" not in pair_lines[:2]
    assert "anc" not in pair_lines[:2]
    assert pair_lines[2:3] == ["3+4@2"]
    assert hard_lines[1].rstrip().endswith("/10")
    assert len(moderate_body_cells) == 1
    assert [moderate_body_cells[0][index].strip() for index in (1, 2, 3, 4)] == [
        "3+4@2",
        ".123 → .987 +.864",
        ".111 → .999 +.888; 1 → 9",
        "+8/-0 +8",
    ]

    monkeypatch.setattr(
        stats_report.shutil,
        "get_terminal_size",
        lambda fallback=(120, 24): os.terminal_size((90, 24)),
    )
    tight = render_table(rows)
    tight_lines = plain_lines(tight)
    body_cells = body_cell_rows(tight)

    assert tight_lines
    assert len(body_cells) == 1
    assert [body_cells[0][index].strip() for index in (1, 2, 3, 4)] == [
        "123+456@789",
        ".123 → .987 +.864",
        ".111 → .999 +.888; 1 → 9",
        "+8/-0 +8",
    ]


def test_eval_stats_from_eval_artifact(tmp_path: Path):
    report = {
        "config": {"reasoning_effort": "medium"},
        "total_tasks": 2,
        "soft_restriction_avg": 0.75,
        "hard_restriction_avg": 0.5,
        "tasks_all_passing": 1,
        "duration_seconds": 125,
        "total_cost_usd": 1.23,
        "costs": [
            {
                "role": "main",
                "model": "dummy-main",
                "calls": 3,
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "cost_usd": 1.0,
            },
            {
                "role": "sub",
                "model": "dummy-sub",
                "calls": 2,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cost_usd": 0.23,
            },
        ],
        "per_task": [
            {
                "task_id": "a",
                "soft": 1.0,
                "hard": 1,
                "cases": [
                    {"passed": True, "message": "All 10 cells match"},
                    {"passed": True, "message": "All 5 cells match"},
                ],
            },
            {
                "task_id": "b",
                "soft": 0.5,
                "hard": 0,
                "cases": [
                    {"passed": False, "message": "Sheet 'A' range A1:A6: 3/6 cells match"}
                ],
            },
        ],
    }
    (tmp_path / "eval.json").write_text(json.dumps(report))

    tasks = eval_task_rows(tmp_path)
    costs = eval_cost_rows(tmp_path)
    rendered = render_stats(tmp_path, table="all")

    assert tasks[0] == {
        "task": "a",
        "soft": "1.000 (15 /15)",
        "hard": "1.000 (2 /2)",
        "_align": {"soft": "left"},
    }
    assert tasks[1] == {
        "task": "b",
        "soft": "0.500 (3 /6)",
        "hard": "0.000 (0 /1)",
        "_align": {"soft": "left"},
    }
    assert costs[0]["scope"] == "executor"
    assert costs[1]["scope"] == "  - main"
    assert costs[1]["model"] == "dummy-main-medium"
    assert costs[2]["scope"] == "  - sub"
    assert costs[2]["model"] == "dummy-sub-none"
    assert "eval: tasks=2, soft=0.750, hard=0.500 (1/2), cost=$1.23, duration=2m 5s" in rendered
    assert "tasks:" in rendered
    assert "costs:" in rendered


def test_render_table_outputs_github_markdown():
    rendered = render_table([{"a": "x|y", "b": "z\nw"}], output_format="markdown")

    assert rendered.splitlines()[0].startswith("| a")
    assert rendered.splitlines()[1].startswith("| -")
    assert "x\\|y" in rendered
    assert "z<br>w" in rendered


def test_render_table_compacts_fractional_decimal_columns():
    rows = [
        {"mean": "0.123", "delta": "+0.045", "mixed": "0.123 → 1.000", "model": "gpt-0.5"},
        {"mean": "0.456", "delta": "-0.012", "mixed": "0.456 → 1.000", "model": "gpt-0.7"},
    ]

    markdown = render_table(rows, output_format="markdown")
    terminal = render_table(rows, output_format="terminal")

    assert ".123" in markdown
    assert "+.045" in markdown
    assert "-.012" in markdown
    assert "0.123 → 1.000" in markdown
    assert "gpt-0.5" in markdown
    assert ".456" in terminal
    assert "gpt-0.7" in terminal


def test_project_cli_check_with_dummy_lms(capsys):
    config = OptimizeConfig(
        executor_lm=_DummyLM(),
        executor_sub_lm=_DummyLM(),
        proposer_lm=_DummyLM(),
        proposer_sub_lm=_DummyLM(),
    )
    status = run_project_cli(lambda: _Project(), config, argv=["optimize", "--check"])

    assert status == 0
    assert "check ok" in capsys.readouterr().out


def test_project_cli_accepts_stat_alias(monkeypatch, capsys, tmp_path: Path):
    def render_stats(run_dir, table="all", output_format="terminal"):
        return f"stats {Path(run_dir).name} {table} {output_format}"

    monkeypatch.setattr(cli_module, "render_stats", render_stats)

    status = run_project_cli(lambda: _Project(), OptimizeConfig(), argv=["stat", str(tmp_path)])

    assert status == 0
    assert f"stats {tmp_path.name} all terminal" in capsys.readouterr().out


def test_public_api_exports_expected_helpers():
    assert run_optimization is not None
    assert check_optimization is not None
    assert agent_spec_from_rlm is not None
    assert build_proposer_for_rlm is not None


def test_apply_optimize_args_does_not_mutate_default_config():
    config = OptimizeConfig(executor_lm="before")
    args = argparse.Namespace(
        executor_lm="after",
        executor_sub_lm=None,
        executor_reasoning_effort=None,
        executor_sub_lm_reasoning_effort=None,
        proposer_lm=None,
        proposer_sub_lm=None,
        proposer_reasoning_effort=None,
        proposer_sub_lm_reasoning_effort=None,
        max_metric_calls=None,
        minibatch_size=None,
        concurrency=None,
        max_iterations=None,
        task_timeout=None,
        proposer_timeout=None,
        heartbeat_interval_seconds=None,
        run_dir=None,
        candidate_selection_strategy=None,
        component_selection_strategy=None,
        max_merge_attempts=None,
        resume=False,
        cache=False,
        verbose_rlm=False,
        merge_proposer=True,
    )

    updated = apply_optimize_args(config, args)

    assert config.executor_lm == "before"
    assert updated.executor_lm == "after"
    assert updated.merge_proposer is True


class _TimeoutProject(_Project):
    async def evaluate_example(self, candidate, example, context):
        await asyncio.sleep(1)
        return RLMGepaExampleResult(score=1.0, feedback="ok", traces=[])


class _ImmediateProject(_Project):
    async def evaluate_example(self, candidate, example, context):
        return RLMGepaExampleResult(
            score=1.0,
            feedback="",
            traces=[{"status": "ok"}],
            example_id=str(example),
        )


def test_adapter_progress_bar_updates_per_example(tmp_path: Path, monkeypatch):
    import rlm_gepa.runtime.adapter as adapter_module

    events: list[tuple[str, object]] = []

    class FakeTqdm:
        def __init__(self, **kwargs):
            events.append(("init", kwargs))

        def set_postfix_str(self, value):
            events.append(("postfix", value))

        def update(self, value):
            events.append(("update", value))

        def close(self):
            events.append(("close", None))

    monkeypatch.setattr(adapter_module, "tqdm", FakeTqdm, raising=False)
    adapter = RLMGepaAdapter(
        project=_ImmediateProject(),
        lm=_DummyLM(),
        sub_lm=_DummyLM(),
        max_iterations=1,
        concurrency=2,
        task_timeout=1,
        output_dir=tmp_path,
        run_id="run_test",
        display_progress_bar=True,
    )

    batch = adapter.evaluate(["a", "b"], {"skill_instructions": "seed"}, capture_traces=True)

    assert batch.scores == [1.0, 1.0]
    assert events[0] == (
        "init",
        {"total": 2, "desc": "  MB 0000 (2 tasks)", "leave": False, "unit": "task"},
    )
    assert [event for event in events if event[0] == "update"] == [("update", 1), ("update", 1)]
    assert events[-1] == ("close", None)


def test_adapter_progress_bar_labels_valset(tmp_path: Path, monkeypatch):
    import rlm_gepa.runtime.adapter as adapter_module

    events: list[tuple[str, object]] = []

    class FakeTqdm:
        def __init__(self, **kwargs):
            events.append(("init", kwargs))

        def set_postfix_str(self, value):
            events.append(("postfix", value))

        def update(self, value):
            events.append(("update", value))

        def close(self):
            events.append(("close", None))

    monkeypatch.setattr(adapter_module, "tqdm", FakeTqdm, raising=False)
    adapter = RLMGepaAdapter(
        project=_ImmediateProject(),
        lm=_DummyLM(),
        sub_lm=_DummyLM(),
        max_iterations=1,
        concurrency=2,
        task_timeout=1,
        output_dir=tmp_path,
        run_id="run_test",
        display_progress_bar=True,
        valset_size=2,
    )

    adapter.evaluate(["a", "b"], {"skill_instructions": "seed"}, capture_traces=False)

    assert events[0] == (
        "init",
        {"total": 2, "desc": "  VALSET 0000 (2 tasks)", "leave": False, "unit": "task"},
    )


def test_adapter_classifies_no_trace_repeat_batch_as_minibatch(tmp_path: Path, monkeypatch):
    import rlm_gepa.runtime.adapter as adapter_module

    descriptions: list[str] = []

    class FakeTqdm:
        def __init__(self, **kwargs):
            descriptions.append(kwargs["desc"])

        def set_postfix_str(self, value):
            pass

        def update(self, value):
            pass

        def close(self):
            pass

    monkeypatch.setattr(adapter_module, "tqdm", FakeTqdm, raising=False)
    adapter = RLMGepaAdapter(
        project=_ImmediateProject(),
        lm=_DummyLM(),
        sub_lm=_DummyLM(),
        max_iterations=1,
        concurrency=2,
        task_timeout=1,
        output_dir=tmp_path,
        run_id="run_test",
        display_progress_bar=True,
        valset_size=2,
    )

    adapter.evaluate(["a", "b"], {"skill_instructions": "seed"}, capture_traces=True)
    adapter.evaluate(["a", "b"], {"skill_instructions": "seed"}, capture_traces=False)

    assert descriptions == ["  MB 0000 (2 tasks)", "  MB 0001 (2 tasks)"]


def test_adapter_progress_bar_uses_reflective_context(tmp_path: Path, monkeypatch):
    import rlm_gepa.runtime.adapter as adapter_module

    descriptions: list[str] = []

    class FakeTqdm:
        def __init__(self, **kwargs):
            descriptions.append(kwargs["desc"])

        def set_postfix_str(self, value):
            pass

        def update(self, value):
            pass

        def close(self):
            pass

    monkeypatch.setattr(adapter_module, "tqdm", FakeTqdm, raising=False)
    adapter = RLMGepaAdapter(
        project=_ImmediateProject(),
        lm=_DummyLM(),
        sub_lm=_DummyLM(),
        max_iterations=1,
        concurrency=2,
        task_timeout=1,
        output_dir=tmp_path,
        run_id="run_test",
        display_progress_bar=True,
        valset_size=2,
    )

    adapter.set_reflective_progress_context(iteration=13, parent_idx=4, child_idx=7)
    adapter.evaluate(["a", "b"], {"skill_instructions": "parent"}, capture_traces=True)
    adapter.evaluate(["a", "b"], {"skill_instructions": "child"}, capture_traces=False)
    adapter.evaluate(["c", "d"], {"skill_instructions": "child"}, capture_traces=False)

    assert descriptions == [
        "  Iteration 13 Parent #4 Minibatch (2 tasks)",
        "  Iteration 13 Child #7 Minibatch (2 tasks)",
        "  Candidate #7 Valset (2 tasks)",
    ]


class _ContextProject(_Project):
    def __init__(self):
        self.verbose_values: list[bool] = []

    async def evaluate_example(self, candidate, example, context):
        self.verbose_values.append(context.verbose_rlm)
        return RLMGepaExampleResult(
            score=1.0,
            feedback="",
            traces=[{"status": "ok"}],
            example_id=str(example),
        )


def test_adapter_propagates_verbose_rlm_to_every_example(tmp_path: Path):
    project = _ContextProject()
    adapter = RLMGepaAdapter(
        project=project,
        lm=_DummyLM(),
        sub_lm=_DummyLM(),
        max_iterations=1,
        concurrency=2,
        task_timeout=1,
        output_dir=tmp_path,
        run_id="run_test",
        verbose_rlm=True,
    )

    adapter.evaluate(["a", "b"], {"skill_instructions": "seed"}, capture_traces=False)

    assert project.verbose_values == [True, True]


class _TelemetryProject(_Project):
    def __init__(self):
        self.contexts: list[EvaluationContext] = []

    async def evaluate_example(self, candidate, example, context):
        self.contexts.append(context)
        context.telemetry_context.write_span(
            "test.case",
            event_domain="test",
            attributes={"example": example},
        )
        return RLMGepaExampleResult(
            score=1.0,
            feedback="",
            traces=[{"status": "ok"}],
            example_id=str(example),
        )


def test_adapter_threads_telemetry_context_and_persists_candidate_hash(tmp_path: Path):
    project = _TelemetryProject()
    telemetry_context = TelemetryContext(
        sink=JsonlTelemetrySink(tmp_path / "telemetry" / "events.jsonl"),
        trace_id="run_test",
        run_id="run_test",
    )
    adapter = RLMGepaAdapter(
        project=project,
        lm=_DummyLM(),
        sub_lm=_DummyLM(),
        max_iterations=1,
        concurrency=2,
        task_timeout=1,
        output_dir=tmp_path,
        run_id="run_test",
        telemetry_context=telemetry_context,
    )

    adapter.evaluate(
        [SimpleNamespace(task_id="example")],
        {"skill_instructions": "seed"},
        capture_traces=False,
    )

    context = project.contexts[0].telemetry_context
    assert context.run_id == "run_test"
    assert context.eval_kind == "valset"
    assert context.eval_idx == 0
    assert context.attempt_id == "attempt_0000"
    assert context.example_id == "example"
    assert context.candidate_id is None
    assert context.candidate_hash.startswith("cand_sha256_")
    events = [
        json.loads(line)
        for line in (tmp_path / "telemetry" / "events.jsonl").read_text().splitlines()
    ]
    assert events[0]["attributes"]["rlm.candidate_hash"] == context.candidate_hash
    trace_rows = [
        json.loads(line)
        for line in (
            tmp_path / "task_traces" / "run_test_eval_valset_attempt_0000_valset.jsonl"
        )
        .read_text()
        .splitlines()
    ]
    assert trace_rows[0]["candidate_id"] is None
    assert trace_rows[0]["candidate_hash"] == context.candidate_hash
    assert trace_rows[0]["telemetry_ref"]["trace_id"] == context.trace_id


class _FailingTelemetryProject(_Project):
    async def evaluate_example(self, candidate, example, context):
        context.telemetry_context.write_span(
            "host_tool.recalculate.timeout",
            event_domain="host_tool",
            status={"code": "ERROR", "message": "tool timed out"},
            attributes={"failure.class": "host_tool_timeout_or_leak"},
        )
        return RLMGepaExampleResult(
            score=0.0,
            feedback="tool failed",
            traces=[],
            example_id=str(example),
            error="tool timed out",
        )


def test_adapter_trace_rows_include_compact_failure_metadata(tmp_path: Path):
    telemetry_context = TelemetryContext(
        sink=JsonlTelemetrySink(tmp_path / "telemetry" / "events.jsonl"),
        trace_id="run_test",
        run_id="run_test",
    )
    adapter = RLMGepaAdapter(
        project=_FailingTelemetryProject(),
        lm=_DummyLM(),
        sub_lm=_DummyLM(),
        max_iterations=1,
        concurrency=1,
        task_timeout=1,
        output_dir=tmp_path,
        run_id="run_test",
        telemetry_context=telemetry_context,
    )

    adapter.evaluate(["example"], {"skill_instructions": "seed"}, capture_traces=False)

    trace_rows = [
        json.loads(line)
        for line in (
            tmp_path / "task_traces" / "run_test_eval_valset_attempt_0000_valset.jsonl"
        )
        .read_text()
        .splitlines()
    ]
    row = trace_rows[0]
    assert row["candidate_id"] is None
    assert row["candidate_hash"].startswith("cand_sha256_")
    assert row["failure_class"] == "host_tool_timeout_or_leak"
    assert row["failure_reason"] == "tool timed out"
    assert row["telemetry_ref"]["events_path"] == "telemetry/events.jsonl"
    assert row["telemetry_ref"]["trace_id"].endswith(":0")


def test_gepa_failure_metadata_includes_lm_truncation_fields():
    events = [
        {
            "name": "rlm.action_generation.parse_error",
            "status": {"code": "ERROR", "message": "parse failed"},
            "attributes": {
                "failure.class": "model_output_truncated",
                "lm.truncated": True,
                "lm.truncation_reason": "max_tokens",
                "lm.finish_reason": "length",
                "lm.max_tokens": 50000,
                "lm.output_tokens": 50000,
            },
        }
    ]

    metadata = _row_failure_metadata(
        {"score": 0.0},
        events,
        telemetry_context=None,
    )

    assert metadata["failure_class"] == "model_output_truncated"
    assert metadata["truncated"] is True
    assert metadata["truncation_reason"] == "max_tokens"
    assert metadata["finish_reason"] == "length"
    assert metadata["max_tokens"] == 50000
    assert metadata["output_tokens"] == 50000


def test_gepa_failure_metadata_infers_truncation_from_structured_trace_finish_reason():
    metadata = _row_failure_metadata(
        {
            "score": 0.0,
            "trace": {
                "usage": {
                    "main": {
                        "output_tokens": 50000,
                        "max_tokens": 50000,
                    }
                },
                "steps": [{"lm": {"finish_reason": "length"}}],
            },
        },
        [],
        telemetry_context=None,
    )

    assert metadata["failure_class"] == "model_output_truncated"
    assert metadata["truncated"] is True
    assert metadata["truncation_reason"] == "max_tokens"
    assert metadata["finish_reason"] == "length"
    assert metadata["max_tokens"] == 50000
    assert metadata["output_tokens"] == 50000


def test_telemetry_classifier_uses_truncation_without_error_text():
    assert classify_failure({"score": 0.0, "truncated": True}, []) == "model_output_truncated"


def test_adapter_enforces_per_example_timeout(tmp_path: Path):
    adapter = RLMGepaAdapter(
        project=_TimeoutProject(),
        lm=_DummyLM(),
        sub_lm=_DummyLM(),
        max_iterations=1,
        concurrency=1,
        task_timeout=0.01,
        output_dir=tmp_path,
        run_id="run_test",
    )

    batch = adapter.evaluate(["example"], {"skill_instructions": "seed"}, capture_traces=True)

    assert batch.scores == [0.0]
    assert batch.trajectories[0]["record"]["Feedback"] == "evaluation timeout at 0.01s"


class _ErrorProject(_Project):
    async def evaluate_example(self, candidate, example, context):
        return RLMGepaExampleResult(
            score=0.0,
            feedback="expected failure",
            traces=[],
            example_id="example",
            error="expected failure",
        )


def test_resume_uses_unique_event_namespace_for_write_once_artifacts(tmp_path: Path):
    run_dir = tmp_path / "run"
    config = OptimizeConfig(run_dir=run_dir)
    _run_dir, first_run_id = prepare_run_dir(_Project(), config, command="first")
    assert (run_dir / "telemetry").is_dir()
    (run_dir / "gepa_state.bin").write_bytes(b"checkpoint")
    old_trace = run_dir / "task_traces" / f"{first_run_id}_eval_valset_attempt_0000_valset.jsonl"
    old_trace.write_text("existing\n")

    resume_config = OptimizeConfig(run_dir=run_dir, resume=True)
    _run_dir, resume_run_id = prepare_run_dir(_Project(), resume_config, command="resume")

    assert resume_run_id.startswith(f"{first_run_id}_resume_")
    adapter = RLMGepaAdapter(
        project=_ErrorProject(),
        lm=_DummyLM(),
        sub_lm=_DummyLM(),
        max_iterations=1,
        concurrency=1,
        task_timeout=1,
        output_dir=run_dir,
        run_id=resume_run_id,
    )

    batch = adapter.evaluate(["example"], {"skill_instructions": "seed"})

    assert batch.scores == [0.0]
    assert old_trace.read_text() == "existing\n"
    new_trace = run_dir / "task_traces" / f"{resume_run_id}_eval_valset_attempt_0000_valset.jsonl"
    assert new_trace.exists()


def test_patch_merge_adapter_uses_patch_signature_and_persists_metadata(
    tmp_path: Path,
    monkeypatch,
):
    import rlm_gepa.proposer.rlm as proposer_module
    import rlm_gepa.runtime.adapter as adapter_module

    captured: dict[str, object] = {}
    proposer_lm = _DummyLM()
    proposer_sub_lm = _DummyLM()

    class FakePredictRLM:
        def __init__(self, signature, *, lm, sub_lm, skills, max_iterations, verbose, debug):
            captured.update(
                {
                    "signature": signature,
                    "lm": lm,
                    "sub_lm": sub_lm,
                    "skills": skills,
                    "max_iterations": max_iterations,
                    "verbose": verbose,
                    "debug": debug,
                }
            )

        async def acall(self, **kwargs):
            captured["inputs"] = kwargs
            return SimpleNamespace(
                base_parent_id=10,
                patch_summary="imported one clause",
                selected_capability={
                    "name": "validated tool usage",
                    "evidence_task_ids": ["train-a"],
                    "trigger": "inputs require tool use",
                    "action": "validate inputs before invoking the tool",
                    "non_application_boundary": "do not apply to unrelated formatting tasks",
                    "preservation_note": "preserves base wins by only applying to tool-use rows",
                },
                imported_from_other=[
                    {
                        "clause": "Use the tool only after validating inputs.",
                        "evidence_task_ids": ["train-a"],
                        "reason": "source wins train-a",
                    }
                ],
                rejected_from_other=["Do not copy unrelated formatting advice."],
                new_instructions="base plus patch",
                trace=None,
                trajectory=[],
            )

    monkeypatch.setattr(adapter_module, "PredictRLM", FakePredictRLM)
    monkeypatch.setattr(adapter_module, "progress_write", lambda _message: None)
    monkeypatch.setattr(proposer_module, "progress_write", lambda _message: None)
    (tmp_path / "proposer_traces").mkdir()
    paired_trace = tmp_path / "paired_patch.jsonl"
    paired_trace.write_text("{}\n")
    adapter = RLMGepaAdapter(
        project=_Project(),
        lm=_DummyLM(),
        sub_lm=_DummyLM(),
        max_iterations=1,
        concurrency=1,
        task_timeout=1,
        output_dir=tmp_path,
        run_id="run_test",
        proposer_lm=proposer_lm,
        proposer_sub_lm=proposer_sub_lm,
        proposer_max_iterations=17,
    )

    new_text, metadata = adapter._rlm_propose_patch_merge_texts(
        call_idx=4,
        attempt_idx=2,
        base_parent_id=10,
        patch_source_parent_id=11,
        base_parent_instructions="base",
        patch_source_parent_instructions="source",
        paired_disagreement_traces_file=SimpleNamespace(path=str(paired_trace)),
        trace_task_ids=["train-a"],
    )

    assert new_text == "base plus patch"
    assert metadata["patch_summary"] == "imported one clause"
    assert metadata["selected_capability"]["name"] == "validated tool usage"
    assert metadata["base_instruction_chars"] == len("base")
    assert metadata["new_instruction_chars"] == len("base plus patch")
    assert metadata["instruction_char_delta"] == len("base plus patch") - len("base")
    assert captured["signature"].input_fields.keys() >= {
        "base_parent_id",
        "base_parent_instructions",
        "patch_source_parent_id",
        "patch_source_parent_instructions",
        "paired_disagreement_traces_file",
    }
    assert "selected_capability" in captured["signature"].output_fields
    assert "common_ancestor_instructions" not in captured["inputs"]
    assert captured["inputs"]["base_parent_id"] == 10
    assert captured["inputs"]["patch_source_parent_id"] == 11
    artifacts = list((tmp_path / "proposer_traces").glob("*_patch_from_cand_10_using_cand_11.json"))
    assert len(artifacts) == 1
    payload = json.loads(artifacts[0].read_text())
    assert payload["kind"] == "patch_merge_proposer"
    patch_output = payload["patch_output"]
    _assert_valid_patch_output(
        patch_output,
        trace_task_ids=["train-a"],
        base_instructions="base",
        new_instructions="base plus patch",
    )
    assert patch_output["imported_from_other"][0]["evidence_task_ids"] == ["train-a"]


def _assert_valid_patch_output(
    patch_output: dict[str, object],
    *,
    trace_task_ids: list[str],
    base_instructions: str,
    new_instructions: str,
) -> None:
    selected_capability = patch_output["selected_capability"]
    assert isinstance(selected_capability, dict)
    assert set(selected_capability) >= {
        "name",
        "evidence_task_ids",
        "trigger",
        "action",
        "non_application_boundary",
        "preservation_note",
    }
    assert set(selected_capability["evidence_task_ids"]) <= set(trace_task_ids)
    assert patch_output["base_instruction_chars"] == len(base_instructions)
    assert patch_output["new_instruction_chars"] == len(new_instructions)
    assert patch_output["instruction_char_delta"] == len(new_instructions) - len(
        base_instructions
    )
    for task_id in trace_task_ids:
        if len(task_id) >= 4:
            assert task_id not in new_instructions
    for audit_label in ("base_win", "patch_source_win", "both_success_guardrail"):
        assert audit_label not in new_instructions
