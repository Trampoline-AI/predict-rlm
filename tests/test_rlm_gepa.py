from __future__ import annotations

import asyncio
import json
import pickle
import random
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from predict_rlm.telemetry import JsonlTelemetrySink, TelemetryContext
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
from rlm_gepa import AgentSpec, OptimizeConfig, RLMGepaProject
from rlm_gepa.cli import run_project_cli
from rlm_gepa.proposer.merge import RlmMergeProposer
from rlm_gepa.proposer.rlm import RLMInstructionProposer
from rlm_gepa.proposer.selection import pick_patch_merge_pair
from rlm_gepa.reporting.cost import CostRow, aggregate_costs_from_log, append_cost_rows
from rlm_gepa.reporting.plots import load_plot_data
from rlm_gepa.reporting.stats import (
    candidate_rows,
    cost_rows,
    iteration_rows,
    merge_rows,
    render_stats,
)
from rlm_gepa.runtime.acceptance import should_accept_reflective_candidate
from rlm_gepa.runtime.adapter import RLMGepaAdapter
from rlm_gepa.schema import RLMGepaExampleResult, validate_project
from rlm_gepa.service import (
    _build_minibatch_sampler,
    _coerce_reflection_lm_text,
    prepare_run_dir,
)

pytestmark = pytest.mark.gepa


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


def test_group_aware_batch_sampler_keeps_groups_intact():
    from gepa.core.data_loader import ensure_loader

    class GroupedProject(_Project):
        def minibatch_group_id(self, example) -> str | None:
            return example["group"]

    examples = [
        {"task_id": f"group_{group}_{index}", "group": f"group_{group}"}
        for group in range(11)
        for index in range(3)
    ]
    loader = ensure_loader(examples)
    sampler = _build_minibatch_sampler(
        GroupedProject(),
        loader,
        minibatch_size=30,
        rng=random.Random(7),
    )

    batch_ids = sampler.next_minibatch_ids(loader, SimpleNamespace(i=0))
    batch_examples = loader.fetch(batch_ids)
    group_counts: dict[str, int] = {}
    for example in batch_examples:
        group_counts[example["group"]] = group_counts.get(example["group"], 0) + 1

    assert len(batch_examples) == 30
    assert len(group_counts) == 10
    assert set(group_counts.values()) == {3}


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
        self.patch_calls = 0

    def progress_label(self, _label):
        return nullcontext()

    def _rlm_propose_patch_merge_texts(self, **_kwargs):
        self.patch_calls += 1
        return "patched instructions", {"patch_summary": "imported one clause"}

    def evaluate(self, batch, _candidate, *, capture_traces, kind):
        scores = self.base_scores if self.evaluate_calls == 0 else self.source_scores
        self.evaluate_calls += 1
        selected_scores = scores[: len(batch)]
        trajectories = [
            {
                "task_id": str(item).replace(" ", "_"),
                "record": {
                    "Inputs": f"input for {item}",
                    "Traces": [{"steps": [{"code": f"solve({item!r})", "error": False}]}],
                    "Failure Metadata": {
                        "failure_class": "host_tool_timeout_or_leak",
                        "failure_reason": "tool timed out",
                        "candidate_hash": "cand_sha256_deadbeef",
                        "telemetry_ref": {
                            "trace_id": "run_test:cand_sha256_deadbeef:minibatch:0:0",
                            "candidate_hash": "cand_sha256_deadbeef",
                            "events_path": "telemetry/events.jsonl",
                        },
                    },
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
            {"skill_instructions": "ancestor", "other": "ancestor kept"},
            {"skill_instructions": "base", "other": "base kept"},
            {"skill_instructions": "source", "other": "source ignored"},
        ],
        parent_program_for_candidate=[[None], [0], [0]],
        prog_candidate_val_subscores=[
            {},
            {"v1": 1.0, "v2": 1.0, "v3": 0.0, "v4": 0.0},
            {"v1": 0.0, "v2": 0.0, "v3": 1.0, "v4": 1.0},
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


@pytest.mark.parametrize(
    ("base_scores", "source_scores", "min_each"),
    [
        ([1.0, 1.0], [0.0, 0.0], 1),
        ([1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0], 2),
    ],
    ids=["missing-source-wins", "cap-drops-required-evidence"],
)
def test_patch_merge_rejects_unbalanced_selected_evidence(
    tmp_path: Path, base_scores, source_scores, min_each
):
    proposer = _make_patch_evidence_proposer(
        tmp_path,
        base_scores=base_scores,
        source_scores=source_scores,
        merge_minibatch_size=2,
        min_each=min_each,
    )
    state = _patch_evidence_state()

    proposal = proposer._propose_patch_merge(
        state=state, iteration=0, merge_candidates=[1, 2], tracked_scores=[0.0, 0.5, 0.5]
    )

    assert proposal is None
    assert proposer.adapter.patch_calls == 0
    assert state.full_program_trace[-1]["rlm_merge_status"] == "preflight_failed"


def test_patch_disagreement_trace_jsonl_contains_patch_schema(tmp_path: Path):
    proposer = _make_patch_evidence_proposer(
        tmp_path,
        base_scores=[1.0, 0.0, 1.0],
        source_scores=[0.0, 1.0, 1.0],
        merge_minibatch_size=3,
    )

    evidence = proposer._build_patch_disagreement_evidence(
        state=_patch_evidence_state(),
        iteration=1,
        attempt_idx=0,
        base_parent_id=1,
        patch_source_parent_id=2,
    )
    records = [
        json.loads(line) for line in Path(evidence.paired_trace_path).read_text().splitlines()
    ]

    assert {record["task_id"]: record["evidence_role"] for record in records} == {
        "train_0": "base_win",
        "train_1": "patch_source_win",
        "train_2": "both_success_guardrail",
    }
    assert records[0]["schema_version"] == 1
    assert records[0]["winner"] in {"base", "patch_source"}
    assert records[0]["evidence_role"] in {"base_win", "patch_source_win"}
    assert records[0]["abs_delta"] == pytest.approx(1.0)
    assert records[0]["base_parent_id"] == 1
    assert records[0]["patch_source_parent_id"] == 2
    assert "generated_outputs" not in records[0]["base_parent"]
    assert "trace_preview" not in records[0]["base_parent"]
    assert records[0]["base_parent"]["traces"]
    records_text = json.dumps(records)
    assert "candidate_hash" not in records_text
    assert "telemetry_ref" not in records_text
    assert "events_path" not in records_text
    assert "trace_id" not in records_text
    assert records[0]["patch_source_parent"]["feedback"]


@pytest.mark.parametrize("child_scores", [[1.0, 1.0], [0.0, 1.0]], ids=["improved", "tied"])
def test_patch_merge_requires_improvement_and_preserves_base_components(
    tmp_path: Path, child_scores
):
    proposer = _make_patch_evidence_proposer(
        tmp_path, base_scores=[1.0, 0.0], source_scores=[0.0, 1.0], merge_minibatch_size=2
    )
    state = _patch_evidence_state()

    def cached_evaluate(candidate, ids, fetch, evaluator):
        assert candidate == {"skill_instructions": "patched instructions", "other": "base kept"}
        return child_scores, len(child_scores)

    state.cached_evaluate = cached_evaluate
    proposal = proposer._propose_patch_merge(
        state=state, iteration=0, merge_candidates=[1, 2], tracked_scores=[0.0, 0.5, 0.5]
    )

    if sum(child_scores) > 1:
        assert proposal.candidate == {
            "skill_instructions": "patched instructions",
            "other": "base kept",
        }
        assert proposal.parent_program_ids == [1, 2]
        assert state.full_program_trace[-1]["rlm_merge_status"] == "accepted"
    else:
        assert proposal is None
        assert state.full_program_trace[-1]["rlm_merge_status"] == "subsample_rejected"
    assert state.program_candidates[1] == {"skill_instructions": "base", "other": "base kept"}


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


def test_merge_sidecar_preserves_attempts_and_pairs_on_resume(tmp_path: Path):
    proposer = _make_patch_evidence_proposer(
        tmp_path,
        base_scores=[1.0],
        source_scores=[0.0],
        merge_minibatch_size=1,
    )
    proposer.use_merge = False
    proposer.rlm_merge_attempts_used = 1
    proposer.merges_performed[0].append((1, 2, 0))
    state = SimpleNamespace(i=0, full_program_trace=[{}])

    assert proposer.propose(state) is None

    resumed = _make_patch_evidence_proposer(
        tmp_path,
        base_scores=[1.0],
        source_scores=[0.0],
        merge_minibatch_size=1,
    )
    assert resumed.rlm_merge_attempts_used == 1
    assert resumed.merges_performed[0] == [(1, 2, 0)]


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
    raw = aggregate_costs_from_log(path)
    assert raw[0].calls == 7
    assert raw[0].cost_usd == pytest.approx(0.4)
    total = next(row for row in cost_rows(tmp_path) if row["scope"] == "TOTAL")
    assert total["total_cost"] == "$0.40"
    assert total["repeat_cost"] == "$0.10"
    assert total["effective_cost"] == "$0.30"


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


def test_merge_rows_use_subsample_scores_for_table_metrics_not_full_val_details(tmp_path: Path):
    state = {
        "prog_candidate_val_subscores": [
            {"a": 1.0, "b": 1.0, "c": 1.0},
            {"a": 1.0, "b": 0.0, "c": 1.0},
            {"a": 1.0, "b": 0.0, "c": 0.0},
        ],
        "full_program_trace": [
            {
                "i": 12,
                "rlm_merge_candidate_pair": (0, 1),
                "rlm_merge_ancestor": 0,
                "rlm_merge_status": "accepted",
                "rlm_merge_base_parent": 0,
                "rlm_merge_patch_source_parent": 1,
                "rlm_merge_new_program_idx": 2,
                "id1_subsample_scores": [0.0, 0.0],
                "id2_subsample_scores": [1.0, 0.0],
                "new_program_subsample_scores": [1.0, 1.0],
            }
        ],
    }
    with (tmp_path / "gepa_state.bin").open("wb") as f:
        pickle.dump(state, f)

    rows = merge_rows(tmp_path)

    assert rows[0]["soft: best(par) -> merge"] == "0.500 → 1.000 +0.500"
    assert rows[0]["hard: best(par) -> merge"] == "0.500 → 1.000 +0.500; 1 → 2"
    assert rows[0]["flips"] == "+1/-0 +1"
    assert rows[0]["p"] == "1.00"
    assert rows[0]["_merge_hard_denominator"] == 2
    assert rows[0]["_detail"] == "→ cand 2"


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

    assert rows[0]["soft: best(par) -> merge"] == "0.500 → 0.500 +0.000"
    assert rows[0]["hard: best(par) -> merge"] == "0.500 → 0.500 +0.000; 1 → 1"
    assert rows[0]["flips"] == "+1/-1 +0"
    assert rows[0]["p"] == "1.00"
    assert rows[0]["outcome"] == "rejected"
    assert rows[0]["_detail"] == "not better than best parent"


def test_iteration_rows_include_attempts_without_child_scores(tmp_path: Path):
    state = {
        "full_program_trace": [
            {"i": 0, "selected_program_candidate": 0, "subsample_scores": [1.0, 0.0]},
            {"i": 1, "selected_program_candidate": 0, "subsample_scores": [1.0, 0.0]},
            {
                "i": 2,
                "selected_program_candidate": 0,
                "subsample_scores": [1.0, 0.0],
                "new_subsample_scores": [0.0, 0.0],
            },
        ]
    }
    with (tmp_path / "gepa_state.bin").open("wb") as f:
        pickle.dump(state, f)

    rows = iteration_rows(tmp_path)

    assert [row["iter"] for row in rows] == ["1 [0]", "2 [0]"]
    assert rows[0]["outcome"] == "NO CHILD"
    assert rows[0]["soft: par → child"] == "-"
    assert rows[1]["outcome"] == "REJECTED"


def test_candidate_rows_show_flips_against_each_parent(tmp_path: Path):
    state = {
        "parent_program_for_candidate": [[None], [0], [0, 1]],
        "prog_candidate_val_subscores": [
            {"a": 1.0, "b": 0.0, "c": 0.0},
            {"a": 0.0, "b": 1.0, "c": 0.0},
            {"a": 1.0, "b": 1.0, "c": 0.0},
        ],
    }
    with (tmp_path / "gepa_state.bin").open("wb") as f:
        pickle.dump(state, f)

    rows = candidate_rows(tmp_path)

    assert rows[2]["soft: par → child"] == "0.333 → 0.667 +0.333\n0.333 → 0.667 +0.333"
    assert rows[2]["hard: par → child"] == "0.333 → 0.667 +0.333\n0.333 → 0.667 +0.333"
    assert rows[2]["flips vs par"] == "+1/-0 +1\n+1/-0 +1"


def test_live_state_best_candidate_overrides_stale_summary(tmp_path: Path):
    state = {
        "program_candidates": [{}, {}, {}],
        "parent_program_for_candidate": [[None], [0], [1]],
        "prog_candidate_val_subscores": [
            {"a": 0.0, "b": 0.0},
            {"a": 0.6, "b": 0.6},
            {"a": 1.0, "b": 1.0},
        ],
        "program_full_scores_val_set": [0.0, 0.6, 1.0],
        "num_metric_calls_by_discovery": [0, 10, 20],
    }
    with (tmp_path / "gepa_state.bin").open("wb") as f:
        pickle.dump(state, f)
    (tmp_path / "optimization_summary.json").write_text(
        json.dumps(
            {
                "best_idx": 1,
                "val_aggregate_scores": [0.0, 0.6, 0.5],
            }
        )
    )

    plot_data = load_plot_data(tmp_path)

    assert plot_data["best_idx"] == 2
    assert plot_data["scores"] == [0.0, 0.6, 1.0]


def test_plot_data_repairs_truncated_live_full_scores_from_subscores(tmp_path: Path):
    state = {
        "program_candidates": [{} for _ in range(8)],
        "parent_program_for_candidate": [[None], [0], [1], [2], [3], [4], [5], [6]],
        "prog_candidate_val_subscores": [
            {"a": 0.0, "b": 0.2},
            {"a": 0.1, "b": 0.3},
            {"a": 0.2, "b": 0.4},
            {"a": 0.3, "b": 0.5},
            {"a": 0.4, "b": 0.6},
            {"a": 0.5, "b": 0.7},
            {"a": 0.8, "b": 0.9},
            {"a": 0.95, "b": 1.0},
        ],
        "program_full_scores_val_set": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "num_metric_calls_by_discovery": [0, 10, 20, 30, 40, 50],
    }
    with (tmp_path / "gepa_state.bin").open("wb") as f:
        pickle.dump(state, f)
    (tmp_path / "optimization_summary.json").write_text(
        json.dumps(
            {
                "best_idx": 5,
                "val_aggregate_scores": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            }
        )
    )

    plot_data = load_plot_data(tmp_path)

    assert plot_data["n"] == 8
    assert plot_data["scores"][6] == pytest.approx(0.85)
    assert plot_data["scores"][7] == pytest.approx(0.975)
    assert plot_data["best_idx"] == 7
    assert plot_data["eval_counts"] == [0, 10, 20, 30, 40, 50, 51, 52]


def test_eval_stats_reports_attempt_outcomes_and_latency_percentiles(tmp_path: Path):
    report = {
        "total_tasks": 4,
        "soft_restriction_avg": 0.25,
        "hard_restriction_avg": 0.25,
        "tasks_all_passing": 1,
        "duration_seconds": 10,
        "total_cost_usd": 0.0,
        "per_task": [],
    }
    (tmp_path / "eval.json").write_text(json.dumps(report))
    trace_dir = tmp_path / "task_traces"
    trace_dir.mkdir()
    (trace_dir / "eval_attempts.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "example_id": "ok",
                    "status": "completed",
                    "feedback": "passed",
                    "trace": {
                        "status": "completed",
                        "iterations": 2,
                        "max_iterations": 5,
                        "duration_ms": 1000,
                    },
                },
                {
                    "example_id": "outer-timeout",
                    "status": "error",
                    "feedback": "evaluation timeout at 300s",
                    "trace": None,
                },
                {
                    "example_id": "maxed",
                    "status": "completed",
                    "feedback": "submitted by fallback",
                    "trace": {
                        "status": "max_iterations",
                        "iterations": 5,
                        "max_iterations": 5,
                        "duration_ms": 5000,
                    },
                },
                {
                    "example_id": "project-timeout",
                    "status": "error",
                    "error": "RLM timeout at 300s",
                    "trace": {
                        "status": "error",
                        "iterations": 1,
                        "max_iterations": 5,
                        "duration_ms": 2000,
                    },
                },
            ]
        )
    )

    terminal = render_stats(tmp_path, table="all")
    markdown = render_stats(tmp_path, table="all", output_format="markdown")

    expected = (
        "attempts=4, timeouts=2, max_iter_hits=1, latency p50=2.0s p90=5.0s p95=5.0s max=5.0s"
    )
    assert expected in terminal
    assert expected in markdown


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


class _TimeoutProject(_Project):
    cancelled = False

    async def evaluate_example(self, candidate, example, context):
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return RLMGepaExampleResult(score=1.0, feedback="ok", traces=[])


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


def test_reflective_record_visible_to_gepa_includes_failure_metadata(tmp_path: Path):
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

    batch = adapter.evaluate(["example"], {"skill_instructions": "seed"}, capture_traces=True)

    record = batch.trajectories[0]["record"]
    assert record["Score"] == 0.0
    assert record["Error"] == "tool timed out"
    assert record["Failure Metadata"]["failure_class"] == "host_tool_timeout_or_leak"
    assert record["Failure Metadata"]["failure_reason"] == "tool timed out"
    assert "candidate_hash" not in record["Failure Metadata"]
    assert "telemetry_ref" not in record["Failure Metadata"]

    task_trace_path = (
        tmp_path / "task_traces" / "run_test_eval_minibatch_attempt_0000_minibatch.jsonl"
    )
    task_row = json.loads(task_trace_path.read_text().splitlines()[0])
    assert task_row["candidate_hash"].startswith("cand_sha256_")
    assert task_row["telemetry_ref"]["trace_id"].endswith(":0")


class _StructuredTraceProject(_Project):
    async def evaluate_example(self, candidate, example, context):
        trace = RunTrace(
            status="completed",
            model="dummy/main",
            sub_model="dummy/sub",
            iterations=1,
            max_iterations=1,
            duration_ms=25,
            usage=LMUsage(
                main=TokenUsage(input_tokens=100, output_tokens=50, cost=0.01),
                sub=TokenUsage(input_tokens=20, output_tokens=10, cost=0.002),
            ),
            steps=[
                IterationStep(
                    iteration=1,
                    reasoning="try the tool and ask the helper",
                    code="answer = tool('x')\nSUBMIT(answer)",
                    output="truncated output",
                    untruncated_output="full output",
                    duration_ms=25,
                    lm=LMFinishMetadata(finish_reason="stop"),
                    tool_calls=[
                        ToolCall(
                            name="tool",
                            args=["x"],
                            kwargs={"mode": "fast"},
                            result={"raw": "tool output"},
                            error="tool exploded",
                            duration_ms=3,
                        )
                    ],
                    predict_calls=[
                        PredictCallGroup(
                            signature="question -> answer",
                            model="dummy/sub",
                            total_usage=TokenUsage(
                                input_tokens=20,
                                output_tokens=10,
                                cost=0.002,
                            ),
                            calls=[
                                PredictCallDetail(
                                    duration_ms=7,
                                    usage=TokenUsage(
                                        input_tokens=20,
                                        output_tokens=10,
                                        cost=0.002,
                                    ),
                                    input={"question": "q"},
                                    output={},
                                    error="helper failed",
                                    lm=LMFinishMetadata(finish_reason="length"),
                                )
                            ],
                        )
                    ],
                )
            ],
        )
        return RLMGepaExampleResult(
            score=0.5,
            feedback="partial",
            traces=[trace],
            rlm_inputs={"example": example},
            example_id=str(example),
        )


def test_adapter_reflective_records_include_structured_run_traces(tmp_path: Path):
    adapter = RLMGepaAdapter(
        project=_StructuredTraceProject(),
        lm=_DummyLM(),
        sub_lm=_DummyLM(),
        max_iterations=1,
        concurrency=1,
        task_timeout=1,
        output_dir=tmp_path,
        run_id="run_test",
    )

    batch = adapter.evaluate(["example"], {"skill_instructions": "seed"}, capture_traces=True)

    record = batch.trajectories[0]["record"]
    trace = record["Traces"][0]
    step = trace["steps"][0]
    assert record["Score"] == 0.5
    assert "Trace Preview" not in record
    assert "Generated Outputs" not in record
    assert step["code"] == "answer = tool('x')\nSUBMIT(answer)"
    assert step["output"] == "truncated output"
    assert step["untruncated_output"] == "full output"
    assert step["lm"] == {"finish_reason": "stop"}
    assert step["tool_calls"][0]["args"] == ["x"]
    assert step["tool_calls"][0]["kwargs"] == {"mode": "fast"}
    assert step["tool_calls"][0]["result"] == {"raw": "tool output"}
    assert step["tool_calls"][0]["error"] == "tool exploded"
    assert step["predict_calls"][0]["calls"][0]["error"] == "helper failed"
    assert step["predict_calls"][0]["calls"][0]["lm"] == {"finish_reason": "length"}
    assert "usage" not in trace
    assert "duration_ms" not in trace
    assert "duration_ms" not in step
    assert "duration_ms" not in step["tool_calls"][0]
    assert "usage" not in step["predict_calls"][0]
    assert "total_usage" not in step["predict_calls"][0]
    assert "duration_ms" not in step["predict_calls"][0]["calls"][0]
    assert "usage" not in step["predict_calls"][0]["calls"][0]

    task_trace_path = (
        tmp_path / "task_traces" / "run_test_eval_minibatch_attempt_0000_minibatch.jsonl"
    )
    task_row = json.loads(task_trace_path.read_text().splitlines()[0])
    archival_trace = task_row["trace"]
    archival_step = archival_trace["steps"][0]
    assert archival_trace["usage"]["main"]["input_tokens"] == 100
    assert archival_trace["usage"]["main"]["cost"] == 0.01
    assert archival_trace["duration_ms"] == 25
    assert archival_step["duration_ms"] == 25
    assert archival_step["tool_calls"][0]["duration_ms"] == 3
    assert archival_step["predict_calls"][0]["total_usage"]["input_tokens"] == 20
    assert archival_step["predict_calls"][0]["calls"][0]["usage"]["input_tokens"] == 20
    assert task_row["traces"][0]["usage"]["sub"]["cost"] == 0.002


def test_adapter_enforces_per_example_timeout(tmp_path: Path):
    project = _TimeoutProject()
    adapter = RLMGepaAdapter(
        project=project,
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
    assert project.cancelled
    row = json.loads(next((tmp_path / "task_traces").glob("*.jsonl")).read_text())
    assert row["score"] == 0.0
    assert "timeout" in row["error"]


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
    old_trace = (
        run_dir / "task_traces" / f"{first_run_id}_eval_valset_attempt_0000_valset.jsonl"
    )
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
    new_trace = (
        run_dir / "task_traces" / f"{resume_run_id}_eval_valset_attempt_0000_valset.jsonl"
    )
    row = json.loads(new_trace.read_text())
    assert row["score"] == 0.0
    assert row["error"] == "expected failure"


def test_rlm_patch_merge_no_op_patch_persists_compact_audit(tmp_path: Path, monkeypatch):
    import rlm_gepa.proposer.rlm as proposer_module
    import rlm_gepa.runtime.adapter as adapter_module

    base_instructions = "base rules"

    class FakePredictRLM:
        def __init__(self, *_args, **_kwargs):
            pass

        async def acall(self, **_kwargs):
            return SimpleNamespace(
                base_parent_id=10,
                patch_summary="left base unchanged because the source capability duplicates base",
                selected_capability={
                    "decision": "no-op",
                    "summary": "source behavior duplicates the base",
                    "evidence_task_ids": [],
                    "trigger": "no concrete source-win-only trigger identified",
                    "non_application_boundary": (
                        "base-win rows already cover the proposed behavior, so leave base unchanged"
                    ),
                },
                patch_audit={
                    "supported_source_win_ids": [],
                    "guardrail_hazards": ["candidate duplicates the base rule"],
                    "notes": "no-op: duplicate source behavior, no clean missing facet",
                },
                new_instructions=base_instructions,
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
        proposer_lm=_DummyLM(),
        proposer_sub_lm=_DummyLM(),
        proposer_max_iterations=1,
    )

    new_text, metadata = adapter._rlm_propose_patch_merge_texts(
        call_idx=4,
        attempt_idx=2,
        base_parent_id=10,
        patch_source_parent_id=11,
        base_parent_instructions=base_instructions,
        patch_source_parent_instructions="source rules",
        paired_disagreement_traces_file=SimpleNamespace(path=str(paired_trace)),
        trace_task_ids=["train-a"],
    )

    assert new_text == base_instructions
    assert metadata["new_instructions"] == base_instructions
    assert metadata["selected_capability"]["evidence_task_ids"] == []
    assert metadata["instruction_char_delta"] == 0
    assert metadata["patch_audit"]["supported_source_win_ids"] == []
    artifacts = list(
        (tmp_path / "proposer_traces").glob("*_patch_from_cand_10_using_cand_11.json")
    )
    assert len(artifacts) == 1
    patch_output = json.loads(artifacts[0].read_text())["patch_output"]
    assert patch_output["new_instructions"] == base_instructions
    assert patch_output["instruction_char_delta"] == 0
    assert patch_output["patch_audit"]["supported_source_win_ids"] == []
    assert patch_output["selected_capability"]["decision"] == "no-op"


def test_rlm_instruction_proposer_serializes_proposer_trace_records(
    tmp_path: Path, monkeypatch
):
    import rlm_gepa.proposer.rlm as proposer_module

    captured: dict[str, object] = {}

    class FakePredictRLM:
        def __init__(self, *_args, **_kwargs):
            pass

        async def acall(self, **kwargs):
            traces_file = kwargs["traces_file"]
            captured["records"] = json.loads(Path(traces_file.path).read_text())
            return SimpleNamespace(
                new_instructions="updated rules",
                generalization_check=[],
                trajectory=[],
                trace=None,
            )

    monkeypatch.setattr(proposer_module, "PredictRLM", FakePredictRLM)
    monkeypatch.setattr(proposer_module, "progress_write", lambda _message: None)
    proposer = RLMInstructionProposer(
        spec=_spec(),
        lm=_DummyLM(),
        sub_lm=_DummyLM(),
        output_dir=tmp_path,
        max_iterations=1,
        timeout=1,
        heartbeat_interval_seconds=60,
        run_id="run_test",
    )
    records = [
        {
            "Inputs": "input",
            "Score": 0.0,
            "Traces": [
                RunTrace(
                    status="completed",
                    model="dummy/main",
                    sub_model="dummy/sub",
                    iterations=1,
                    max_iterations=1,
                    duration_ms=10,
                    usage=LMUsage(
                        main=TokenUsage(input_tokens=100, output_tokens=50, cost=0.01),
                        sub=TokenUsage(input_tokens=20, output_tokens=10, cost=0.002),
                    ),
                    steps=[
                        IterationStep(
                            iteration=1,
                            reasoning="inspect image",
                            code="x = 1",
                            output="ok",
                            untruncated_output="ok",
                            duration_ms=10,
                            tool_calls=[ToolCall(name="tool", error="boom", duration_ms=1)],
                            predict_calls=[
                                PredictCallGroup(
                                    signature="image -> answer",
                                    model="dummy/sub",
                                    calls=[
                                        PredictCallDetail(
                                            duration_ms=1,
                                            input={
                                                "image": "data:image/png;base64,QUJDREVGRw=="
                                            },
                                            output={},
                                            error="predict boom",
                                        )
                                    ],
                                )
                            ],
                        )
                    ],
                )
            ],
            "Failure Metadata": {
                "failure_class": "host_tool_timeout_or_leak",
                "failure_reason": "tool timed out",
                "candidate_hash": "cand_sha256_deadbeef",
                "telemetry_ref": {
                    "trace_id": "run_test:cand_sha256_deadbeef:minibatch:0:0",
                    "candidate_hash": "cand_sha256_deadbeef",
                    "events_path": "telemetry/events.jsonl",
                },
            },
            "Trace Preview": "rendered preview",
            "Generated Outputs": "rendered preview",
            "Feedback": "failed",
            "Error": "tool timed out",
        }
    ]

    new_text = proposer.propose_one_component("skill_instructions", "seed", records)

    assert new_text == "updated rules"
    serialized = captured["records"]
    assert isinstance(serialized, list)
    assert serialized[0]["Traces"][0]["steps"][0]["tool_calls"][0]["error"] == "boom"
    assert (
        serialized[0]["Traces"][0]["steps"][0]["predict_calls"][0]["calls"][0]["error"]
        == "predict boom"
    )
    serialized_text = json.dumps(serialized)
    assert "QUJDREVGRw==" not in serialized_text
    assert "data:image/png;base64,<IMAGE_BASE_64_ENCODED(12)>" in serialized_text
    assert "usage" not in serialized_text
    assert "duration_ms" not in serialized_text
    assert "cost" not in serialized_text
    assert "cache_hits" not in serialized_text
    assert serialized[0]["Failure Metadata"]["failure_class"] == "host_tool_timeout_or_leak"
    assert "candidate_hash" not in serialized_text
    assert "telemetry_ref" not in serialized_text
    assert "events_path" not in serialized_text
    assert "trace_id" not in serialized_text
    assert "Trace Preview" not in serialized[0]
    assert "Generated Outputs" not in serialized[0]
