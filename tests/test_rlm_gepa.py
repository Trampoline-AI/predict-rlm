from __future__ import annotations

import argparse
import asyncio
import json
import pickle
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlm_gepa import (
    AgentSpec,
    OptimizeConfig,
    RLMGepaProject,
    agent_spec_from_rlm,
    build_merge_signature,
    build_proposer_for_rlm,
    build_proposer_signature,
    check_optimization,
    run_optimization,
)
from rlm_gepa.cli import apply_optimize_args, run_project_cli
from rlm_gepa.proposer.merge import VALID_STATUSES, RlmMergeProposer
from rlm_gepa.proposer.selection import pick_merge_triplet, walk_ancestors
from rlm_gepa.reporting.cost import CostRow, aggregate_costs_from_log, append_cost_rows
from rlm_gepa.reporting.plots import resolve_plot_output_paths
from rlm_gepa.reporting.stats import (
    candidate_rows,
    cost_rows,
    eval_cost_rows,
    eval_task_rows,
    iteration_rows,
    render_stats,
    render_table,
)
from rlm_gepa.runtime.adapter import RLMGepaAdapter
from rlm_gepa.schema import RLMGepaExampleResult, validate_project
from rlm_gepa.service import _coerce_reflection_lm_text, prepare_run_dir


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


def test_build_signatures_render_agent_spec():
    spec = _spec()
    proposer = build_proposer_signature(spec)
    merge = build_merge_signature(spec)

    assert "test agent" in proposer.instructions
    assert "{{" not in proposer.instructions
    assert "paired_traces_file" in merge.input_fields


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


def test_pick_merge_triplet_finds_divergent_siblings():
    parents = [[None], [0], [0]]
    candidates = [
        {"skill_instructions": "seed"},
        {"skill_instructions": "a"},
        {"skill_instructions": "b"},
    ]
    scores_a = {f"t{i}": 1.0 if i % 2 else 0.0 for i in range(12)}
    scores_b = {f"t{i}": 0.0 if i % 2 else 1.0 for i in range(12)}

    assert walk_ancestors(parents, 2) == {0}
    triplet = pick_merge_triplet(
        merge_candidates=[1, 2],
        program_candidates=candidates,
        parent_program_for_candidate=parents,
        prog_candidate_val_subscores=[{}, scores_a, scores_b],
        tracked_scores=[0.3, 0.5, 0.5],
        merges_performed=[],
        rng=random.Random(0),
        component_name="skill_instructions",
        min_each=3,
    )
    assert triplet == (1, 2, 0)


def test_pick_merge_triplet_dedups_previously_attempted_triplets():
    parents = [[None], [0], [0]]
    candidates = [
        {"skill_instructions": "seed"},
        {"skill_instructions": "a"},
        {"skill_instructions": "b"},
    ]
    scores_a = {f"t{i}": 1.0 if i % 2 else 0.0 for i in range(12)}
    scores_b = {f"t{i}": 0.0 if i % 2 else 1.0 for i in range(12)}

    triplet = pick_merge_triplet(
        merge_candidates=[1, 2],
        program_candidates=candidates,
        parent_program_for_candidate=parents,
        prog_candidate_val_subscores=[{}, scores_a, scores_b],
        tracked_scores=[0.3, 0.5, 0.5],
        merges_performed=[(1, 2, 0)],
        rng=random.Random(0),
        component_name="skill_instructions",
        min_each=3,
    )

    assert triplet is None


def test_pick_merge_triplet_skips_identical_candidate_text():
    parents = [[None], [0], [0]]
    candidates = [
        {"skill_instructions": "seed"},
        {"skill_instructions": "same"},
        {"skill_instructions": "same"},
    ]
    scores_a = {f"t{i}": 1.0 if i % 2 else 0.0 for i in range(12)}
    scores_b = {f"t{i}": 0.0 if i % 2 else 1.0 for i in range(12)}

    triplet = pick_merge_triplet(
        merge_candidates=[1, 2],
        program_candidates=candidates,
        parent_program_for_candidate=parents,
        prog_candidate_val_subscores=[{}, scores_a, scores_b],
        tracked_scores=[0.3, 0.5, 0.5],
        merges_performed=[],
        rng=random.Random(0),
        component_name="skill_instructions",
        min_each=3,
    )

    assert triplet is None


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
def test_rlm_merge_scores_candidate_without_merge_only_post_checks(tmp_path: Path, monkeypatch):
    from gepa.core.data_loader import ensure_loader

    import rlm_gepa.proposer.merge as merge_module

    monkeypatch.setattr(merge_module, "find_dominator_programs", lambda *_args: [1, 2])
    monkeypatch.setattr(merge_module, "pick_merge_triplet", lambda **_kwargs: (1, 2, 0))

    class FakeAdapter:
        proposer_trace_dir = tmp_path
        run_id = "run_test"

        def __init__(self):
            self.evaluate_calls = 0

        def evaluate(self, _batch, _candidate, *, capture_traces, kind):
            self.evaluate_calls += 1
            scores = [1.0, 0.0] if self.evaluate_calls == 1 else [0.0, 1.0]
            trajectories = [
                {"task_id": "task_1", "record": {"Feedback": "a wins"}},
                {"task_id": "task_2", "record": {"Feedback": "b wins"}},
            ]
            return SimpleNamespace(scores=scores, trajectories=trajectories)

        def make_reflective_dataset(self, _candidate, eval_batch, components):
            records = [trajectory["record"] for trajectory in eval_batch.trajectories]
            return {component: records for component in components}

        def _reserve_merge_proposer_call_idx(self):
            return 7

        def _rlm_propose_merge_texts(self, **_kwargs):
            return "merged instructions " * 200, ["[NEW] missing task_id citation"]

        def progress_label(self, _label):
            class NoopContext:
                def __enter__(self):
                    return None

                def __exit__(self, *_args):
                    return False

            return NoopContext()

        def queue_valset_progress_label(self, _label):
            pass

    def evaluator(_inputs, _candidate):
        return [], [], None

    proposer = RlmMergeProposer(
        logger=_Logger(),
        valset=ensure_loader(["v1", "v2"]),
        evaluator=evaluator,
        adapter=FakeAdapter(),
        trainset=ensure_loader(["train_1", "train_2"]),
        use_merge=True,
        max_merge_invocations=1,
        max_rlm_merge_attempts=5,
        min_each=1,
        merge_minibatch_size=2,
        rlm_merge_state_path=tmp_path / "rlm_merge_state.json",
        rng=random.Random(0),
    )
    proposer.last_iter_found_new_program = True
    proposer.merges_due = 1
    proposer.select_eval_subsample_for_merged_program = lambda *_args, **_kwargs: ["v1", "v2"]

    state = SimpleNamespace(
        i=0,
        full_program_trace=[{}],
        program_candidates=[
            {"skill_instructions": "ancestor"},
            {"skill_instructions": "parent a"},
            {"skill_instructions": "parent b"},
        ],
        parent_program_for_candidate=[[None], [0], [0]],
        prog_candidate_val_subscores=[
            {},
            {"v1": 1.0, "v2": 0.0},
            {"v1": 0.0, "v2": 1.0},
        ],
        program_full_scores_val_set=[0.0, 0.5, 0.5],
        per_program_tracked_scores=[0.0, 0.5, 0.5],
        total_num_evals=0,
    )
    state.get_pareto_front_mapping = lambda: {}
    state.cached_evaluate = lambda *_args, **_kwargs: ([1.0, 0.0], 2)

    proposal = proposer.propose(state)

    assert proposal is None
    assert state.full_program_trace[-1]["rlm_merge_status"] == "subsample_rejected"
    assert state.full_program_trace[-1]["new_program_subsample_scores"] == [1.0, 0.0]
    assert "rlm_merge_reject_reason" not in state.full_program_trace[-1]


def test_rlm_merge_paired_trace_file_contains_rich_records(tmp_path: Path):
    proposer = _make_merge_helper(tmp_path)
    proposer.adapter = SimpleNamespace(proposer_trace_dir=tmp_path, run_id="run_test")
    eval_a = SimpleNamespace(scores=[1.0, 0.0])
    eval_b = SimpleNamespace(scores=[0.0, 1.0])

    path = proposer._write_paired_trace_file(
        id1=1,
        id2=2,
        ancestor=0,
        attempt_idx=0,
        task_data_ids=["data_0", "data_1"],
        trace_task_ids=["task_a", "task_b"],
        reflective_a=[
            {
                "Inputs": "Instruction: do A",
                "Generated Outputs": "parent A trace full reasoning + code",
                "Feedback": "Task score: 1.000",
            },
            {
                "Inputs": "Instruction: do B",
                "Generated Outputs": "parent A trace for task_b, failed",
                "Feedback": "Task score: 0.000",
            },
        ],
        reflective_b=[
            {
                "Inputs": "Instruction: do A",
                "Generated Outputs": "parent B trace, failed",
                "Feedback": "Task score: 0.000",
            },
            {
                "Inputs": "Instruction: do B",
                "Generated Outputs": "parent B trace, worked",
                "Feedback": "Task score: 1.000",
            },
        ],
        eval_a=eval_a,
        eval_b=eval_b,
        eps=1e-6,
    )

    records = [json.loads(line) for line in Path(path).read_text().splitlines()]

    assert records[0]["winner"] == "a"
    assert records[1]["winner"] == "b"
    assert records[0]["task_id"] == "task_a"
    assert records[0]["parent_a"]["generated_outputs"].startswith("parent A trace")
    assert records[0]["parent_b"]["feedback"] == "Task score: 0.000"
    assert "trajectory_summary" not in records[0]["parent_a"]


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
                "id1_subsample_scores": [0.0, 1.0],
                "id2_subsample_scores": [1.0, 0.0],
                "new_program_subsample_scores": [1.0, 1.0],
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
    assert rows[0]["hard: par → child"] == "0.500 → 1.000 +0.500; 1 → 2 /2"
    assert rows[0]["flips"] == "+1/-0 +1"
    assert rows[0]["p"] == "1.00"
    assert rows[0]["iter"] == "0 [0]"
    assert rows[0]["_highlight"] is True
    assert rows[1]["iter"] == "1 [0, 1]"
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
    assert "| iter" in rendered
    assert "| soft: par → child" in rendered
    assert "| hard: par → child" in rendered
    assert "| cand [par]" in rendered
    assert "| Δ-seed" in rendered
    assert "| ----" in rendered
    assert "**1 [0]**" in rendered
    terminal = render_stats(tmp_path)
    assert "┌" in terminal
    assert "\033[3m" in terminal
    assert "\033[38;5;248m" in terminal
    assert "\033[38;5;248m0.500 → 1.000\033[0m\033[1;38;5;220m +0.500" in terminal
    assert "\033[38;5;248m0.500 → 1.000\033[0m\033[1;38;5;220m +0.500; 1 → 2 /2" in terminal
    assert "\033[38;5;248m+1/-0\033[0m\033[1;38;5;220m +1" in terminal
    assert "\033[1;38;5;220m" in terminal
    assert "**1**" not in terminal
    assert "costs:" in terminal
    assert "total_cost" in terminal
    assert "repeat_cost" in terminal
    assert "effective_cost" in terminal
    assert "costs (raw spend: all logged LM calls):" not in terminal
    assert "costs (deduped spend: stable operation ids only; legacy rows counted raw):" not in terminal


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


def test_merge_proposer_uses_configured_proposer_lms(tmp_path: Path, monkeypatch):
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
                new_instructions="merged instructions",
                generalization_check=["task-a: combines both parents"],
                trace=None,
                trajectory=[],
            )

    monkeypatch.setattr(adapter_module, "PredictRLM", FakePredictRLM)
    monkeypatch.setattr(adapter_module, "progress_write", lambda _message: None)
    monkeypatch.setattr(proposer_module, "progress_write", lambda _message: None)
    (tmp_path / "proposer_traces").mkdir()
    paired_trace = tmp_path / "paired.jsonl"
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

    new_text, audit = adapter._rlm_propose_merge_texts(
        call_idx=3,
        attempt_idx=2,
        id1=10,
        id2=11,
        ancestor=4,
        current_instructions_a="parent a",
        current_instructions_b="parent b",
        common_ancestor_instructions="ancestor",
        paired_traces_file=SimpleNamespace(path=str(paired_trace)),
        trace_task_ids=["task-a"],
    )

    assert new_text == "merged instructions"
    assert audit == ["task-a: combines both parents"]
    assert captured["lm"] is proposer_lm
    assert captured["sub_lm"] is proposer_sub_lm
    assert captured["max_iterations"] == 17
    assert captured["skills"] == []
