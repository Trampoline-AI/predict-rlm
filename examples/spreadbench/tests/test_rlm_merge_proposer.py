"""Unit tests for the RLM merge proposer pieces that live behind the
pair-selection gate.

The smoke run on 2026-04-22 hit a bug where _write_paired_trace_file
wrote thin summary stubs instead of the rich Generated Outputs /
Feedback text the RLM needs for synthesis. The bug didn't manifest
because the pair selector kept returning None on a 2-candidate pool
(pair_skipped) — step 9 (trace writing) lives behind step 4 (pair
selection). A unit test for _write_paired_trace_file would have caught
this before launch, so these tests now cover:

  - pick_merge_triplet: None on insufficient candidates; real triplet
    on Pareto + mutual-divergence setup; dedup against merges_performed
  - walk_ancestors: BFS correctness on a linear + branching parent chain
  - _record_merge_status: enum enforcement + rlm_merge_ namespace check
  - _structural_new_audit_check: rejects [NEW] without citation; rejects
    unknown task_id citations; accepts valid citations
  - _write_paired_trace_file: uses adapter.make_reflective_dataset rich
    fields, NOT summary stubs (the regression the smoke caught)
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import pytest

from lib.merge_selection import pick_merge_triplet, walk_ancestors
from lib.rlm_merge_proposer import (
    RlmMergeProposer,
    VALID_STATUSES,
)


# ---------------------------------------------------------------------------
# walk_ancestors
# ---------------------------------------------------------------------------


def test_walk_ancestors_linear():
    # 0 (seed) <- 1 <- 2 <- 3
    parents = [[None], [0], [1], [2]]
    assert walk_ancestors(parents, 3) == {0, 1, 2}
    assert walk_ancestors(parents, 2) == {0, 1}
    assert walk_ancestors(parents, 1) == {0}
    assert walk_ancestors(parents, 0) == set()


def test_walk_ancestors_branching():
    # seed 0 <- 1 <- 2
    #       \- 3
    #           \- 4
    parents = [[None], [0], [1], [0], [3]]
    assert walk_ancestors(parents, 2) == {0, 1}
    assert walk_ancestors(parents, 4) == {0, 3}
    # Descendants of seed that share seed as common ancestor:
    # cand 2 and cand 4 → common = {0}
    assert walk_ancestors(parents, 2) & walk_ancestors(parents, 4) == {0}


# ---------------------------------------------------------------------------
# pick_merge_triplet
# ---------------------------------------------------------------------------


def _candidates(n: int, component="skill_instructions") -> list[dict]:
    """n distinct single-component candidates with unique text per index."""
    return [{component: f"cand_{i}_text"} for i in range(n)]


def test_pick_merge_triplet_none_on_too_few_candidates():
    # Only 2 Pareto-dominators → no triplet possible (no common ancestor
    # option that isn't one of them).
    result = pick_merge_triplet(
        merge_candidates=[0, 1],
        program_candidates=_candidates(2),
        parent_program_for_candidate=[[None], [0]],
        prog_candidate_val_subscores=[
            {f"t{i}": 0.5 for i in range(12)},
            {f"t{i}": 0.8 for i in range(12)},
        ],
        tracked_scores=[0.5, 0.8],
        merges_performed=[],
        rng=random.Random(0),
        component_name="skill_instructions",
        min_each=3,
    )
    assert result is None, "single non-seed candidate can't form a triplet"


def test_pick_merge_triplet_finds_divergent_pair():
    # Three candidates: seed 0, cand 1<-0, cand 2<-0 with mutual wins
    parents = [[None], [0], [0]]
    # cand 1 wins odd tasks, cand 2 wins even tasks
    s1 = {f"t{i}": 1.0 if i % 2 else 0.0 for i in range(12)}
    s2 = {f"t{i}": 0.0 if i % 2 else 1.0 for i in range(12)}
    result = pick_merge_triplet(
        merge_candidates=[1, 2],  # seed is not a Pareto-dominator in real runs
        program_candidates=_candidates(3),
        parent_program_for_candidate=parents,
        prog_candidate_val_subscores=[{}, s1, s2],
        tracked_scores=[0.3, 0.5, 0.5],  # ancestor seed dominated
        merges_performed=[],
        rng=random.Random(0),
        component_name="skill_instructions",
        min_each=3,
    )
    assert result is not None
    id1, id2, ancestor = result
    assert {id1, id2} == {1, 2}
    assert id1 < id2, "ids should be normalized id1 <= id2"
    assert ancestor == 0


def test_pick_merge_triplet_dedup_against_merges_performed():
    """A (min, max, anc) triple already in merges_performed should be
    skipped, even if we'd otherwise pick it."""
    parents = [[None], [0], [0]]
    s1 = {f"t{i}": 1.0 if i % 2 else 0.0 for i in range(12)}
    s2 = {f"t{i}": 0.0 if i % 2 else 1.0 for i in range(12)}
    merges_performed = [(1, 2, 0)]  # already attempted
    result = pick_merge_triplet(
        merge_candidates=[1, 2],
        program_candidates=_candidates(3),
        parent_program_for_candidate=parents,
        prog_candidate_val_subscores=[{}, s1, s2],
        tracked_scores=[0.3, 0.5, 0.5],
        merges_performed=merges_performed,
        rng=random.Random(0),
        component_name="skill_instructions",
        min_each=3,
    )
    assert result is None, "already-attempted triplet must be skipped"


def test_pick_merge_triplet_skips_text_identical_pair():
    """Even if val scores differ, candidates with identical text on the
    target component are not worth merging."""
    parents = [[None], [0], [0]]
    identical_candidates = [
        {"skill_instructions": "seed"},
        {"skill_instructions": "identical_text"},
        {"skill_instructions": "identical_text"},  # same text as cand 1
    ]
    s1 = {f"t{i}": 1.0 if i % 2 else 0.0 for i in range(12)}
    s2 = {f"t{i}": 0.0 if i % 2 else 1.0 for i in range(12)}
    result = pick_merge_triplet(
        merge_candidates=[1, 2],
        program_candidates=identical_candidates,
        parent_program_for_candidate=parents,
        prog_candidate_val_subscores=[{}, s1, s2],
        tracked_scores=[0.3, 0.5, 0.5],
        merges_performed=[],
        rng=random.Random(0),
        component_name="skill_instructions",
        min_each=3,
    )
    assert result is None, "text-identical pair must be skipped"


def test_pick_merge_triplet_mutual_wins_threshold():
    """If one side dominates (no mutual wins), return None. Asymmetric
    divergence is a reflective-proposer case, not a merge case."""
    parents = [[None], [0], [0]]
    # cand 1 wins everything, cand 2 wins nothing
    s1 = {f"t{i}": 1.0 for i in range(12)}
    s2 = {f"t{i}": 0.0 for i in range(12)}
    result = pick_merge_triplet(
        merge_candidates=[1, 2],
        program_candidates=_candidates(3),
        parent_program_for_candidate=parents,
        prog_candidate_val_subscores=[{}, s1, s2],
        tracked_scores=[0.3, 0.5, 0.3],
        merges_performed=[],
        rng=random.Random(0),
        component_name="skill_instructions",
        min_each=3,
    )
    assert result is None, "asymmetric dominance must be skipped"


# ---------------------------------------------------------------------------
# _record_merge_status
# ---------------------------------------------------------------------------


def _make_proposer_stub(tmp_path):
    """Create an instance with minimal state for helper methods only."""
    # Skip __init__ because it wants an adapter etc.; we're only testing
    # pure helpers that don't touch self.adapter/trainset.
    instance = RlmMergeProposer.__new__(RlmMergeProposer)
    instance.rlm_merge_state_path = tmp_path / "rlm_merge_state.json"
    instance.rlm_merge_attempts_used = 0
    instance.merges_performed = ([], [])
    instance._min_each = 3
    return instance


def test_record_merge_status_valid_enum():
    inst = _make_proposer_stub(Path("/tmp"))
    state = SimpleNamespace(full_program_trace=[{"i": 0}])
    inst._record_merge_status(state, "pair_skipped", attempt_idx=None)
    entry = state.full_program_trace[-1]
    assert entry["rlm_merge_status"] == "pair_skipped"
    assert entry["rlm_merge_attempt_idx"] is None
    assert entry["rlm_merge_considered"] is True  # only for pair_skipped / cap


def test_record_merge_status_invalid_enum_raises():
    inst = _make_proposer_stub(Path("/tmp"))
    state = SimpleNamespace(full_program_trace=[{"i": 0}])
    with pytest.raises(ValueError, match="invalid merge status"):
        inst._record_merge_status(state, "not_a_real_status", attempt_idx=None)


def test_record_merge_status_rejects_unnamespaced_kwargs():
    inst = _make_proposer_stub(Path("/tmp"))
    state = SimpleNamespace(full_program_trace=[{"i": 0}])
    with pytest.raises(ValueError, match="must start with 'rlm_merge_'"):
        inst._record_merge_status(
            state, "accepted", attempt_idx=0, bad_field_name="value",
        )


def test_record_merge_status_accepts_namespaced_kwargs():
    inst = _make_proposer_stub(Path("/tmp"))
    state = SimpleNamespace(full_program_trace=[{"i": 0}])
    inst._record_merge_status(
        state,
        "accepted",
        attempt_idx=0,
        rlm_merge_candidate_pair=(1, 2),
        rlm_merge_new_sum=5.5,
    )
    entry = state.full_program_trace[-1]
    assert entry["rlm_merge_candidate_pair"] == (1, 2)
    assert entry["rlm_merge_new_sum"] == 5.5


def test_all_valid_statuses_in_enum():
    """Smoke-check: the 9 statuses the propose flow uses all exist in
    VALID_STATUSES. Catches drift if propose adds a new status but forgets
    to whitelist it."""
    expected = {
        "attempt_cap_exhausted", "pair_skipped", "preflight_failed",
        "structural_rejected", "duplicate", "length_rejected",
        "subsample_rejected", "accepted", "error",
    }
    assert VALID_STATUSES == frozenset(expected)


# ---------------------------------------------------------------------------
# _structural_new_audit_check
# ---------------------------------------------------------------------------


def test_new_audit_check_accepts_valid_citation():
    inst = _make_proposer_stub(Path("/tmp"))
    lines = [
        "[COMMON] rule one | api: ... | use case: ...",
        "[NEW] new rule citing task_id=42 | api: ... | ...",
    ]
    ok, reason = inst._structural_new_audit_check(lines, {"42", "17"})
    assert ok is True
    assert reason == ""


def test_new_audit_check_rejects_missing_citation():
    inst = _make_proposer_stub(Path("/tmp"))
    lines = [
        "[NEW] new rule without a task_id anywhere",
    ]
    ok, reason = inst._structural_new_audit_check(lines, {"42"})
    assert ok is False
    assert "missing task_id citation" in reason


def test_new_audit_check_rejects_unknown_citation():
    inst = _make_proposer_stub(Path("/tmp"))
    lines = [
        "[NEW] rule citing task_id=999 | ...",
    ]
    ok, reason = inst._structural_new_audit_check(lines, {"42", "17"})
    assert ok is False
    assert "999" in reason


def test_new_audit_check_ignores_non_new_lines():
    """Only [NEW] lines need citations. [COMMON], [A_ONLY], [B_ONLY],
    [MERGED] lines are allowed without task_id."""
    inst = _make_proposer_stub(Path("/tmp"))
    lines = [
        "[COMMON] inherited rule | ...",
        "[A_ONLY] kept from A | ...",
        "[B_ONLY] kept from B | ...",
        "[MERGED] reconciled | ...",
    ]
    ok, reason = inst._structural_new_audit_check(lines, set())
    assert ok is True
    assert reason == ""


# ---------------------------------------------------------------------------
# _write_paired_trace_file — the regression the smoke run caught
# ---------------------------------------------------------------------------


def test_paired_trace_file_contains_rich_reflective_fields(tmp_path):
    """Regression test for the 2026-04-22 bug: _write_paired_trace_file
    must write per-record "Generated Outputs" and "Feedback" from
    adapter.make_reflective_dataset, NOT just summary stubs
    (num_passed/num_total). The thin-trace bug would have fed the merge
    RLM insufficient context to synthesize anything.
    """
    inst = RlmMergeProposer.__new__(RlmMergeProposer)
    # Minimal adapter stub: has the proposer_trace_dir attr and a
    # make_reflective_dataset method that returns rich records.
    inst.adapter = SimpleNamespace(
        proposer_trace_dir=str(tmp_path),
        make_reflective_dataset=lambda cand, eval_batch, comps: {
            comps[0]: [
                {
                    "Inputs": "Instruction type: X\n\nInstruction: do Y",
                    "Generated Outputs": "banner\n\nstep-by-step trace text",
                    "Feedback": "Task score: 0.500\n\nPer-case: PASS / FAIL mismatches",
                }
                for _ in range(eval_batch.trajectories or [])
            ]
        },
    )
    trace_task_ids = ["task_a", "task_b"]
    # Build mock eval batches — reflective_a/b will be passed directly
    eval_a = SimpleNamespace(
        scores=[1.0, 0.0],
        trajectories=[{"task_id": "task_a"}, {"task_id": "task_b"}],
    )
    eval_b = SimpleNamespace(
        scores=[0.0, 1.0],
        trajectories=[{"task_id": "task_a"}, {"task_id": "task_b"}],
    )
    reflective_a = [
        {
            "Inputs": "Instruction type: A\n\nInstruction: do X",
            "Generated Outputs": "parent A trace full reasoning + code",
            "Feedback": "Task score: 1.000 | per-case pass",
        },
        {
            "Inputs": "Instruction type: B\n\nInstruction: do Z",
            "Generated Outputs": "parent A trace for task_b, failed",
            "Feedback": "Task score: 0.000 | per-case fail mismatch at B5",
        },
    ]
    reflective_b = [
        {
            "Inputs": "Instruction type: A\n\nInstruction: do X",
            "Generated Outputs": "parent B trace, different approach",
            "Feedback": "Task score: 0.000 | per-case fail",
        },
        {
            "Inputs": "Instruction type: B\n\nInstruction: do Z",
            "Generated Outputs": "parent B trace, worked",
            "Feedback": "Task score: 1.000 | all cases pass",
        },
    ]

    path = inst._write_paired_trace_file(
        id1=1,
        id2=2,
        ancestor=0,
        attempt_idx=0,
        task_data_ids=["data_0", "data_1"],
        trace_task_ids=trace_task_ids,
        reflective_a=reflective_a,
        reflective_b=reflective_b,
        eval_a=eval_a,
        eval_b=eval_b,
        eps=1e-6,
    )

    records = [
        json.loads(line) for line in Path(path).read_text().splitlines() if line
    ]
    assert len(records) == 2

    # REGRESSION ASSERTS: generated_outputs + feedback must be non-empty
    # and must be the actual rich rendering, not summary stubs.
    for rec in records:
        for side in ("parent_a", "parent_b"):
            assert "generated_outputs" in rec[side], (
                f"record missing {side}.generated_outputs — "
                "the thin-trace bug would have produced "
                "'trajectory_summary' with just num_passed/num_total"
            )
            assert "feedback" in rec[side]
            assert "trajectory_summary" not in rec[side], (
                "regression: old thin-trace code used 'trajectory_summary' "
                "key; new code must emit 'generated_outputs' + 'feedback'"
            )
            assert rec[side]["generated_outputs"], "generated_outputs empty"
            assert rec[side]["feedback"], "feedback empty"

    # Winner field is derived correctly
    assert records[0]["winner"] == "a"  # A=1.0, B=0.0
    assert records[1]["winner"] == "b"  # A=0.0, B=1.0
    # task_id used for citations
    assert records[0]["task_id"] == "task_a"
    assert records[1]["task_id"] == "task_b"
    # inputs carried through from the rich record
    assert records[0]["inputs"].startswith("Instruction type:")
