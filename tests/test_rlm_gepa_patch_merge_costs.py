from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from predict_rlm.trace import (
    LMUsage,
    PredictCallDetail,
    PredictCallGroup,
    RunTrace,
    TokenUsage,
)
from rlm_gepa import AgentSpec, RLMGepaProject
from rlm_gepa.reporting.cost import CostRow, append_cost_rows
from rlm_gepa.reporting.stats import cost_rows
from rlm_gepa.runtime.adapter import RLMGepaAdapter

pytestmark = pytest.mark.gepa


class _DummyLM:
    model = "dummy/model"


class _Project(RLMGepaProject):
    project_name = "test-project"
    components = ("skill_instructions",)
    agent_spec = AgentSpec(
        agent_type="test agent",
        use_cases=["case a", "case b"],
        runtime_grounding_examples={"tools": ["tool()"], "env": ["sandbox"], "spec": ["protocol"]},
        tool_signatures="tool() -> str",
        target_signature="input: str -> output: str",
        scoring_description="exact match",
    )

    def seed_candidate(self) -> dict[str, str]:
        return {"skill_instructions": "seed"}

    def load_trainset(self) -> list[str]:
        return ["train"]

    def load_valset(self) -> list[str]:
        return ["val"]

    async def evaluate_example(self, candidate, example, context):
        raise NotImplementedError


def _trace_with_proposer_usage() -> RunTrace:
    return RunTrace(
        status="completed",
        model="dummy-main",
        sub_model="dummy-sub",
        iterations=1,
        max_iterations=1,
        duration_ms=10,
        usage=LMUsage(
            main=TokenUsage(input_tokens=100, output_tokens=20, cost=0.03),
            sub=TokenUsage(input_tokens=50, output_tokens=10, cost=0.02),
        ),
        steps=[
            {
                "iteration": 1,
                "reasoning": "use helper",
                "code": "predict()",
                "output": "ok",
                "untruncated_output": "ok",
                "duration_ms": 10,
                "predict_calls": [
                    PredictCallGroup(
                        signature="Helper",
                        model="dummy-sub",
                        calls=[
                            PredictCallDetail(
                                duration_ms=5,
                                usage=TokenUsage(input_tokens=50, output_tokens=10, cost=0.02),
                            )
                        ],
                    )
                ],
            }
        ],
    )


def test_patch_merge_proposer_logs_merge_proposer_cost_roles(tmp_path: Path, monkeypatch):
    import rlm_gepa.runtime.adapter as adapter_module

    class FakePredictRLM:
        def __init__(self, *_args, **_kwargs):
            pass

        async def acall(self, **_kwargs):
            return SimpleNamespace(
                base_parent_id=1,
                patch_summary="patched",
                selected_capability={
                    "decision": "grafted",
                    "summary": "capability",
                    "evidence_task_ids": ["train-a"],
                    "trigger": "task requires the source parent's missing capability",
                    "non_application_boundary": (
                        "do not apply on base-win or both-success rows that already solve it"
                    ),
                },
                patch_audit={
                    "supported_source_win_ids": ["train-a"],
                    "guardrail_hazards": [],
                    "notes": "base lacks the selected facet",
                },
                new_instructions="patched instructions",
                trace=_trace_with_proposer_usage(),
                trajectory=[],
            )

    monkeypatch.setattr(adapter_module, "PredictRLM", FakePredictRLM)
    monkeypatch.setattr(adapter_module, "progress_write", lambda _message: None)
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
        proposer_lm=_DummyLM(),
        proposer_sub_lm=_DummyLM(),
        proposer_max_iterations=1,
    )

    adapter._rlm_propose_patch_merge_texts(
        call_idx=1,
        attempt_idx=0,
        base_parent_id=1,
        patch_source_parent_id=2,
        base_parent_instructions="base",
        patch_source_parent_instructions="source",
        paired_disagreement_traces_file=SimpleNamespace(path=str(paired_trace)),
        trace_task_ids=["train-a"],
    )

    cost_log = [json.loads(line) for line in (tmp_path / "cost_log.jsonl").read_text().splitlines()]
    assert [row["role"] for row in cost_log] == ["merge_proposer", "merge_proposer_sub_lm"]
    assert [row["cost_usd"] for row in cost_log] == [0.03, 0.02]

    rows = cost_rows(tmp_path)
    assert any(row.get("scope") == "merge" and row.get("_category") for row in rows)
    assert any(
        row.get("scope") == "  - proposer main" and row.get("total_cost") == "$0.03"
        for row in rows
    )
    assert any(
        row.get("scope") == "  - proposer sub" and row.get("total_cost") == "$0.02"
        for row in rows
    )
    assert not any(row.get("scope") == "patch-merge" for row in rows)


def test_cost_rows_group_legacy_patch_merge_roles_under_merge_proposer(tmp_path: Path):
    append_cost_rows(
        tmp_path / "cost_log.jsonl",
        [
            CostRow(
                event_id="e",
                operation_id="op",
                attempt_id="a",
                event="patch_merge_proposer_call",
                role="patch_merge_proposer",
                model="dummy-main",
                calls=1,
                input_tokens=10,
                output_tokens=2,
                cost_usd=0.01,
            ),
            CostRow(
                event_id="e",
                operation_id="op",
                attempt_id="a",
                event="patch_merge_proposer_call",
                role="patch_merge_proposer_sub_lm",
                model="dummy-sub",
                calls=1,
                input_tokens=11,
                output_tokens=3,
                cost_usd=0.02,
            ),
        ],
    )

    rows = cost_rows(tmp_path)

    assert any(row.get("scope") == "merge" and row.get("_category") for row in rows)
    assert any(
        row.get("scope") == "  - proposer main" and row.get("model") == "dummy-main"
        for row in rows
    )
    assert any(
        row.get("scope") == "  - proposer sub" and row.get("model") == "dummy-sub"
        for row in rows
    )
    assert not any(row.get("scope") == "patch-merge" for row in rows)
