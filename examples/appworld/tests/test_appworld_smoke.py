import argparse
import asyncio
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from appworld_rlm import AppWorldRLM
from appworld_rlm.agent import service as service_module
from appworld_rlm.agent.skills import (
    APPWORLD_SKILL_INSTRUCTIONS,
    load_official_icl_demo_task_ids,
    render_official_icl_demos,
)
from appworld_rlm.bench import cli as bench_cli
from appworld_rlm.bench import evaluation
from appworld_rlm.bench.config import EvalConfig
from appworld_rlm.bench.dataset import load_dataset, load_train_validation
from appworld_rlm.bench.scoring import score_prediction_result, score_runner_result
from appworld_rlm.gepa import cli as gepa_cli
from appworld_rlm.gepa import project as gepa_project_module
from appworld_rlm.gepa.config import APPWORLD_SPEC, AppWorldGepaConfig
from appworld_rlm.gepa.project import COMPONENT_SKILL, AppWorldGepaProject
from appworld_rlm.tools.appworld_worker import (
    JsonlAppWorldWorker,
    _appworld_root_from_data_root,
    to_jsonable,
)
from appworld_rlm.tools.runner import (
    AppWorldSessionClient,
    RunnerResult,
    _default_appworld_python,
)

from appworld_rlm.bench.cli import appworld_eval_header_summary
from rlm_gepa.reporting.stats import render_stats
from rlm_gepa.schema import EvaluationContext, validate_project

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "appworld_data"


def test_service_constructs():
    service = AppWorldRLM(max_iterations=3, verbose=False)
    assert service.max_iterations == 3


def test_official_icl_demo_asset_loads_and_renders(tmp_path):
    manifest = load_official_icl_demo_task_ids()
    task_ids = manifest["demo_task_ids"]

    assert task_ids == ["82e2fac_1", "29caf6f_1", "d0b1f43_1"]
    assert "demos" not in manifest
    assert "official_appworld_demo_task_ids" == manifest["source"]["type"]

    data_root = tmp_path / "appworld_data"
    for task_id in task_ids:
        task_dir = data_root / "tasks" / task_id
        task_dir.mkdir(parents=True)
        (task_dir / "specs.json").write_text(
            json.dumps({"instruction": f"fixture instruction for {task_id}"})
        )

    rendered = render_official_icl_demos(manifest, data_root=data_root)
    for task_id in task_ids:
        assert f"official demo task ID `{task_id}`" in rendered
        assert f"fixture instruction for {task_id}" in rendered
    assert "data/tasks/<task_id>/specs.json" not in rendered
    assert "canary" not in rendered.lower()
    assert "private data" in rendered.lower()
    assert "Stock task-ID examples" in APPWORLD_SKILL_INSTRUCTIONS


def test_load_official_split_files():
    train = load_dataset("train", FIXTURE_ROOT)
    test = load_dataset("test_normal", FIXTURE_ROOT)
    assert train[0].task_id == "aaa111_1"
    assert test[0].task_id == "fff666_1"


def test_load_dataset_reads_task_instruction_from_specs_json():
    train = load_dataset("train", FIXTURE_ROOT)

    assert train[0].instruction == "Use the fixture apps to complete task aaa111_1."
    assert train[1].task_id == "aaa111_2"
    assert train[1].instruction == ""


def test_train_validation_split_is_group_disjoint_and_deterministic():
    train_a, val_a = load_train_validation(FIXTURE_ROOT, val_ratio=0.25, seed=7)
    train_b, val_b = load_train_validation(FIXTURE_ROOT, val_ratio=0.25, seed=7)
    assert train_a == train_b
    assert val_a == val_b
    assert val_a
    assert {item.group_id for item in train_a}.isdisjoint({item.group_id for item in val_a})


def test_score_runner_result_feedback():
    score, feedback = score_runner_result({"score": 0.25, "feedback": "missing email"})
    assert score == 0.25
    assert "missing email" in feedback


def test_runner_result_tool_text_round_trip():
    text = RunnerResult(
        task_id="aaa111_1",
        success=True,
        score=1.0,
        stdout="ok",
        stderr="",
        feedback="done",
    ).to_tool_text()
    assert score_runner_result(text) == (1.0, "done")


def test_worker_maps_data_root_to_appworld_root():
    assert _appworld_root_from_data_root("data") == "."
    assert _appworld_root_from_data_root("/tmp/appworld/data") == "/tmp/appworld"


def test_worker_to_jsonable_handles_nested_objects():
    class MunchLike:
        def to_dict(self):
            return {"items": [SimpleNamespace(value=3)]}

    assert to_jsonable({"outer": MunchLike()}) == {"outer": {"items": [{"value": 3}]}}


def test_default_appworld_python_prefers_local_runtime(tmp_path, monkeypatch):
    appworld_python = tmp_path / ".appworld-venv" / "bin" / "python"
    appworld_python.parent.mkdir(parents=True)
    appworld_python.write_text("")
    monkeypatch.chdir(tmp_path)

    assert _default_appworld_python() == str(appworld_python)


def test_session_client_call_api_rejects_non_object_kwargs():
    client = AppWorldSessionClient()

    text = client.call_appworld_api("aaa111_1", "venmo", "search_friends", "[]")

    payload = __import__("json").loads(text)
    assert payload["success"] is False
    assert "JSON object" in payload["feedback"]


def test_session_client_formats_direct_tool_response(monkeypatch):
    client = AppWorldSessionClient()
    monkeypatch.setattr(client, "_ensure_task", lambda task_id: None)
    monkeypatch.setattr(
        client,
        "request",
        lambda payload: {
            "task_id": payload["task_id"],
            "session_id": payload["session_id"],
            "operation": payload["op"],
            "success": True,
            "result": {"apps": ["venmo"]},
            "stdout": "ok",
            "stderr": "",
            "feedback": "",
        },
    )

    payload = __import__("json").loads(client.list_appworld_apps("aaa111_1"))

    assert payload["operation"] == "list_apps"
    assert payload["result"] == {"apps": ["venmo"]}


def test_direct_api_call_persists_before_evaluate():
    class FakeApi:
        def __init__(self, world):
            self.world = world

        def mutate_friendship(self):
            self.world.mutated = True
            return {"message": "changed"}

    class FakeWorld:
        task_id = "aaa111_1"
        output_db_home_path_on_disk = "/tmp/appworld-output"

        def __init__(self):
            self.mutated = False
            self.saved = False
            self.logs_saved = False
            self.apis = SimpleNamespace(venmo=FakeApi(self))

        def _save_state(self, output_db_home_path):
            assert output_db_home_path == self.output_db_home_path_on_disk
            self.saved = True

        def save_logs(self):
            self.logs_saved = True

        def evaluate(self):
            return {
                "success": self.mutated and self.saved,
                "score": 1.0 if self.mutated and self.saved else 0.0,
                "feedback": "persisted" if self.saved else "not persisted",
            }

    worker = JsonlAppWorldWorker()
    world = FakeWorld()
    worker._sessions["aaa111_1"] = {
        "task_id": "aaa111_1",
        "world": world,
        "manager": SimpleNamespace(__exit__=lambda *_args: None),
    }

    call_response = worker.handle(
        {
            "op": "call_api",
            "task_id": "aaa111_1",
            "session_id": "aaa111_1",
            "app_name": "venmo",
            "api_name": "mutate_friendship",
            "kwargs": {},
        }
    )
    eval_response = worker.handle(
        {"op": "evaluate_task", "task_id": "aaa111_1", "session_id": "aaa111_1"}
    )

    assert call_response["success"] is True
    assert world.saved is True
    assert world.logs_saved is True
    assert eval_response["success"] is True
    assert eval_response["score"] == 1.0


def test_score_prediction_result_uses_last_runner_tool_result():
    prediction = SimpleNamespace(
        final_answer="looks good",
        trace=SimpleNamespace(
            steps=[
                SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            name="run_appworld_program",
                            result=RunnerResult(
                                task_id="aaa111_1",
                                success=False,
                                score=0.25,
                                stdout="",
                                stderr="",
                                feedback="missing email",
                            ).to_tool_text(),
                            error=None,
                        )
                    ]
                )
            ]
        ),
    )

    score, feedback = score_prediction_result(prediction)

    assert score == 0.25
    assert "missing email" in feedback


def test_score_prediction_result_uses_evaluate_appworld_task_result():
    prediction = SimpleNamespace(
        final_answer="looks good",
        trace=SimpleNamespace(
            steps=[
                SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            name="evaluate_appworld_task",
                            result=RunnerResult(
                                task_id="aaa111_1",
                                success=True,
                                score=1.0,
                                stdout="",
                                stderr="",
                                feedback="done",
                            ).to_tool_text(),
                            error=None,
                        )
                    ]
                )
            ]
        ),
    )

    score, feedback = score_prediction_result(prediction)

    assert score == 1.0
    assert feedback == "done"


def test_score_prediction_result_rejects_final_answer_without_runner_tool():
    prediction = SimpleNamespace(final_answer="done", trace=SimpleNamespace(steps=[]))

    score, feedback = score_prediction_result(prediction)

    assert score == 0.0
    assert "did not call evaluate_appworld_task or run_appworld_program" in feedback


class _AutoCompleteClient:
    def __init__(self):
        self.calls = []

    def list_appworld_apps(self, task_id):
        return task_id

    def show_appworld_api_descriptions(self, task_id, app_name):
        return task_id + app_name

    def show_appworld_api_doc(self, task_id, app_name, api_name):
        return task_id + app_name + api_name

    def search_appworld_api_docs(self, task_id, query):
        return task_id + query

    def call_appworld_api(self, task_id, app_name, api_name, kwargs_json):
        self.calls.append((task_id, app_name, api_name, kwargs_json))
        return json.dumps({"success": True, "feedback": "completed"})

    def close_appworld_task(self, task_id):
        return task_id

    def run_appworld_program(self, task_id, program):
        return task_id + program


def _run_appworld_rlm_with_prediction(monkeypatch, prediction, client):
    class FakePredictRLM:
        def __init__(self, *_args, **_kwargs):
            pass

        async def acall(self, **_kwargs):
            return prediction

    monkeypatch.setattr(service_module, "PredictRLM", FakePredictRLM)
    agent = AppWorldRLM(appworld_client=client)
    return asyncio.run(agent.aforward(task_id="aaa111_1", instruction="do it"))


def test_appworld_rlm_completes_task_from_final_answer(monkeypatch):
    client = _AutoCompleteClient()

    result = _run_appworld_rlm_with_prediction(
        monkeypatch,
        SimpleNamespace(final_answer="foo"),
        client,
    )

    assert result.final_answer == "foo"
    assert client.calls == [
        ("aaa111_1", "supervisor", "complete_task", json.dumps({"answer": "foo"}))
    ]


def test_appworld_rlm_completes_task_without_answer_for_empty_submission(monkeypatch):
    for final_answer in ("null", " NULL ", "none", " None ", None):
        client = _AutoCompleteClient()

        _run_appworld_rlm_with_prediction(
            monkeypatch,
            SimpleNamespace(final_answer=final_answer),
            client,
        )

        assert client.calls == [("aaa111_1", "supervisor", "complete_task", "{}")]


def test_appworld_rlm_does_not_double_complete_after_successful_trace_call(monkeypatch):
    client = _AutoCompleteClient()
    prediction = SimpleNamespace(
        final_answer="foo",
        trace=SimpleNamespace(
            steps=[
                SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            name="call_appworld_api",
                            args=[
                                "supervisor",
                                "complete_task",
                                json.dumps({"answer": "foo"}),
                            ],
                            kwargs={},
                            result=json.dumps({"success": True}),
                            error=None,
                        )
                    ]
                )
            ]
        ),
    )

    _run_appworld_rlm_with_prediction(monkeypatch, prediction, client)

    assert client.calls == []


def test_service_binds_current_task_appworld_tools(monkeypatch):
    captured_tools = {}

    class FakeClient:
        def __init__(self):
            self.calls = []

        def list_appworld_apps(self, task_id):
            self.calls.append(("list_appworld_apps", task_id))
            return json.dumps({"success": True, "result": {"apps": ["venmo"]}})

        def show_appworld_api_descriptions(self, task_id, app_name):
            self.calls.append(("show_appworld_api_descriptions", task_id, app_name))
            return json.dumps({"success": True, "result": {"search": "Search Venmo friends."}})

        def show_appworld_api_doc(self, task_id, app_name, api_name):
            self.calls.append(("show_appworld_api_doc", task_id, app_name, api_name))
            return json.dumps({"success": True, "result": {"api_name": api_name}})

        def search_appworld_api_docs(self, task_id, query):
            self.calls.append(("search_appworld_api_docs", task_id, query))
            return json.dumps({"success": True, "result": ["venmo.search"]})

        def call_appworld_api(self, task_id, app_name, api_name, kwargs_json):
            self.calls.append((task_id, app_name, api_name, kwargs_json))
            if app_name == "supervisor" and api_name == "complete_task":
                return json.dumps({"success": True})
            return task_id + app_name + api_name + kwargs_json

        def evaluate_appworld_task(self, task_id):
            return task_id

        def close_appworld_task(self, task_id):
            return task_id

        def run_appworld_program(self, task_id, program):
            return task_id + program

    class FakePredictRLM:
        def __init__(self, _signature, *, skills, **_kwargs):
            captured_tools.update(skills[0].tools)

        async def acall(self, **_kwargs):
            return SimpleNamespace(final_answer="done")

    monkeypatch.setattr(service_module, "PredictRLM", FakePredictRLM)
    client = FakeClient()
    agent = AppWorldRLM(appworld_client=client)

    asyncio.run(agent.aforward(task_id="aaa111_1", instruction="do it"))

    assert {
        "list_appworld_apps",
        "show_appworld_api_descriptions",
        "show_appworld_api_doc",
        "search_appworld_api_docs",
        "call_appworld_api",
    } == set(captured_tools)
    assert captured_tools["list_appworld_apps"]() == json.dumps(
        {"success": True, "result": {"apps": ["venmo"]}}
    )
    assert captured_tools["show_appworld_api_descriptions"]("venmo") == json.dumps(
        {"success": True, "result": {"search": "Search Venmo friends."}}
    )
    assert captured_tools["show_appworld_api_doc"]("venmo", "search") == json.dumps(
        {"success": True, "result": {"api_name": "search"}}
    )
    assert captured_tools["search_appworld_api_docs"]("friends") == json.dumps(
        {"success": True, "result": ["venmo.search"]}
    )
    assert captured_tools["call_appworld_api"](
        "venmo",
        "search",
        '{"query": "alice", "page_index": 2}',
    ) == 'aaa111_1venmosearch{"query": "alice", "page_index": 2}'
    assert client.calls[-5:] == [
        ("list_appworld_apps", "aaa111_1"),
        ("show_appworld_api_descriptions", "aaa111_1", "venmo"),
        ("show_appworld_api_doc", "aaa111_1", "venmo", "search"),
        ("search_appworld_api_docs", "aaa111_1", "friends"),
        ("aaa111_1", "venmo", "search", '{"query": "alice", "page_index": 2}'),
    ]
    assert "close_appworld_task" not in captured_tools
    assert "evaluate_appworld_task" not in captured_tools
    assert "run_appworld_program" not in captured_tools
    assert "venmo__search" not in captured_tools


def test_gepa_project_validates_with_fixture_data():
    config = AppWorldGepaConfig(data_root=FIXTURE_ROOT, val_ratio=0.25, val_limit=3)
    project = AppWorldGepaProject(config)
    validation = validate_project(project)
    assert len(validation.trainset) > len(validation.valset) > 0
    assert project.components == (COMPONENT_SKILL,)
    assert set(project.seed_candidate()) == {COMPONENT_SKILL}


def test_gepa_agent_spec_is_derived_from_clean_rlm_tools():
    for tool_name in {
        "list_appworld_apps",
        "show_appworld_api_descriptions",
        "show_appworld_api_doc",
        "search_appworld_api_docs",
        "call_appworld_api",
    }:
        assert tool_name in APPWORLD_SPEC.tool_signatures
    assert "close_appworld_task" not in APPWORLD_SPEC.tool_signatures
    assert "evaluate_appworld_task" not in APPWORLD_SPEC.tool_signatures
    assert "run_appworld_program" not in APPWORLD_SPEC.tool_signatures
    assert "venmo__search" not in APPWORLD_SPEC.tool_signatures
    assert "SolveAppWorldTask" in APPWORLD_SPEC.target_signature
    assert "harness-side" in APPWORLD_SPEC.scoring_description


def test_gepa_project_scores_from_harness_side_evaluator(monkeypatch, tmp_path):
    events = []

    class FakeAppWorldClient:
        def evaluate_appworld_task(self, task_id):
            events.append(("evaluate", task_id))
            return RunnerResult(
                task_id=task_id,
                success=True,
                score=1.0,
                stdout="",
                stderr="",
                feedback="harness score",
            ).to_tool_text()

        def close_appworld_task(self, task_id):
            events.append(("close_task", task_id))
            return ""

        def close(self):
            events.append(("close_client", None))

    class FakeAppWorldRLM:
        def __init__(self, **_kwargs):
            self.appworld_client = FakeAppWorldClient()

        async def acall(self, **kwargs):
            events.append(("agent", kwargs["task_id"]))
            return SimpleNamespace(
                trace=SimpleNamespace(
                    steps=[
                        SimpleNamespace(
                            tool_calls=[
                                SimpleNamespace(
                                    name="evaluate_appworld_task",
                                    result=RunnerResult(
                                        task_id=kwargs["task_id"],
                                        success=False,
                                        score=0.0,
                                        stdout="",
                                        stderr="",
                                        feedback="stale trace result",
                                    ).to_tool_text(),
                                    error=None,
                                )
                            ]
                        )
                    ]
                )
            )

    monkeypatch.setattr(gepa_project_module, "AppWorldRLM", FakeAppWorldRLM)
    project = AppWorldGepaProject(AppWorldGepaConfig(data_root=FIXTURE_ROOT))
    result = asyncio.run(
        project.evaluate_example(
            project.seed_candidate(),
            evaluation.AppWorldExample("aaa111_1", "train", "do it"),
            EvaluationContext(
                lm=object(),
                sub_lm=object(),
                max_iterations=3,
                task_timeout=1,
                output_dir=tmp_path,
                kind="train",
            ),
        )
    )

    assert result.score == 1.0
    assert result.feedback == "harness score"
    assert events == [
        ("agent", "aaa111_1"),
        ("evaluate", "aaa111_1"),
        ("close_task", "aaa111_1"),
        ("close_client", None),
    ]


def test_eval_builds_lms_before_constructing_appworld_rlm(monkeypatch, tmp_path):
    built_lms = {
        "openai/gpt-5.4": object(),
        "openai/gpt-5.4-mini": object(),
    }
    build_calls = []
    agent_calls = []
    eval_calls = []
    close_calls = []

    def fake_build_lm(model, *, reasoning_effort=None):
        build_calls.append((model, reasoning_effort))
        return built_lms[model]

    class FakeAppWorldRLM:
        def __init__(self, *, lm, sub_lm, max_iterations, verbose, skill, data_root):
            self.appworld_client = SimpleNamespace(
                evaluate_appworld_task=lambda task_id: eval_calls.append(task_id)
                or RunnerResult(
                    task_id=task_id,
                    success=True,
                    score=1.0,
                    stdout="",
                    stderr="",
                    feedback="ok",
                ).to_tool_text(),
                close_appworld_task=lambda task_id: close_calls.append(task_id) or "",
            )
            agent_calls.append(
                {
                    "lm": lm,
                    "sub_lm": sub_lm,
                    "max_iterations": max_iterations,
                    "verbose": verbose,
                    "skill": skill,
                    "data_root": data_root,
                }
            )

        async def acall(self, **_kwargs):
            return SimpleNamespace(
                trace=SimpleNamespace(
                    steps=[
                        SimpleNamespace(
                            tool_calls=[
                                SimpleNamespace(
                                    name="evaluate_appworld_task",
                                    result=RunnerResult(
                                        task_id="aaa111_1",
                                        success=False,
                                        score=0.0,
                                        stdout="",
                                        stderr="",
                                        feedback="stale trace result",
                                    ).to_tool_text(),
                                    error=None,
                                )
                            ]
                        )
                    ]
                )
            )

    monkeypatch.setattr(evaluation, "build_lm", fake_build_lm)
    monkeypatch.setattr(gepa_project_module, "AppWorldRLM", FakeAppWorldRLM)

    report = asyncio.run(
        evaluation.run_evaluation(
            EvalConfig(
                data_root=FIXTURE_ROOT,
                dataset="test_normal",
                limit=1,
                reasoning_effort="low",
                task_timeout=1,
                output_dir=tmp_path / "eval-run",
            )
        )
    )

    assert report.count == 1
    assert build_calls == [
        ("openai/gpt-5.4", "low"),
        ("openai/gpt-5.4-mini", None),
    ]
    assert report.results[0].score == 1.0
    assert report.results[0].feedback == "ok"
    assert eval_calls == ["fff666_1"]
    assert close_calls == ["fff666_1"]
    assert agent_calls[0]["lm"] is built_lms["openai/gpt-5.4"]
    assert agent_calls[0]["sub_lm"] is built_lms["openai/gpt-5.4-mini"]
    assert not isinstance(agent_calls[0]["lm"], dict)
    assert not isinstance(agent_calls[0]["sub_lm"], dict)
    assert (tmp_path / "eval-run" / "eval.json").exists()
    assert list((tmp_path / "eval-run" / "task_traces").glob("*_eval.jsonl"))


def test_run_evaluation_scores_from_harness_side_evaluator(monkeypatch, tmp_path):
    events = []

    class FakeAppWorldClient:
        def evaluate_appworld_task(self, task_id):
            events.append(("evaluate", task_id))
            return RunnerResult(
                task_id=task_id,
                success=True,
                score=1.0,
                stdout="",
                stderr="",
                feedback="harness score",
            ).to_tool_text()

        def close_appworld_task(self, task_id):
            events.append(("close", task_id))
            return ""

    class FakeAppWorldRLM:
        def __init__(self, **_kwargs):
            self.appworld_client = FakeAppWorldClient()

        async def acall(self, **kwargs):
            events.append(("agent", kwargs["task_id"]))
            return SimpleNamespace(
                final_answer="done",
                trace=SimpleNamespace(
                    steps=[
                        SimpleNamespace(
                            tool_calls=[
                                SimpleNamespace(
                                    name="evaluate_appworld_task",
                                    result=RunnerResult(
                                        task_id=kwargs["task_id"],
                                        success=False,
                                        score=0.0,
                                        stdout="",
                                        stderr="",
                                        feedback="rlm trace should be ignored",
                                    ).to_tool_text(),
                                    error=None,
                                )
                            ]
                        )
                    ]
                ),
            )

    monkeypatch.setattr(evaluation, "build_lm", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(gepa_project_module, "AppWorldRLM", FakeAppWorldRLM)
    monkeypatch.setattr(
        evaluation,
        "load_eval_dataset",
        lambda _config: [
            evaluation.AppWorldExample("aaa111_1", "test_normal", "do it")
        ],
    )

    report = asyncio.run(
        evaluation.run_evaluation(
            EvalConfig(
                dataset="test_normal",
                task_timeout=1,
                output_dir=tmp_path / "eval-run",
            )
        )
    )

    assert report.results[0].score == 1.0
    assert report.results[0].feedback == "harness score"
    assert events == [
        ("agent", "aaa111_1"),
        ("evaluate", "aaa111_1"),
        ("close", "aaa111_1"),
    ]
    eval_payload = __import__("json").loads((tmp_path / "eval-run" / "eval.json").read_text())
    assert eval_payload["total_tasks"] == 1
    assert eval_payload["soft_restriction_avg"] == 1.0
    assert eval_payload["hard_restriction_avg"] == 1.0
    assert eval_payload["tasks_all_passing"] == 1
    assert eval_payload["task_goal_completion"] == 1.0
    assert eval_payload["scenario_goal_completion"] == 1.0
    assert eval_payload["scenarios_all_passing"] == 1
    assert eval_payload["total_scenarios"] == 1
    assert eval_payload["per_task"][0]["cases"][0]["message"] == "harness score"
    assert "eval: tasks=1, soft=1.000, hard=1.000" in render_stats(tmp_path / "eval-run")
    assert "TGC=100.0% (1/1), SGC=100.0% (1/1)" in appworld_eval_header_summary(
        tmp_path / "eval-run"
    )


def test_codex_lm_args_are_registered_for_eval_and_optimize():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    bench_cli.add_eval_subcommand(subparsers)
    eval_args = parser.parse_args(
        ["eval", "--codex-lm", "--codex-lm-exclude", "gpt-4.1"]
    )
    assert eval_args.codex_lm is True
    assert eval_args.codex_lm_exclude == ["gpt-4.1"]

    optimize_parser = argparse.ArgumentParser()
    gepa_cli._add_project_args(optimize_parser)
    optimize_args = optimize_parser.parse_args(["--no-codex-lm"])
    assert optimize_args.codex_lm is False


def test_install_codex_lm_explicit_missing_has_appworld_uv_hint(monkeypatch):
    monkeypatch.setattr(bench_cli.importlib.util, "find_spec", lambda name: None)

    try:
        bench_cli.install_codex_lm(
            argparse.Namespace(codex_lm=True, codex_lm_exclude=[])
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected RuntimeError")

    assert "--with-editable /Users/gabriel/Workspace/dspy-codex-lm" in message
    assert "uv run --project examples/appworld" in message


def test_install_codex_lm_auto_enables_and_sets_default_key(monkeypatch):
    calls = []
    package = types.ModuleType("dspy_codex_lm")
    cli_module = types.ModuleType("dspy_codex_lm.cli")
    cli_module.install_monkeypatch = lambda *, exclude: calls.append(tuple(exclude))
    monkeypatch.setitem(sys.modules, "dspy_codex_lm", package)
    monkeypatch.setitem(sys.modules, "dspy_codex_lm.cli", cli_module)
    monkeypatch.setattr(bench_cli.importlib.util, "find_spec", lambda name: object())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    bench_cli.install_codex_lm(
        argparse.Namespace(codex_lm=None, codex_lm_exclude=["anthropic/"])
    )

    assert calls == [("anthropic/",)]
    assert bench_cli.os.environ["OPENAI_API_KEY"] == "codex-lm"


def test_install_codex_lm_no_flag_disables_auto_enable(monkeypatch):
    calls = []
    cli_module = types.ModuleType("dspy_codex_lm.cli")
    cli_module.install_monkeypatch = lambda *, exclude: calls.append(tuple(exclude))
    monkeypatch.setitem(sys.modules, "dspy_codex_lm.cli", cli_module)
    monkeypatch.setattr(bench_cli.importlib.util, "find_spec", lambda name: object())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    bench_cli.install_codex_lm(
        argparse.Namespace(codex_lm=False, codex_lm_exclude=["openai/gpt-4"])
    )

    assert calls == []
    assert "OPENAI_API_KEY" not in bench_cli.os.environ
