from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_EXAMPLE_DIR = Path(__file__).resolve().parent.parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

from spreadsheet_rlm.bench import cli as eval_cli  # noqa: E402
from spreadsheet_rlm.bench import evaluation  # noqa: E402
from spreadsheet_rlm.bench.config import EvalConfig  # noqa: E402
from spreadsheet_rlm.bench.dataset import SpreadsheetTask  # noqa: E402


def test_eval_config_defaults_keep_jspi_backend() -> None:
    parser = eval_cli.build_eval_parser()
    args = parser.parse_args([])

    config = eval_cli.build_eval_config(args, log_dir=Path("logs"))

    assert config.sandbox_backend == "jspi"
    assert config.sbx_pool_size is None
    assert config.sbx_template is None
    assert config.sbx_preinstall_packages is True


def test_eval_config_parses_sbx_pool_flags() -> None:
    parser = eval_cli.build_eval_parser()
    args = parser.parse_args(
        [
            "--sandbox-backend",
            "sbx",
            "--sbx-pool-size",
            "5",
            "--sbx-template",
            "custom/template:tag",
            "--no-sbx-preinstall-packages",
        ]
    )

    config = eval_cli.build_eval_config(args, log_dir=None)

    assert config.sandbox_backend == "sbx"
    assert config.sbx_pool_size == 5
    assert config.sbx_template == "custom/template:tag"
    assert config.sbx_preinstall_packages is False


def test_eval_config_parses_rlm_logging_flags() -> None:
    parser = eval_cli.build_eval_parser()
    args = parser.parse_args(["--verbose-rlm", "--debug-rlm"])

    config = eval_cli.build_eval_config(args, log_dir=None)

    assert config.verbose_rlm is True
    assert config.debug_rlm is True


def test_sbx_pool_size_requires_sbx_backend() -> None:
    with pytest.raises(ValueError, match="sbx_pool_size.*sandbox_backend"):
        EvalConfig(sandbox_backend="jspi", sbx_pool_size=5)

    parser = eval_cli.build_eval_parser()
    args = parser.parse_args(["--sandbox-backend", "jspi", "--sbx-pool-size", "5"])
    with pytest.raises(ValueError, match="sbx_pool_size.*sandbox_backend"):
        eval_cli.build_eval_config(args, log_dir=None)


def test_run_tasks_async_creates_sbx_pool_and_passes_it_to_cases(monkeypatch) -> None:
    events: list[tuple[str, object]] = []

    class FakeSbxConfig:
        def __init__(self, *, template=None):
            self.template = template

    class FakeSbxPool:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            events.append(("init", kwargs))

        def __enter__(self):
            events.append(("enter", self))
            return self

        def __exit__(self, exc_type, exc, traceback):
            events.append(("exit", self))

    async def fake_run_case(
        task,
        idx,
        input_path,
        answer_path,
        sig_cls,
        skill,
        lm,
        sub_lm,
        sem,
        tmp_dir,
        config,
        sbx_pool=None,
    ):
        events.append(("case_pool", sbx_pool))
        return evaluation.CaseResult(idx=idx, score=1.0, passed=True, message="ok")

    monkeypatch.setattr(evaluation, "SbxConfig", FakeSbxConfig, raising=False)
    monkeypatch.setattr(evaluation, "SbxPool", FakeSbxPool, raising=False)
    monkeypatch.setattr(evaluation, "_run_case", fake_run_case)

    task = SpreadsheetTask(
        task_id="task1",
        instruction="Do it",
        instruction_type="instruction",
        answer_position="Sheet1!A1",
        spreadsheet_dir="/tmp/task1",
        test_cases=((1, "/tmp/input.xlsx", "/tmp/answer.xlsx"),),
    )
    config = EvalConfig(
        sandbox_backend="sbx",
        sbx_pool_size=5,
        sbx_template="custom/template:tag",
        sbx_preinstall_packages=False,
        concurrency=5,
    )

    results = asyncio.run(
        evaluation._run_tasks_async(
            [task], object, SimpleNamespace(packages=["openpyxl"]), object(), object(), config
        )
    )

    assert results[0].hard == 1
    init_kwargs = events[0][1]
    assert init_kwargs["size"] == 5
    assert init_kwargs["config"].template == "custom/template:tag"
    assert init_kwargs["preinstall_packages"] is False
    assert init_kwargs["skill_packages"] == ["openpyxl"]
    assert events[1][0] == "enter"
    assert events[2] == ("case_pool", events[1][1])
    assert events[3] == ("exit", events[1][1])


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (EvalConfig(), {}),
        (EvalConfig(sandbox_backend="sbx"), {"sandbox_backend": "sbx"}),
    ],
)
def test_run_case_passes_predict_rlm_sandbox_kwargs(
    monkeypatch,
    tmp_path,
    config,
    expected,
) -> None:
    calls: list[dict] = []
    produced = tmp_path / "produced.xlsx"
    produced.write_text("workbook")

    class FakePredictRLM:
        def __init__(self, sig_cls, **kwargs):
            calls.append(kwargs)

        async def acall(self, **kwargs):
            return SimpleNamespace(output_spreadsheet=SimpleNamespace(path=str(produced)))

    monkeypatch.setattr(evaluation, "PredictRLM", FakePredictRLM)
    monkeypatch.setattr(evaluation, "parse_answer_position", lambda *args: ("Sheet1", "A1"))
    monkeypatch.setattr(evaluation, "_build_instruction", lambda *args: "formatted")
    monkeypatch.setattr(
        evaluation,
        "recalculate",
        lambda path: SimpleNamespace(source="test"),
    )
    monkeypatch.setattr(evaluation, "score_workbooks", lambda *args: (1.0, "ok"))

    task = SpreadsheetTask(
        task_id="task1",
        instruction="Do it",
        instruction_type="instruction",
        answer_position="Sheet1!A1",
        spreadsheet_dir=str(tmp_path),
        test_cases=(),
    )

    result = asyncio.run(
        evaluation._run_case(
            task,
            1,
            str(tmp_path / "input.xlsx"),
            str(tmp_path / "answer.xlsx"),
            object,
            object(),
            object(),
            object(),
            asyncio.Semaphore(1),
            str(tmp_path),
            config,
        )
    )

    assert result.passed is True
    assert calls[0]["verbose"] is config.verbose_rlm
    assert calls[0]["debug"] is config.debug_rlm
    for key, value in expected.items():
        assert calls[0][key] == value
    for absent_key in {"sandbox_backend", "sbx_pool"} - set(expected):
        assert absent_key not in calls[0]
