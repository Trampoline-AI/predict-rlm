from __future__ import annotations

from pathlib import Path

import dspy
from dspy.utils.dummies import DummyLM

from predict_rlm import PredictRLM
from predict_rlm.backends.supervisor.runner import DirectPythonBackend


def test_sandbox_path_global_does_not_poison_later_execute(tmp_path: Path) -> None:
    backend = DirectPythonBackend(
        runner_path=str(tmp_path / "predict_rlm_runner.py"),
        workdir=str(tmp_path),
        exec_timeout=10,
    )
    try:
        first = backend.execute(
            "from pathlib import Path\n"
            "p = Path('/sandbox/output/foo')\n"
            "print(type(p).__name__)",
            timeout=10,
        )
        second = backend.execute("print('second ok')", timeout=10)
    finally:
        backend.shutdown()

    assert first == "_SandboxPath\n"
    assert second == "second ok\n"


def test_direct_backend_copies_directory_to_sandbox_path(tmp_path: Path) -> None:
    source = tmp_path / "workspace"
    (source / "nested").mkdir(parents=True)
    (source / "nested" / "value.txt").write_text("directory copy", encoding="utf-8")
    backend = DirectPythonBackend(
        runner_path=str(tmp_path / "predict_rlm_runner.py"),
        workdir=str(tmp_path),
        exec_timeout=10,
    )
    try:
        backend.mount_file_at(str(source), "/sandbox/input/workspace")
        output = backend.execute(
            "from pathlib import Path\n"
            "print(Path('/sandbox/input/workspace/nested/value.txt').read_text())",
            timeout=10,
        )
    finally:
        backend.shutdown()

    assert output == "directory copy\n"


def test_regular_path_global_survives_timeout_recovery_as_path(tmp_path: Path) -> None:
    backend = DirectPythonBackend(
        runner_path=str(tmp_path / "predict_rlm_runner.py"),
        workdir=str(tmp_path),
        exec_timeout=10,
    )
    try:
        first = backend.execute(
            "from pathlib import Path\np = Path('/app/model.xml')\nprint(type(p).__name__)",
            timeout=10,
        )
        backend.execute(
            "import signal\n"
            "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
            "while True:\n"
            "    pass",
            timeout=0.05,
        )
        recovered = backend.execute(
            "print(type(p).__name__)\nprint(hasattr(p, 'read_text'))\nprint(p / 'child.txt')",
            timeout=10,
        )
    finally:
        backend.shutdown()

    assert first == "_SandboxPath\n"
    assert recovered == "_SandboxPath\nTrue\n/app/model.xml/child.txt\n"


class CountingDummyLM(DummyLM):
    calls: list[object] = []

    def forward(self, *args, **kwargs):
        self.calls.append(kwargs.get("messages"))
        return super().forward(*args, **kwargs)


def test_predict_on_direct_backend_uses_the_rlm_adapter(tmp_path: Path) -> None:
    # A lost adapter falls back to ChatAdapter, fails to parse the JSON reply, and
    # re-sends the request through JSONAdapter: two LM calls for one predict().
    CountingDummyLM.calls = []
    sub_lm = CountingDummyLM([{"answer": "4"}, {"answer": "4"}], adapter=dspy.JSONAdapter())
    code = (
        "result = await predict('question -> answer', question='2+2?')\n"
        "SUBMIT(answer=result['answer'])"
    )
    rlm = PredictRLM(
        "query -> answer",
        lm=DummyLM([{"plan": "ask", "code": code}], adapter=dspy.JSONAdapter()),
        sub_lm=sub_lm,
        max_iterations=1,
        interpreter=DirectPythonBackend(
            runner_path=str(tmp_path / "predict_rlm_runner.py"),
            workdir=str(tmp_path),
            exec_timeout=10,
        ),
    )

    assert rlm(query="add").answer == "4"
    assert len(CountingDummyLM.calls) == 1
