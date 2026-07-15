from __future__ import annotations

from pathlib import Path

from predict_rlm.backends.supervisor._payload import (
    _pickleable_globals_snapshot,
    _SandboxPath,
)
from predict_rlm.backends.supervisor.runner import DirectPythonBackend


def test_sandbox_path_snapshot_uses_virtual_string() -> None:
    snapshot = _pickleable_globals_snapshot({
        "path": _SandboxPath("/sandbox/output/foo"),
    })

    assert snapshot["globals"] == {"path": "/sandbox/output/foo"}
    assert snapshot["restored_globals"] == ["path"]
    assert snapshot["lost_globals"] == []


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
            "from pathlib import Path\n"
            "p = Path('/app/model.xml')\n"
            "print(type(p).__name__)",
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
            "print(type(p).__name__)\n"
            "print(hasattr(p, 'read_text'))\n"
            "print(p / 'child.txt')",
            timeout=10,
        )
    finally:
        backend.shutdown()

    assert first == "_SandboxPath\n"
    assert recovered == "_SandboxPath\nTrue\n/app/model.xml/child.txt\n"
