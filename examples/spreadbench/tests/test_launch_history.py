"""Pins the launch_history.log contract.

Every optimize.py invocation — fresh or resume — should append one line
to ``<run_dir>/launch_history.log`` with the timestamp, cwd, and full
argv so a glance at any run directory tells you which commands produced
its state bin.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


def test_launch_history_append_writes_argv_line(tmp_path, monkeypatch):
    from optimize import _append_launch_history

    fake_argv = [
        "examples/spreadbench/scripts/optimize.py",
        "--lm", "openai/gpt-5.4-mini",
        "--rlm_proposer",
    ]
    monkeypatch.setattr(sys, "argv", fake_argv)

    _append_launch_history(tmp_path)

    log = tmp_path / "launch_history.log"
    assert log.exists()
    content = log.read_text()
    # One line, timestamp + cwd + command
    lines = content.strip().splitlines()
    assert len(lines) == 1
    # Timestamp column is ISO-ish (starts with 20-digits)
    assert lines[0][:4].isdigit()
    # argv survives verbatim
    assert "--lm openai/gpt-5.4-mini" in lines[0]
    assert "--rlm_proposer" in lines[0]
    # Tab-separated columns
    assert lines[0].count("\t") == 2


def test_launch_history_appends_across_invocations(tmp_path, monkeypatch):
    """A resume reruns optimize.py on the same run_dir; the log must
    accumulate entries, not overwrite the original.
    """
    from optimize import _append_launch_history

    monkeypatch.setattr(sys, "argv", ["opt.py", "--first"])
    _append_launch_history(tmp_path)
    monkeypatch.setattr(sys, "argv", ["opt.py", "--resume"])
    _append_launch_history(tmp_path)

    lines = (tmp_path / "launch_history.log").read_text().strip().splitlines()
    assert len(lines) == 2
    assert "--first" in lines[0]
    assert "--resume" in lines[1]


def test_launch_history_never_raises(tmp_path, monkeypatch):
    """A broken launch_history.log must not block a real run. Point the
    log at a read-only location and confirm the helper swallows the error.
    """
    from optimize import _append_launch_history

    monkeypatch.setattr(sys, "argv", ["opt.py"])
    # Non-existent nested path with an unwritable parent — still shouldn't raise
    bad_dir = tmp_path / "does" / "not" / "exist"
    # Helper is best-effort; should not raise even if mkdir somehow fails
    _append_launch_history(bad_dir)
    # After fixing the parent, the helper should succeed next time
    _append_launch_history(tmp_path)
    assert (tmp_path / "launch_history.log").exists()
