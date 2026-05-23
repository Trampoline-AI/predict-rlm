"""Shared Python runner payload helpers for Terminal-Bench containers."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path


def runner_script_path() -> Path:
    """Return the shared runner script copied by both SBX and Terminal-Bench."""
    spec = importlib.util.find_spec("predict_rlm.sandbox.python_runner")
    if spec is None or spec.origin is None:
        raise RuntimeError("Could not locate predict_rlm.sandbox.python_runner")
    return Path(spec.origin).resolve()


def runner_source() -> str:
    return runner_script_path().read_text(encoding="utf-8")


if __name__ == "__main__":
    from predict_rlm.sandbox.python_runner import _main

    asyncio.run(_main())
