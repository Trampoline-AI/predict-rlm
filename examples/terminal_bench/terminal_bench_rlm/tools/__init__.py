"""Tooling for running PredictRLM code inside Terminal-Bench containers."""

from typing import Any

__all__ = ["TerminalBenchRunnerClientAdapter"]


def __getattr__(name: str) -> Any:
    if name == "TerminalBenchRunnerClientAdapter":
        from .container_runner import TerminalBenchRunnerClientAdapter

        return TerminalBenchRunnerClientAdapter
    raise AttributeError(name)
