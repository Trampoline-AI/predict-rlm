"""Tooling for running PredictRLM code in Terminal-Bench environments."""

from typing import Any

__all__ = ["TerminalBenchEnvironmentAdapter", "TerminalBenchRunnerClientAdapter"]


def __getattr__(name: str) -> Any:
    if name == "TerminalBenchEnvironmentAdapter":
        from .container_runner import TerminalBenchEnvironmentAdapter

        return TerminalBenchEnvironmentAdapter
    if name == "TerminalBenchRunnerClientAdapter":
        from .container_runner import TerminalBenchRunnerClientAdapter

        return TerminalBenchRunnerClientAdapter
    raise AttributeError(name)
