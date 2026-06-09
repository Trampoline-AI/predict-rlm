"""Tooling for running PredictRLM code inside Terminal-Bench containers."""

from typing import Any

__all__ = ["TerminalBenchRunnerInterpreter"]


def __getattr__(name: str) -> Any:
    if name == "TerminalBenchRunnerInterpreter":
        from .container_runner import TerminalBenchRunnerInterpreter

        return TerminalBenchRunnerInterpreter
    raise AttributeError(name)
