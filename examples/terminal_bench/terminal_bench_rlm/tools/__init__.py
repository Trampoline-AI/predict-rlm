"""Tooling for running PredictRLM code inside Terminal-Bench containers."""

from typing import Any

__all__ = ["HarborEnvironmentInterpreter", "TerminalBenchRunnerInterpreter"]


def __getattr__(name: str) -> Any:
    if name == "HarborEnvironmentInterpreter":
        from .container_runner import HarborEnvironmentInterpreter

        return HarborEnvironmentInterpreter
    if name == "TerminalBenchRunnerInterpreter":
        from .container_runner import TerminalBenchRunnerInterpreter

        return TerminalBenchRunnerInterpreter
    raise AttributeError(name)
