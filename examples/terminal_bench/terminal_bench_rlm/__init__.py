"""PredictRLM integration helpers for Terminal-Bench task containers."""

from typing import Any

__all__ = [
    "HarborEnvironmentInterpreter",
    "HarborPredictRLMAgent",
    "TerminalBenchRLMBaseAgent",
    "TerminalBenchRLMAgent",
    "TerminalBenchRunnerInterpreter",
    "feedback",
    "hard_score",
    "score_details",
    "soft_score",
    "to_gepa_example_result",
]


def __getattr__(name: str) -> Any:
    if name == "HarborEnvironmentInterpreter":
        from .tools.container_runner import HarborEnvironmentInterpreter

        return HarborEnvironmentInterpreter
    if name == "HarborPredictRLMAgent":
        from .tools.tbench_agent import HarborPredictRLMAgent

        return HarborPredictRLMAgent
    if name == "TerminalBenchRunnerInterpreter":
        from .tools.container_runner import TerminalBenchRunnerInterpreter

        return TerminalBenchRunnerInterpreter
    if name == "TerminalBenchRLMBaseAgent":
        from .tools.tbench_agent import TerminalBenchRLMBaseAgent

        return TerminalBenchRLMBaseAgent
    if name == "TerminalBenchRLMAgent":
        from .tools.tbench_agent import TerminalBenchRLMAgent

        return TerminalBenchRLMAgent
    if name in {"feedback", "hard_score", "score_details", "soft_score", "to_gepa_example_result"}:
        from . import scoring

        return getattr(scoring, name)
    raise AttributeError(name)
