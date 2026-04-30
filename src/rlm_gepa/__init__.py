"""RLM-GEPA: GEPA optimization for PredictRLM projects."""

from .cli import run_project_cli
from .proposer.rlm import (
    agent_spec_from_rlm,
    build_merge_signature,
    build_patch_merge_signature,
    build_proposer_for_rlm,
    build_proposer_signature,
)
from .schema import (
    AgentSpec,
    EvaluationContext,
    OptimizeConfig,
    OptimizeReport,
    RLMGepaExampleResult,
    RLMGepaProject,
)
from .service import check_optimization, run_optimization

__all__ = [
    "AgentSpec",
    "EvaluationContext",
    "OptimizeConfig",
    "OptimizeReport",
    "RLMGepaExampleResult",
    "RLMGepaProject",
    "agent_spec_from_rlm",
    "build_merge_signature",
    "build_patch_merge_signature",
    "build_proposer_for_rlm",
    "build_proposer_signature",
    "check_optimization",
    "run_project_cli",
    "run_optimization",
]
