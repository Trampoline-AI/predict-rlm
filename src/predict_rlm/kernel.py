"""Compatibility imports for the versioned small-kernel runtime."""

from .runtime import (
    RUNTIME_SPI_VERSION,
    RunContext,
    RuntimeContribution,
    RuntimeModule,
    RuntimeSpec,
    current_run_context,
    resolve_runtime_spec,
)

__all__ = [
    "RUNTIME_SPI_VERSION",
    "RunContext",
    "RuntimeContribution",
    "RuntimeModule",
    "RuntimeSpec",
    "current_run_context",
    "resolve_runtime_spec",
]
