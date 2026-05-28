"""Opt-in runtime function hooks for sandbox execution."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

RuntimeHookPhase = Literal["before", "after", "error"]


class RuntimeHook(BaseModel):
    """Dotted Python function target to observe inside the sandbox runtime."""

    target: str
    phases: set[RuntimeHookPhase] = Field(default_factory=lambda: {"before"})


class RuntimeHookEvent(BaseModel):
    """Sanitized host-side event emitted by an observed sandbox function call."""

    target: str
    phase: RuntimeHookPhase
    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    error: str | None = None
    duration_ms: int | None = None
    timestamp: float
