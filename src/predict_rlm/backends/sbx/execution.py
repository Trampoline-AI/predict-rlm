"""Final execution lifecycle for maintained SBX backends."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Callable

from predict_rlm.backends.adapters import NativeInterpreterExecutionSession
from predict_rlm.runtime import (
    ExecutionSession,
    ExecutionSpec,
    RunContext,
    SessionOwnership,
)

from .backend import SbxBackend
from .config import SbxConfig
from .pool import SbxPool


class SbxExecutionBackend:
    """Own one websocket SBX interpreter for each invocation."""

    name = "sbx"
    supports_mirror_workspaces = True
    supports_direct_workspaces = True

    def __init__(
        self,
        *,
        config: SbxConfig | None = None,
        runtime_hooks: list[Any] | None = None,
        on_runtime_hook_event: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config or SbxConfig()
        self.runtime_hooks = list(runtime_hooks or ())
        self.on_runtime_hook_event = on_runtime_hook_event

    @asynccontextmanager
    async def start(
        self,
        spec: ExecutionSpec,
        ctx: RunContext,
    ) -> AsyncIterator[ExecutionSession]:
        interpreter = SbxBackend(
            config=self.config,
            tools=dict(spec.tools),
            output_fields=[dict(field) for field in spec.output_fields],
            allowed_domains=list(spec.allowed_domains) or None,
            debug=spec.debug,
            verbose=spec.verbose,
            extra_read_paths=list(spec.extra_read_paths) or None,
            extra_write_paths=list(spec.extra_write_paths) or None,
            runtime_hooks=self.runtime_hooks,
            on_runtime_hook_event=self.on_runtime_hook_event,
        )
        session = NativeInterpreterExecutionSession(
            interpreter,
            name=self.name,
            ownership=SessionOwnership.OWNED,
        )
        ctx.session = session
        ctx.ownership = SessionOwnership.OWNED
        try:
            yield session
        finally:
            ctx.session = None
            aretire = getattr(interpreter, "aretire_when_host_work_finishes", None)
            retired = (
                await aretire()
                if callable(aretire)
                else interpreter.retire_when_host_work_finishes()
            )
            if not retired:
                await interpreter.ashutdown()


class SbxPoolExecutionBackend:
    """Lease one maintained websocket interpreter from an SBX pool."""

    name = "sbx"
    supports_mirror_workspaces = True
    supports_direct_workspaces = False
    direct_workspace_error = (
        "Workspace(mode='direct') requires a per-call SBX interpreter; "
        "prewarmed SbxPool instances cannot add workspace mounts after creation."
    )

    def __init__(
        self,
        pool: SbxPool,
        *,
        runtime_hooks: list[Any] | None = None,
        on_runtime_hook_event: Callable[..., Any] | None = None,
    ) -> None:
        self.pool = pool
        self.runtime_hooks = list(runtime_hooks or ())
        self.on_runtime_hook_event = on_runtime_hook_event

    @asynccontextmanager
    async def start(
        self,
        spec: ExecutionSpec,
        ctx: RunContext,
    ) -> AsyncIterator[ExecutionSession]:
        async with self.pool.alease(
            tools=dict(spec.tools),
            output_fields=[dict(field) for field in spec.output_fields],
            debug=spec.debug,
            verbose=spec.verbose,
            runtime_hooks=self.runtime_hooks,
            on_runtime_hook_event=self.on_runtime_hook_event,
        ) as interpreter:
            session = NativeInterpreterExecutionSession(
                interpreter,
                name=self.name,
                ownership=SessionOwnership.POOLED,
            )
            ctx.session = session
            ctx.ownership = SessionOwnership.POOLED
            try:
                yield session
            finally:
                ctx.session = None
