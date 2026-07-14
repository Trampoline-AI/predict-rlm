"""Final execution lifecycle for maintained SBX backends."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any, Callable

from predict_rlm.backends.adapters import NativeInterpreterExecutionSession
from predict_rlm.runtime import (
    ExecutionSession,
    ExecutionSpec,
    HostDirectoryMount,
    RunContext,
    SessionOwnership,
    SessionRequirements,
    UnsupportedOperationError,
)

from .backend import SbxBackend
from .config import SbxConfig
from .pool import SbxPool


def _requirements_key(
    requirements: SessionRequirements,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    return (
        tuple(sorted(set(requirements.allowed_domains))),
        tuple(sorted(set(requirements.extra_read_paths))),
        tuple(sorted(set(requirements.extra_write_paths))),
    )


def _spec_requirements(spec: ExecutionSpec) -> SessionRequirements:
    return SessionRequirements(
        allowed_domains=spec.allowed_domains,
        extra_read_paths=spec.extra_read_paths,
        extra_write_paths=spec.extra_write_paths,
    )


class SbxExecutionBackend:
    """Own one websocket SBX interpreter for each invocation."""

    name = "sbx"

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

    async def validate_host_directory_mounts(
        self,
        mounts: Sequence[HostDirectoryMount],
        ctx: RunContext,
    ) -> None:
        del ctx
        if self.config.reuse and mounts:
            raise UnsupportedOperationError(
                "Named/reused SBX sandboxes do not accept per-invocation host mounts."
            )

    @asynccontextmanager
    async def start(
        self,
        spec: ExecutionSpec,
        ctx: RunContext,
    ) -> AsyncIterator[ExecutionSession]:
        if self.config.reuse and (
            spec.host_directory_mounts
            or spec.allowed_domains
            or spec.extra_read_paths
            or spec.extra_write_paths
        ):
            raise UnsupportedOperationError(
                "Named/reused SBX sandboxes do not accept per-invocation host mounts, "
                "allowed domains, or host path requirements."
            )
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
            direct_workspace_mounts=list(spec.host_directory_mounts) or None,
        )
        session = NativeInterpreterExecutionSession(
            interpreter,
            name=self.name,
            ownership=SessionOwnership.OWNED,
            host_directory_mounts=spec.host_directory_mounts,
            sandbox_roots=spec.sandbox_roots,
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

    async def validate_host_directory_mounts(
        self,
        mounts: Sequence[HostDirectoryMount],
        ctx: RunContext,
    ) -> None:
        del ctx
        if mounts:
            raise UnsupportedOperationError(
                "Host directory mounts require a per-call SBX interpreter; "
                "prewarmed SbxPool instances cannot add mounts after creation."
            )

    @asynccontextmanager
    async def start(
        self,
        spec: ExecutionSpec,
        ctx: RunContext,
    ) -> AsyncIterator[ExecutionSession]:
        if _requirements_key(_spec_requirements(spec)) != _requirements_key(
            self.pool.session_requirements
        ):
            raise UnsupportedOperationError(
                "Per-invocation allowed domains and host path requirements must "
                "match the prewarmed SbxPool fixed policy."
            )
        if spec.host_directory_mounts:
            raise UnsupportedOperationError(
                "Host directory mounts require a per-call SBX interpreter; "
                "prewarmed SbxPool instances cannot add mounts after creation."
            )
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
                sandbox_roots=spec.sandbox_roots,
            )
            ctx.session = session
            ctx.ownership = SessionOwnership.POOLED
            try:
                yield session
            finally:
                ctx.session = None
