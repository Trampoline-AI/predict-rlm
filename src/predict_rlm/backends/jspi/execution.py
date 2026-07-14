"""Final execution lifecycle for the maintained JSPI backend."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any, Callable

from predict_rlm.backends.adapters import NativeInterpreterExecutionSession, _artifact_path
from predict_rlm.runtime import (
    Artifact,
    ArtifactBinding,
    ExecutionSession,
    ExecutionSpec,
    HostDirectoryMount,
    RunContext,
    SandboxRootReservation,
    SessionOwnership,
    UnsupportedOperationError,
)

from .backend import JspiBackend


class JspiExecutionSession(NativeInterpreterExecutionSession):
    """JSPI session with host-call-safe artifact transfer operations."""

    def __init__(
        self,
        interpreter: JspiBackend,
        *,
        sandbox_roots: Sequence[SandboxRootReservation] = (),
    ) -> None:
        super().__init__(
            interpreter,
            name="jspi",
            ownership=SessionOwnership.OWNED,
            sandbox_roots=sandbox_roots,
        )
        self._running_code = False

    async def run_code(self, code, variables=None, *, timeout=None):
        self._running_code = True
        try:
            return await super().run_code(code, variables, timeout=timeout)
        finally:
            self._running_code = False

    async def mount(self, artifact: Artifact) -> ArtifactBinding:
        if self._running_code and artifact.kind == "compat.file":
            source_path = _artifact_path(artifact, "source_path")
            sandbox_path = _artifact_path(artifact, "sandbox_path")
            await self.interpreter._mount_file_during_tool(source_path, sandbox_path)
            return ArtifactBinding(artifact_id=artifact.id, path=sandbox_path)
        return await super().mount(artifact)

    async def collect(self, artifact: Artifact) -> Any:
        if self._running_code and artifact.kind == "compat.file":
            sandbox_path = _artifact_path(artifact, "sandbox_path")
            destination_path = _artifact_path(artifact, "destination_path")
            await self.interpreter._sync_file_during_tool(
                sandbox_path,
                destination_path,
            )
            return destination_path
        return await super().collect(artifact)

    async def cancel(self) -> None:
        if self._cancelled:
            return
        self._cancelled = True
        await self.interpreter.acancel_execution()


class JspiExecutionBackend:
    """Own one JSPI interpreter for each PredictRLM invocation."""

    name = "jspi"

    def __init__(
        self,
        *,
        allowed_domains: list[str] | None = None,
        telemetry_context: Callable[[], Any] | None = None,
    ) -> None:
        self.allowed_domains = allowed_domains
        self.telemetry_context = telemetry_context

    async def validate_host_directory_mounts(
        self,
        mounts: Sequence[HostDirectoryMount],
        ctx: RunContext,
    ) -> None:
        del ctx
        if mounts:
            raise UnsupportedOperationError(
                "The JSPI backend does not support host directory mounts"
            )

    @asynccontextmanager
    async def start(
        self,
        spec: ExecutionSpec,
        ctx: RunContext,
    ) -> AsyncIterator[ExecutionSession]:
        if spec.host_directory_mounts:
            raise UnsupportedOperationError(
                "The JSPI backend does not support host directory mounts"
            )
        interpreter = JspiBackend(
            tools=dict(spec.tools),
            output_fields=[dict(field) for field in spec.output_fields],
            allowed_domains=list(spec.allowed_domains) or self.allowed_domains,
            debug=spec.debug,
            verbose=spec.verbose,
            extra_read_paths=list(spec.extra_read_paths) or None,
            extra_write_paths=list(spec.extra_write_paths) or None,
            telemetry_context=(
                self.telemetry_context() if self.telemetry_context is not None else None
            ),
        )
        session = JspiExecutionSession(
            interpreter,
            sandbox_roots=spec.sandbox_roots,
        )
        ctx.session = session
        ctx.ownership = SessionOwnership.OWNED
        try:
            yield session
        finally:
            ctx.session = None
            await_host_work = getattr(interpreter, "await_host_work", None)
            if callable(await_host_work):
                await await_host_work(cancel=True)
            if not interpreter.retire_when_sync_workers_finish():
                await interpreter.ashutdown()
