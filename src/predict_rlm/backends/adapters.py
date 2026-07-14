"""Adapters from existing interpreter objects to the async session contract."""

from __future__ import annotations

import asyncio
import inspect
import os
import threading
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import AbstractContextManager, asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from predict_rlm.runtime import (
    Artifact,
    ArtifactBinding,
    ExecutionResult,
    ExecutionSession,
    ExecutionSpec,
    RunContext,
    SessionOwnership,
)


def _artifact_path(artifact: Artifact, name: str) -> str:
    value = artifact.metadata.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Artifact {artifact.id!r} requires string metadata {name!r}")
    return value


def _validate_sandbox_path(path: str) -> None:
    parsed = PurePosixPath(path)
    if not parsed.is_absolute() or ".." in parsed.parts:
        raise ValueError(f"Artifact sandbox path must be absolute and traversal-free: {path!r}")


_UNSET_WORKSPACE_KEY = object()


def _direct_workspace_key(ctx: RunContext) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (
                str(artifact.metadata["source_path"]),
                str(artifact.metadata["sandbox_path"]),
            )
            for prepared in ctx.prepared_inputs.values()
            for artifact in prepared.artifacts
            if artifact.kind == "compat.workspace.direct"
        )
    )


def _validate_injected_workspace_reuse(
    backend: Any,
    ctx: RunContext,
) -> tuple[tuple[str, str], ...] | None:
    if backend._ownership is not SessionOwnership.INJECTED:
        return None
    current = _direct_workspace_key(ctx)
    previous = backend._direct_workspace_key
    if previous is not _UNSET_WORKSPACE_KEY and previous != current and (previous or current):
        raise ValueError(
            "Injected interpreters cannot change direct Workspace mounts across "
            "sequential invocations"
        )
    return current


class InterpreterExecutionSession:
    """Compatibility session over a caller-supplied synchronous interpreter."""

    def __init__(
        self,
        interpreter: Any,
        *,
        name: str,
        ownership: SessionOwnership,
    ) -> None:
        self.interpreter = interpreter
        self.name = name
        self.ownership = ownership
        self._finalized = False
        self._cancelled = False
        self._sync_tasks: set[asyncio.Task[Any]] = set()
        self._post_execute_hooks: list[Callable[[Any], Any]] = []
        self._direct_workspace_mounts: list[Any] = []

    async def _run_sync(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        self._sync_tasks.add(task)
        task.add_done_callback(self._sync_tasks.discard)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError as cancellation:
            try:
                await task
            except BaseException as worker_error:
                setattr(cancellation, "sync_worker_error", worker_error)
            raise

    async def wait_for_idle(self) -> None:
        while self._sync_tasks:
            await asyncio.gather(*tuple(self._sync_tasks), return_exceptions=True)
        await_host_work = getattr(self.interpreter, "await_host_work", None)
        if callable(await_host_work):
            result = await_host_work()
            if inspect.isawaitable(result):
                await result

    async def install_packages(self, packages: Sequence[str]) -> None:
        if not packages:
            return
        install = getattr(self.interpreter, "ensure_skill_packages", None)
        if callable(install):
            await self._run_sync(install, list(packages))

    async def mount(self, artifact: Artifact) -> ArtifactBinding:
        source_path = _artifact_path(artifact, "source_path")
        sandbox_path = _artifact_path(artifact, "sandbox_path")
        _validate_sandbox_path(sandbox_path)

        if artifact.kind == "compat.workspace.direct":
            configure = getattr(self.interpreter, "configure_direct_workspace_mounts", None)
            if not callable(configure):
                raise ValueError("Workspace(mode='direct') requires the SBX backend.")
            self._direct_workspace_mounts.append(artifact.metadata["workspace_binding"])
            await self._run_sync(configure, self._direct_workspace_mounts)
            directory = True
        elif artifact.kind == "compat.workspace.mirror":
            state = artifact.metadata["workspace_binding"]
            await self._run_sync(self.interpreter.mkdir_p, sandbox_path)
            for host_path, target in await self._run_sync(state.iter_mounts):
                await self._run_sync(self.interpreter.mount_file_at, host_path, target)
            add_hook = getattr(self.interpreter, "add_post_execute_hook", None)
            if not callable(add_hook):
                raise ValueError(
                    "The selected interpreter does not support Workspace mirror sync."
                )
            hook = state.sync_from_sandbox
            add_hook(hook)
            self._post_execute_hooks.append(hook)
            directory = True
        elif artifact.kind == "compat.output.directory":
            await self._run_sync(self.interpreter.mkdir_p, sandbox_path)
            directory = True
        elif os.path.isdir(source_path):
            await self._mount_directory(source_path, sandbox_path)
            directory = True
        elif os.path.isfile(source_path):
            await self._run_sync(
                self.interpreter.mount_file_at,
                source_path,
                sandbox_path,
            )
            directory = False
        else:
            raise FileNotFoundError(f"Artifact source does not exist: {source_path}")

        return ArtifactBinding(
            artifact_id=artifact.id,
            path=sandbox_path,
            metadata={"directory": directory},
        )

    async def run_code(
        self,
        code: str,
        variables: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> ExecutionResult:
        kwargs: dict[str, Any] = {"variables": dict(variables or {})}
        if timeout is not None:
            kwargs["timeout"] = timeout
        execute_async = getattr(self.interpreter, "aexecute", None)
        uses_websocket = getattr(self.interpreter, "_uses_websocket_transport", None)
        prefer_sync = callable(uses_websocket) and not uses_websocket()
        if callable(execute_async) and not prefer_sync:
            value = await execute_async(code, **kwargs)
        else:
            value = await self._run_sync(self.interpreter.execute, code, **kwargs)
        return ExecutionResult(value=value)

    async def collect(self, artifact: Artifact) -> Any:
        sandbox_path = _artifact_path(artifact, "sandbox_path")
        destination_path = _artifact_path(artifact, "destination_path")
        _validate_sandbox_path(sandbox_path)
        if bool(artifact.metadata.get("directory")):
            return await self._collect_directory(sandbox_path, destination_path)
        destination = Path(destination_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        await self._run_sync(
            self.interpreter.sync_file_to,
            sandbox_path,
            str(destination),
        )
        return str(destination)

    async def finalize(self) -> None:
        remove_hook = getattr(self.interpreter, "remove_post_execute_hook", None)
        if callable(remove_hook):
            for hook in self._post_execute_hooks:
                remove_hook(hook)
        self._post_execute_hooks.clear()
        self._finalized = True

    async def cancel(self) -> None:
        if self._cancelled:
            return
        self._cancelled = True
        interrupt_async = getattr(self.interpreter, "ainterrupt", None)
        if callable(interrupt_async):
            await interrupt_async()
            return
        interrupt = getattr(self.interpreter, "interrupt", None)
        if callable(interrupt):
            await self._run_sync(interrupt)

    async def _mount_directory(self, source_path: str, sandbox_path: str) -> None:
        await self._run_sync(self.interpreter.mkdir_p, sandbox_path)
        source_root = Path(source_path)
        for host_path in sorted(path for path in source_root.rglob("*") if path.is_file()):
            relative = host_path.relative_to(source_root).as_posix()
            target = str(PurePosixPath(sandbox_path) / relative)
            _validate_sandbox_path(target)
            await self._run_sync(
                self.interpreter.mount_file_at,
                str(host_path),
                target,
            )

    async def _collect_directory(self, sandbox_path: str, destination_path: str) -> str:
        destination = Path(destination_path)
        destination.mkdir(parents=True, exist_ok=True)
        files = await self._run_sync(self.interpreter.list_dir, sandbox_path)
        for source in sorted(files):
            relative = PurePosixPath(source).relative_to(PurePosixPath(sandbox_path))
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            await self._run_sync(self.interpreter.sync_file_to, source, str(target))
        return str(destination)


AcquireInterpreter = Callable[
    [ExecutionSpec, RunContext], AbstractContextManager[Any]
]


class InterpreterBackendAdapter:
    """Thread-bridged boundary for a caller-supplied synchronous interpreter."""

    def __init__(
        self,
        name: str,
        acquire: AcquireInterpreter,
        *,
        ownership: SessionOwnership,
        supports_direct_workspaces: bool = False,
        supports_mirror_workspaces: bool = False,
        direct_workspace_error: str | None = None,
    ) -> None:
        self.name = name
        self._acquire = acquire
        self._ownership = ownership
        self._invocation_lock = (
            threading.Lock() if ownership is SessionOwnership.INJECTED else None
        )
        self._direct_workspace_key: object = _UNSET_WORKSPACE_KEY
        self.supports_direct_workspaces = supports_direct_workspaces
        self.supports_mirror_workspaces = supports_mirror_workspaces
        self.direct_workspace_error = direct_workspace_error

    @asynccontextmanager
    async def start(
        self,
        spec: ExecutionSpec,
        ctx: RunContext,
    ) -> AsyncIterator[ExecutionSession]:
        lock_acquired = False
        manager: AbstractContextManager[Any] | None = None
        entered = False
        try:
            if self._invocation_lock is not None:
                await self._acquire_invocation_lock()
                lock_acquired = True
            workspace_key = _validate_injected_workspace_reuse(self, ctx)
            manager = self._acquire(spec, ctx)
            enter_task = asyncio.create_task(asyncio.to_thread(manager.__enter__))
            try:
                interpreter = await asyncio.shield(enter_task)
            except asyncio.CancelledError:
                await enter_task
                entered = True
                entered = False
                await self._exit_manager(manager, None, None, None)
                raise
            entered = True
            if workspace_key is not None and self._direct_workspace_key is _UNSET_WORKSPACE_KEY:
                self._direct_workspace_key = workspace_key
            session = InterpreterExecutionSession(
                interpreter,
                name=self.name,
                ownership=self._ownership,
            )
            ctx.session = session
            ctx.ownership = self._ownership
            try:
                yield session
            except BaseException as exc:
                await session.wait_for_idle()
                entered = False
                suppress = await self._exit_manager(
                    manager,
                    type(exc),
                    exc,
                    exc.__traceback__,
                )
                if not suppress:
                    raise
            else:
                await session.wait_for_idle()
                entered = False
                await self._exit_manager(manager, None, None, None)
        finally:
            try:
                if entered and manager is not None:
                    entered = False
                    await self._exit_manager(manager, None, None, None)
            finally:
                ctx.session = None
                if lock_acquired and self._invocation_lock is not None:
                    self._invocation_lock.release()

    async def _acquire_invocation_lock(self) -> None:
        if self._invocation_lock is None:
            return
        while True:
            acquire_task = asyncio.create_task(
                asyncio.to_thread(self._invocation_lock.acquire, True, 0.05)
            )
            try:
                acquired = await asyncio.shield(acquire_task)
            except asyncio.CancelledError:
                if await acquire_task:
                    self._invocation_lock.release()
                raise
            if acquired:
                return

    async def _exit_manager(
        self,
        manager: AbstractContextManager[Any],
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> Any:
        exit_task = asyncio.create_task(
            asyncio.to_thread(manager.__exit__, exc_type, exc, traceback)
        )
        try:
            return await asyncio.shield(exit_task)
        except asyncio.CancelledError:
            await exit_task
            raise


class NativeInterpreterExecutionSession:
    """Session over a maintained interpreter's native async operations."""

    def __init__(
        self,
        interpreter: Any,
        *,
        name: str,
        ownership: SessionOwnership,
    ) -> None:
        self.interpreter = interpreter
        self.name = name
        self.ownership = ownership
        self._finalized = False
        self._cancelled = False
        self._post_execute_hooks: list[Callable[[Any], Any]] = []
        self._direct_workspace_mounts: list[Any] = []

    async def install_packages(self, packages: Sequence[str]) -> None:
        if packages:
            await self.interpreter.aensure_skill_packages(list(packages))

    async def mount(self, artifact: Artifact) -> ArtifactBinding:
        source_path = _artifact_path(artifact, "source_path")
        sandbox_path = _artifact_path(artifact, "sandbox_path")
        _validate_sandbox_path(sandbox_path)

        if artifact.kind == "compat.workspace.direct":
            self._direct_workspace_mounts.append(artifact.metadata["workspace_binding"])
            await self.interpreter.aconfigure_direct_workspace_mounts(
                self._direct_workspace_mounts
            )
            directory = True
        elif artifact.kind == "compat.workspace.mirror":
            state = artifact.metadata["workspace_binding"]
            await self.interpreter.amkdir_p(sandbox_path)
            for host_path, target in await state.aiter_mounts():
                await self.interpreter.amount_file_at(host_path, target)
            hook = state.async_sync_from_sandbox
            self.interpreter.add_post_execute_hook(hook)
            self._post_execute_hooks.append(hook)
            directory = True
        elif artifact.kind == "compat.output.directory":
            await self.interpreter.amkdir_p(sandbox_path)
            directory = True
        elif os.path.isdir(source_path):
            await self._mount_directory(source_path, sandbox_path)
            directory = True
        elif os.path.isfile(source_path):
            await self.interpreter.amount_file_at(source_path, sandbox_path)
            directory = False
        else:
            raise FileNotFoundError(f"Artifact source does not exist: {source_path}")

        return ArtifactBinding(
            artifact_id=artifact.id,
            path=sandbox_path,
            metadata={"directory": directory},
        )

    async def run_code(
        self,
        code: str,
        variables: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> ExecutionResult:
        value = await self.interpreter.aexecute(
            code,
            variables=dict(variables or {}),
            timeout=timeout,
        )
        return ExecutionResult(value=value)

    async def collect(self, artifact: Artifact) -> Any:
        sandbox_path = _artifact_path(artifact, "sandbox_path")
        destination_path = _artifact_path(artifact, "destination_path")
        _validate_sandbox_path(sandbox_path)
        if bool(artifact.metadata.get("directory")):
            return await self._collect_directory(sandbox_path, destination_path)
        destination = Path(destination_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        await self.interpreter.async_file_to(sandbox_path, str(destination))
        return str(destination)

    async def finalize(self) -> None:
        await_host_work = getattr(self.interpreter, "await_host_work", None)
        if callable(await_host_work):
            result = await_host_work(cancel=True)
            if inspect.isawaitable(result):
                await result
        for hook in self._post_execute_hooks:
            self.interpreter.remove_post_execute_hook(hook)
        self._post_execute_hooks.clear()
        self._finalized = True

    async def cancel(self) -> None:
        if self._cancelled:
            return
        self._cancelled = True
        await self.interpreter.ainterrupt()

    async def _mount_directory(self, source_path: str, sandbox_path: str) -> None:
        await self.interpreter.amkdir_p(sandbox_path)
        source_root = Path(source_path)
        for host_path in sorted(path for path in source_root.rglob("*") if path.is_file()):
            await asyncio.sleep(0)
            relative = host_path.relative_to(source_root).as_posix()
            target = str(PurePosixPath(sandbox_path) / relative)
            _validate_sandbox_path(target)
            await self.interpreter.amount_file_at(str(host_path), target)

    async def _collect_directory(self, sandbox_path: str, destination_path: str) -> str:
        destination = Path(destination_path)
        destination.mkdir(parents=True, exist_ok=True)
        files = await self.interpreter.alist_dir(sandbox_path)
        for source in sorted(files):
            await asyncio.sleep(0)
            relative = PurePosixPath(source).relative_to(PurePosixPath(sandbox_path))
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            await self.interpreter.async_file_to(source, str(target))
        return str(destination)


class ExistingExecutionBackendAdapter:
    """Validate and expose a user-supplied final execution backend."""

    def __init__(self, backend: Any) -> None:
        if not callable(getattr(backend, "start", None)):
            raise TypeError("execution must implement start(spec, ctx)")
        self.backend = backend
        self.name = str(getattr(backend, "name", type(backend).__name__))

    def start(self, spec: ExecutionSpec, ctx: RunContext) -> Any:
        return self.backend.start(spec, ctx)
