"""Thread-safe pool for Docker Sandboxes backends."""

from __future__ import annotations

import asyncio
import concurrent.futures
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, Callable

from predict_rlm._logging import configure_predict_rlm_logging
from predict_rlm.trace import ms_since

from .backend import SbxBackend
from .config import SbxConfig
from .logging import log_pool_lifecycle


class SbxPool:
    """Thread-safe pool of prewarmed Docker Sandboxes backends."""

    supports_direct_workspaces = False
    supports_mirror_workspaces = True

    def __init__(
        self,
        *,
        size: int,
        config: SbxConfig | None = None,
        allowed_domains: list[str] | None = None,
        tools: dict[str, Callable[..., Any]] | None = None,
        output_fields: list[dict] | None = None,
        preinstall_packages: bool = True,
        skill_packages: list[str] | None = None,
        debug: bool = False,
        verbose: bool = False,
        extra_read_paths: list[str] | None = None,
        extra_write_paths: list[str] | None = None,
        _supervisor_command: list[str] | None = None,
        _runner_command: list[str] | None = None,
        _staging_root: str | Path | None = None,
    ) -> None:
        if size < 1:
            raise ValueError("SbxPool size must be at least 1")
        self.size = size
        self.config = config or SbxConfig()
        self._pool_name_prefix = (
            self.config.name or f"predict-rlm-sbx-pool-{uuid.uuid4().hex[:12]}"
        )
        self.debug = debug
        self.verbose = verbose
        configure_predict_rlm_logging(
            debug=True if debug else None,
            verbose=True if verbose else None,
        )
        self._interpreter_kwargs = {
            "config": self.config,
            "allowed_domains": allowed_domains,
            "tools": tools,
            "output_fields": output_fields,
            "preinstall_packages": preinstall_packages,
            "skill_packages": skill_packages,
            "debug": debug,
            "verbose": verbose,
            "extra_read_paths": extra_read_paths,
            "extra_write_paths": extra_write_paths,
            "_supervisor_command": _supervisor_command,
            "_runner_command": _runner_command,
        }
        self._staging_root = Path(_staging_root) if _staging_root is not None else None
        self._available: queue.Queue[SbxBackend] = queue.Queue(maxsize=size)
        self._all_interpreters: list[SbxBackend] = []
        self._lock = threading.Lock()
        self._state_changed = threading.Condition(self._lock)
        self._async_waiters: set[
            tuple[asyncio.AbstractEventLoop, asyncio.Future[None]]
        ] = set()
        self._started = False
        self._starting = False
        self._shutdown = False
        self._shutdown_requested = False
        self._shutting_down = False
        self._replacement_failure: BaseException | None = None

    def configure_debug(self, enabled: bool) -> None:
        self.debug = enabled
        configure_predict_rlm_logging(debug=enabled)
        self._interpreter_kwargs["debug"] = enabled
        with self._state_changed:
            interpreters = list(self._all_interpreters)
        for interpreter in interpreters:
            configure = getattr(interpreter, "configure_debug", None)
            if callable(configure):
                configure(enabled)

    def configure_verbose(self, enabled: bool) -> None:
        self.verbose = enabled
        configure_predict_rlm_logging(verbose=enabled)
        self._interpreter_kwargs["verbose"] = enabled
        with self._state_changed:
            interpreters = list(self._all_interpreters)
        for interpreter in interpreters:
            configure = getattr(interpreter, "configure_verbose", None)
            if callable(configure):
                configure(enabled)

    def _log_lifecycle(self, event: str, **fields: Any) -> None:
        log_pool_lifecycle(
            enabled=self.debug,
            event=event,
            pool_name=self._pool_name_prefix,
            pool_size=self.size,
            **fields,
        )

    def _notify_state_changed_locked(self) -> None:
        self._state_changed.notify_all()
        waiters = tuple(self._async_waiters)
        self._async_waiters.clear()
        for loop, waiter in waiters:
            try:
                loop.call_soon_threadsafe(self._resolve_async_waiter, waiter)
            except RuntimeError:
                pass

    @staticmethod
    def _resolve_async_waiter(waiter: asyncio.Future[None]) -> None:
        if not waiter.done():
            waiter.set_result(None)

    def _register_async_waiter_locked(
        self,
    ) -> tuple[asyncio.AbstractEventLoop, asyncio.Future[None]]:
        loop = asyncio.get_running_loop()
        registration = (loop, loop.create_future())
        self._async_waiters.add(registration)
        return registration

    async def _wait_for_state_change_async(
        self,
        registration: tuple[asyncio.AbstractEventLoop, asyncio.Future[None]],
    ) -> None:
        try:
            await registration[1]
        finally:
            with self._state_changed:
                self._async_waiters.discard(registration)

    @staticmethod
    def _reject_direct_workspace_mounts(direct_workspace_mounts: list[Any] | None) -> None:
        if direct_workspace_mounts:
            raise ValueError(
                "Workspace(mode='direct') requires a per-call SBX interpreter; "
                "prewarmed SbxPool instances cannot add workspace mounts after creation."
            )

    def __enter__(self) -> SbxPool:
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.shutdown()

    async def __aenter__(self) -> SbxPool:
        await self.astart()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.ashutdown()

    def start(self) -> None:
        if self._begin_start(allow_restart=True):
            self._finish_start()

    async def astart(self) -> None:
        if await self._begin_start_async(allow_restart=True):
            await self._finish_start_async()

    def _begin_start(self, *, allow_restart: bool) -> bool:
        with self._state_changed:
            if allow_restart:
                while self._shutting_down:
                    self._state_changed.wait()
            elif self._is_stopping_locked():
                raise RuntimeError("SbxPool is shut down")
            if self._started:
                return False
            while self._starting:
                self._state_changed.wait()
                if not allow_restart and self._is_stopping_locked():
                    raise RuntimeError("SbxPool is shut down")
                if self._started:
                    return False
            if not allow_restart and self._is_stopping_locked():
                raise RuntimeError("SbxPool is shut down")
            self._starting = True
            self._shutdown = False
            self._shutdown_requested = False
            return True

    async def _begin_start_async(self, *, allow_restart: bool) -> bool:
        while True:
            registration = None
            with self._state_changed:
                if allow_restart and self._shutting_down:
                    registration = self._register_async_waiter_locked()
                elif not allow_restart and self._is_stopping_locked():
                    raise RuntimeError("SbxPool is shut down")
                elif self._started:
                    return False
                elif self._starting:
                    registration = self._register_async_waiter_locked()
                else:
                    self._starting = True
                    self._shutdown = False
                    self._shutdown_requested = False
                    return True
            await self._wait_for_state_change_async(registration)

    def _finish_start(self) -> None:
        interpreters: list[SbxBackend] = []
        self._log_lifecycle("sbx.pool.start", size=self.size)
        try:
            for index in range(self.size):
                interpreters.append(self._create_interpreter(index))
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.size) as executor:
                futures = [executor.submit(interpreter.prewarm) for interpreter in interpreters]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
        except Exception:
            self._log_lifecycle("sbx.pool.start.error", status="error")
            self._shutdown_interpreters(interpreters, suppress_errors=True)
            with self._state_changed:
                self._drain_available_locked()
                self._all_interpreters.clear()
                self._started = False
                self._starting = False
                self._notify_state_changed_locked()
            raise

        with self._state_changed:
            self._drain_available_locked()
            self._all_interpreters = interpreters
            for interpreter in interpreters:
                self._available.put(interpreter)
            self._started = True
            self._starting = False
            self._shutdown = False
            self._notify_state_changed_locked()
        self._log_lifecycle("sbx.pool.ready", interpreters=len(interpreters))

    async def _finish_start_async(self) -> None:
        interpreters: list[SbxBackend] = []
        self._log_lifecycle("sbx.pool.start", size=self.size)
        try:
            for index in range(self.size):
                interpreters.append(self._create_interpreter(index))
            results = await asyncio.gather(
                *(interpreter.aprewarm() for interpreter in interpreters),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result
        except BaseException:
            self._log_lifecycle("sbx.pool.start.error", status="error")
            await self._ashutdown_interpreters(interpreters, suppress_errors=True)
            with self._state_changed:
                self._drain_available_locked()
                self._all_interpreters.clear()
                self._started = False
                self._starting = False
                self._notify_state_changed_locked()
            raise

        with self._state_changed:
            self._drain_available_locked()
            self._all_interpreters = interpreters
            for interpreter in interpreters:
                self._available.put(interpreter)
            self._started = True
            self._starting = False
            self._shutdown = False
            self._notify_state_changed_locked()
        self._log_lifecycle("sbx.pool.ready", interpreters=len(interpreters))

    @contextmanager
    def lease(
        self,
        *,
        tools: dict[str, Callable[..., Any]] | None = None,
        output_fields: list[dict] | None = None,
        skill_packages: list[str] | None = None,
        debug: bool | None = None,
        verbose: bool | None = None,
        runtime_hooks: list[Any] | None = None,
        on_runtime_hook_event: Callable[..., Any] | None = None,
        direct_workspace_mounts: list[Any] | None = None,
    ):
        self._reject_direct_workspace_mounts(direct_workspace_mounts)
        effective_debug = self.debug if debug is None else debug
        effective_verbose = self.verbose if verbose is None else verbose
        self._ensure_started_for_lease()
        lease_start = time.perf_counter()
        self._log_lifecycle("sbx.pool.lease.wait")
        interpreter = self._acquire_interpreter()
        self._log_lifecycle(
            "sbx.pool.lease.acquired",
            duration_ms=ms_since(lease_start),
            interpreter=type(interpreter).__name__,
        )
        try:
            interpreter.configure_runtime(
                tools=tools,
                output_fields=output_fields,
                skill_packages=skill_packages,
                debug=effective_debug,
                verbose=effective_verbose,
                runtime_hooks=runtime_hooks,
                on_runtime_hook_event=on_runtime_hook_event,
            )
            yield interpreter
        finally:
            with self._state_changed:
                stopping = (
                    self._is_stopping_locked() or interpreter not in self._all_interpreters
                )
            if stopping:
                return
            try:
                self._log_lifecycle("sbx.pool.lease.reset_start")
                interpreter.reset()
                self._log_lifecycle("sbx.pool.lease.reset_ok")
            except Exception:
                self._log_lifecycle("sbx.pool.lease.reset_error", status="error")
                interpreter = self._replace_interpreter(interpreter)
            with self._state_changed:
                if self._is_stopping_locked() or interpreter not in self._all_interpreters:
                    self._notify_state_changed_locked()
                    return
                self._available.put_nowait(interpreter)
                self._notify_state_changed_locked()
            self._log_lifecycle("sbx.pool.lease.released")

    @asynccontextmanager
    async def alease(
        self,
        *,
        tools: dict[str, Callable[..., Any]] | None = None,
        output_fields: list[dict] | None = None,
        skill_packages: list[str] | None = None,
        debug: bool | None = None,
        verbose: bool | None = None,
        runtime_hooks: list[Any] | None = None,
        on_runtime_hook_event: Callable[..., Any] | None = None,
        direct_workspace_mounts: list[Any] | None = None,
    ):
        self._reject_direct_workspace_mounts(direct_workspace_mounts)
        effective_debug = self.debug if debug is None else debug
        effective_verbose = self.verbose if verbose is None else verbose
        await self._ensure_started_for_alease()
        lease_start = time.perf_counter()
        self._log_lifecycle("sbx.pool.lease.wait")
        interpreter = await self._acquire_interpreter_async()
        try:
            interpreter = await self._ensure_interpreter_event_loop(interpreter)
        except BaseException:
            with self._state_changed:
                if interpreter in self._all_interpreters and not self._is_stopping_locked():
                    self._available.put_nowait(interpreter)
                    self._notify_state_changed_locked()
            raise
        self._log_lifecycle(
            "sbx.pool.lease.acquired",
            duration_ms=ms_since(lease_start),
            interpreter=type(interpreter).__name__,
        )
        try:
            await interpreter.aconfigure_runtime(
                tools=tools,
                output_fields=output_fields,
                skill_packages=skill_packages,
                debug=effective_debug,
                verbose=effective_verbose,
                runtime_hooks=runtime_hooks,
                on_runtime_hook_event=on_runtime_hook_event,
            )
            yield interpreter
        finally:
            release_task = asyncio.create_task(self._release_interpreter_async(interpreter))
            try:
                await asyncio.shield(release_task)
            except asyncio.CancelledError:
                await release_task
                raise

    async def _ensure_interpreter_event_loop(
        self,
        interpreter: SbxBackend,
    ) -> SbxBackend:
        previous_loop = getattr(interpreter, "_async_loop", None)
        current_loop = asyncio.get_running_loop()
        if previous_loop is None or previous_loop is current_loop:
            return interpreter
        if not previous_loop.is_closed():
            raise RuntimeError("Pooled SBX interpreter is active on another event loop")

        return await self._replace_interpreter_async(
            interpreter,
            close_loop_transport=True,
        )

    async def _release_interpreter_async(self, interpreter: SbxBackend) -> None:
        with self._state_changed:
            stopping = self._is_stopping_locked() or interpreter not in self._all_interpreters
        if stopping:
            return
        try:
            self._log_lifecycle("sbx.pool.lease.reset_start")
            await interpreter.areset()
            self._log_lifecycle("sbx.pool.lease.reset_ok")
        except Exception:
            self._log_lifecycle("sbx.pool.lease.reset_error", status="error")
            interpreter = await self._replace_interpreter_async(interpreter)
        with self._state_changed:
            if self._is_stopping_locked() or interpreter not in self._all_interpreters:
                self._notify_state_changed_locked()
                return
            self._available.put_nowait(interpreter)
            self._notify_state_changed_locked()
        self._log_lifecycle("sbx.pool.lease.released")

    async def _replace_interpreter_async(
        self,
        interpreter: SbxBackend,
        *,
        close_loop_transport: bool = False,
    ) -> SbxBackend:
        with self._state_changed:
            if interpreter not in self._all_interpreters or self._is_stopping_locked():
                raise RuntimeError("SbxPool is shut down")
            index = self._all_interpreters.index(interpreter)
            self._all_interpreters.pop(index)
            self._notify_state_changed_locked()

        async def replace() -> SbxBackend:
            replacement: SbxBackend | None = None
            try:
                if close_loop_transport:
                    interpreter._shutdown_async_transport_after_loop_closed()
                retire = getattr(interpreter, "aretire_when_host_work_finishes", None)
                if retire is None or not await retire():
                    await interpreter.ashutdown()
                replacement = self._create_interpreter(index)
                self._log_lifecycle("sbx.pool.replacement.prewarm", index=index)
                await replacement.aprewarm()
                with self._state_changed:
                    if self._is_stopping_locked():
                        discard = True
                    else:
                        self._all_interpreters.insert(index, replacement)
                        self._notify_state_changed_locked()
                        discard = False
                if discard:
                    await replacement.ashutdown()
                    raise RuntimeError("SbxPool is shut down")
                return replacement
            except BaseException as exc:
                if replacement is not None:
                    await self._ashutdown_interpreters(
                        [replacement],
                        suppress_errors=True,
                    )
                with self._state_changed:
                    self._replacement_failure = exc
                    self._started = False
                    self._shutdown_requested = True
                    self._notify_state_changed_locked()
                raise

        replacement_task = asyncio.create_task(replace())
        try:
            return await asyncio.shield(replacement_task)
        except asyncio.CancelledError:
            replacement = await replacement_task
            with self._state_changed:
                if replacement in self._all_interpreters and not self._is_stopping_locked():
                    self._available.put_nowait(replacement)
                    self._notify_state_changed_locked()
            raise

    def _replace_interpreter(self, interpreter: SbxBackend) -> SbxBackend:
        with self._state_changed:
            if interpreter not in self._all_interpreters or self._is_stopping_locked():
                raise RuntimeError("SbxPool is shut down")
            index = self._all_interpreters.index(interpreter)
            self._all_interpreters.pop(index)
            self._notify_state_changed_locked()

        replacement: SbxBackend | None = None
        try:
            retire = getattr(interpreter, "retire_when_host_work_finishes", None)
            if retire is None or not retire():
                interpreter.shutdown()
            replacement = self._create_interpreter(index)
            self._log_lifecycle("sbx.pool.replacement.prewarm", index=index)
            replacement.prewarm()
            with self._state_changed:
                if self._is_stopping_locked():
                    discard = True
                else:
                    self._all_interpreters.insert(index, replacement)
                    self._notify_state_changed_locked()
                    discard = False
            if discard:
                replacement.shutdown()
                raise RuntimeError("SbxPool is shut down")
            return replacement
        except BaseException as exc:
            if replacement is not None:
                self._shutdown_interpreters([replacement], suppress_errors=True)
            with self._state_changed:
                self._replacement_failure = exc
                self._started = False
                self._shutdown_requested = True
                self._notify_state_changed_locked()
            raise

    def shutdown(self) -> None:
        self._log_lifecycle("sbx.pool.shutdown.start")
        with self._state_changed:
            while self._starting:
                self._shutdown_requested = True
                self._notify_state_changed_locked()
                self._state_changed.wait()
            if self._shutdown:
                return
            self._shutdown = True
            self._shutdown_requested = False
            self._shutting_down = True
            interpreters = list(self._all_interpreters)
            self._drain_available_locked()
            self._all_interpreters.clear()
            self._started = False
            self._notify_state_changed_locked()

        try:
            self._shutdown_interpreters(interpreters)
        finally:
            with self._state_changed:
                self._shutting_down = False
                self._notify_state_changed_locked()
        self._log_lifecycle("sbx.pool.shutdown.complete", interpreters=len(interpreters))

    async def ashutdown(self) -> None:
        shutdown_task = asyncio.create_task(self._ashutdown())
        try:
            await asyncio.shield(shutdown_task)
        except asyncio.CancelledError:
            await shutdown_task
            raise

    async def _ashutdown(self) -> None:
        self._log_lifecycle("sbx.pool.shutdown.start")
        while True:
            with self._state_changed:
                if not self._starting:
                    break
                self._shutdown_requested = True
                self._notify_state_changed_locked()
                registration = self._register_async_waiter_locked()
            await self._wait_for_state_change_async(registration)

        with self._state_changed:
            if self._shutdown:
                return
            self._shutdown = True
            self._shutdown_requested = False
            self._shutting_down = True
            interpreters = list(self._all_interpreters)
            self._drain_available_locked()
            self._all_interpreters.clear()
            self._started = False
            self._notify_state_changed_locked()

        try:
            await self._ashutdown_interpreters(interpreters)
        finally:
            with self._state_changed:
                self._shutting_down = False
                self._notify_state_changed_locked()
        self._log_lifecycle("sbx.pool.shutdown.complete", interpreters=len(interpreters))

    def _ensure_started_for_lease(self) -> None:
        if self._begin_start(allow_restart=False):
            self._finish_start()
        with self._state_changed:
            if self._is_stopping_locked() or not self._started:
                raise RuntimeError("SbxPool is shut down")

    async def _ensure_started_for_alease(self) -> None:
        if await self._begin_start_async(allow_restart=False):
            await self._finish_start_async()
        with self._state_changed:
            if self._is_stopping_locked() or not self._started:
                raise RuntimeError("SbxPool is shut down")

    def _acquire_interpreter(self) -> SbxBackend:
        with self._state_changed:
            while True:
                if self._is_stopping_locked() or not self._started:
                    self._raise_unavailable_locked()
                try:
                    return self._available.get_nowait()
                except queue.Empty:
                    self._state_changed.wait()

    async def _acquire_interpreter_async(self) -> SbxBackend:
        while True:
            with self._state_changed:
                if self._is_stopping_locked() or not self._started:
                    self._raise_unavailable_locked()
                try:
                    return self._available.get_nowait()
                except queue.Empty:
                    registration = self._register_async_waiter_locked()
            await self._wait_for_state_change_async(registration)

    def _raise_unavailable_locked(self) -> None:
        if self._replacement_failure is not None:
            raise RuntimeError("SbxPool replacement failed") from self._replacement_failure
        raise RuntimeError("SbxPool is shut down")

    def _is_stopping_locked(self) -> bool:
        return self._shutdown or self._shutdown_requested or self._shutting_down

    def _create_interpreter(self, index: int) -> SbxBackend:
        kwargs = dict(self._interpreter_kwargs)
        if self.size > 1:
            kwargs["config"] = self.config.model_copy(
                update={"name": f"{self._pool_name_prefix}-{index}"}
            )
        if self._staging_root is not None:
            kwargs["_staging_root"] = self._staging_root / f"runner-{index}"
        return SbxBackend(**kwargs)

    def _drain_available_locked(self) -> None:
        while True:
            try:
                self._available.get_nowait()
            except queue.Empty:
                break

    def _shutdown_interpreters(
        self,
        interpreters: list[SbxBackend],
        *,
        suppress_errors: bool = False,
    ) -> None:
        first_error: BaseException | None = None
        def shutdown(interpreter: SbxBackend) -> None:
            retire = getattr(interpreter, "retire_when_host_work_finishes", None)
            if retire is None or not retire():
                interpreter.shutdown()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, len(interpreters))
        ) as executor:
            futures = [executor.submit(shutdown, interpreter) for interpreter in interpreters]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
        if first_error is not None and not suppress_errors:
            raise first_error

    async def _ashutdown_interpreters(
        self,
        interpreters: list[SbxBackend],
        *,
        suppress_errors: bool = False,
    ) -> None:
        async def shutdown(interpreter: SbxBackend) -> None:
            retire = getattr(interpreter, "aretire_when_host_work_finishes", None)
            if retire is None or not await retire():
                await interpreter.ashutdown()

        results = await asyncio.gather(
            *(shutdown(interpreter) for interpreter in interpreters),
            return_exceptions=True,
        )
        first_error = next(
            (result for result in results if isinstance(result, BaseException)),
            None,
        )
        if first_error is not None and not suppress_errors:
            raise first_error
