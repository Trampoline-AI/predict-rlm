"""Thread-safe pool for SBX interpreters."""

from __future__ import annotations

import concurrent.futures
import queue
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from predict_rlm._logging import configure_predict_rlm_logging
from predict_rlm.trace import ms_since

from .base import SbxConfig
from .sbx_logging import log_pool_lifecycle

if TYPE_CHECKING:
    from .sbx import SbxInterpreter


class SbxPool:
    """Thread-safe pool of prewarmed Docker Sandboxes interpreters."""

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
        self._available: queue.Queue[SbxInterpreter] = queue.Queue(maxsize=size)
        self._all_interpreters: list[SbxInterpreter] = []
        self._lock = threading.Lock()
        self._state_changed = threading.Condition(self._lock)
        self._started = False
        self._starting = False
        self._shutdown = False
        self._shutdown_requested = False
        self._shutting_down = False

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

    def __enter__(self) -> SbxPool:
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.shutdown()

    def start(self) -> None:
        if self._begin_start(allow_restart=True):
            self._finish_start()

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

    def _finish_start(self) -> None:
        interpreters: list[SbxInterpreter] = []
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
                self._state_changed.notify_all()
            raise

        with self._state_changed:
            self._drain_available_locked()
            self._all_interpreters = interpreters
            for interpreter in interpreters:
                self._available.put(interpreter)
            self._started = True
            self._starting = False
            self._shutdown = False
            self._state_changed.notify_all()
        self._log_lifecycle("sbx.pool.ready", interpreters=len(interpreters))

    @contextmanager
    def lease(
        self,
        *,
        tools: dict[str, Callable[..., Any]] | None = None,
        output_fields: list[dict] | None = None,
        debug: bool | None = None,
        verbose: bool | None = None,
    ):
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
                debug=effective_debug,
                verbose=effective_verbose,
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
                interpreter.shutdown()
                with self._state_changed:
                    if self._is_stopping_locked() or interpreter not in self._all_interpreters:
                        self._state_changed.notify_all()
                        return
                    index = self._all_interpreters.index(interpreter)
                    replacement = self._create_interpreter(index)
                    self._log_lifecycle("sbx.pool.replacement.prewarm", index=index)
                    replacement.prewarm()
                    self._all_interpreters[index] = replacement
                    interpreter = replacement
            with self._state_changed:
                if self._is_stopping_locked() or interpreter not in self._all_interpreters:
                    self._state_changed.notify_all()
                    return
                self._available.put_nowait(interpreter)
                self._state_changed.notify()
            self._log_lifecycle("sbx.pool.lease.released")

    def shutdown(self) -> None:
        self._log_lifecycle("sbx.pool.shutdown.start")
        with self._state_changed:
            while self._starting:
                self._shutdown_requested = True
                self._state_changed.notify_all()
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
            self._state_changed.notify_all()

        try:
            self._shutdown_interpreters(interpreters)
        finally:
            with self._state_changed:
                self._shutting_down = False
                self._state_changed.notify_all()
        self._log_lifecycle("sbx.pool.shutdown.complete", interpreters=len(interpreters))

    def _ensure_started_for_lease(self) -> None:
        if self._begin_start(allow_restart=False):
            self._finish_start()
        with self._state_changed:
            if self._is_stopping_locked() or not self._started:
                raise RuntimeError("SbxPool is shut down")

    def _acquire_interpreter(self) -> SbxInterpreter:
        with self._state_changed:
            while True:
                if self._is_stopping_locked() or not self._started:
                    raise RuntimeError("SbxPool is shut down")
                try:
                    return self._available.get_nowait()
                except queue.Empty:
                    self._state_changed.wait()

    def _is_stopping_locked(self) -> bool:
        return self._shutdown or self._shutdown_requested or self._shutting_down

    def _create_interpreter(self, index: int) -> SbxInterpreter:
        from .sbx import SbxInterpreter

        kwargs = dict(self._interpreter_kwargs)
        if self.size > 1:
            kwargs["config"] = self.config.model_copy(
                update={"name": f"{self._pool_name_prefix}-{index}"}
            )
        if self._staging_root is not None:
            kwargs["_staging_root"] = self._staging_root / f"runner-{index}"
        return SbxInterpreter(**kwargs)

    def _drain_available_locked(self) -> None:
        while True:
            try:
                self._available.get_nowait()
            except queue.Empty:
                break

    def _shutdown_interpreters(
        self,
        interpreters: list[SbxInterpreter],
        *,
        suppress_errors: bool = False,
    ) -> None:
        first_error: BaseException | None = None
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, len(interpreters))
        ) as executor:
            futures = [executor.submit(interpreter.shutdown) for interpreter in interpreters]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
        if first_error is not None and not suppress_errors:
            raise first_error
