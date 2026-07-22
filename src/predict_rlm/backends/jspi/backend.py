"""JSPI-enabled Python interpreter with concurrent async tool execution.

JSPI (JavaScript Promise Integration) enables synchronous-looking calls from
Python to JavaScript in the Pyodide/WASM environment. This interpreter extends
DSPy's PythonInterpreter with:
- JSPI V8 flag for tool calls in WASM
- Concurrent tool execution using asyncio
- All tools are async for use with asyncio.gather()
- Skill package installation in the sandbox

Protocol: JSON-RPC 2.0 (matching dspy 3.1.3+)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import inspect
import json
import logging
import os
import select
import shutil
import signal
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput
from dspy.primitives.python_interpreter import PythonInterpreter
from pydantic import BaseModel

from predict_rlm._logging import (
    configure_predict_rlm_logging,
    emit_trace_error,
    emit_trace_result,
    emit_trace_tool_call,
    format_log_fields,
    interpreter_result_logging_enabled,
    live_tool_call_logging_enabled,
)
from predict_rlm._shared import strip_code_fences
from predict_rlm.backends.base import (
    STALE_RESPONSE_DISCARD_LIMIT,
    BackendExecutionGate,
    SandboxExecutionError,
    SandboxFatalError,
)
from predict_rlm.execution_timeout import (
    ITERATION_TIMEOUT_FAILURE_CLASS,
    RecoverableExecutionTimeout,
    format_recoverable_timeout_result,
    recoverable_timeout_host_deadline_seconds,
    resolve_execution_timeout,
)
from predict_rlm.runtime import (
    ArtifactFileInfo,
    SyncWorker,
    SyncWorkerTracker,
    callable_has_sync_leaf,
    host_sync_worker_policy,
    invoke_host_callable,
)
from predict_rlm.serialization import to_plain_data
from predict_rlm.telemetry import (
    TelemetryContext,
    make_span_id,
    reset_current_telemetry_context,
    set_current_telemetry_context,
)

if TYPE_CHECKING:
    from os import PathLike
    from typing import Callable


logger = logging.getLogger(__name__)


def _set_future_result(future: asyncio.Future[Any], value: Any) -> None:
    if not future.done():
        future.set_result(value)


def _set_future_exception(future: asyncio.Future[Any], exc: BaseException) -> None:
    if not future.done():
        future.set_exception(exc)



# JSON-RPC 2.0 helpers (local to avoid coupling to dspy internals)
JSONRPC_APP_ERRORS = {
    "SyntaxError": -32000,
    "NameError": -32001,
    "TypeError": -32002,
    "ValueError": -32003,
    "AttributeError": -32004,
    "IndexError": -32005,
    "KeyError": -32006,
    "RuntimeError": -32007,
    "CodeInterpreterError": -32008,
    "Unknown": -32099,
}

def _jsonrpc_result(result: Any, id: int | str) -> str:
    return json.dumps({"jsonrpc": "2.0", "result": result, "id": id})


def _jsonrpc_error(code: int, message: str, id: int | str, data: dict | None = None) -> str:
    err = {"code": code, "message": message}
    if data:
        err["data"] = data
    return json.dumps({"jsonrpc": "2.0", "error": err, "id": id})


# Default domains the sandbox can access
# PyPI domains are required for micropip to install packages (pydantic, etc.)
DEFAULT_ALLOWED_DOMAINS: list[str] = [
    "pypi.org",
    "files.pythonhosted.org",
    "cdn.jsdelivr.net",  # Pyodide packages
]

# Path to our custom runner.js with concurrent tool support
RUNNER_PATH = Path(__file__).with_name("payload.js")

# Wall-clock budget for a single JSON-RPC request/response round-trip with the
# deno subprocess (health_check, mount, sync_file, etc). Without this ceiling,
# ``_send_request`` used to pass ``timeout=None`` and block indefinitely if
# deno emitted partial bytes then stalled — freezing the entire asyncio event
# loop when called from ``_health_check`` inside an ``aexecute`` coroutine.
# Exposed as a module-level knob so callers / tests can override.
DENO_REQUEST_TIMEOUT_SEC: float = 30.0

# Wall-clock budget for a single host-side tool call (recalculate, render,
# predict, user-provided tools). Exceeding this returns an error value to
# the sandbox rather than letting the overall ``_execute_with_timeout``
# wall expire and kill the Deno subprocess. The contract is: tool errors
# are RECOVERABLE (the RLM sees "[Error] tool timed out after …s" and
# can rewrite its code), while sandbox death is NOT (mounts lost, etc).
# Must be less than ``exec_timeout`` so the tool guard fires first.
TOOL_CALL_TIMEOUT_SEC: float = 180.0


def _needs_jspi_flag() -> bool:
    """Check if Deno's V8 needs --experimental-wasm-jspi.

    JSPI shipped unflagged in V8 13.7 (Chrome 137). Older V8 versions
    need the flag; newer ones don't recognize it and will error.
    """
    import subprocess

    try:
        out = subprocess.check_output(["deno", "--version"], text=True)
        for line in out.splitlines():
            if line.startswith("v8 "):
                # e.g. "v8 13.0.245.12-rusty"
                parts = line.split()[1].split(".")
                major, minor = int(parts[0]), int(parts[1])
                return (major, minor) < (13, 7)
    except Exception:
        pass
    # If we can't determine, include the flag — it's needed on older versions
    # and will produce a clear error on newer ones.
    return True


class JspiBackend(SyncWorkerTracker, PythonInterpreter):
    """PythonInterpreter with JSPI and concurrent async tool execution.

    JSPI (JavaScript Promise Integration) allows Python code running in the
    Pyodide WASM sandbox to make synchronous-looking calls to async JavaScript
    functions (like our tool bridge). Without JSPI, tool calls would fail with
    "can't call async function from sync context" errors.

    Concurrent Tool Execution:
        All tools are async. Multiple tool calls via asyncio.gather() are
        executed concurrently on the host side. Example:

        ```python
        import asyncio
        results = await asyncio.gather(
            predict("img: dspy.Image -> text", img=img1),
            predict("img: dspy.Image -> text", img=img2),
            predict("img: dspy.Image -> text", img=img3),
        )
        ```

    Network Access:
        By default, the sandbox has no network access. Use ``allowed_domains``
        to specify which domains the sandbox can reach.

    Skill Packages:
        Pass ``skill_packages`` to pre-install additional PyPI packages in the
        sandbox at startup. These are installed alongside the default packages
        (pandas, pydantic) via micropip.

    Example:
        ```python
        interpreter = JspiBackend(
            tools={"my_tool": my_tool_func},
            allowed_domains=["api.example.com"],
            skill_packages=["pdfplumber", "openpyxl"],
        )
        result = interpreter.execute(code)
        ```
    """

    # Max concurrent Deno/Pyodide sandboxes system-wide.
    # Each uses ~800MB RAM; 50 × 800MB ≈ 40GB.
    MAX_CONCURRENT_SANDBOXES = 50
    _sandbox_semaphore: asyncio.Semaphore | None = None

    @classmethod
    def _get_semaphore(cls) -> asyncio.Semaphore:
        if cls._sandbox_semaphore is None:
            cls._sandbox_semaphore = asyncio.Semaphore(cls.MAX_CONCURRENT_SANDBOXES)
        return cls._sandbox_semaphore

    def __init__(
        self,
        *,
        allowed_domains: list[str] | None = None,
        tools: dict[str, Callable[..., Any]] | None = None,
        output_fields: list[dict] | None = None,
        preinstall_packages: bool = True,
        skill_packages: list[str] | None = None,
        debug: bool = False,
        verbose: bool = False,
        # Advanced options (passed through to PythonInterpreter)
        deno_command: list[str] | None = None,
        enable_read_paths: list[PathLike | str] | None = None,
        enable_write_paths: list[PathLike | str] | None = None,
        extra_read_paths: list[PathLike | str] | None = None,
        extra_write_paths: list[PathLike | str] | None = None,
        enable_env_vars: list[str] | None = None,
        sync_files: bool = True,
        exec_timeout: float = 300.0,
        telemetry_context: TelemetryContext | None = None,
    ) -> None:
        """Initialize interpreter with JSPI and concurrent tool support.

        Args:
            allowed_domains: Domains/IPs the sandbox can access via network.
                            Empty list (default) means no network access.
                            Example: ["api.example.com", "192.168.1.100:8080"]
            tools: Tool functions callable from sandbox code. Can be sync or async.
            output_fields: Output field definitions for typed SUBMIT.
            preinstall_packages: If True, pre-install pandas and pydantic at startup.
                                Set to False for faster startup in tests that don't need them.
            skill_packages: Additional PyPI packages to install in the sandbox.
                           These come from Skill definitions and are installed
                           alongside the default packages via micropip.
            debug: If True, emit sandbox lifecycle diagnostics through logging.
                  Defaults to False.
            verbose: If True, print REPL output, errors, SUBMIT payloads, and
                  tool calls to stderr in real-time. Defaults to False.
            deno_command: Custom Deno command. If provided, JSPI flag is NOT
                         automatically added (assumes you've configured it).
            enable_read_paths: Files/directories to allow reading from.
                              These are mounted automatically by the parent class.
            enable_write_paths: Files/directories to allow writing to.
                               These are synced automatically by the parent class.
            extra_read_paths: Additional paths for Deno --allow-read permissions
                             without triggering parent's auto-mount.
            extra_write_paths: Additional paths for Deno --allow-write permissions
                              without triggering parent's auto-sync.
            enable_env_vars: Environment variable names to expose.
            sync_files: Whether to sync file changes back to host.
            exec_timeout: Wall-clock seconds to allow for a single REPL
                         execute call before killing the sandbox subprocess.
                         Defaults to 300s. Must exceed the slowest expected
                         host-side tool call: e.g. ``recalculate()`` on a
                         workbook with whole-column formulas can take 2-3
                         minutes via LibreOffice. When this fires it also
                         tears down the Deno subprocess (mounts are lost —
                         the interpreter can't be reused), so the value
                         should be generous; the guard is only meant to
                         catch genuinely pathological code (regex
                         catastrophic backtracking, infinite loops, or a
                         host tool that never returns).
            telemetry_context: Optional best-effort OTel-shaped telemetry
                         context. Telemetry failures are always ignored.
        """
        self._telemetry_context = telemetry_context
        self._interpreter_id = make_span_id("jspi")
        self._last_semaphore_attrs: dict[str, Any] = {}
        # Merge default domains with user-provided domains
        network_access = list(DEFAULT_ALLOWED_DOMAINS) if preinstall_packages else []
        if allowed_domains:
            network_access.extend(allowed_domains)

        # Pass preinstall flag and skill packages to runner via env vars
        env_vars = list(enable_env_vars or [])

        if preinstall_packages:
            os.environ["PYODIDE_PREINSTALL"] = "1"
            env_vars.append("PYODIDE_PREINSTALL")
        else:
            os.environ["PYODIDE_PREINSTALL"] = "0"
            env_vars.append("PYODIDE_PREINSTALL")

        # Skill packages are passed as a comma-separated env var
        if skill_packages:
            os.environ["SKILL_PACKAGES"] = ",".join(skill_packages)
            env_vars.append("SKILL_PACKAGES")
        elif "SKILL_PACKAGES" in os.environ:
            del os.environ["SKILL_PACKAGES"]

        self._debug = debug
        self._verbose = verbose
        self._preinstall_packages = preinstall_packages
        self.skill_packages = list(dict.fromkeys(skill_packages or ()))
        self._installed_skill_packages = (
            set(self.skill_packages) if preinstall_packages else set()
        )
        configure_predict_rlm_logging(
            debug=True if debug else None,
            verbose=True if verbose else None,
        )

        # Merge extra paths into Deno permissions (but NOT into parent's
        # enable_read_paths/enable_write_paths which trigger auto-mount/sync)
        all_read_paths = list(enable_read_paths or []) + list(extra_read_paths or [])
        all_write_paths = list(enable_write_paths or []) + list(extra_write_paths or [])

        # Scan tools for SyncedFile annotations with custom host_dir paths
        # and add them to Deno permissions so the runner can write there.
        if tools:
            from predict_rlm.files import get_synced_file_params

            for tool_fn in tools.values():
                if getattr(tool_fn, "__predict_rlm_synced_file_operation__", False):
                    continue
                for sf in get_synced_file_params(tool_fn).values():
                    if sf.host_dir is not None:
                        all_write_paths.append(sf.host_dir)
                        all_read_paths.append(sf.host_dir)

        # Build custom deno command if not provided
        if deno_command is None:
            deno_command = self._build_deno_command(
                all_read_paths,
                all_write_paths,
                network_access,
                env_vars,
            )

        super().__init__(
            deno_command=deno_command,
            enable_read_paths=enable_read_paths,
            enable_write_paths=enable_write_paths,
            enable_env_vars=enable_env_vars,
            enable_network_access=network_access if network_access else None,
            sync_files=sync_files,
            tools=tools,
            output_fields=output_fields,
        )
        # Raw-fd I/O state (initialised in _ensure_deno_process)
        self._stdout_fd: int = -1
        self._stdin_fd: int = -1
        self._read_buf: str = ""
        # Per-interpreter thread pool for sync tool calls (avoids starving
        # the shared default executor when many interpreters run concurrently)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        # Pending file-sync operations requested by tools during execution.
        # Maps request ID → asyncio.Future resolved by the execute loop.
        self._pending_file_ops: dict[int, asyncio.Future] = {}
        self._host_tool_tasks: set[asyncio.Task[Any]] = set()
        self._active_tool_count = 0
        # Per-execute wall-clock timeout (see __init__ docstring).
        self._exec_timeout = exec_timeout
        self._execution_gate = BackendExecutionGate("JSPI backend")
        self._post_execute_hooks: list[Callable[[Any], Any]] = []
        self._active_execute_request_id: int | None = None

    def configure_debug(self, enabled: bool) -> None:
        self._debug = enabled
        configure_predict_rlm_logging(debug=enabled)

    def configure_verbose(self, enabled: bool) -> None:
        self._verbose = enabled
        configure_predict_rlm_logging(verbose=enabled)

    def configure_runtime(
        self,
        *,
        debug: bool | None = None,
        verbose: bool | None = None,
        tools: dict[str, Callable[..., Any]] | None = None,
        output_fields: list[dict] | None = None,
    ) -> None:
        if debug is not None:
            self.configure_debug(debug)
        if verbose is not None:
            self.configure_verbose(verbose)
        if tools is not None:
            self.tools = tools
        if output_fields is not None:
            self.output_fields = output_fields

    def _log_lifecycle(self, event: str, **fields: Any) -> None:
        if not getattr(self, "_debug", False):
            return
        logger.debug(
            "%s%s",
            event,
            format_log_fields(
                {
                    "backend": "jspi",
                    "interpreter_id": getattr(self, "_interpreter_id", None),
                    "process_pid": self._telemetry_process_pid(),
                    **fields,
                }
            ),
        )

    def _log_partial_output(self, output: str, **fields: Any) -> None:
        if not getattr(self, "_debug", False) or not output:
            return
        logger.debug(
            "sandbox.partial_output%s\n%s",
            format_log_fields(
                {
                    "backend": "jspi",
                    "interpreter_id": getattr(self, "_interpreter_id", None),
                    "process_pid": self._telemetry_process_pid(),
                    "chars": len(output),
                    **fields,
                }
            ),
            output.rstrip(),
        )

    def _telemetry_process_pid(self) -> int | None:
        process = getattr(self, "deno_process", None)
        pid = getattr(process, "pid", None)
        return pid if isinstance(pid, int) else None

    def _telemetry_pending_file_ops_count(self) -> int:
        pending_file_ops = getattr(self, "_pending_file_ops", {})
        return len(pending_file_ops) if pending_file_ops is not None else 0

    def _telemetry_pending_tool_count(self) -> int:
        return len(self._live_host_tool_tasks())

    def _host_tool_task_set(self) -> set[asyncio.Task[Any]]:
        tasks = getattr(self, "_host_tool_tasks", None)
        if tasks is None:
            tasks = set()
            self._host_tool_tasks = tasks
        return tasks

    def _live_host_tool_tasks(self) -> tuple[asyncio.Task[Any], ...]:
        return tuple(
            task
            for task in JspiBackend._host_tool_task_set(self)
            if not task.done()
        )

    def _track_host_tool_task(self, task: asyncio.Task[Any]) -> None:
        tasks = JspiBackend._host_tool_task_set(self)
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    def _live_host_sync_workers(self) -> tuple[SyncWorker, ...]:
        live_sync_workers = getattr(self, "_live_sync_workers", None)
        return live_sync_workers() if callable(live_sync_workers) else ()

    async def await_host_work(self, *, cancel: bool = False) -> None:
        async def wait() -> None:
            while True:
                tasks = JspiBackend._live_host_tool_tasks(self)
                if cancel:
                    for task in tasks:
                        task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

                workers = JspiBackend._live_host_sync_workers(self)
                if workers:
                    await asyncio.gather(
                        *(worker.wait() for worker in workers),
                        return_exceptions=True,
                    )

                if (
                    not JspiBackend._live_host_tool_tasks(self)
                    and not JspiBackend._live_host_sync_workers(self)
                ):
                    return

        if not cancel:
            await wait()
            return

        cleanup = asyncio.create_task(wait())
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise

    def _wait_for_host_work_sync(self, *, cancel: bool = False) -> None:
        tasks = JspiBackend._live_host_tool_tasks(self)
        workers = JspiBackend._live_host_sync_workers(self)
        if not tasks and not workers:
            return
        if not tasks:
            for worker in workers:
                try:
                    worker.future.result()
                except BaseException:
                    pass
            return
        loop = tasks[0].get_loop()
        if loop.is_closed():
            raise RuntimeError("JSPI host tool cleanup outlived its event loop")
        import nest_asyncio

        nest_asyncio.apply(loop)
        loop.run_until_complete(JspiBackend.await_host_work(self, cancel=cancel))

    def _telemetry_attrs(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        attrs: dict[str, Any] = {
            "predict_rlm.interpreter_id": getattr(self, "_interpreter_id", None),
        }
        pid = self._telemetry_process_pid()
        if pid is not None:
            attrs["process.pid"] = pid
        if extra:
            attrs.update(extra)
        return {key: value for key, value in attrs.items() if value is not None}

    def _write_telemetry_span(
        self,
        name: str,
        *,
        status: str | dict[str, Any] = "OK",
        attributes: dict[str, Any] | None = None,
        start_time_unix_nano: int | None = None,
        end_time_unix_nano: int | None = None,
    ) -> None:
        telemetry_context = getattr(self, "_telemetry_context", None)
        if telemetry_context is None:
            return
        try:
            telemetry_context.write_span(
                name,
                event_domain="sandbox",
                status=status,
                start_time_unix_nano=start_time_unix_nano,
                end_time_unix_nano=end_time_unix_nano,
                attributes=self._telemetry_attrs(attributes),
            )
        except Exception:
            return

    def _ensure_deno_process(self) -> None:
        """Override to capture raw fds for non-blocking I/O."""
        # Clean up old event loop watchers before replacing fds
        try:
            loop = asyncio.get_running_loop()
            if self._stdout_fd >= 0:
                loop.remove_reader(self._stdout_fd)
            if self._stdin_fd >= 0:
                loop.remove_writer(self._stdin_fd)
        except (RuntimeError, ValueError):
            pass  # No running loop or fd already closed

        prev = self.deno_process
        if prev is None or prev.poll() is not None:
            self._stdin_fd = -1
            self._stdout_fd = -1
            self._read_buf = ""
            self._installed_skill_packages = (
                set(self.skill_packages) if self._preinstall_packages else set()
            )
            self._log_lifecycle("jspi.deno.starting")
        super()._ensure_deno_process()
        if self.deno_process is not None and self.deno_process is not prev:
            self._stdout_fd = self.deno_process.stdout.fileno()
            self._stdin_fd = self.deno_process.stdin.fileno()
            self._read_buf = ""
            # Set stdin to non-blocking for async writes
            import fcntl

            flags = fcntl.fcntl(self._stdin_fd, fcntl.F_GETFL)
            fcntl.fcntl(self._stdin_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            # Reset executor — old threads may be stuck on a dead process
            self._executor.shutdown(wait=False)
            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            self._write_telemetry_span("sandbox.deno.start")
            self._log_lifecycle("jspi.deno.started")

    async def _aensure_deno_process(self) -> None:
        """Start Deno and complete its health check without sync RPC helpers."""
        process = self.deno_process
        if process is not None and process.poll() is None:
            return

        loop = asyncio.get_running_loop()
        if self._stdout_fd >= 0:
            loop.remove_reader(self._stdout_fd)
        if self._stdin_fd >= 0:
            loop.remove_writer(self._stdin_fd)

        self._stdin_fd = -1
        self._stdout_fd = -1
        self._read_buf = ""
        self._tools_registered = False
        self._mounted_files = False
        self._installed_skill_packages = (
            set(self.skill_packages) if self._preinstall_packages else set()
        )
        self._log_lifecycle("jspi.deno.starting")

        import subprocess

        try:
            self.deno_process = subprocess.Popen(
                self.deno_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="UTF-8",
                env=os.environ.copy(),
            )
        except FileNotFoundError as exc:
            raise CodeInterpreterError(
                "Deno executable not found. Please install Deno to proceed.\n"
                "Installation instructions:\n"
                "> curl -fsSL https://deno.land/install.sh | sh\n"
                "*or*, on macOS with Homebrew:\n"
                "> brew install deno\n"
                "For additional configurations: "
                "https://docs.deno.com/runtime/getting_started/installation/"
            ) from exc

        if self.deno_process.stdin is None or self.deno_process.stdout is None:
            raise CodeInterpreterError("Deno subprocess did not expose stdin/stdout pipes")
        self._stdout_fd = self.deno_process.stdout.fileno()
        self._stdin_fd = self.deno_process.stdin.fileno()
        import fcntl

        flags = fcntl.fcntl(self._stdin_fd, fcntl.F_GETFL)
        fcntl.fcntl(self._stdin_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        self._executor.shutdown(wait=False)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            await self._ahealth_check()
        except BaseException:
            await self._akill_sandbox()
            raise
        self._write_telemetry_span("sandbox.deno.start")
        self._log_lifecycle("jspi.deno.started")

    async def aensure_ready(self) -> None:
        """Ensure the Deno/Pyodide runtime is ready to accept control requests."""
        await self._aensure_deno_process()

    async def astart(self) -> None:
        await self._aensure_deno_process()

    def shutdown(self) -> None:
        """Shut down the Deno subprocess with timeout guards.

        Overrides PythonInterpreter.shutdown() which has no timeout on
        stdin.flush() or .wait(), causing hangs when Deno is unresponsive.
        """
        import subprocess

        self._wait_for_host_work_sync(cancel=True)

        process = self.deno_process
        if process and process.poll() is None:
            self._write_telemetry_span("sandbox.shutdown.start")
            self._log_lifecycle("jspi.shutdown.start")
            killed = False
            try:
                self._write_stdin(json.dumps({"jsonrpc": "2.0", "method": "shutdown"}) + "\n")
                self.deno_process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                shutdown_wait_timeout = (
                    0.2
                    if isinstance(process, subprocess.Popen)
                    or getattr(self, "_prefer_fast_shutdown", False)
                    else 5
                )
                process.wait(timeout=shutdown_wait_timeout)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                    killed = True
                    self._write_telemetry_span(
                        "sandbox.shutdown.kill",
                        attributes={"process.kill_result": "sent"},
                    )
                    self._log_lifecycle("jspi.shutdown.kill", kill_result="sent")
                except Exception as exc:
                    self._write_telemetry_span(
                        "sandbox.shutdown.kill",
                        status={"code": "ERROR", "message": str(exc)},
                        attributes={"process.kill_result": "error"},
                    )
                    self._log_lifecycle(
                        "jspi.shutdown.kill",
                        status="error",
                        error_type=type(exc).__name__,
                    )
                process.wait()
            self._write_telemetry_span(
                "sandbox.shutdown.complete",
                attributes={"process.kill_result": "killed" if killed else "not_needed"},
            )
            self._log_lifecycle(
                "jspi.shutdown.complete",
                kill_result="killed" if killed else "not_needed",
            )
        self.deno_process = None
        self._owner_thread = None
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)

    async def ashutdown(self) -> None:
        """Shut down Deno without blocking the event loop on process waits."""
        await self.await_host_work(cancel=True)
        process = self.deno_process
        if process is not None and process.poll() is None:
            self._write_telemetry_span("sandbox.shutdown.start")
            self._log_lifecycle("jspi.shutdown.start")
            killed = False
            try:
                message = json.dumps({"jsonrpc": "2.0", "method": "shutdown"}) + "\n"
                await self._write_stdin_async(message)
            except (BrokenPipeError, OSError, CodeInterpreterError):
                pass

            if self._stdin_fd >= 0:
                try:
                    process.stdin.close()
                except (AttributeError, OSError):
                    pass
                self._stdin_fd = -1

            shutdown_wait_timeout = (
                0.2
                if getattr(self, "_prefer_fast_shutdown", False)
                or type(process).__module__ == "subprocess"
                else 5.0
            )
            if not await self._await_process_exit(process, shutdown_wait_timeout):
                try:
                    process.kill()
                    killed = True
                except Exception:
                    pass
                await self._await_process_exit(process, 1.0)
            self._write_telemetry_span(
                "sandbox.shutdown.complete",
                attributes={"process.kill_result": "killed" if killed else "not_needed"},
            )
            self._log_lifecycle(
                "jspi.shutdown.complete",
                kill_result="killed" if killed else "not_needed",
            )

        self.deno_process = None
        self._owner_thread = None
        self._stdin_fd = -1
        self._stdout_fd = -1
        self._read_buf = ""
        for future in list(self._pending_file_ops.values()):
            if not future.done():
                future.cancel()
        self._pending_file_ops.clear()
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)

    async def _await_process_exit(self, process: Any, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(0.01, remaining))
        return True

    def _build_deno_command(
        self,
        read_paths: list[PathLike | str],
        write_paths: list[PathLike | str],
        network_access: list[str],
        env_vars: list[str],
    ) -> list[str]:
        """Build Deno command with JSPI flag and our custom runner."""
        args = ["deno", "run"]

        # JSPI shipped unflagged in V8 13.7 (Chrome 137); older V8 needs the flag
        if _needs_jspi_flag():
            args.append("--v8-flags=--experimental-wasm-jspi")

        # Allow reading our runner and explicitly enabled paths
        allowed_read = [str(RUNNER_PATH), str(RUNNER_PATH.parent)]

        # Add Deno cache directories (may be multiple on different platforms)
        allowed_read.extend(self._get_deno_dir())

        # Add user-specified read/write paths
        allowed_read.extend(str(p) for p in read_paths)
        allowed_read.extend(str(p) for p in write_paths)

        # Allow reading temp dirs so @file_sync tools can mount files back
        import tempfile as _tempfile

        allowed_read.append(_tempfile.gettempdir())
        allowed_read.append("/tmp")

        if allowed_read:
            args.append(f"--allow-read={','.join(allowed_read)}")

        # Allow writing to Deno cache (for Pyodide package caching),
        # the sandbox dir (for nodeModulesDir micropip), temp dirs (for Pyodide
        # temporary files during package installation), and user paths
        import tempfile

        allowed_write = list(self._get_deno_dir())
        allowed_write.append(str(RUNNER_PATH.parent))
        allowed_write.append(tempfile.gettempdir())
        allowed_write.append("/tmp")
        allowed_write.extend(str(p) for p in write_paths)

        if allowed_write:
            args.append(f"--allow-write={','.join(allowed_write)}")

        if network_access:
            args.append(f"--allow-net={','.join(network_access)}")

        # Allow env var access — Deno and Pyodide need HOME, DENO_DIR, etc.
        # during startup, so we allow all env vars rather than restricting.
        args.append("--allow-env")

        # Use Deno's global cache for npm: specifiers instead of a parent
        # node_modules (e.g. one created by `prisma generate`). Without this,
        # Deno 2.x fails to resolve npm:pyodide when a package.json exists
        # in a parent directory that doesn't list pyodide.
        args.append("--node-modules-dir=none")
        args.append("--no-prompt")
        args.append(str(RUNNER_PATH))

        if env_vars:
            args.append(",".join(env_vars))

        return args

    def _sync_files(self) -> None:
        """Sync modified files back to the host without blocking writes."""
        if not self.enable_write_paths or not self.sync_files:
            return

        for path in self.enable_write_paths:
            virtual_path = f"/sandbox/{os.path.basename(path)}"
            msg = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "sync_file",
                    "params": {
                        "virtual_path": virtual_path,
                        "host_path": str(path),
                    },
                }
            )
            self._write_stdin(msg + "\n")

    async def _async_files(self) -> None:
        if not self.enable_write_paths or not self.sync_files:
            return
        for path in self.enable_write_paths:
            virtual_path = f"/sandbox/{os.path.basename(path)}"
            message = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "sync_file",
                    "params": {
                        "virtual_path": virtual_path,
                        "host_path": str(path),
                    },
                }
            )
            await self._write_stdin_async(message + "\n")

    def _send_request(self, method: str, params: dict, context: str) -> dict:
        """Send a JSON-RPC request without blocking the OS pipe.

        Skips non-JSON lines (e.g. Pyodide package-loading messages) matching
        the parent PythonInterpreter behaviour.
        """
        self._request_id += 1
        request_id = self._request_id
        start_time = time.time_ns()
        if method == "health_check":
            self._write_telemetry_span(
                "sandbox.health_check.start",
                attributes={
                    "rpc.request_id": request_id,
                    "timeout.seconds": DENO_REQUEST_TIMEOUT_SEC,
                },
            )
        msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
                "id": request_id,
            }
        )
        self._write_stdin(msg + "\n")

        # Resync loop: after a timed-out ``_send_request`` (the
        # DENO_REQUEST_TIMEOUT_SEC fix), deno may still eventually
        # deliver its response, leaving a stale JSON-RPC frame buffered
        # in stdout. The NEXT request would read THAT frame first and
        # see an old id. Instead of raising (which sends the RLM into
        # an infinite "retry the same code" loop because the model
        # thinks the mismatch is a code-format bug), discard any
        # stale-id responses until we find one that matches or hit the
        # safety cap.
        max_stale_discards = STALE_RESPONSE_DISCARD_LIMIT
        for _attempt in range(max_stale_discards + 1):
            response_line = self._read_with_timeout(timeout=DENO_REQUEST_TIMEOUT_SEC)
            if not response_line:
                exit_code = self.deno_process.poll()
                if exit_code is not None:
                    stderr = getattr(self.deno_process, "stderr", None)
                    stderr_text = ""
                    if stderr is not None:
                        try:
                            stderr_text = stderr.read() or ""
                        except Exception:
                            stderr_text = ""
                    detail = f" {stderr_text}" if stderr_text else ""
                    raise CodeInterpreterError(
                        f"Deno exited (code {exit_code}) {context}{detail}"
                    )
                if method == "health_check":
                    self._write_telemetry_span(
                        "sandbox.health_check.no_response",
                        status={
                            "code": "ERROR",
                            "message": f"No response {context}",
                        },
                        attributes={
                            "rpc.request_id": request_id,
                            "timeout.seconds": DENO_REQUEST_TIMEOUT_SEC,
                            "failure.class": "sandbox_lifecycle_failure",
                        },
                        start_time_unix_nano=start_time,
                    )
                raise CodeInterpreterError(f"No response {context}")

            if not response_line.startswith("{"):
                continue

            try:
                response = json.loads(response_line)
            except json.JSONDecodeError:
                continue

            resp_id = response.get("id")
            if resp_id == request_id:
                break
            logger.warning(
                "Discarding stale deno response (id=%s, expected %s) "
                "in %s — likely a prior request's late arrival",
                resp_id,
                request_id,
                context,
            )
        else:
            raise CodeInterpreterError(
                f"Too many stale responses in {context}: couldn't resync to "
                f"id={request_id} after {max_stale_discards} discards"
            )

        if "error" in response:
            raise CodeInterpreterError(
                f"Error {context}: {response['error'].get('message', 'Unknown error')}"
            )
        if method == "health_check":
            self._write_telemetry_span(
                "sandbox.health_check.ok",
                attributes={
                    "rpc.request_id": request_id,
                    "timeout.seconds": DENO_REQUEST_TIMEOUT_SEC,
                },
                start_time_unix_nano=start_time,
            )
        return response

    async def _asend_request(self, method: str, params: dict, context: str) -> dict:
        """Send one JSON-RPC control request through the nonblocking fd path."""
        self._request_id += 1
        request_id = self._request_id
        start_time = time.time_ns()
        if method == "health_check":
            self._write_telemetry_span(
                "sandbox.health_check.start",
                attributes={
                    "rpc.request_id": request_id,
                    "timeout.seconds": DENO_REQUEST_TIMEOUT_SEC,
                },
            )
        message = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
                "id": request_id,
            }
        )
        await self._write_stdin_async(message + "\n")

        max_stale_discards = STALE_RESPONSE_DISCARD_LIMIT
        for _attempt in range(max_stale_discards + 1):
            response_line = await self._read_with_timeout_async(DENO_REQUEST_TIMEOUT_SEC)
            if not response_line:
                process = self.deno_process
                exit_code = process.poll() if process is not None else None
                if exit_code is not None:
                    stderr = getattr(process, "stderr", None)
                    stderr_text = stderr.read() if stderr is not None else ""
                    detail = f" {stderr_text}" if stderr_text else ""
                    raise CodeInterpreterError(
                        f"Deno exited (code {exit_code}) {context}{detail}"
                    )
                if method == "health_check":
                    self._write_telemetry_span(
                        "sandbox.health_check.no_response",
                        status={"code": "ERROR", "message": f"No response {context}"},
                        attributes={
                            "rpc.request_id": request_id,
                            "timeout.seconds": DENO_REQUEST_TIMEOUT_SEC,
                            "failure.class": "sandbox_lifecycle_failure",
                        },
                        start_time_unix_nano=start_time,
                    )
                raise CodeInterpreterError(f"No response {context}")
            if not response_line.startswith("{"):
                continue
            try:
                response = json.loads(response_line)
            except json.JSONDecodeError:
                continue
            if response.get("id") == request_id:
                break
            logger.warning(
                "Discarding stale deno response (id=%s, expected %s) in %s "
                "- likely a prior request's late arrival",
                response.get("id"),
                request_id,
                context,
            )
        else:
            raise CodeInterpreterError(
                f"Too many stale responses in {context}: couldn't resync to "
                f"id={request_id} after {max_stale_discards} discards"
            )

        if "error" in response:
            raise CodeInterpreterError(
                f"Error {context}: {response['error'].get('message', 'Unknown error')}"
            )
        if method == "health_check":
            self._write_telemetry_span(
                "sandbox.health_check.ok",
                attributes={
                    "rpc.request_id": request_id,
                    "timeout.seconds": DENO_REQUEST_TIMEOUT_SEC,
                },
                start_time_unix_nano=start_time,
            )
        return response

    async def _ahealth_check(self) -> None:
        response = await self._asend_request(
            "execute",
            {"code": "print(1+1)"},
            "during health check",
        )
        if response.get("result", {}).get("output", "").strip() != "2":
            raise CodeInterpreterError(f"Unexpected ping response: {response}")

    def _get_deno_dir(self) -> list[str]:
        """Get Deno cache directory paths (may have multiple on different platforms)."""
        dirs = []
        deno_dir = os.environ.get("DENO_DIR")
        if deno_dir:
            dirs.append(deno_dir)

        home = os.environ.get("HOME") or os.environ.get("USERPROFILE")
        if home:
            # Linux/default location
            dirs.append(os.path.join(home, ".cache", "deno"))
            # macOS with Homebrew location
            dirs.append(os.path.join(home, "Library", "Caches", "deno"))

        return dirs

    # --- File I/O helpers (for LocalFile/OutputFile support) ---

    def ensure_skill_packages(self, packages: list[str]) -> None:
        requested = list(dict.fromkeys(packages))
        missing = [package for package in requested if package not in self._installed_skill_packages]
        if not missing:
            return
        self._ensure_deno_process()
        response = self._send_request(
            "install_packages",
            {"packages": missing},
            "installing skill packages",
        )
        self._installed_skill_packages.update(response.get("result", {}).get("installed", ()))

    async def aensure_skill_packages(self, packages: list[str]) -> None:
        requested = list(dict.fromkeys(packages))
        missing = [package for package in requested if package not in self._installed_skill_packages]
        if not missing:
            return
        await self._aensure_deno_process()
        response = await self._asend_request(
            "install_packages",
            {"packages": missing},
            "installing skill packages",
        )
        self._installed_skill_packages.update(response.get("result", {}).get("installed", ()))

    async def _amount_files(self) -> None:
        if self._mounted_files:
            return
        paths_to_mount = [*self.enable_read_paths, *self.enable_write_paths]
        if not paths_to_mount:
            return
        for path in paths_to_mount:
            if not path:
                continue
            if not os.path.exists(path):
                if path in self.enable_write_paths:
                    Path(path).touch()
                else:
                    raise FileNotFoundError(f"Cannot mount non-existent file: {path}")
            virtual_path = f"/sandbox/{os.path.basename(path)}"
            await self._asend_request(
                "mount_file",
                {"host_path": str(path), "virtual_path": virtual_path},
                f"mounting {path}",
            )
        self._mounted_files = True

    async def _aregister_tools(self) -> None:
        if self._tools_registered:
            return
        params: dict[str, Any] = {}
        if self.tools:
            params["tools"] = [
                {"name": name, "parameters": self._extract_parameters(function)}
                for name, function in self.tools.items()
            ]
        if self.output_fields:
            params["outputs"] = self.output_fields
        if params:
            await self._asend_request("register", params, "registering tools/outputs")
        self._tools_registered = True

    async def _ainject_pending_variables(self) -> None:
        for name, value in self._pending_large_vars.items():
            await self._asend_request(
                "inject_var",
                {"name": name, "value": value},
                f"injecting variable {name!r}",
            )

    def mount_file_at(self, host_path: str, virtual_path: str) -> None:
        """Mount a host file at a specific virtual path in the sandbox."""
        self._ensure_deno_process()
        self._send_request(
            "mount_file",
            {"host_path": host_path, "virtual_path": virtual_path},
            f"mounting {host_path} at {virtual_path}",
        )

    async def amount_file_at(self, host_path: str, virtual_path: str) -> None:
        await self._aensure_deno_process()
        await self._asend_request(
            "mount_file",
            {"host_path": host_path, "virtual_path": virtual_path},
            f"mounting {host_path} at {virtual_path}",
        )

    def mkdir_p(self, virtual_path: str) -> None:
        """Create a directory (and parents) in the sandbox MEMFS."""
        self._ensure_deno_process()
        self._send_request("mkdir_p", {"path": virtual_path}, f"creating dir {virtual_path}")

    async def amkdir_p(self, virtual_path: str) -> None:
        await self._aensure_deno_process()
        await self._asend_request(
            "mkdir_p",
            {"path": virtual_path},
            f"creating dir {virtual_path}",
        )

    def list_dir(self, virtual_path: str) -> list[str]:
        """Recursively list all files under a virtual directory."""
        self._ensure_deno_process()
        response = self._send_request(
            "list_dir", {"path": virtual_path}, f"listing {virtual_path}"
        )
        return response.get("result", {}).get("files", [])

    async def alist_dir(self, virtual_path: str) -> list[str]:
        await self._aensure_deno_process()
        response = await self._asend_request(
            "list_dir",
            {"path": virtual_path},
            f"listing {virtual_path}",
        )
        return response.get("result", {}).get("files", [])

    def workspace_manifest(self, virtual_path: str) -> dict[str, ArtifactFileInfo]:
        """Return a recursive file manifest under a sandbox workspace path."""
        self._ensure_deno_process()
        response = self._send_request(
            "workspace_manifest",
            {"path": virtual_path},
            f"building workspace manifest for {virtual_path}",
        )
        files = response.get("result", {}).get("files", {})
        return {
            rel_path: ArtifactFileInfo(
                type=info["type"],
                sha256=info["sha256"],
                size=info["size"],
            )
            for rel_path, info in files.items()
        }

    async def aworkspace_manifest(
        self,
        virtual_path: str,
    ) -> dict[str, ArtifactFileInfo]:
        await self._aensure_deno_process()
        response = await self._asend_request(
            "workspace_manifest",
            {"path": virtual_path},
            f"building workspace manifest for {virtual_path}",
        )
        files = response.get("result", {}).get("files", {})
        return {
            rel_path: ArtifactFileInfo(
                type=info["type"],
                sha256=info["sha256"],
                size=info["size"],
            )
            for rel_path, info in files.items()
        }

    def add_post_execute_hook(self, hook: Callable[[Any], Any]) -> None:
        """Register a hook called after each completed sandbox execution."""
        self._post_execute_hooks.append(hook)

    def remove_post_execute_hook(self, hook: Callable[[Any], Any]) -> None:
        if hook in self._post_execute_hooks:
            self._post_execute_hooks.remove(hook)

    def _run_post_execute_hooks(self) -> None:
        for hook in getattr(self, "_post_execute_hooks", ()):
            hook(self)

    async def _arun_post_execute_hooks(self) -> None:
        for hook in getattr(self, "_post_execute_hooks", ()):
            result = hook(self)
            if inspect.isawaitable(result):
                await result

    def sync_file_to(self, virtual_path: str, host_path: str) -> None:
        """Sync a single file from the sandbox MEMFS back to the host.

        Sent as a request (not a fire-and-forget notification) so we block until the
        Deno side has finished writing the host file. Callers read the host file right
        after (workspace sync-back recomputes the host manifest, output extraction
        reads submitted files), and a fire-and-forget write races those reads.
        """
        self._ensure_deno_process()
        self._send_request(
            "sync_file",
            {"virtual_path": virtual_path, "host_path": host_path},
            f"syncing {virtual_path} to {host_path}",
        )

    async def async_file_to(self, virtual_path: str, host_path: str) -> None:
        await self._aensure_deno_process()
        await self._asend_request(
            "sync_file",
            {"virtual_path": virtual_path, "host_path": host_path},
            f"syncing {virtual_path} to {host_path}",
        )

    def _strip_code_fences(self, code: str) -> str:
        return strip_code_fences(code)

    def _to_python(self, value: Any) -> Any:
        """Recursively convert Pydantic models to plain Python dicts."""
        if isinstance(value, BaseModel):
            return value.model_dump()
        if isinstance(value, list):
            return [self._to_python(v) for v in value]
        if isinstance(value, dict):
            return {k: self._to_python(v) for k, v in value.items()}
        return value

    def _serialize_value(self, value: Any) -> str:
        """Serialize a Python value to a string representation for injection.

        Extends parent to support Pydantic models (converted via model_dump())
        and uses repr() instead of json.dumps() for dicts/lists so that
        None/True/False stay as valid Python (json.dumps produces null/true/false).
        """
        value = self._to_python(value)

        if isinstance(value, (dict, list)):
            return repr(value)

        # Fall back to parent implementation for other types
        return PythonInterpreter._serialize_value(self, value)

    async def aexecute(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        """Async execute — runs on the current event loop without blocking.

        Unlike execute(), this directly awaits _execute_async so other
        coroutines in an asyncio.gather can make progress concurrently.
        """
        variables = variables or {}
        code = self._strip_code_fences(code)
        code = self._inject_variables(code, variables)

        async with self._execution_gate.atop_level():
            await self.await_host_work()
            # Limit concurrent Deno sandboxes to prevent OOM.
            # Acquired per-execute, released when done — allows other
            # interpreters to run between iterations.
            sem = self._get_semaphore()
            wait_start = time.perf_counter()
            await sem.acquire()
            wait_ms = round((time.perf_counter() - wait_start) * 1000)
            sem_value = getattr(sem, "_value", None)
            self._last_semaphore_attrs = {
                "sandbox.semaphore_wait_ms": wait_ms,
                "sandbox.active_count": (
                    self.MAX_CONCURRENT_SANDBOXES - sem_value
                    if isinstance(sem_value, int)
                    else None
                ),
            }
            try:
                if timeout is None:
                    result = await self._aexecute_inner(code, variables)
                else:
                    result = await self._aexecute_inner(code, variables, timeout=timeout)
            except (asyncio.CancelledError, SandboxFatalError):
                raise
            except BaseException as exc:
                try:
                    await self._arun_post_execute_hooks()
                except BaseException as post_execute_error:
                    setattr(exc, "post_execute_error", post_execute_error)
                raise
            else:
                await self._arun_post_execute_hooks()
                return result
            finally:
                self._last_semaphore_attrs = {}
                sem.release()

    def interrupt(self) -> None:
        """Abort in-flight work by invalidating the current Deno session."""
        self._kill_sandbox()

    async def ainterrupt(self) -> None:
        await self._akill_sandbox()

    async def acancel_execution(self) -> None:
        """Quiesce active generated code without destroying the sandbox filesystem."""
        execute_request_id = self._active_execute_request_id
        if execute_request_id is None:
            return
        process = self.deno_process
        if process is None or process.poll() is not None:
            raise SandboxFatalError(
                "Deno stopped before generated execution could be quiesced"
            )
        response_task = asyncio.create_task(self._execute_async(execute_request_id))
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5
        try:
            while not response_task.done():
                process.send_signal(signal.SIGUSR1)
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise asyncio.TimeoutError
                await asyncio.wait(
                    (response_task,),
                    timeout=min(0.05, remaining),
                )
            await response_task
        except SandboxExecutionError:
            pass
        except asyncio.TimeoutError as exc:
            await self._akill_sandbox()
            raise SandboxFatalError(
                "Deno did not quiesce cancelled generated execution; the sandbox "
                "was force-killed and its filesystem was lost"
            ) from exc
        finally:
            if not response_task.done():
                response_task.cancel()
                await asyncio.gather(response_task, return_exceptions=True)

    def _resolve_execution_timeout(self, timeout: float | None) -> tuple[float, str]:
        return resolve_execution_timeout(timeout, default_timeout=self._exec_timeout)

    def _execute_params(
        self,
        code: str,
        timeout: float | None,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"code": code}
        if timeout is not None:
            params["execution_timeout_seconds"] = timeout_seconds
        return params

    def _host_watchdog_timeout(
        self,
        timeout_seconds: float,
        timeout_failure_class: str,
    ) -> float:
        return recoverable_timeout_host_deadline_seconds(
            timeout_seconds,
            timeout_failure_class,
        )

    def _format_recoverable_timeout_result(
        self,
        result: dict[str, Any],
    ) -> RecoverableExecutionTimeout:
        return format_recoverable_timeout_result(result)

    async def _aexecute_inner(
        self,
        code: str,
        variables: dict[str, Any] | None,
        *,
        timeout: float | None = None,
    ) -> Any:
        """Inner async execute — runs within the sandbox semaphore."""
        timeout_seconds, timeout_failure_class = self._resolve_execution_timeout(timeout)
        await self._aensure_deno_process()
        await self._amount_files()
        await self._aregister_tools()
        await self._ainject_pending_variables()

        self._request_id += 1
        execute_request_id = self._request_id
        execute_start_time = time.time_ns()
        self._write_telemetry_span(
            "sandbox.execute.start",
            attributes={
                "rpc.request_id": execute_request_id,
                "timeout.seconds": timeout_seconds,
                "sandbox.pending_tool_count": self._telemetry_pending_tool_count(),
                "sandbox.pending_file_op_count": self._telemetry_pending_file_ops_count(),
                **getattr(self, "_last_semaphore_attrs", {}),
            },
        )
        self._log_lifecycle(
            "jspi.execute.start",
            request_id=execute_request_id,
            code_chars=len(code),
            timeout_seconds=timeout_seconds,
            pending_tool_count=self._telemetry_pending_tool_count(),
            pending_file_op_count=self._telemetry_pending_file_ops_count(),
        )
        input_data = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "execute",
                "params": self._execute_params(code, timeout, timeout_seconds),
                "id": execute_request_id,
            }
        )
        try:
            await self._write_stdin_async(input_data + "\n")
        except BrokenPipeError as e:
            self._log_lifecycle(
                "jspi.execute.broken_pipe",
                request_id=execute_request_id,
                status="error",
            )
            await self._akill_sandbox()
            raise SandboxFatalError(
                "Sandbox subprocess died before receiving the execute "
                "request (BrokenPipeError). Per-run mounts cannot be "
                "restored on a fresh process, so this run is unrecoverable."
            ) from e

        self._active_execute_request_id = execute_request_id
        try:
            if timeout is None:
                return await self._execute_with_timeout(
                    execute_request_id,
                    execute_start_time,
                )
            return await self._execute_with_timeout(
                execute_request_id,
                execute_start_time,
                timeout_seconds=timeout_seconds,
                timeout_failure_class=timeout_failure_class,
            )
        except asyncio.CancelledError as cancellation:
            try:
                await asyncio.shield(self.acancel_execution())
            except BaseException as quiesce_error:
                setattr(cancellation, "jspi_quiesce_error", quiesce_error)
            raise
        finally:
            if self._active_execute_request_id == execute_request_id:
                self._active_execute_request_id = None

    async def _execute_with_timeout(
        self,
        execute_request_id: int,
        execute_start_time: int | None = None,
        *,
        timeout_seconds: float | None = None,
        timeout_failure_class: str = "sandbox_exec_timeout",
    ) -> Any:
        """Run the execute loop under a wall-clock deadline.

        Constructor-level sandbox timeouts are fatal. LM-selected
        per-iteration timeouts are enforced inside Pyodide with an
        interrupt buffer; the host deadline here is only a recovery
        backstop so the runner has time to return buffered output.
        """
        timeout_seconds = self._exec_timeout if timeout_seconds is None else timeout_seconds
        host_timeout = self._host_watchdog_timeout(
            timeout_seconds,
            timeout_failure_class,
        )
        try:
            execute_async = self._execute_async
            execute_async_params = inspect.signature(execute_async).parameters
            execute_coro = (
                execute_async(
                    execute_request_id,
                    timeout_seconds=timeout_seconds,
                    timeout_failure_class=timeout_failure_class,
                )
                if "timeout_seconds" in execute_async_params
                else execute_async(execute_request_id)
            )
            result = await asyncio.wait_for(
                execute_coro,
                timeout=host_timeout,
            )
            if isinstance(result, RecoverableExecutionTimeout):
                self._write_telemetry_span(
                    "sandbox.execute.timeout",
                    status={
                        "code": "ERROR",
                        "message": (
                            "Iteration execution timed out after "
                            f"{result.timeout_seconds:g}s"
                        ),
                    },
                    attributes={
                        "rpc.request_id": execute_request_id,
                        "timeout.seconds": result.timeout_seconds,
                        "sandbox.pending_tool_count": self._telemetry_pending_tool_count(),
                        "sandbox.pending_file_op_count": (
                            self._telemetry_pending_file_ops_count()
                        ),
                        "failure.class": ITERATION_TIMEOUT_FAILURE_CLASS,
                        "timeout.recoverable": True,
                    },
                    start_time_unix_nano=execute_start_time,
                )
                self._log_lifecycle(
                    "jspi.execute.timeout_recovered",
                    status="timeout_recovered",
                    request_id=execute_request_id,
                    duration_ms=(
                        round((time.time_ns() - execute_start_time) / 1_000_000)
                        if execute_start_time is not None
                        else None
                    ),
                    timeout_seconds=result.timeout_seconds,
                    pending_tool_count=self._telemetry_pending_tool_count(),
                    pending_file_op_count=self._telemetry_pending_file_ops_count(),
                )
                return result
            self._write_telemetry_span(
                "sandbox.execute.ok",
                attributes={
                    "rpc.request_id": execute_request_id,
                    "timeout.seconds": timeout_seconds,
                    "sandbox.pending_tool_count": self._telemetry_pending_tool_count(),
                    "sandbox.pending_file_op_count": self._telemetry_pending_file_ops_count(),
                },
                start_time_unix_nano=execute_start_time,
            )
            self._log_lifecycle(
                "jspi.execute.ok",
                request_id=execute_request_id,
                timeout_seconds=timeout_seconds,
                pending_tool_count=self._telemetry_pending_tool_count(),
                pending_file_op_count=self._telemetry_pending_file_ops_count(),
            )
            return result
        except asyncio.TimeoutError:
            self._write_telemetry_span(
                "sandbox.execute.timeout",
                status={
                    "code": "ERROR",
                    "message": f"Sandbox exec timed out after {host_timeout}s",
                },
                attributes={
                    "rpc.request_id": execute_request_id,
                    "timeout.seconds": host_timeout,
                    **(
                        {
                            "rlm_iteration_timeout.seconds": timeout_seconds,
                            "timeout.recovery_failed": True,
                        }
                        if timeout_failure_class == ITERATION_TIMEOUT_FAILURE_CLASS
                        else {}
                    ),
                    "sandbox.pending_tool_count": self._telemetry_pending_tool_count(),
                    "sandbox.pending_file_op_count": self._telemetry_pending_file_ops_count(),
                    "failure.class": (
                        "sandbox_exec_timeout"
                        if timeout_failure_class == ITERATION_TIMEOUT_FAILURE_CLASS
                        else timeout_failure_class
                    ),
                },
                start_time_unix_nano=execute_start_time,
            )
            self._log_lifecycle(
                "jspi.execute.timeout",
                request_id=execute_request_id,
                duration_ms=(
                    round((time.time_ns() - execute_start_time) / 1_000_000)
                    if execute_start_time is not None
                    else None
                ),
                timeout_seconds=host_timeout,
                pending_tool_count=self._telemetry_pending_tool_count(),
                pending_file_op_count=self._telemetry_pending_file_ops_count(),
                status="error",
            )
            await self._akill_sandbox()
            if timeout_failure_class == ITERATION_TIMEOUT_FAILURE_CLASS:
                raise SandboxFatalError(
                    f"Sandbox failed to recover from iteration timeout after "
                    f"{timeout_seconds}s (request id {execute_request_id}); "
                    f"waited {host_timeout}s before force-killing Deno."
                )
            raise SandboxFatalError(
                f"Sandbox exec timed out after {host_timeout}s "
                f"(request id {execute_request_id}). The Deno subprocess "
                f"was force-killed; per-run mounts are gone, so this "
                f"interpreter is dead. Usually indicates pathological code "
                f"in the sandbox (regex catastrophic backtracking, infinite "
                f"loop, or a host-side tool that never returns)."
            )

    def _kill_sandbox(self) -> None:
        """Force-kill the Deno subprocess for shutdown.

        Called when we're tearing the sandbox down for good (timeout or
        crash). No state bookkeeping — the interpreter is not expected to
        be reused after this; the context manager's ``shutdown`` will run
        next and the raised ``SandboxFatalError`` propagates out of the
        RLM run.
        """
        if self.deno_process is not None:
            self._write_telemetry_span("sandbox.shutdown.start")
            self._log_lifecycle("jspi.kill.start")
            kill_result = "sent"
            try:
                self.deno_process.kill()
            except Exception:
                kill_result = "error"
            try:
                self.deno_process.wait(timeout=0.2)
            except Exception:
                pass
            self._write_telemetry_span(
                "sandbox.shutdown.kill",
                attributes={"process.kill_result": kill_result},
                status="OK" if kill_result == "sent" else "ERROR",
            )
            self._write_telemetry_span(
                "sandbox.shutdown.complete",
                attributes={"process.kill_result": kill_result},
                status="OK" if kill_result == "sent" else "ERROR",
            )
            self._log_lifecycle(
                "jspi.kill.complete",
                kill_result=kill_result,
                status="ok" if kill_result == "sent" else "error",
            )
        self.deno_process = None
        self._stdin_fd = -1
        self._stdout_fd = -1
        self._read_buf = ""
        # Cancel any pending file-sync futures; they can't complete now.
        for fut in list(self._pending_file_ops.values()):
            if not fut.done():
                fut.cancel()
        self._pending_file_ops.clear()

    async def _akill_sandbox(self) -> None:
        """Force-kill Deno and reap it without a blocking process wait."""
        process = self.deno_process
        if process is not None:
            self._write_telemetry_span("sandbox.shutdown.start")
            self._log_lifecycle("jspi.kill.start")
            kill_result = "sent"
            try:
                process.kill()
            except Exception:
                kill_result = "error"
            await self._await_process_exit(process, 0.2)
            self._write_telemetry_span(
                "sandbox.shutdown.kill",
                attributes={"process.kill_result": kill_result},
                status="OK" if kill_result == "sent" else "ERROR",
            )
            self._write_telemetry_span(
                "sandbox.shutdown.complete",
                attributes={"process.kill_result": kill_result},
                status="OK" if kill_result == "sent" else "ERROR",
            )
            self._log_lifecycle(
                "jspi.kill.complete",
                kill_result=kill_result,
                status="ok" if kill_result == "sent" else "error",
            )
        self.deno_process = None
        self._stdin_fd = -1
        self._stdout_fd = -1
        self._read_buf = ""
        for future in list(self._pending_file_ops.values()):
            if not future.done():
                future.cancel()
        self._pending_file_ops.clear()

    def execute(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        """Execute code with concurrent async tool support.

        All tools are async. Multiple tool calls via asyncio.gather()
        are executed concurrently on the host side.
        """
        with self._execution_gate.top_level():
            self._wait_for_host_work_sync()
            try:
                result = self._execute_top_level(code, variables, timeout=timeout)
            except SandboxFatalError:
                raise
            except BaseException as exc:
                try:
                    self._run_post_execute_hooks()
                except BaseException as post_execute_error:
                    setattr(exc, "post_execute_error", post_execute_error)
                raise
            else:
                self._run_post_execute_hooks()
                return result

    def _execute_top_level(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        variables = variables or {}
        timeout_seconds, timeout_failure_class = self._resolve_execution_timeout(timeout)

        # Strip markdown code fences that models often add
        code = self._strip_code_fences(code)

        code = self._inject_variables(code, variables)
        self._ensure_deno_process()
        self._mount_files()
        self._register_tools()

        # Send the code as JSON-RPC 2.0 request
        self._request_id += 1
        execute_request_id = self._request_id
        execute_start_time = time.time_ns()
        self._write_telemetry_span(
            "sandbox.execute.start",
            attributes={
                "rpc.request_id": execute_request_id,
                "timeout.seconds": timeout_seconds,
                "sandbox.pending_tool_count": self._telemetry_pending_tool_count(),
                "sandbox.pending_file_op_count": self._telemetry_pending_file_ops_count(),
            },
        )
        input_data = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "execute",
                "params": self._execute_params(code, timeout, timeout_seconds),
                "id": execute_request_id,
            }
        )
        try:
            self._write_stdin(input_data + "\n")
        except BrokenPipeError as e:
            self._kill_sandbox()
            raise SandboxFatalError(
                "Sandbox subprocess died before receiving the execute "
                "request (BrokenPipeError). Per-run mounts cannot be "
                "restored on a fresh process, so this run is unrecoverable."
            ) from e

        # Run the async execute loop. Use nest_asyncio to allow
        # run_until_complete even inside an already-running loop (e.g. marimo).
        import nest_asyncio

        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        if timeout is None:
            return loop.run_until_complete(
                self._execute_with_timeout(execute_request_id, execute_start_time)
            )
        return loop.run_until_complete(
            self._execute_with_timeout(
                execute_request_id,
                execute_start_time,
                timeout_seconds=timeout_seconds,
                timeout_failure_class=timeout_failure_class,
            )
        )

    async def _execute_async(
        self,
        execute_request_id: int,
        *,
        timeout_seconds: float | None = None,
        timeout_failure_class: str = "sandbox_exec_timeout",
    ) -> Any:
        """Read messages and handle tool calls concurrently using asyncio."""
        await JspiBackend.await_host_work(self)
        pending_tasks: dict[str, asyncio.Task] = {}  # request_id -> Task
        try:
            return await self._execute_async_loop(
                execute_request_id,
                pending_tasks,
                timeout_seconds=timeout_seconds,
                timeout_failure_class=timeout_failure_class,
            )
        finally:
            for task in pending_tasks.values():
                task.cancel()
            if pending_tasks:
                await asyncio.gather(*pending_tasks.values(), return_exceptions=True)
            self._active_tool_count = len(JspiBackend._live_host_tool_tasks(self))

    async def _execute_async_loop(
        self,
        execute_request_id: int,
        pending_tasks: dict[str, asyncio.Task],
        *,
        timeout_seconds: float | None = None,
        timeout_failure_class: str = "sandbox_exec_timeout",
    ) -> Any:
        stale_discards = 0
        pending_tool_deadline = (
            time.monotonic() + timeout_seconds
            if timeout_failure_class == ITERATION_TIMEOUT_FAILURE_CLASS
            and timeout_seconds is not None
            else None
        )

        while True:
            self._active_tool_count = len(pending_tasks)
            if (
                pending_tasks
                and pending_tool_deadline is not None
                and time.monotonic() >= pending_tool_deadline
            ):
                for request_id, task in list(pending_tasks.items()):
                    task.cancel()
                    response = _jsonrpc_error(
                        JSONRPC_APP_ERRORS["RuntimeError"],
                        "iteration timed out while awaiting host tool callback",
                        request_id,
                    )
                    await self._write_stdin_async(response + "\n")
                pending_tasks.clear()
                self._active_tool_count = len(JspiBackend._live_host_tool_tasks(self))
                return self._format_recoverable_timeout_result({
                    "timeout": {"seconds": timeout_seconds},
                    "stdout": "",
                    "stderr": (
                        "Deno/Pyodide was awaiting host tool callbacks when the "
                        "iteration timeout elapsed. Pending host callbacks were "
                        "rejected so the live sandbox can continue."
                    ),
                    "state": {
                        "preserved": True,
                        "source": "live_pyodide",
                        "scope": "full_live",
                    },
                })
            # Check for completed tool calls and send responses
            await self._send_completed_responses(pending_tasks)
            self._active_tool_count = len(pending_tasks)

            # Read next message — non-blocking via event loop fd watching
            if pending_tasks:
                output_line = await self._read_with_timeout_async(0.01)
                if output_line is None:
                    await asyncio.sleep(0)
                    continue
            else:
                output_line = await self._read_with_timeout_async(1.0)
                if output_line is None:
                    await asyncio.sleep(0)
                    continue

            if not output_line:
                await self._akill_sandbox()
                raise SandboxFatalError(
                    "Deno subprocess stopped producing stdout during async execute. "
                    "The sandbox was force-killed; per-run mounts are gone, so "
                    "this interpreter is dead."
                )

            # Skip non-JSON lines (e.g., Pyodide package loading messages)
            if not output_line.startswith("{"):
                logger.debug(f"Skipping non-JSON output: {output_line}")
                self._log_lifecycle(
                    "jspi.protocol.non_json_stdout",
                    preview=output_line[:200],
                )
                continue

            try:
                result = json.loads(output_line)
            except json.JSONDecodeError:
                logger.info(f"Skipping malformed JSON: {output_line[:100]}")
                self._log_lifecycle(
                    "jspi.protocol.malformed_json",
                    preview=output_line[:200],
                    status="error",
                )
                continue

            # Route file-sync responses to pending futures (from _execute_tool_async)
            resp_id = result.get("id")
            if resp_id is not None and resp_id in self._pending_file_ops:
                future = self._pending_file_ops.pop(resp_id)
                if not future.done():
                    future.set_result(result)
                continue

            # JSON-RPC request from sandbox (tool call)
            if "method" in result:
                if result["method"] == "tool_call":
                    if (
                        pending_tool_deadline is not None
                        and time.monotonic() >= pending_tool_deadline
                    ):
                        continue
                    request_id = result["id"]
                    params = result.get("params", {})
                    task = asyncio.create_task(
                        self._execute_tool_async(params["name"], params, request_id)
                    )
                    JspiBackend._track_host_tool_task(self, task)
                    pending_tasks[request_id] = task
                    self._active_tool_count = len(pending_tasks)
                    continue

            if "result" in result and result.get("id") != execute_request_id:
                stale_discards += 1
                if stale_discards > STALE_RESPONSE_DISCARD_LIMIT:
                    raise CodeInterpreterError(
                        "Too many stale top-level responses while resyncing "
                        f"execute request id={execute_request_id}"
                    )
                logger.warning(
                    "Discarding stale deno execute response (id=%s, expected %s)",
                    result.get("id"),
                    execute_request_id,
                )
                self._log_lifecycle(
                    "jspi.protocol.stale_response",
                    request_id=execute_request_id,
                    response_id=result.get("id"),
                    stale_discards=stale_discards,
                )
                continue

            if (
                "error" in result
                and result.get("id") is not None
                and result.get("id") != execute_request_id
            ):
                stale_discards += 1
                if stale_discards > STALE_RESPONSE_DISCARD_LIMIT:
                    raise CodeInterpreterError(
                        "Too many stale top-level errors while resyncing "
                        f"execute request id={execute_request_id}"
                    )
                logger.warning(
                    "Discarding stale deno execute error (id=%s, expected %s)",
                    result.get("id"),
                    execute_request_id,
                )
                self._log_lifecycle(
                    "jspi.protocol.stale_error",
                    request_id=execute_request_id,
                    response_id=result.get("id"),
                    stale_discards=stale_discards,
                )
                continue

            if "error" in result and result.get("id") is None:
                error = result.get("error") or {}
                logger.warning(
                    "Discarding uncorrelated deno error while waiting for "
                    "execute response id=%s: %s",
                    execute_request_id,
                    error.get("message", error),
                )
                continue

            if "result" in result:
                res = result["result"]
                if isinstance(res, dict) and "timeout" in res:
                    for task in pending_tasks.values():
                        task.cancel()
                    pending_tasks.clear()
                    self._active_tool_count = len(JspiBackend._live_host_tool_tasks(self))
                    return self._format_recoverable_timeout_result(res)

            # Before returning, ensure all pending tool calls complete
            await self._wait_and_send_all_responses(pending_tasks)
            self._active_tool_count = len(pending_tasks)

            # JSON-RPC success response
            if "result" in result:
                res = result["result"]
                await self._async_files()
                if interpreter_result_logging_enabled(getattr(self, "_verbose", False)):
                    emit_trace_result(res)
                if "final" in res:
                    return FinalOutput(res["final"])
                return res.get("output", None)

            # JSON-RPC error response
            if "error" in result:
                error = result["error"]
                error_data = error.get("data", {})
                error_type = error_data.get("type", "Sandbox Error")
                error_args = error_data.get("args", [])
                error_msg = error.get("message", "")
                partial_output = error_data.get("output") or ""

                if partial_output:
                    self._log_partial_output(partial_output, error_type=error_type)
                if interpreter_result_logging_enabled(getattr(self, "_verbose", False)):
                    if partial_output:
                        emit_trace_result({"output": partial_output})
                    emit_trace_error(error_type, error_msg or error_args)

                if error_type == "SyntaxError":
                    # Format a helpful message from the args tuple:
                    # SyntaxError.args = (msg, (filename, lineno, offset, text, ...))
                    detail = error_msg
                    if error_args and len(error_args) >= 2 and isinstance(error_args[1], list):
                        info = error_args[1]
                        lineno = info[1] if len(info) > 1 else None
                        offset = info[2] if len(info) > 2 else None
                        text = (info[3] or "").rstrip("\n") if len(info) > 3 else None
                        parts = [error_args[0] or "invalid syntax"]
                        if lineno:
                            parts.append(f"line {lineno}")
                        if offset:
                            parts.append(f"col {offset}")
                        if text:
                            parts.append(repr(text))
                        detail = ", ".join(parts)
                    raise SyntaxError(detail or "Invalid Python syntax")
                else:
                    raise SandboxExecutionError(
                        f"{error_type}: {error_args or error_msg}",
                        partial_output=partial_output,
                    )

            raise CodeInterpreterError(f"Unexpected message format from sandbox: {result}")

    def _read_with_timeout(self, timeout: float | None) -> str | None:
        """Sync read — used by _send_request (registration, mount, etc.).

        The ``timeout`` is a **wall-clock budget** for the entire line
        (including any inter-chunk gaps), not just the initial "wait
        until fd is readable". If the deno subprocess streams partial
        bytes then stalls before the newline, the budget still fires
        and this returns ``None`` — freeing the caller (and the asyncio
        event loop) from an unbounded wait on a silent pipe.
        """
        if timeout is not None:
            if "\n" in self._read_buf:
                line, self._read_buf = self._read_buf.split("\n", 1)
                return line.strip()
            if self._stdout_fd >= 0:
                ready, _, _ = select.select([self._stdout_fd], [], [], timeout)
                if not ready:
                    return None
            else:
                stdout = getattr(self.deno_process, "stdout", None)
                if stdout is None:
                    return None
                ready, _, _ = select.select([stdout.fileno()], [], [], timeout)
                if not ready:
                    return None
        try:
            return self._read_line_raw(timeout=timeout)
        except TimeoutError:
            return None

    async def _read_with_timeout_async(self, timeout: float | None) -> str | None:
        """Async read using event loop fd watching — zero threads.

        Each ``os.read()`` is preceded by an fd-readiness notification, including
        when a line arrives in multiple chunks.
        """
        if "\n" in self._read_buf:
            line, self._read_buf = self._read_buf.split("\n", 1)
            return line.strip()

        if self._stdout_fd < 0:
            raise CodeInterpreterError("Deno stdout fd is unavailable for async reads")

        deadline = None if timeout is None else time.monotonic() + timeout
        loop = asyncio.get_running_loop()
        while "\n" not in self._read_buf:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return None
            ready = asyncio.Event()
            loop.add_reader(self._stdout_fd, ready.set)
            try:
                if remaining is None:
                    await ready.wait()
                else:
                    await asyncio.wait_for(ready.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                return None
            finally:
                loop.remove_reader(self._stdout_fd)

            try:
                chunk = os.read(self._stdout_fd, 65536)
            except (BlockingIOError, InterruptedError):
                continue
            if not chunk:
                remainder = self._read_buf
                self._read_buf = ""
                return remainder.strip()
            self._read_buf += chunk.decode("utf-8", errors="replace")

        line, self._read_buf = self._read_buf.split("\n", 1)
        return line.strip()

    def _read_line_raw(self, timeout: float | None = None) -> str:
        """Read one line from the raw stdout fd, accumulating in _read_buf.

        When ``timeout`` is given, each ``os.read`` iteration is gated by
        a ``select`` on the remaining wall-clock budget. If the budget
        expires before a newline arrives, raise ``TimeoutError``.
        Without this bound, a deno subprocess that emits partial bytes
        then hangs freezes the caller indefinitely — and if the caller
        is the asyncio event loop (via ``_ensure_deno_process``'s sync
        health check) the whole process goes 0% CPU with no progress.
        """
        stdout = getattr(self.deno_process, "stdout", None)
        if self._stdout_fd < 0 or stdout is None:
            return (stdout.readline() if stdout else "").strip()

        deadline = None if timeout is None else time.monotonic() + timeout

        while "\n" not in self._read_buf:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "deno stdout read exceeded heartbeat budget before newline arrived"
                    )
                ready, _, _ = select.select([self._stdout_fd], [], [], remaining)
                if not ready:
                    raise TimeoutError(
                        "deno stdout read exceeded heartbeat budget before newline arrived"
                    )
            chunk = os.read(self._stdout_fd, 65536)
            if not chunk:
                remainder = self._read_buf
                self._read_buf = ""
                return remainder.strip()
            self._read_buf += chunk.decode("utf-8", errors="replace")

        line, self._read_buf = self._read_buf.split("\n", 1)
        return line.strip()

    async def _sync_file_during_tool(self, virtual_path: str, host_path: str) -> None:
        """Sync a file from sandbox MEMFS to host during a tool call.

        Sends a sync_file request to the Deno runner's responseReader (which
        handles it during tool execution) and awaits the response via a Future
        resolved by the _execute_async loop.
        """
        self._request_id += 1
        req_id = self._request_id
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending_file_ops[req_id] = future
        msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "sync_file",
                "params": {"virtual_path": virtual_path, "host_path": host_path},
                "id": req_id,
            }
        )
        await self._write_stdin_async(msg + "\n")
        result = await future
        if "error" in result:
            raise CodeInterpreterError(
                f"sync_file failed: {result['error'].get('message', result['error'])}"
            )

    async def _mount_file_during_tool(self, host_path: str, virtual_path: str) -> None:
        """Mount a file from host into sandbox MEMFS during a tool call."""
        self._request_id += 1
        req_id = self._request_id
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending_file_ops[req_id] = future
        msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "mount_file",
                "params": {"host_path": host_path, "virtual_path": virtual_path},
                "id": req_id,
            }
        )
        await self._write_stdin_async(msg + "\n")
        result = await future
        if "error" in result:
            raise CodeInterpreterError(
                f"mount_file failed: {result['error'].get('message', result['error'])}"
            )

    async def _execute_tool_async(
        self,
        tool_name: str,
        call_args: dict,
        tool_request_id: int | str | None = None,
    ) -> dict:
        """Execute a tool asynchronously and return the response dict."""
        from predict_rlm.trace import ToolCall, ms_since, record_tool_call

        if not hasattr(self, "_execution_gate"):
            self._execution_gate = BackendExecutionGate("JSPI backend")

        if getattr(self, "_verbose", False) or live_tool_call_logging_enabled():
            emit_trace_tool_call(
                tool_name,
                args=call_args.get("args", []),
                kwargs=call_args.get("kwargs", {}),
            )

        call_start = time.perf_counter()
        telemetry_start = time.time_ns()
        self._write_telemetry_span(
            "sandbox.tool_call.start",
            attributes={
                "tool.name": tool_name,
                "tool.id": tool_request_id,
                "rpc.request_id": tool_request_id,
                "timeout.seconds": TOOL_CALL_TIMEOUT_SEC,
                "sandbox.pending_tool_count": self._telemetry_pending_tool_count(),
                "sandbox.pending_file_op_count": self._telemetry_pending_file_ops_count(),
            },
        )
        self._log_lifecycle(
            "jspi.tool_call.start",
            tool=tool_name,
            request_id=tool_request_id,
            timeout_seconds=TOOL_CALL_TIMEOUT_SEC,
            pending_tool_count=self._telemetry_pending_tool_count(),
        )
        # Copy to mutable containers so the SyncedFile handler below can
        # rewrite sandbox paths to host paths before invoking the tool.
        args = list(call_args.get("args", []))
        kwargs = dict(call_args.get("kwargs", {}))
        temp_dir: str | None = None

        try:
            if tool_name not in self.tools:
                raise CodeInterpreterError(f"Unknown tool: {tool_name}")

            tool_fn = self.tools[tool_name]

            # Pass pydantic_schemas through to predict tool if present
            pydantic_schemas = call_args.get("pydantic_schemas")
            if pydantic_schemas and tool_name == "predict":
                kwargs["pydantic_schemas"] = pydantic_schemas

            # Handle SyncedFile-annotated tool parameters: sync sandbox files
            # to host before calling, and mount modified files back after.
            from predict_rlm.files import get_synced_file_params

            synced_params = (
                {}
                if getattr(tool_fn, "__predict_rlm_synced_file_operation__", False)
                else get_synced_file_params(tool_fn)
            )
            temp_dir = None
            # (sandbox_path, host_path, writeback) for each synced param
            synced_entries: list[tuple[str, str, bool]] = []

            if synced_params:
                sig = inspect.signature(tool_fn)
                param_names = list(sig.parameters.keys())

                for param_name, sf in synced_params.items():
                    # Resolve the sandbox path from args or kwargs
                    sandbox_path = kwargs.get(param_name)
                    if sandbox_path is None and param_name in param_names:
                        idx = param_names.index(param_name)
                        if idx < len(args):
                            sandbox_path = args[idx]
                    if not sandbox_path or not isinstance(sandbox_path, str):
                        continue

                    # Determine host directory
                    if sf.host_dir is not None:
                        host_dir = sf.host_dir
                        os.makedirs(host_dir, exist_ok=True)
                    else:
                        if temp_dir is None:
                            temp_dir = tempfile.mkdtemp(prefix="tool-file-sync-")
                        host_dir = temp_dir

                    host_path = os.path.join(host_dir, os.path.basename(sandbox_path))
                    await self._sync_file_during_tool(sandbox_path, host_path)
                    synced_entries.append((sandbox_path, host_path, sf.writeback))

                    # Replace the sandbox path with the host path in args/kwargs
                    if param_name in kwargs:
                        kwargs[param_name] = host_path
                    elif param_name in param_names:
                        idx = param_names.index(param_name)
                        if idx < len(args):
                            args[idx] = host_path

            # Check if tool is async or sync. Wrap in a per-call wall-
            # clock budget so a slow host tool (e.g. recalculate() on a
            # whole-column formula → 2-3 min via LibreOffice) can't
            # blow past the outer ``_exec_timeout`` ceiling. On
            # timeout the guard converts the hang into a ``TimeoutError``
            # which the ``except Exception`` handler below turns into
            # a normal ``{"error": "tool timed out …"}`` response —
            # routed back to the sandbox's ``await tool()`` as a
            # recoverable error. Sandbox stays alive, RLM sees the
            # error and can rewrite its code to avoid the slow path.
            if asyncio.iscoroutinefunction(tool_fn):
                telemetry_context = getattr(self, "_telemetry_context", None)
                token = set_current_telemetry_context(telemetry_context)
                try:
                    policy = (
                        host_sync_worker_policy(
                            self,
                            timeout=TOOL_CALL_TIMEOUT_SEC,
                            detach_on_cancel=True,
                        )
                        if callable_has_sync_leaf(tool_fn)
                        else contextlib.nullcontext()
                    )
                    with self._execution_gate.async_tool_callback(), policy:
                        result = await asyncio.wait_for(
                            tool_fn(*args, **kwargs),
                            timeout=TOOL_CALL_TIMEOUT_SEC,
                        )
                except asyncio.TimeoutError as e:
                    raise TimeoutError(
                        f"tool {tool_name!r} timed out after "
                        f"{TOOL_CALL_TIMEOUT_SEC}s (per-call budget)"
                    ) from e
                finally:
                    reset_current_telemetry_context(token)
            else:
                telemetry_context = getattr(self, "_telemetry_context", None)
                token = set_current_telemetry_context(telemetry_context)

                def call_tool() -> Any:
                    with self._execution_gate.tool_callback():
                        return tool_fn(*args, **kwargs)

                try:
                    with host_sync_worker_policy(
                        self,
                        timeout=TOOL_CALL_TIMEOUT_SEC,
                        detach_on_cancel=True,
                    ):
                        result = await invoke_host_callable(call_tool)
                except TimeoutError as e:
                    raise TimeoutError(
                        f"tool {tool_name!r} timed out after "
                        f"{TOOL_CALL_TIMEOUT_SEC}s (per-call budget)"
                    ) from e
                finally:
                    reset_current_telemetry_context(token)

            if inspect.isawaitable(result):
                try:
                    result = await asyncio.wait_for(
                        result,
                        timeout=TOOL_CALL_TIMEOUT_SEC,
                    )
                except asyncio.TimeoutError as e:
                    raise TimeoutError(
                        f"tool {tool_name!r} timed out after "
                        f"{TOOL_CALL_TIMEOUT_SEC}s (per-call budget)"
                    ) from e

            # Mount modified files back into the sandbox (only for writeback params)
            if synced_entries:
                for sandbox_path, host_path, writeback in synced_entries:
                    if writeback and os.path.isfile(host_path):
                        await self._mount_file_during_tool(host_path, sandbox_path)
                if temp_dir:
                    shutil.rmtree(temp_dir, ignore_errors=True)

            result = to_plain_data(result)

            is_json = result is None or isinstance(result, (list, dict, int, float, bool))
            response = {
                "value": json.dumps(result) if is_json else str(result or ""),
                "type": "json" if is_json else "string",
            }

            # Record non-predict tool calls (predict records itself with richer detail)
            if tool_name != "predict":
                record_tool_call(
                    ToolCall(
                        name=tool_name,
                        args=args,
                        kwargs={k: v for k, v in kwargs.items() if k != "pydantic_schemas"},
                        result=result,
                        duration_ms=ms_since(call_start),
                    )
                )

            self._write_telemetry_span(
                "sandbox.tool_call.ok",
                attributes={
                    "tool.name": tool_name,
                    "tool.id": tool_request_id,
                    "rpc.request_id": tool_request_id,
                    "timeout.seconds": TOOL_CALL_TIMEOUT_SEC,
                    "sandbox.pending_tool_count": self._telemetry_pending_tool_count(),
                    "sandbox.pending_file_op_count": self._telemetry_pending_file_ops_count(),
                },
                start_time_unix_nano=telemetry_start,
            )
            self._log_lifecycle(
                "jspi.tool_call.ok",
                tool=tool_name,
                request_id=tool_request_id,
                duration_ms=ms_since(call_start),
            )
            return response
        except asyncio.CancelledError as cancellation:
            if temp_dir:
                worker = getattr(cancellation, "sync_worker", None)
                deferred_dir = temp_dir
                if not self.defer_until_sync_workers_finish(
                    lambda: shutil.rmtree(deferred_dir, ignore_errors=True),
                    worker,
                ):
                    shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        except Exception as e:
            # Clean up any SyncedFile temp dir before returning
            if temp_dir:
                worker = getattr(e, "sync_worker", None)
                if worker is not None and not worker.done:
                    deferred_dir = temp_dir
                    worker.add_done_callback(
                        lambda _worker: shutil.rmtree(deferred_dir, ignore_errors=True)
                    )
                else:
                    shutil.rmtree(temp_dir, ignore_errors=True)
            if tool_name != "predict":
                record_tool_call(
                    ToolCall(
                        name=tool_name,
                        args=args,
                        kwargs={k: v for k, v in kwargs.items() if k != "pydantic_schemas"},
                        result=None,
                        error=str(e),
                        duration_ms=ms_since(call_start),
                    )
                )
            if isinstance(e, TimeoutError):
                self._write_telemetry_span(
                    "sandbox.tool_call.timeout",
                    status={"code": "ERROR", "message": str(e)},
                    attributes={
                        "tool.name": tool_name,
                        "tool.id": tool_request_id,
                        "rpc.request_id": tool_request_id,
                        "timeout.seconds": TOOL_CALL_TIMEOUT_SEC,
                        "sandbox.pending_tool_count": self._telemetry_pending_tool_count(),
                        "sandbox.pending_file_op_count": self._telemetry_pending_file_ops_count(),
                        "failure.class": "host_tool_timeout_or_leak",
                    },
                    start_time_unix_nano=telemetry_start,
                )
                lifecycle_event = "jspi.tool_call.timeout"
            else:
                lifecycle_event = "jspi.tool_call.error"
            self._log_lifecycle(
                lifecycle_event,
                tool=tool_name,
                request_id=tool_request_id,
                duration_ms=ms_since(call_start),
                error_type=type(e).__name__,
                status="error",
            )
            return {"error": str(e)}

    def _write_stdin(self, data: str) -> None:
        """Sync write — used by _send_request, shutdown, etc."""
        if self.deno_process is None or self.deno_process.poll() is not None:
            raise CodeInterpreterError(
                "Deno process is no longer running — cannot write to stdin"
            )
        fd = self._stdin_fd
        if fd < 0 or self.deno_process.stdin is None:
            # During interpreter bootstrapping the raw fd has not been
            # captured yet. Fall back to the blocking TextIO write that
            # PythonInterpreter used so the health check can succeed.
            self.deno_process.stdin.write(data)
            self.deno_process.stdin.flush()
            return

        encoded = data.encode("utf-8")
        view = memoryview(encoded)

        while view:
            try:
                written = os.write(fd, view)
                view = view[written:]
            except BlockingIOError:
                select.select([], [fd], [])
            except InterruptedError:
                continue

    async def _write_stdin_async(self, data: str) -> None:
        """Async write using event loop fd watching — zero threads.

        Sets the stdin fd to non-blocking and writes in chunks, yielding
        to the event loop when the pipe buffer is full.
        """
        if self.deno_process is None or self.deno_process.poll() is not None:
            raise CodeInterpreterError(
                "Deno process is no longer running — cannot write to stdin"
            )
        encoded = data.encode("utf-8")
        fd = self._stdin_fd
        loop = asyncio.get_running_loop()
        while encoded:
            try:
                n = os.write(fd, encoded)
                encoded = encoded[n:]
            except BlockingIOError:
                writable = asyncio.Event()
                loop.add_writer(fd, writable.set)
                try:
                    await writable.wait()
                finally:
                    loop.remove_writer(fd)

    async def _send_completed_responses(self, pending_tasks: dict[str, asyncio.Task]) -> None:
        """Send a JSON-RPC response for one completed tool call.

        Sends at most one response per invocation to prevent filling the OS
        pipe buffer when multiple large responses complete simultaneously.
        The caller (_execute_async) loops, interleaving reads between writes
        so Deno's JS event loop gets time to process JSPI promise resolutions.
        """
        for request_id, task in list(pending_tasks.items()):
            if not task.done():
                continue

            pending_tasks.pop(request_id)
            try:
                result = task.result()
            except Exception as e:
                result = {"error": str(e)}

            if "error" in result:
                response = _jsonrpc_error(
                    JSONRPC_APP_ERRORS["RuntimeError"],
                    result["error"],
                    request_id,
                )
            else:
                response = _jsonrpc_result(
                    {"value": result["value"], "type": result["type"]},
                    request_id,
                )
            await self._write_stdin_async(response + "\n")
            return  # Send at most one — interleave reads between writes

    async def _wait_and_send_all_responses(
        self, pending_tasks: dict[str, asyncio.Task]
    ) -> None:
        """Wait for all pending tool calls and send their JSON-RPC responses.

        Yields to the event loop between writes so the pipe buffer can drain
        and Deno's JSPI can process promise resolutions.
        """
        for request_id, task in list(pending_tasks.items()):
            try:
                result = await task  # Wait for completion
            except Exception as e:
                result = {"error": str(e)}

            if "error" in result:
                response = _jsonrpc_error(
                    JSONRPC_APP_ERRORS["RuntimeError"],
                    result["error"],
                    request_id,
                )
            else:
                response = _jsonrpc_result(
                    {"value": result["value"], "type": result["type"]},
                    request_id,
                )
            await self._write_stdin_async(response + "\n")

        pending_tasks.clear()
