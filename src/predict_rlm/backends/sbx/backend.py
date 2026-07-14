"""Docker Sandboxes execution backend."""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import contextlib
import contextvars
import hashlib
import inspect
import json
import os
import queue
import re
import secrets
import select
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput

try:
    from websockets.asyncio.client import ClientConnection as AsyncClientConnection
    from websockets.asyncio.client import connect as async_websocket_connect
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.client import ClientConnection
    from websockets.sync.client import connect as websocket_connect
except ImportError as exc:  # pragma: no cover - exercised by optional-dep import tests
    raise ImportError(
        "The SBX backend requires optional dependency `predict-rlm[sbx]`. "
        "Install with `pip install 'predict-rlm[sbx]'` or "
        "`uv pip install 'predict-rlm[sbx]'`."
    ) from exc

from predict_rlm._logging import (
    configure_predict_rlm_logging,
    emit_trace_error,
    emit_trace_result,
    emit_trace_tool_call,
    interpreter_result_logging_enabled,
    live_tool_call_logging_enabled,
)
from predict_rlm._shared import strip_code_fences
from predict_rlm.execution_timeout import (
    ITERATION_TIMEOUT_FAILURE_CLASS,
    format_recoverable_timeout_result,
    recoverable_timeout_host_deadline_seconds,
    resolve_execution_timeout,
)
from predict_rlm.files import get_synced_file_params
from predict_rlm.runtime import (
    SyncWorkerTracker,
    host_sync_worker_policy,
    invoke_host_callable,
)
from predict_rlm.runtime_hooks import RuntimeHook, RuntimeHookEvent
from predict_rlm.serialization import to_plain_data
from predict_rlm.trace import ToolCall, ms_since, record_tool_call
from predict_rlm.workspace import DirectWorkspaceMount, WorkspaceFileInfo

from ..base import (
    BackendExecutionGate,
    LegacyExecutionBackend,
    SandboxExecutionError,
    SandboxFatalError,
    SupervisorClient,
    SupervisorProcess,
)
from .config import SbxConfig
from .logging import log_interpreter_lifecycle, log_partial_output

SUPERVISOR_PAYLOAD_SOURCE_PATH = Path(__file__).parents[1] / "supervisor" / "_payload.py"
DEFAULT_PACKAGE_DOMAINS = ["pypi.org", "files.pythonhosted.org"]
SBX_PYTHON_EXECUTABLE = "python3"
SBX_TRANSPORT_PACKAGES = ["websockets"]
_LOCALHOST_ENDPOINT_RE = re.compile(
    r"(?:(?P<scheme>https?)://)?(?P<host>localhost|127\.0\.0\.1|\[::1\])(?::(?P<port>\d+))"
)

_owned_staging_roots_pending_cleanup: set[str] = set()


def _cleanup_pending_staging_roots() -> None:
    for path in list(_owned_staging_roots_pending_cleanup):
        shutil.rmtree(path, ignore_errors=True)
        try:
            Path(path).parent.rmdir()
        except OSError:
            pass


atexit.register(_cleanup_pending_staging_roots)


def _dedupe_packages(packages: list[str]) -> list[str]:
    return list(dict.fromkeys(packages))


class _DetachedWebSocketSupervisorProcess:
    stdin = None
    stdout = None
    stderr = None

    def poll(self) -> int | None:
        return None


class SbxBackend(SyncWorkerTracker, SupervisorClient, LegacyExecutionBackend):
    """Execution backend backed by Docker Sandboxes.

    The backend starts a Python JSON-RPC supervisor inside a Docker Sandbox and
    maps predict-rlm virtual paths under a per-run workspace staging root.
    """

    def __init__(
        self,
        *,
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
        direct_workspace_mounts: list[DirectWorkspaceMount] | None = None,
        runtime_hooks: list[RuntimeHook] | None = None,
        on_runtime_hook_event: Callable[[RuntimeHookEvent], Any] | None = None,
        _runner_command: list[str] | None = None,
        _websocket_supervisor_command: list[str] | None = None,
        _websocket_url: str | None = None,
        _staging_root: str | Path | None = None,
    ) -> None:
        SupervisorClient.__init__(self, supervisor_name="Sbx supervisor")
        self.config = config or SbxConfig()
        self.allowed_domains = allowed_domains
        self.tools = tools or {}
        self.output_fields = output_fields or []
        self.preinstall_packages = preinstall_packages
        self.skill_packages = _dedupe_packages(skill_packages or [])
        self._installed_skill_packages: set[str] = set()
        self.debug = debug
        self.verbose = verbose
        configure_predict_rlm_logging(
            debug=True if debug else None,
            verbose=True if verbose else None,
        )
        self.extra_read_paths = extra_read_paths or []
        self.extra_write_paths = extra_write_paths or []
        self._supervisor_command = _supervisor_command or _runner_command
        self._websocket_supervisor_command = _websocket_supervisor_command
        self._websocket_url = _websocket_url
        self._websocket_path = f"/predict-rlm/{secrets.token_urlsafe(32)}"
        self._direct_workspace_mounts = list(direct_workspace_mounts or [])
        self.runtime_hooks = list(runtime_hooks or [])
        self.on_runtime_hook_event = on_runtime_hook_event
        self._host_workspace = Path.cwd()
        self._owns_staging_root = _staging_root is None
        if _staging_root is not None:
            self._staging_root = Path(_staging_root)
        elif self.config.reuse and self.config.name:
            self._staging_root = (
                self._host_workspace / ".predict_rlm_sbx" / self.config.name
            )
        else:
            self._staging_root = (
                self._host_workspace / ".predict_rlm_sbx" / uuid.uuid4().hex
            )
        self._staging_root.mkdir(parents=True, exist_ok=True)
        if self._owns_staging_root and not self.config.persist:
            _owned_staging_roots_pending_cleanup.add(str(self._staging_root))
        self._proc: subprocess.Popen[str] | None = None
        self._stdout_lines: queue.Queue[str] = queue.Queue()
        self._stdout_reader: threading.Thread | None = None
        self._ws: ClientConnection | None = None
        self._async_proc: asyncio.subprocess.Process | None = None
        self._async_ws: AsyncClientConnection | None = None
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._async_pending_tool_calls: dict[asyncio.Task[dict[str, Any]], int] = {}
        self._quarantined_async_tool_calls: set[asyncio.Task[Any]] = set()
        self._host_work_retirement: asyncio.Task[None] | None = None
        self._active_async_request: dict[str, Any] | None = None
        self._pending_tool_calls: dict[concurrent.futures.Future[dict[str, Any]], int] = {}
        self._quarantined_tool_calls: set[
            concurrent.futures.Future[dict[str, Any]]
        ] = set()
        self._active_execute_timeout_deadline: float | None = None
        self.cancellation_interrupt_timeout: float = 10.0
        self._execution_gate = BackendExecutionGate("SBX backend")
        self._sandbox_name: str | None = None
        self._prepared_supervisor_path: Path | None = None
        self._published_websocket_url: str | None = None
        self._active_websocket_port: int | None = None
        self._shutdown = False
        self._post_execute_hooks: list[Callable[[Any], Any]] = []
        self._owned_direct_aliases: list[Path] = []
        self._relocate_owned_staging_root_if_nested_in_direct_workspace()

    def configure_debug(self, enabled: bool) -> None:
        self.debug = enabled
        configure_predict_rlm_logging(debug=enabled)

    def configure_verbose(self, enabled: bool) -> None:
        self.verbose = enabled
        configure_predict_rlm_logging(verbose=enabled)

    def _log_lifecycle(self, event: str, **fields: Any) -> None:
        fields.setdefault("sandbox_name", getattr(self, "_sandbox_name", None))
        fields.setdefault("process_pid", getattr(self._proc, "pid", None) if self._proc else None)
        fields.setdefault("staging_root", getattr(self, "_staging_root", None))
        log_interpreter_lifecycle(
            enabled=getattr(self, "debug", False),
            event=event,
            **fields,
        )

    def _log_partial_output(self, output: str, **fields: Any) -> None:
        fields.setdefault("sandbox_name", getattr(self, "_sandbox_name", None))
        fields.setdefault("process_pid", getattr(self._proc, "pid", None) if self._proc else None)
        fields.setdefault("staging_root", getattr(self, "_staging_root", None))
        log_partial_output(
            enabled=getattr(self, "debug", False),
            output=output,
            **fields,
        )

    def _uses_websocket_transport(self) -> bool:
        return self._supervisor_command is None

    def _transport_running(self) -> bool:
        if self._uses_websocket_transport():
            return self._ws is not None or self._async_ws is not None
        return bool(self._proc and self._proc.poll() is None)

    def execute(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        with self._execution_gate.top_level():
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
        code = strip_code_fences(code)
        if variables:
            mapped_variables = {
                name: self._map_variable_value(value) for name, value in variables.items()
            }
            assignments = "\n".join(
                f"{name} = {value!r}" for name, value in mapped_variables.items()
            )
            code = f"{assignments}\n{code}"
        params: dict[str, Any] = {"code": code}
        if timeout is not None:
            execution_timeout, _ = self._resolve_execution_timeout(timeout)
            params["execution_timeout_seconds"] = execution_timeout
        response = self._send_request("execute", params, timeout=timeout)
        return self._unwrap_execute_response(response)

    async def aexecute(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        if not self._uses_websocket_transport():
            raise RuntimeError("SbxBackend.aexecute requires the maintained websocket transport")
        async with self._execution_gate.atop_level():
            try:
                result = await self._aexecute_top_level(code, variables, timeout=timeout)
            except asyncio.CancelledError:
                await self._abort_async_execution_after_cancellation()
                raise
            except SandboxFatalError:
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

    async def _aexecute_top_level(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        code = strip_code_fences(code)
        if variables:
            mapped_variables = {
                name: self._map_variable_value(value) for name, value in variables.items()
            }
            assignments = "\n".join(
                f"{name} = {value!r}" for name, value in mapped_variables.items()
            )
            code = f"{assignments}\n{code}"
        params: dict[str, Any] = {"code": code}
        if timeout is not None:
            execution_timeout, _ = self._resolve_execution_timeout(timeout)
            params["execution_timeout_seconds"] = execution_timeout
        response = await self._asend_websocket_request("execute", params, timeout=timeout)
        return self._unwrap_execute_response(response)

    async def _abort_async_execution_after_cancellation(self) -> None:
        active_request = self._active_async_request
        if active_request is None:
            return
        try:
            await self._asend_interrupt_frame()
            await self._areceive_websocket_response(
                **active_request,
                deadline=time.monotonic() + self.cancellation_interrupt_timeout,
            )
        except BaseException:
            await self._ahard_abort_websocket_after_failed_interrupt()

    def interrupt(self, *, timeout: float | None = 10.0) -> bool:
        """Abort the currently-running cell while keeping the warm sandbox.

        Sends an out-of-band ``interrupt`` frame over the websocket. ``ws.send``
        is thread-safe and we never call ``ws.recv`` here: the execute loop
        blocked in ``recv`` (possibly on another thread) delivers the resulting
        interrupted result through its normal path. Returns whether a cell was
        running when the interrupt was issued.
        """
        if not self._uses_websocket_transport():
            return False
        was_running = self._execution_gate.is_running()
        ws = self._ws
        if ws is None:
            return False
        payload = {
            "jsonrpc": "2.0",
            "method": "interrupt",
            "params": {},
            "id": self._next_request_id(),
        }
        try:
            ws.send(self._serialize_supervisor_message(payload), text=True)
        except TypeError:
            ws.send(self._serialize_supervisor_message(payload))
        except Exception as exc:
            raise SandboxFatalError(
                f"Failed to send interrupt to Sbx WebSocket supervisor: {exc}"
            ) from exc
        if was_running and not self._execution_gate.wait_until_idle(timeout):
            raise SandboxFatalError(
                "Interrupt frame sent but the running cell did not release the "
                f"execution gate within {timeout}s."
            )
        return was_running

    async def ainterrupt(self, *, timeout: float | None = 10.0) -> bool:
        if not self._uses_websocket_transport():
            return False
        was_running = self._execution_gate.is_running()
        if self._async_ws is None:
            return False
        await self._asend_interrupt_frame()
        if not was_running:
            return False
        deadline = None if timeout is None else time.monotonic() + timeout
        while self._execution_gate.is_running():
            if deadline is not None and time.monotonic() >= deadline:
                raise SandboxFatalError(
                    "Interrupt frame sent but the running cell did not release the "
                    f"execution gate within {timeout}s."
                )
            await asyncio.sleep(0.01)
        return True

    async def _asend_interrupt_frame(self) -> None:
        ws = self._require_async_websocket()
        payload = {
            "jsonrpc": "2.0",
            "method": "interrupt",
            "params": {},
            "id": self._next_request_id(),
        }
        try:
            await ws.send(self._serialize_supervisor_message(payload), text=True)
        except TypeError:
            await ws.send(self._serialize_supervisor_message(payload))
        except Exception as exc:
            raise SandboxFatalError(
                f"Failed to send interrupt to Sbx WebSocket supervisor: {exc}"
            ) from exc

    async def _ahard_abort_websocket_after_failed_interrupt(self) -> None:
        with contextlib.suppress(Exception):
            await self._akill_websocket_supervisor()

    def _hard_abort_websocket_after_failed_interrupt(self) -> None:
        """Tear down the websocket + supervisor when a graceful interrupt fails.

        This sacrifices the warm sandbox instead of reusing a connection with an
        execute still draining in another thread.
        """
        with contextlib.suppress(Exception):
            self._discard_supervisor_process()

    def mount_file_at(self, host_path: str, virtual_path: str) -> None:
        source = Path(host_path)
        target = self._host_path_for_sandbox_path(virtual_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    async def amount_file_at(self, host_path: str, virtual_path: str) -> None:
        source = Path(host_path)
        target = self._host_path_for_sandbox_path(virtual_path)
        await self._acopy_file(source, target)

    def mkdir_p(self, virtual_path: str) -> None:
        self._host_path_for_sandbox_path(virtual_path).mkdir(parents=True, exist_ok=True)

    async def amkdir_p(self, virtual_path: str) -> None:
        self._host_path_for_sandbox_path(virtual_path).mkdir(parents=True, exist_ok=True)

    def list_dir(self, virtual_path: str) -> list[str]:
        root = self._host_path_for_sandbox_path(virtual_path)
        if not root.exists():
            return []
        return [
            self._sandbox_path_for_host_path(path)
            for path in sorted(root.rglob("*"))
            if path.is_file()
        ]

    async def alist_dir(self, virtual_path: str) -> list[str]:
        root = self._host_path_for_sandbox_path(virtual_path)
        if not root.exists():
            return []
        files = []
        for path in sorted(root.rglob("*")):
            await asyncio.sleep(0)
            if path.is_file():
                files.append(self._sandbox_path_for_host_path(path))
        return files

    def workspace_manifest(self, virtual_path: str) -> dict[str, WorkspaceFileInfo]:
        root = self._host_path_for_sandbox_path(virtual_path)
        if not root.exists():
            raise FileNotFoundError(f"Workspace mount does not exist: {virtual_path}")
        if not root.is_dir():
            raise NotADirectoryError(f"Workspace mount is not a directory: {virtual_path}")
        files: dict[str, WorkspaceFileInfo] = {}
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel_path = path.relative_to(root).as_posix()
            files[rel_path] = WorkspaceFileInfo(
                type="file",
                sha256=self._sha256_file(path),
                size=path.stat().st_size,
            )
        return files

    async def aworkspace_manifest(
        self, virtual_path: str
    ) -> dict[str, WorkspaceFileInfo]:
        root = self._host_path_for_sandbox_path(virtual_path)
        if not root.exists():
            raise FileNotFoundError(f"Workspace mount does not exist: {virtual_path}")
        if not root.is_dir():
            raise NotADirectoryError(f"Workspace mount is not a directory: {virtual_path}")
        files: dict[str, WorkspaceFileInfo] = {}
        for path in sorted(root.rglob("*")):
            await asyncio.sleep(0)
            if not path.is_file():
                continue
            rel_path = path.relative_to(root).as_posix()
            files[rel_path] = WorkspaceFileInfo(
                type="file",
                sha256=await self._asha256_file(path),
                size=path.stat().st_size,
            )
        return files

    def add_post_execute_hook(self, hook: Callable[[Any], Any]) -> None:
        self._post_execute_hooks.append(hook)

    def remove_post_execute_hook(self, hook: Callable[[Any], Any]) -> None:
        if hook in self._post_execute_hooks:
            self._post_execute_hooks.remove(hook)

    def _run_post_execute_hooks(self) -> None:
        for hook in self._post_execute_hooks:
            hook(self)

    async def _arun_post_execute_hooks(self) -> None:
        for hook in self._post_execute_hooks:
            result = hook(self)
            if inspect.isawaitable(result):
                await result

    def sync_file_to(self, virtual_path: str, host_path: str) -> None:
        source = self._host_path_for_sandbox_path(virtual_path)
        target = Path(host_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    async def async_file_to(self, virtual_path: str, host_path: str) -> None:
        source = self._host_path_for_sandbox_path(virtual_path)
        target = Path(host_path)
        await self._acopy_file(source, target)

    async def _acopy_file(self, source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(source, "rb") as source_file, open(target, "wb") as target_file:
            while chunk := source_file.read(1024 * 1024):
                target_file.write(chunk)
                await asyncio.sleep(0)
        shutil.copystat(source, target)

    async def _asha256_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
                await asyncio.sleep(0)
        return digest.hexdigest()

    def _sha256_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _host_path_for_virtual_path(self, virtual_path: str) -> Path:
        sandbox_root = (self._staging_root / "sandbox").resolve()
        rel = virtual_path.removeprefix("/sandbox").lstrip("/")
        host_path = (sandbox_root / rel).resolve()
        try:
            host_path.relative_to(sandbox_root)
        except ValueError as exc:
            raise ValueError(f"Sbx virtual path escapes /sandbox: {virtual_path}") from exc
        return host_path

    def _host_path_for_sandbox_path(self, sandbox_path: str) -> Path:
        for mount in self._direct_workspace_mounts:
            rel_path = self._relative_to_prefix(sandbox_path, mount.sandbox_path)
            if rel_path is not None:
                return Path(mount.host_path, *rel_path.parts)
        if sandbox_path == "/sandbox" or sandbox_path.startswith("/sandbox/"):
            return self._host_path_for_virtual_path(sandbox_path)
        raise ValueError(
            "Sbx path must be under /sandbox or a direct workspace mount: "
            f"{sandbox_path}"
        )

    def _map_variable_value(self, value: Any) -> Any:
        # Normalize rich objects (pydantic models, dataclasses, sets) to plain data
        # first -- they are injected into the sandbox via repr(), and e.g. a model's
        # repr is a constructor call (RfpAnalysis(...)) referencing a class the
        # sandbox doesn't have, which would raise NameError.
        value = to_plain_data(value)
        if isinstance(value, str) and (value == "/sandbox" or value.startswith("/sandbox/")):
            return str(self._host_path_for_virtual_path(value))
        if isinstance(value, list):
            return [self._map_variable_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._map_variable_value(item) for item in value)
        if isinstance(value, dict):
            return {key: self._map_variable_value(item) for key, item in value.items()}
        return value

    def _virtual_path_for_host_path(self, host_path: Path) -> str:
        sandbox_root = (self._staging_root / "sandbox").resolve()
        rel = host_path.resolve().relative_to(sandbox_root)
        return "/sandbox/" + rel.as_posix()

    def _sandbox_path_for_host_path(self, host_path: Path) -> str:
        for mount in self._direct_workspace_mounts:
            try:
                rel = host_path.resolve().relative_to(Path(mount.host_path).resolve())
            except ValueError:
                continue
            if rel.as_posix() == ".":
                return mount.sandbox_path
            return f"{mount.sandbox_path.rstrip('/')}/{rel.as_posix()}"
        return self._virtual_path_for_host_path(host_path)

    def _relative_to_prefix(self, path: str, prefix: str) -> Path | None:
        try:
            rel = Path(path).relative_to(Path(prefix))
        except ValueError:
            return None
        return Path() if rel.as_posix() == "." else rel

    def configure_direct_workspace_mounts(
        self, mounts: list[DirectWorkspaceMount]
    ) -> None:
        mounts = list(mounts)
        if self._same_direct_workspace_mounts(mounts):
            return
        if self._transport_running():
            raise RuntimeError(
                "Direct workspace mounts must be configured before the SBX runner starts"
            )
        self._direct_workspace_mounts = mounts
        self._relocate_owned_staging_root_if_nested_in_direct_workspace()

    async def aconfigure_direct_workspace_mounts(
        self, mounts: list[DirectWorkspaceMount]
    ) -> None:
        mounts = list(mounts)
        if self._same_direct_workspace_mounts(mounts):
            return
        if self._transport_running():
            raise RuntimeError(
                "Direct workspace mounts must be configured before the SBX runner starts"
            )
        self._direct_workspace_mounts = mounts
        self._relocate_owned_staging_root_if_nested_in_direct_workspace()

    def _relocate_owned_staging_root_if_nested_in_direct_workspace(self) -> None:
        """Move an owned staging root out of any direct workspace mount.

        The default staging root lives under the invoking cwd. When a direct
        workspace mount covers that cwd, the staging root would be nested inside
        the user's mounted workspace (polluting it and its sync-back manifest),
        so relocate it to a private system-temp dir instead.
        """
        if not self._owns_staging_root:
            return
        staging_root = self._staging_root.resolve()
        for mount in self._direct_workspace_mounts:
            direct_root = Path(mount.host_path).resolve()
            try:
                staging_root.relative_to(direct_root)
            except ValueError:
                continue
            old_staging_root = self._staging_root
            _owned_staging_roots_pending_cleanup.discard(str(old_staging_root))
            self._staging_root = self._relocated_staging_root()
            self._staging_root.mkdir(parents=True, exist_ok=True)
            if not self.config.persist:
                _owned_staging_roots_pending_cleanup.add(str(self._staging_root))
            shutil.rmtree(old_staging_root, ignore_errors=True)
            try:
                old_staging_root.parent.rmdir()
            except OSError:
                pass
            return

    def _relocated_staging_root(self) -> Path:
        if self.config.reuse and self.config.name:
            return Path(tempfile.gettempdir()) / f"predict-rlm-sbx-{self.config.name}"
        return Path(tempfile.mkdtemp(prefix="predict-rlm-sbx-"))

    def _same_direct_workspace_mounts(self, mounts: list[DirectWorkspaceMount]) -> bool:
        return self._direct_workspace_mount_keys(mounts) == self._direct_workspace_mount_keys(
            self._direct_workspace_mounts
        )

    def _direct_workspace_mount_keys(
        self, mounts: list[DirectWorkspaceMount]
    ) -> list[tuple[str, str]]:
        return [
            (
                os.path.abspath(mount.host_path),
                os.path.normpath(mount.sandbox_path),
            )
            for mount in mounts
        ]

    def configure_runtime(
        self,
        *,
        tools: dict[str, Callable[..., Any]] | None = None,
        output_fields: list[dict] | None = None,
        skill_packages: list[str] | None = None,
        debug: bool | None = None,
        verbose: bool | None = None,
        runtime_hooks: list[RuntimeHook] | None = None,
        on_runtime_hook_event: Callable[[RuntimeHookEvent], Any] | None = None,
    ) -> None:
        if debug is not None:
            self.configure_debug(debug)
        if verbose is not None:
            self.configure_verbose(verbose)
        if tools is not None and tools is not self.tools:
            self.tools = tools
        if output_fields is not None:
            self.output_fields = output_fields
        if skill_packages is not None:
            self.ensure_skill_packages(skill_packages)
        if runtime_hooks is not None:
            self.runtime_hooks = list(runtime_hooks)
            self.on_runtime_hook_event = on_runtime_hook_event
        if self._transport_running():
            if self.output_fields:
                self._send_request("register_output_fields", {"fields": self.output_fields})
            if self.tools:
                self._send_request("register_tools", {"tools": list(self.tools)})
            if runtime_hooks is not None or self.runtime_hooks:
                self._register_runtime_hooks()
        self._log_lifecycle(
            "sbx.runtime.configured",
            tools=len(self.tools),
            output_fields=len(self.output_fields),
            skill_packages=len(self.skill_packages),
            process_running=bool(self._proc and self._proc.poll() is None),
        )

    async def aconfigure_runtime(
        self,
        *,
        tools: dict[str, Callable[..., Any]] | None = None,
        output_fields: list[dict] | None = None,
        skill_packages: list[str] | None = None,
        debug: bool | None = None,
        verbose: bool | None = None,
        runtime_hooks: list[RuntimeHook] | None = None,
        on_runtime_hook_event: Callable[[RuntimeHookEvent], Any] | None = None,
    ) -> None:
        if debug is not None:
            self.configure_debug(debug)
        if verbose is not None:
            self.configure_verbose(verbose)
        if tools is not None and tools is not self.tools:
            self.tools = tools
        if output_fields is not None:
            self.output_fields = output_fields
        if skill_packages is not None:
            await self.aensure_skill_packages(skill_packages)
        if runtime_hooks is not None:
            self.runtime_hooks = list(runtime_hooks)
            self.on_runtime_hook_event = on_runtime_hook_event
        if self._async_ws is not None:
            if self.output_fields:
                await self._asend_websocket_request(
                    "register_output_fields", {"fields": self.output_fields}
                )
            if self.tools:
                await self._asend_websocket_request(
                    "register_tools", {"tools": list(self.tools)}
                )
            if runtime_hooks is not None or self.runtime_hooks:
                await self._asend_websocket_request(
                    "register_runtime_hooks",
                    {"hooks": [hook.model_dump(mode="json") for hook in self.runtime_hooks]},
                )
        self._log_lifecycle(
            "sbx.runtime.configured",
            tools=len(self.tools),
            output_fields=len(self.output_fields),
            skill_packages=len(self.skill_packages),
            process_running=self._async_process_running(),
        )

    def ensure_skill_packages(self, packages: list[str]) -> None:
        requested = _dedupe_packages(packages)
        if not requested:
            return
        known_packages = set(self.skill_packages)
        self.skill_packages.extend(
            package for package in requested if package not in known_packages
        )
        missing = [
            package
            for package in requested
            if package not in self._installed_skill_packages
        ]
        if not missing or self._sandbox_name is None:
            return
        self._install_packages(
            missing,
            event="sbx.skill_packages",
            failure_label="install sbx skill packages",
        )
        self._installed_skill_packages.update(missing)

    async def aensure_skill_packages(self, packages: list[str]) -> None:
        requested = _dedupe_packages(packages)
        if not requested:
            return
        known_packages = set(self.skill_packages)
        self.skill_packages.extend(
            package for package in requested if package not in known_packages
        )
        missing = [
            package
            for package in requested
            if package not in self._installed_skill_packages
        ]
        if not missing or self._sandbox_name is None:
            return
        await self._ainstall_packages(
            missing,
            event="sbx.skill_packages",
            failure_label="install sbx skill packages",
        )
        self._installed_skill_packages.update(missing)

    def _register_runtime_hooks(self) -> None:
        self._send_request(
            "register_runtime_hooks",
            {"hooks": [hook.model_dump(mode="json") for hook in self.runtime_hooks]},
        )

    def prewarm(self) -> None:
        self._log_lifecycle("sbx.prewarm.start")
        self._ensure_process()
        self._log_lifecycle("sbx.prewarm.ok")

    async def aprewarm(self) -> None:
        self._log_lifecycle("sbx.prewarm.start")
        await self._aensure_websocket_supervisor()
        self._log_lifecycle("sbx.prewarm.ok")

    def reset(self) -> None:
        if self.has_live_host_work():
            raise RuntimeError("Cannot reset SBX while host tool work is still active")
        self._log_lifecycle("sbx.reset.start")
        self._send_request("reset", {})
        sandbox_root = self._staging_root / "sandbox"
        shutil.rmtree(sandbox_root, ignore_errors=True)
        sandbox_root.mkdir(parents=True, exist_ok=True)
        self._post_execute_hooks.clear()
        self._log_lifecycle("sbx.reset.ok")

    async def areset(self) -> None:
        if self.has_live_host_work():
            raise RuntimeError("Cannot reset SBX while host tool work is still active")
        self._log_lifecycle("sbx.reset.start")
        await self._asend_websocket_request("reset", {})
        sandbox_root = self._staging_root / "sandbox"
        shutil.rmtree(sandbox_root, ignore_errors=True)
        sandbox_root.mkdir(parents=True, exist_ok=True)
        self._post_execute_hooks.clear()
        self._log_lifecycle("sbx.reset.ok")

    def shutdown(self) -> None:
        if self._async_ws is not None or self._async_proc is not None:
            loop = self._async_loop
            if loop is not None and loop.is_running():
                try:
                    current_loop = asyncio.get_running_loop()
                except RuntimeError:
                    current_loop = None
                if current_loop is loop:
                    raise RuntimeError("Use await SbxBackend.ashutdown() from async code")
                asyncio.run_coroutine_threadsafe(self.ashutdown(), loop).result()
                return
            self._shutdown_async_transport_after_loop_closed()
        if self._shutdown:
            return
        self._shutdown = True
        self._log_lifecycle("sbx.shutdown.start")
        if self._uses_websocket_transport():
            sent_shutdown = False
            if self._ws is not None:
                try:
                    self._send_websocket_request("shutdown", {})
                    sent_shutdown = True
                except Exception:
                    pass
                with contextlib.suppress(Exception):
                    self._ws.close()
                self._ws = None
            if self._proc and self._proc.poll() is None:
                if not sent_shutdown:
                    self._proc.kill()
                    self._proc.wait(timeout=5)
                    self._log_lifecycle("sbx.shutdown.kill", kill_result="sent")
                else:
                    try:
                        self._proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self._proc.kill()
                        self._proc.wait(timeout=5)
                        self._log_lifecycle("sbx.shutdown.kill", kill_result="sent")
        elif self._proc and self._proc.poll() is None:
            try:
                self._send_request("shutdown", {})
            except Exception:
                pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)
                self._log_lifecycle("sbx.shutdown.kill", kill_result="sent")
        self._proc = None

        if (
            self._supervisor_command is None
            and self._sandbox_name
            and self.config.remove_on_shutdown
        ):
            if not self.config.persist:
                subprocess.run(
                    ["sbx", "rm", "--force", self._sandbox_name],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self._log_lifecycle("sbx.shutdown.rm")
        elif (
            self._supervisor_command is None
            and self._sandbox_name
            and self.config.reuse
            and self.config.stop_on_shutdown
        ):
            subprocess.run(
                ["sbx", "stop", self._sandbox_name],
                check=False,
                capture_output=True,
                text=True,
            )
            self._log_lifecycle("sbx.shutdown.stop")
        self._cleanup_direct_workspace_aliases_host_side()
        self._cleanup_staging_root()
        self._log_lifecycle("sbx.shutdown.complete")

    def _shutdown_async_transport_after_loop_closed(self) -> None:
        if self._async_ws is not None:
            transport = getattr(self._async_ws, "transport", None)
            if transport is not None:
                with contextlib.suppress(RuntimeError):
                    transport.abort()
            self._async_ws = None
        if self._async_proc is not None and self._async_proc.returncode is None:
            self._async_proc.kill()
        self._async_proc = None
        self._async_loop = None

    async def ashutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self._log_lifecycle("sbx.shutdown.start")
        sent_shutdown = False
        if self._async_ws is not None:
            try:
                await self._asend_websocket_request("shutdown", {})
                sent_shutdown = True
            except Exception:
                pass
            with contextlib.suppress(Exception):
                await self._async_ws.close()
            self._async_ws = None

        await self._astop_supervisor_process(sent_shutdown=sent_shutdown)

        if (
            self._supervisor_command is None
            and self._sandbox_name
            and self.config.remove_on_shutdown
        ):
            if not self.config.persist:
                await self._arun_command(
                    ["sbx", "rm", "--force", self._sandbox_name],
                    check=False,
                )
                self._log_lifecycle("sbx.shutdown.rm")
        elif (
            self._supervisor_command is None
            and self._sandbox_name
            and self.config.reuse
            and self.config.stop_on_shutdown
        ):
            await self._arun_command(
                ["sbx", "stop", self._sandbox_name],
                check=False,
            )
            self._log_lifecycle("sbx.shutdown.stop")
        self._cleanup_direct_workspace_aliases_host_side()
        self._cleanup_staging_root()
        self._async_loop = None
        self._log_lifecycle("sbx.shutdown.complete")

    async def _astop_supervisor_process(self, *, sent_shutdown: bool) -> None:
        if self._async_proc is not None:
            process = self._async_proc
            if process.returncode is None:
                if sent_shutdown:
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5)
                    except TimeoutError:
                        process.kill()
                        await process.wait()
                        self._log_lifecycle("sbx.shutdown.kill", kill_result="sent")
                else:
                    process.kill()
                    await process.wait()
                    self._log_lifecycle("sbx.shutdown.kill", kill_result="sent")
            self._async_proc = None
        if self._proc is not None:
            process = self._proc
            if process.poll() is None:
                deadline = time.monotonic() + 5
                while sent_shutdown and process.poll() is None and time.monotonic() < deadline:
                    await asyncio.sleep(0.01)
                if process.poll() is None:
                    process.kill()
                    while process.poll() is None:
                        await asyncio.sleep(0.01)
                    self._log_lifecycle("sbx.shutdown.kill", kill_result="sent")
            self._proc = None

    def _cleanup_staging_root(self) -> None:
        _owned_staging_roots_pending_cleanup.discard(str(self._staging_root))
        if not self._owns_staging_root or self.config.persist:
            return
        shutil.rmtree(self._staging_root, ignore_errors=True)
        try:
            self._staging_root.parent.rmdir()
        except OSError:
            pass

    def destroy(self) -> None:
        """Force-remove the sandbox and delete its staging root."""
        self._log_lifecycle("sbx.destroy.start")
        if not self._shutdown:
            with contextlib.suppress(Exception):
                self.shutdown()
        if self._sandbox_name:
            self.remove(self._sandbox_name)
        _owned_staging_roots_pending_cleanup.discard(str(self._staging_root))
        shutil.rmtree(self._staging_root, ignore_errors=True)
        with contextlib.suppress(OSError):
            self._staging_root.parent.rmdir()
        self._sandbox_name = None
        self._log_lifecycle("sbx.destroy.complete")

    @classmethod
    def remove(cls, name: str) -> None:
        """Force-remove a persisted sandbox by name (no staging-root cleanup)."""
        subprocess.run(
            ["sbx", "rm", "--force", name],
            check=False,
            capture_output=True,
            text=True,
        )

    def _ensure_process(self) -> None:
        if self._uses_websocket_transport():
            self._ensure_websocket_supervisor()
            return
        if self._proc and self._proc.poll() is None:
            return
        if self._proc and self._proc.poll() is not None:
            raise SandboxFatalError("Sbx supervisor process exited unexpectedly")

        start = time.perf_counter()
        try:
            if self._supervisor_command is not None:
                self._setup_direct_workspace_aliases_host_side()
                command = self._supervisor_command
            else:
                command = self._start_sbx_and_build_supervisor_command()
            env = os.environ.copy()
            env["PREDICT_RLM_SBX_ROOT"] = str(self._staging_root)
            self._log_lifecycle(
                "sbx.runner.start",
                command=command[0] if command else None,
            )
            self._proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                bufsize=1,
                start_new_session=True,
            )
            self._start_stdout_reader()
            if self.output_fields:
                self._send_request("register_output_fields", {"fields": self.output_fields})
            if self.tools:
                self._send_request("register_tools", {"tools": list(self.tools)})
            if self.runtime_hooks:
                self._register_runtime_hooks()
        except BaseException as exc:
            self._log_lifecycle(
                "sbx.runner.error",
                status="error",
                error_type=type(exc).__name__,
                duration_ms=round((time.perf_counter() - start) * 1000),
            )
            raise
        self._log_lifecycle(
            "sbx.runner.started",
            status="ok",
            duration_ms=round((time.perf_counter() - start) * 1000),
            sandbox_name=self._sandbox_name,
            process_pid=getattr(self._proc, "pid", None),
        )

    def _start_stdout_reader(self) -> None:
        assert self._proc is not None
        assert self._proc.stdout is not None
        stdout = self._proc.stdout
        self._stdout_lines = queue.Queue()

        def read_stdout() -> None:
            for line in stdout:
                self._stdout_lines.put(line)

        self._stdout_reader = threading.Thread(
            target=read_stdout,
            name="predict-rlm-sbx-stdout",
            daemon=True,
        )
        self._stdout_reader.start()

    def _ensure_websocket_supervisor(self) -> None:
        if self._ws is not None:
            return
        if self._proc and self._proc.poll() is not None:
            raise SandboxFatalError("Sbx supervisor process exited unexpectedly")

        start = time.perf_counter()
        try:
            if self._websocket_supervisor_command is not None and self._proc is None:
                self._setup_direct_workspace_aliases_host_side()
                self._start_local_websocket_supervisor()
            if self._websocket_url is None:
                self._start_sbx_websocket_supervisor()
            assert self._websocket_url is not None
            self._connect_websocket_supervisor(self._websocket_url)
            if self.output_fields:
                self._send_websocket_request(
                    "register_output_fields",
                    {"fields": self.output_fields},
                )
            if self.tools:
                self._send_websocket_request("register_tools", {"tools": list(self.tools)})
            if self.runtime_hooks:
                self._send_websocket_request(
                    "register_runtime_hooks",
                    {"hooks": [hook.model_dump(mode="json") for hook in self.runtime_hooks]},
                )
        except BaseException as exc:
            self._log_lifecycle(
                "sbx.runner.error",
                status="error",
                error_type=type(exc).__name__,
                duration_ms=round((time.perf_counter() - start) * 1000),
            )
            self._teardown_failed_websocket_supervisor()
            raise
        self._log_lifecycle(
            "sbx.runner.started",
            status="ok",
            transport="websocket",
            duration_ms=round((time.perf_counter() - start) * 1000),
            sandbox_name=self._sandbox_name,
            process_pid=getattr(self._proc, "pid", None),
        )

    async def _aensure_websocket_supervisor(self) -> None:
        loop = asyncio.get_running_loop()
        if self._async_loop is not None and self._async_loop is not loop:
            raise RuntimeError("SbxBackend async transport cannot move between event loops")
        self._async_loop = loop
        if self._async_ws is not None:
            return
        if self._async_proc is not None and self._async_proc.returncode is not None:
            raise SandboxFatalError("Sbx supervisor process exited unexpectedly")
        if self._proc is not None and self._proc.poll() is not None:
            raise SandboxFatalError("Sbx supervisor process exited unexpectedly")

        start = time.perf_counter()
        try:
            if (
                self._websocket_supervisor_command is not None
                and self._async_proc is None
                and self._proc is None
            ):
                self._setup_direct_workspace_aliases_host_side()
                await self._astart_local_websocket_supervisor()
            if self._websocket_url is None:
                await self._astart_sbx_websocket_supervisor()
            assert self._websocket_url is not None
            await self._aconnect_websocket_supervisor(self._websocket_url)
            if self.output_fields:
                await self._asend_websocket_request(
                    "register_output_fields",
                    {"fields": self.output_fields},
                )
            if self.tools:
                await self._asend_websocket_request(
                    "register_tools", {"tools": list(self.tools)}
                )
            if self.runtime_hooks:
                await self._asend_websocket_request(
                    "register_runtime_hooks",
                    {"hooks": [hook.model_dump(mode="json") for hook in self.runtime_hooks]},
                )
        except BaseException as exc:
            self._log_lifecycle(
                "sbx.runner.error",
                status="error",
                error_type=type(exc).__name__,
                duration_ms=round((time.perf_counter() - start) * 1000),
            )
            await self._ateardown_failed_websocket_supervisor()
            raise
        self._log_lifecycle(
            "sbx.runner.started",
            status="ok",
            transport="websocket",
            duration_ms=round((time.perf_counter() - start) * 1000),
            sandbox_name=self._sandbox_name,
            process_pid=self._async_process_pid(),
        )

    async def _astart_local_websocket_supervisor(self) -> None:
        assert self._websocket_supervisor_command is not None
        env = os.environ.copy()
        env["PREDICT_RLM_SBX_ROOT"] = str(self._staging_root)
        self._log_lifecycle(
            "sbx.runner.start",
            command=self._websocket_supervisor_command[0],
            transport="websocket",
        )
        self._async_proc = await asyncio.create_subprocess_exec(
            *self._websocket_supervisor_command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            start_new_session=True,
        )

    async def _astart_sbx_websocket_supervisor(self) -> None:
        supervisor_path = await self._astart_sbx_and_prepare_supervisor()
        assert self._sandbox_name is not None
        websocket_port = self._resolve_websocket_port()
        self._active_websocket_port = websocket_port
        runner_root = self._staging_root
        runner_root.mkdir(parents=True, exist_ok=True)
        command = [
            "sbx",
            "exec",
            "-w",
            str(self._staging_root),
            self._sandbox_name,
            "env",
            f"PREDICT_RLM_SBX_ROOT={runner_root}",
            SBX_PYTHON_EXECUTABLE,
            "-u",
            str(supervisor_path),
            "--websocket-host",
            "0.0.0.0",
            "--websocket-port",
            str(websocket_port),
            "--websocket-path",
            self._websocket_path,
            "--websocket-max-message-bytes",
            str(self.config.websocket_max_message_bytes),
        ]
        self._log_lifecycle(
            "sbx.runner.start",
            command=command[0],
            transport="websocket",
        )
        self._async_proc = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        self._websocket_url = await self._apublish_websocket_port(websocket_port)

    async def _aconnect_websocket_supervisor(self, url: str) -> None:
        deadline = time.monotonic() + self.config.websocket_startup_timeout
        last_error: BaseException | None = None
        while True:
            if self._async_process_exited():
                stderr = await self._aread_active_process_stderr()
                diagnostic = stderr.strip() or str(last_error or "process exited")
                raise SandboxFatalError(
                    "Sbx WebSocket supervisor exited before accepting connections at "
                    f"{url}: {diagnostic}"
                )
            try:
                self._async_ws = await async_websocket_connect(
                    url,
                    open_timeout=min(2.0, max(0.1, deadline - time.monotonic())),
                    max_size=self.config.websocket_max_message_bytes,
                    max_queue=32,
                    proxy=None,
                )
                self._log_lifecycle("sbx.websocket.connected", endpoint=url)
                return
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                last_error = exc
                if time.monotonic() >= deadline:
                    raise SandboxFatalError(
                        "Timed out connecting to sbx WebSocket supervisor at "
                        f"{url}: {last_error}"
                    ) from last_error
                await asyncio.sleep(0.1)

    async def _apublish_websocket_port(self, port: int | None = None) -> str:
        assert self._sandbox_name is not None
        port = port or self._active_websocket_port or self.config.websocket_port
        if not port:
            raise SandboxFatalError("Cannot publish sbx WebSocket supervisor without a port")
        result = await self._arun_command(
            [
                "sbx",
                "ports",
                self._sandbox_name,
                "--publish",
                str(port),
            ],
            timeout=self.config.exec_timeout,
            check=False,
        )
        if result.returncode != 0:
            raise SandboxFatalError(
                "Failed to publish sbx WebSocket supervisor port "
                f"{port}: exit code {result.returncode}; "
                f"stdout: {result.stdout.strip()}; stderr: {result.stderr.strip()}"
            )
        endpoint = self._parse_published_websocket_endpoint(result.stdout)
        self._published_websocket_url = endpoint
        self._log_lifecycle("sbx.websocket.published", endpoint=endpoint)
        return endpoint

    def _start_local_websocket_supervisor(self) -> None:
        assert self._websocket_supervisor_command is not None
        env = os.environ.copy()
        env["PREDICT_RLM_SBX_ROOT"] = str(self._staging_root)
        self._log_lifecycle(
            "sbx.runner.start",
            command=self._websocket_supervisor_command[0],
            transport="websocket",
        )
        self._proc = subprocess.Popen(
            self._websocket_supervisor_command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=1,
            start_new_session=True,
        )

    def _start_sbx_websocket_supervisor(self) -> None:
        supervisor_path = self._start_sbx_and_prepare_supervisor()
        assert self._sandbox_name is not None
        websocket_port = self._resolve_websocket_port()
        self._active_websocket_port = websocket_port
        runner_root = self._staging_root
        runner_root.mkdir(parents=True, exist_ok=True)
        command = [
            "sbx",
            "exec",
            "-w",
            str(self._staging_root),
            self._sandbox_name,
            "env",
            f"PREDICT_RLM_SBX_ROOT={runner_root}",
            SBX_PYTHON_EXECUTABLE,
            "-u",
            str(supervisor_path),
            "--websocket-host",
            "0.0.0.0",
            "--websocket-port",
            str(websocket_port),
            "--websocket-path",
            self._websocket_path,
            "--websocket-max-message-bytes",
            str(self.config.websocket_max_message_bytes),
        ]
        self._log_lifecycle(
            "sbx.runner.start",
            command=command[0],
            transport="websocket",
        )
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self._websocket_url = self._publish_websocket_port(websocket_port)

    def _resolve_websocket_port(self) -> int:
        if self.config.websocket_port:
            return self.config.websocket_port
        return self._choose_dynamic_websocket_port()

    def _choose_dynamic_websocket_port(self) -> int:
        return 20_000 + secrets.randbelow(40_000)

    def _connect_websocket_supervisor(self, url: str) -> None:
        deadline = time.monotonic() + self.config.websocket_startup_timeout
        last_error: BaseException | None = None
        while True:
            if self._proc is not None and self._proc.poll() is not None:
                stderr = self._read_stderr_for_process(self._proc)
                diagnostic = stderr.strip() or str(last_error or "process exited")
                raise SandboxFatalError(
                    "Sbx WebSocket supervisor exited before accepting connections at "
                    f"{url}: {diagnostic}"
                )
            try:
                self._ws = websocket_connect(
                    url,
                    open_timeout=min(2.0, max(0.1, deadline - time.monotonic())),
                    max_size=self.config.websocket_max_message_bytes,
                    max_queue=32,
                    proxy=None,
                )
                self._log_lifecycle("sbx.websocket.connected", endpoint=url)
                return
            except BaseException as exc:
                last_error = exc
                if time.monotonic() >= deadline:
                    raise SandboxFatalError(
                        "Timed out connecting to sbx WebSocket supervisor at "
                        f"{url}: {last_error}"
                    ) from last_error
                time.sleep(0.1)

    def _publish_websocket_port(self, port: int | None = None) -> str:
        assert self._sandbox_name is not None
        port = port or self._active_websocket_port or self.config.websocket_port
        if not port:
            raise SandboxFatalError("Cannot publish sbx WebSocket supervisor without a port")
        result = subprocess.run(
            [
                "sbx",
                "ports",
                self._sandbox_name,
                "--publish",
                str(port),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=self.config.exec_timeout,
        )
        if result.returncode != 0:
            raise SandboxFatalError(
                "Failed to publish sbx WebSocket supervisor port "
                f"{port}: exit code {result.returncode}; "
                f"stdout: {result.stdout.strip()}; stderr: {result.stderr.strip()}"
            )
        endpoint = self._parse_published_websocket_endpoint(result.stdout)
        self._published_websocket_url = endpoint
        self._log_lifecycle("sbx.websocket.published", endpoint=endpoint)
        return endpoint

    def _parse_published_websocket_endpoint(self, stdout: str) -> str:
        match = _LOCALHOST_ENDPOINT_RE.search(stdout)
        if not match:
            raise SandboxFatalError(
                "Could not determine sbx published WebSocket endpoint from "
                f"`sbx ports` output: {stdout.strip()}"
            )
        scheme = "wss" if match.group("scheme") == "https" else "ws"
        host = match.group("host")
        port = match.group("port")
        return f"{scheme}://{host}:{port}{self._websocket_path}"

    def _start_sbx_and_build_supervisor_command(self) -> list[str]:
        supervisor_path = self._start_sbx_and_prepare_supervisor()
        assert self._sandbox_name is not None
        runner_root = self._staging_root
        runner_root.mkdir(parents=True, exist_ok=True)
        return [
            "sbx",
            "exec",
            "-i",
            "-w",
            str(self._staging_root),
            self._sandbox_name,
            "env",
            f"PREDICT_RLM_SBX_ROOT={runner_root}",
            SBX_PYTHON_EXECUTABLE,
            "-u",
            str(supervisor_path),
        ]

    def _start_sbx_and_prepare_supervisor(self) -> Path:
        if shutil.which("sbx") is None:
            self._log_lifecycle("sbx.create.missing_cli", status="error")
            raise SandboxFatalError(
                "Docker Sandboxes backend requires the `sbx` CLI. "
                "Install it with `brew install docker/tap/sbx` and run `sbx login`."
            )

        if self._sandbox_name is not None:
            return self._prepared_supervisor_path or self._prepare_supervisor_script()

        supervisor_path = self._prepare_supervisor_script()

        if self.config.reuse and self._try_reattach_named_sandbox():
            if self.skill_packages:
                self._apply_network_policy()
                self.ensure_skill_packages(self.skill_packages)
            return supervisor_path

        self._create_and_bootstrap_sandbox()
        return supervisor_path

    async def _astart_sbx_and_prepare_supervisor(self) -> Path:
        if shutil.which("sbx") is None:
            self._log_lifecycle("sbx.create.missing_cli", status="error")
            raise SandboxFatalError(
                "Docker Sandboxes backend requires the `sbx` CLI. "
                "Install it with `brew install docker/tap/sbx` and run `sbx login`."
            )
        if self._sandbox_name is not None:
            return self._prepared_supervisor_path or self._prepare_supervisor_script()

        supervisor_path = self._prepare_supervisor_script()
        if self.config.reuse and await self._atry_reattach_named_sandbox():
            if self.skill_packages:
                await self._aapply_network_policy()
                await self.aensure_skill_packages(self.skill_packages)
            return supervisor_path
        await self._acreate_and_bootstrap_sandbox()
        return supervisor_path

    def _create_and_bootstrap_sandbox(self) -> None:
        primary_workspace = str(self._staging_root)
        if self.config.workspace_read_only:
            primary_workspace = f"{primary_workspace}:ro"
        direct_workspaces = self._direct_workspace_args()
        sandbox_name = self.config.name or f"predict-rlm-{uuid.uuid4().hex[:12]}"
        create_cmd = [
            "sbx",
            "create",
            "shell",
            primary_workspace,
            *self.config.extra_workspaces,
            *direct_workspaces,
            "--name",
            sandbox_name,
        ]
        for flag, value in (
            ("--cpus", self.config.cpus),
            ("--memory", self.config.memory),
            ("--template", self.config.template),
            ("--kit", self.config.kit),
            ("--branch", self.config.branch),
        ):
            if value is not None:
                create_cmd.extend([flag, str(value)])
        create_start = time.perf_counter()
        self._log_lifecycle(
            "sbx.create.start",
            create_timeout=self.config.create_timeout,
            workspace_read_only=self.config.workspace_read_only,
            extra_workspaces=len(self.config.extra_workspaces),
        )
        try:
            created = subprocess.run(
                create_cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.config.create_timeout,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            self._log_lifecycle(
                "sbx.create.error",
                duration_ms=ms_since(create_start),
                error_type=type(exc).__name__,
                status="error",
            )
            raise SandboxFatalError(f"Failed to create sbx sandbox: {exc}") from exc

        self._sandbox_name = sandbox_name
        self._log_lifecycle(
            "sbx.create.ok",
            duration_ms=ms_since(create_start),
            stdout_chars=len(created.stdout or ""),
            stderr_chars=len(created.stderr or ""),
        )
        self._apply_network_policy()
        self._bootstrap_packages()
        self._setup_direct_workspace_aliases_in_sandbox()

    async def _acreate_and_bootstrap_sandbox(self) -> None:
        primary_workspace = str(self._staging_root)
        if self.config.workspace_read_only:
            primary_workspace = f"{primary_workspace}:ro"
        sandbox_name = self.config.name or f"predict-rlm-{uuid.uuid4().hex[:12]}"
        create_cmd = [
            "sbx",
            "create",
            "shell",
            primary_workspace,
            *self.config.extra_workspaces,
            *self._direct_workspace_args(),
            "--name",
            sandbox_name,
        ]
        for flag, value in (
            ("--cpus", self.config.cpus),
            ("--memory", self.config.memory),
            ("--template", self.config.template),
            ("--kit", self.config.kit),
            ("--branch", self.config.branch),
        ):
            if value is not None:
                create_cmd.extend([flag, str(value)])
        create_start = time.perf_counter()
        self._log_lifecycle(
            "sbx.create.start",
            create_timeout=self.config.create_timeout,
            workspace_read_only=self.config.workspace_read_only,
            extra_workspaces=len(self.config.extra_workspaces),
        )
        try:
            created = await self._arun_command(
                create_cmd,
                check=True,
                timeout=self.config.create_timeout,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            self._log_lifecycle(
                "sbx.create.error",
                duration_ms=ms_since(create_start),
                error_type=type(exc).__name__,
                status="error",
            )
            raise SandboxFatalError(f"Failed to create sbx sandbox: {exc}") from exc

        self._sandbox_name = sandbox_name
        self._log_lifecycle(
            "sbx.create.ok",
            duration_ms=ms_since(create_start),
            stdout_chars=len(created.stdout or ""),
            stderr_chars=len(created.stderr or ""),
        )
        await self._aapply_network_policy()
        await self._abootstrap_packages()
        await self._asetup_direct_workspace_aliases_in_sandbox()

    async def _aprobe_sandbox_state(self, name: str) -> str:
        result = await self._arun_command(
            ["sbx", "ls"],
            check=False,
            timeout=self.config.exec_timeout,
        )
        if result.returncode != 0:
            return "missing"
        for line in result.stdout.splitlines():
            fields = line.split()
            if not fields or fields[0] != name:
                continue
            rest = " ".join(fields[1:]).lower()
            if "stop" in rest or "exit" in rest:
                return "stopped"
            return "running"
        return "missing"

    async def _asbx_sandbox_healthy(self, name: str) -> bool:
        result = await self._arun_command(
            ["sbx", "exec", name, "true"],
            check=False,
            timeout=self.config.exec_timeout,
        )
        return result.returncode == 0

    async def _atry_reattach_named_sandbox(self) -> bool:
        name = self.config.name
        assert name is not None
        self._log_lifecycle("sbx.reattach.start", sandbox_name=name)
        state = await self._aprobe_sandbox_state(name)
        if state == "missing":
            self._log_lifecycle("sbx.reattach.miss", sandbox_name=name)
            return False
        if state == "stopped":
            start_result = await self._arun_command(
                ["sbx", "start", name],
                check=False,
                timeout=self.config.create_timeout,
            )
            if start_result.returncode != 0:
                self._log_lifecycle(
                    "sbx.reattach.unhealthy.recreate",
                    sandbox_name=name,
                    reason="start_failed",
                )
                await self._aforce_remove_sandbox(name)
                return False
        if not await self._asbx_sandbox_healthy(name):
            self._log_lifecycle(
                "sbx.reattach.unhealthy.recreate",
                sandbox_name=name,
                reason="health_check_failed",
            )
            await self._aforce_remove_sandbox(name)
            return False
        self._sandbox_name = name
        await self._asetup_direct_workspace_aliases_in_sandbox()
        self._log_lifecycle("sbx.reattach.ok", sandbox_name=name)
        return True

    async def _aforce_remove_sandbox(self, name: str) -> None:
        await self._arun_command(
            ["sbx", "rm", "--force", name],
            check=False,
        )

    def _probe_sandbox_state(self, name: str) -> str:
        """Resolve a named sandbox to ``running`` / ``stopped`` / ``missing``."""
        result = subprocess.run(
            ["sbx", "ls"],
            check=False,
            capture_output=True,
            text=True,
            timeout=self.config.exec_timeout,
        )
        if result.returncode != 0:
            return "missing"
        for line in result.stdout.splitlines():
            fields = line.split()
            if not fields or fields[0] != name:
                continue
            rest = " ".join(fields[1:]).lower()
            if "stop" in rest or "exit" in rest:
                return "stopped"
            return "running"
        return "missing"

    def _sbx_sandbox_healthy(self, name: str) -> bool:
        """Cheap liveness probe: a trivial in-container command must succeed."""
        result = subprocess.run(
            ["sbx", "exec", name, "true"],
            check=False,
            capture_output=True,
            text=True,
            timeout=self.config.exec_timeout,
        )
        return result.returncode == 0

    def _try_reattach_named_sandbox(self) -> bool:
        """Return True when an existing named sandbox is ready to reuse."""
        name = self.config.name
        assert name is not None
        self._log_lifecycle("sbx.reattach.start", sandbox_name=name)
        state = self._probe_sandbox_state(name)

        if state == "missing":
            self._log_lifecycle("sbx.reattach.miss", sandbox_name=name)
            return False

        if state == "stopped":
            start_result = subprocess.run(
                ["sbx", "start", name],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.config.create_timeout,
            )
            if start_result.returncode != 0:
                self._log_lifecycle(
                    "sbx.reattach.unhealthy.recreate",
                    sandbox_name=name,
                    reason="start_failed",
                )
                self._force_remove_sandbox(name)
                return False

        if not self._sbx_sandbox_healthy(name):
            self._log_lifecycle(
                "sbx.reattach.unhealthy.recreate",
                sandbox_name=name,
                reason="health_check_failed",
            )
            self._force_remove_sandbox(name)
            return False

        self._sandbox_name = name
        self._setup_direct_workspace_aliases_in_sandbox()
        self._log_lifecycle("sbx.reattach.ok", sandbox_name=name)
        return True

    def _force_remove_sandbox(self, name: str) -> None:
        subprocess.run(
            ["sbx", "rm", "--force", name],
            check=False,
            capture_output=True,
            text=True,
        )

    def _direct_workspace_args(self) -> list[str]:
        seen = {str(self._staging_root)}
        args: list[str] = []
        for mount in self._direct_workspace_mounts:
            if mount.host_path in seen:
                continue
            seen.add(mount.host_path)
            args.append(mount.host_path)
        return args

    def _direct_workspace_aliases(self) -> list[tuple[str, str]]:
        return [
            (mount.host_path, mount.sandbox_path)
            for mount in self._direct_workspace_mounts
            if mount.host_path != mount.sandbox_path
        ]

    def _setup_direct_workspace_aliases_in_sandbox(self) -> None:
        aliases = self._direct_workspace_aliases()
        if not aliases:
            return
        assert self._sandbox_name is not None
        script = (
            "import json, os, pathlib, sys\n"
            "for source, target in json.loads(sys.argv[1]):\n"
            "    source_path = pathlib.Path(source)\n"
            "    target_path = pathlib.Path(target)\n"
            "    if target_path.exists() or target_path.is_symlink():\n"
            "        if target_path.is_symlink() and os.readlink(target_path) == str(source_path):\n"
            "            continue\n"
            "        raise FileExistsError(f'Direct workspace alias already exists: {target}')\n"
            "    target_path.parent.mkdir(parents=True, exist_ok=True)\n"
            "    target_path.symlink_to(source_path, target_is_directory=True)\n"
        )
        result = subprocess.run(
            [
                "sbx",
                "exec",
                "-w",
                str(self._staging_root),
                "-u",
                "root",
                self._sandbox_name,
                SBX_PYTHON_EXECUTABLE,
                "-c",
                script,
                json.dumps(aliases),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=self.config.exec_timeout,
        )
        if result.returncode != 0:
            raise SandboxFatalError(
                "Failed to configure direct workspace aliases: "
                f"stdout: {result.stdout.strip()}; stderr: {result.stderr.strip()}"
            )

    async def _asetup_direct_workspace_aliases_in_sandbox(self) -> None:
        aliases = self._direct_workspace_aliases()
        if not aliases:
            return
        assert self._sandbox_name is not None
        script = (
            "import json, os, pathlib, sys\n"
            "for source, target in json.loads(sys.argv[1]):\n"
            "    source_path = pathlib.Path(source)\n"
            "    target_path = pathlib.Path(target)\n"
            "    if target_path.exists() or target_path.is_symlink():\n"
            "        if target_path.is_symlink() and os.readlink(target_path) == str(source_path):\n"
            "            continue\n"
            "        raise FileExistsError(f'Direct workspace alias already exists: {target}')\n"
            "    target_path.parent.mkdir(parents=True, exist_ok=True)\n"
            "    target_path.symlink_to(source_path, target_is_directory=True)\n"
        )
        result = await self._arun_command(
            [
                "sbx",
                "exec",
                "-w",
                str(self._staging_root),
                "-u",
                "root",
                self._sandbox_name,
                SBX_PYTHON_EXECUTABLE,
                "-c",
                script,
                json.dumps(aliases),
            ],
            check=False,
            timeout=self.config.exec_timeout,
        )
        if result.returncode != 0:
            raise SandboxFatalError(
                "Failed to configure direct workspace aliases: "
                f"stdout: {result.stdout.strip()}; stderr: {result.stderr.strip()}"
            )

    def _setup_direct_workspace_aliases_host_side(self) -> None:
        for source, target in self._direct_workspace_aliases():
            source_path = Path(source)
            target_path = Path(target)
            if target_path.exists() or target_path.is_symlink():
                if target_path.is_symlink() and os.readlink(target_path) == str(source_path):
                    continue
                raise FileExistsError(f"Direct workspace alias already exists: {target}")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.symlink_to(source_path, target_is_directory=True)
            self._owned_direct_aliases.append(target_path)

    def _cleanup_direct_workspace_aliases_host_side(self) -> None:
        for path in reversed(self._owned_direct_aliases):
            try:
                if path.is_symlink():
                    path.unlink()
            except OSError:
                pass
        self._owned_direct_aliases.clear()

    def _start_sbx_and_build_runner_command(self) -> list[str]:
        return self._start_sbx_and_build_supervisor_command()

    def _prepare_supervisor_script(self) -> Path:
        supervisor_dir = self._staging_root / ".predict_rlm_supervisor"
        supervisor_dir.mkdir(parents=True, exist_ok=True)
        supervisor_path = supervisor_dir / "_payload.py"
        shutil.copy2(SUPERVISOR_PAYLOAD_SOURCE_PATH, supervisor_path)
        self._prepared_supervisor_path = supervisor_path
        return supervisor_path

    def _apply_network_policy(self) -> None:
        domains = list(DEFAULT_PACKAGE_DOMAINS)
        domains.extend(self.allowed_domains or [])
        self._log_lifecycle("sbx.network_policy.start", domains=len(domains))
        for domain in domains:
            result = subprocess.run(
                ["sbx", "policy", "allow", "network", domain],
                check=False,
                capture_output=True,
                text=True,
            )
            self._log_lifecycle(
                "sbx.network_policy.domain",
                domain=domain,
                returncode=result.returncode,
                status="ok" if result.returncode == 0 else "error",
            )
        self._log_lifecycle("sbx.network_policy.complete", domains=len(domains))

    async def _aapply_network_policy(self) -> None:
        domains = list(DEFAULT_PACKAGE_DOMAINS)
        domains.extend(self.allowed_domains or [])
        self._log_lifecycle("sbx.network_policy.start", domains=len(domains))
        for domain in domains:
            result = await self._arun_command(
                ["sbx", "policy", "allow", "network", domain],
                check=False,
            )
            self._log_lifecycle(
                "sbx.network_policy.domain",
                domain=domain,
                returncode=result.returncode,
                status="ok" if result.returncode == 0 else "error",
            )
        self._log_lifecycle("sbx.network_policy.complete", domains=len(domains))

    def _bootstrap_packages(self) -> None:
        packages = list(SBX_TRANSPORT_PACKAGES)
        if self.preinstall_packages:
            packages.extend(["pydantic", "pandas"])
        packages.extend(self.skill_packages)
        packages = _dedupe_packages(packages)
        self._install_packages(
            packages,
            event="sbx.bootstrap",
            failure_label="bootstrap sbx packages",
        )
        self._installed_skill_packages.update(self.skill_packages)

    async def _abootstrap_packages(self) -> None:
        packages = list(SBX_TRANSPORT_PACKAGES)
        if self.preinstall_packages:
            packages.extend(["pydantic", "pandas"])
        packages.extend(self.skill_packages)
        packages = _dedupe_packages(packages)
        await self._ainstall_packages(
            packages,
            event="sbx.bootstrap",
            failure_label="bootstrap sbx packages",
        )
        self._installed_skill_packages.update(self.skill_packages)

    def _install_packages(
        self,
        packages: list[str],
        *,
        event: str,
        failure_label: str,
    ) -> None:
        if not packages:
            return
        assert self._sandbox_name is not None
        command = [
            "sbx",
            "exec",
            "-w",
            str(self._staging_root),
            self._sandbox_name,
            SBX_PYTHON_EXECUTABLE,
            "-m",
            "pip",
            "install",
            "--break-system-packages",
            *packages,
        ]
        install_start = time.perf_counter()
        self._log_lifecycle(
            f"{event}.start",
            packages=",".join(packages),
            timeout_seconds=self.config.exec_timeout,
        )
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.config.exec_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            self._log_lifecycle(
                f"{event}.timeout",
                packages=",".join(packages),
                duration_ms=ms_since(install_start),
                status="error",
            )
            raise SandboxFatalError(
                f"Failed to {failure_label} {packages}: timed out after "
                f"{self.config.exec_timeout}s"
            ) from exc
        if result.returncode != 0:
            self._log_lifecycle(
                f"{event}.error",
                packages=",".join(packages),
                duration_ms=ms_since(install_start),
                returncode=result.returncode,
                stdout_chars=len(result.stdout or ""),
                stderr_chars=len(result.stderr or ""),
                status="error",
            )
            raise SandboxFatalError(
                f"Failed to {failure_label} "
                f"{packages}: exit code {result.returncode}; "
                f"stdout: {result.stdout.strip()}; stderr: {result.stderr.strip()}"
            )
        self._log_lifecycle(
            f"{event}.ok",
            packages=",".join(packages),
            duration_ms=ms_since(install_start),
            stdout_chars=len(result.stdout or ""),
            stderr_chars=len(result.stderr or ""),
        )

    async def _ainstall_packages(
        self,
        packages: list[str],
        *,
        event: str,
        failure_label: str,
    ) -> None:
        if not packages:
            return
        assert self._sandbox_name is not None
        command = [
            "sbx",
            "exec",
            "-w",
            str(self._staging_root),
            self._sandbox_name,
            SBX_PYTHON_EXECUTABLE,
            "-m",
            "pip",
            "install",
            "--break-system-packages",
            *packages,
        ]
        install_start = time.perf_counter()
        self._log_lifecycle(
            f"{event}.start",
            packages=",".join(packages),
            timeout_seconds=self.config.exec_timeout,
        )
        try:
            result = await self._arun_command(
                command,
                check=False,
                timeout=self.config.exec_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            self._log_lifecycle(
                f"{event}.timeout",
                packages=",".join(packages),
                duration_ms=ms_since(install_start),
                status="error",
            )
            raise SandboxFatalError(
                f"Failed to {failure_label} {packages}: timed out after "
                f"{self.config.exec_timeout}s"
            ) from exc
        if result.returncode != 0:
            self._log_lifecycle(
                f"{event}.error",
                packages=",".join(packages),
                duration_ms=ms_since(install_start),
                returncode=result.returncode,
                stdout_chars=len(result.stdout or ""),
                stderr_chars=len(result.stderr or ""),
                status="error",
            )
            raise SandboxFatalError(
                f"Failed to {failure_label} "
                f"{packages}: exit code {result.returncode}; "
                f"stdout: {result.stdout.strip()}; stderr: {result.stderr.strip()}"
            )
        self._log_lifecycle(
            f"{event}.ok",
            packages=",".join(packages),
            duration_ms=ms_since(install_start),
            stdout_chars=len(result.stdout or ""),
            stderr_chars=len(result.stderr or ""),
        )

    async def _arun_command(
        self,
        command: list[str],
        *,
        check: bool,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise
        except TimeoutError as exc:
            process.kill()
            stdout_bytes, stderr_bytes = await process.communicate()
            raise subprocess.TimeoutExpired(
                command,
                timeout,
                output=stdout_bytes.decode("utf-8", errors="replace"),
                stderr=stderr_bytes.decode("utf-8", errors="replace"),
            ) from exc
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        result = subprocess.CompletedProcess(command, process.returncode or 0, stdout, stderr)
        if check and result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode,
                command,
                output=stdout,
                stderr=stderr,
            )
        return result

    def _read_stdout_line(self, deadline: float) -> str | None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        try:
            return self._stdout_lines.get(timeout=min(remaining, 0.05))
        except queue.Empty:
            return None

    def _read_supervisor_stdout_line(
        self,
        process: SupervisorProcess,
        *,
        deadline: float,
        timeout: float,
    ) -> str | None:
        if (
            self._active_execute_timeout_deadline is not None
            and self._pending_tool_calls
            and time.monotonic() >= self._active_execute_timeout_deadline
        ):
            return None
        return self._read_stdout_line(deadline) or ""

    def _resolve_execution_timeout(self, timeout: float | None) -> tuple[float, str]:
        return resolve_execution_timeout(timeout, default_timeout=self.config.exec_timeout)

    def _host_watchdog_timeout(
        self,
        timeout_seconds: float,
        timeout_failure_class: str,
    ) -> float:
        return recoverable_timeout_host_deadline_seconds(
            timeout_seconds,
            timeout_failure_class,
        )

    def _fail_timed_out_request(
        self,
        timeout_seconds: float,
        host_timeout_seconds: float,
        timeout_failure_class: str,
        *,
        fatal: bool = True,
    ) -> None:
        if self._uses_websocket_transport():
            self._log_lifecycle(
                "sbx.request.timeout",
                timeout_seconds=self.config.exec_timeout,
                status="error",
            )
            self._kill_websocket_supervisor()
            if not fatal:
                return
            if timeout_failure_class == ITERATION_TIMEOUT_FAILURE_CLASS:
                raise SandboxFatalError(
                    "Sbx supervisor failed to recover from iteration timeout after "
                    f"{timeout_seconds:g}s; waited {host_timeout_seconds:g}s before "
                    "force-killing supervisor"
                )
            raise SandboxFatalError(
                f"Sbx supervisor request timed out after {host_timeout_seconds:g}s"
            )

        assert self._proc is not None
        self._log_lifecycle(
            "sbx.request.timeout",
            timeout_seconds=self.config.exec_timeout,
            status="error",
        )
        self._proc.kill()
        with contextlib.suppress(Exception):
            self._proc.wait(timeout=1)
        if not fatal:
            return
        if timeout_failure_class == ITERATION_TIMEOUT_FAILURE_CLASS:
            raise SandboxFatalError(
                "Sbx supervisor failed to recover from iteration timeout after "
                f"{timeout_seconds:g}s; waited {host_timeout_seconds:g}s before "
                "force-killing supervisor"
            )
        raise SandboxFatalError(
            f"Sbx supervisor request timed out after {host_timeout_seconds:g}s"
        )

    def _kill_websocket_supervisor(self) -> None:
        if self._ws is not None:
            with contextlib.suppress(Exception):
                self._ws.close()
            self._ws = None
        if self._proc and self._proc.poll() is None:
            self._proc.kill()
            with contextlib.suppress(Exception):
                self._proc.wait(timeout=1)
            self._proc = None
            return
        if self._sandbox_name and self._prepared_supervisor_path is not None:
            subprocess.run(
                [
                    "sbx",
                    "exec",
                    "-w",
                    str(self._staging_root),
                    self._sandbox_name,
                    "pkill",
                    "-f",
                    str(self._prepared_supervisor_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=min(self.config.exec_timeout, 5),
            )
            if self._websocket_supervisor_command is None:
                self._websocket_url = None
                self._published_websocket_url = None

    async def _akill_websocket_supervisor(self) -> None:
        if self._async_ws is not None:
            with contextlib.suppress(Exception):
                await self._async_ws.close()
            self._async_ws = None
        if self._async_proc is not None:
            if self._async_proc.returncode is None:
                self._async_proc.kill()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(self._async_proc.wait(), timeout=1)
            self._async_proc = None
        elif self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
            deadline = time.monotonic() + 1
            while self._proc.poll() is None and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            self._proc = None
        elif self._sandbox_name and self._prepared_supervisor_path is not None:
            await self._arun_command(
                [
                    "sbx",
                    "exec",
                    "-w",
                    str(self._staging_root),
                    self._sandbox_name,
                    "pkill",
                    "-f",
                    str(self._prepared_supervisor_path),
                ],
                check=False,
                timeout=min(self.config.exec_timeout, 5),
            )
        if self._websocket_supervisor_command is None:
            self._websocket_url = None
            self._published_websocket_url = None
        self._active_async_request = None

    def _get_supervisor_process(self) -> subprocess.Popen[str] | None:
        return self._proc

    def _request_timeout_seconds(
        self,
        method: str,
        params: dict[str, Any],
        timeout: float | None,
    ) -> float:
        timeout_seconds, timeout_failure_class = self._resolve_execution_timeout(timeout)
        return self._host_watchdog_timeout(timeout_seconds, timeout_failure_class)

    def _execution_timeout_metadata_from_params(
        self,
        params: dict[str, Any],
    ) -> tuple[float, str]:
        if "execution_timeout_seconds" in params:
            return float(params["execution_timeout_seconds"]), ITERATION_TIMEOUT_FAILURE_CLASS
        return self._resolve_execution_timeout(None)

    def _submit_tool_call(self, request: dict[str, Any]) -> None:
        request_id = request.get("id")
        params = request.get("params", {})
        if self.verbose or live_tool_call_logging_enabled():
            emit_trace_tool_call(
                params.get("name"),
                args=params.get("args", []),
                kwargs=params.get("kwargs", {}),
            )
        self._log_lifecycle(
            "sbx.tool_call.start",
            tool=params.get("name"),
            request_id=request_id,
        )
        ctx = contextvars.copy_context()
        worker = self._start_sync_worker(ctx.run, self._build_tool_response, request)
        self._pending_tool_calls[worker.future] = request_id

    def _asubmit_tool_call(self, request: dict[str, Any]) -> None:
        request_id = request.get("id")
        params = request.get("params", {})
        if self.verbose or live_tool_call_logging_enabled():
            emit_trace_tool_call(
                params.get("name"),
                args=params.get("args", []),
                kwargs=params.get("kwargs", {}),
            )
        self._log_lifecycle(
            "sbx.tool_call.start",
            tool=params.get("name"),
            request_id=request_id,
        )
        task = asyncio.create_task(self._abuild_tool_response(request))
        self._async_pending_tool_calls[task] = request_id

    def _drain_completed_tool_calls(self) -> None:
        completed = [future for future in self._pending_tool_calls if future.done()]
        for future in completed:
            self._pending_tool_calls.pop(future)
            self._write_tool_response(future.result())

    async def _adrain_completed_tool_calls(self) -> None:
        completed = [task for task in self._async_pending_tool_calls if task.done()]
        for task in completed:
            self._async_pending_tool_calls.pop(task)
            await self._awrite_tool_response(task.result())

    def _drain_completed_supervisor_work(self) -> None:
        self._drain_completed_tool_calls()

    def _handle_supervisor_control_message(
        self,
        message: dict[str, Any],
        *,
        deadline: float,
    ) -> bool:
        if message.get("method") == "tool_call":
            self._submit_tool_call(message)
            return True
        if message.get("method") == "runtime_hook_event":
            self._handle_runtime_hook_event(message)
            return True
        return False

    def _handle_runtime_hook_event(self, request: dict[str, Any]) -> None:
        if self.on_runtime_hook_event is None:
            return
        try:
            event = RuntimeHookEvent.model_validate(request.get("params") or {})
            self.on_runtime_hook_event(event)
        except Exception:
            return

    async def _ahandle_supervisor_control_message(
        self,
        message: dict[str, Any],
    ) -> bool:
        if message.get("method") == "tool_call":
            self._asubmit_tool_call(message)
            return True
        if message.get("method") == "runtime_hook_event":
            self._handle_runtime_hook_event(message)
            return True
        return False

    def _build_tool_response(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = request.get("id")
        params = request.get("params", {})
        name = params.get("name")
        temp_dir: str | None = None
        call_start = time.perf_counter()
        args: list[Any] = []
        kwargs: dict[str, Any] = {}
        try:
            if name not in self.tools:
                raise CodeInterpreterError(f"Unknown tool: {name}")
            tool = self.tools[name]
            args = list(params.get("args", []))
            kwargs = dict(params.get("kwargs", {}))
            args, kwargs, synced_entries, temp_dir = self._prepare_synced_file_tool_args(
                tool,
                args,
                kwargs,
            )
            with self._execution_gate.tool_callback():
                result = tool(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = asyncio.run(result)
            for sandbox_path, host_path, writeback in synced_entries:
                if writeback and os.path.isfile(host_path):
                    self.mount_file_at(host_path, sandbox_path)
            result = to_plain_data(result)
            is_json = result is None or isinstance(result, (dict, list, int, float, bool))
            response = {
                "jsonrpc": "2.0",
                "result": {
                    "value": json.dumps(result) if is_json else str(result or ""),
                    "type": "json" if is_json else "string",
                },
                "id": request_id,
            }
            if name != "predict":
                record_tool_call(
                    ToolCall(
                        name=name,
                        args=args,
                        kwargs={k: v for k, v in kwargs.items() if k != "pydantic_schemas"},
                        result=result,
                        duration_ms=ms_since(call_start),
                    )
                )
            self._log_lifecycle(
                "sbx.tool_call.ok",
                tool=name,
                request_id=request_id,
                duration_ms=ms_since(call_start),
            )
            return response
        except Exception as exc:
            if name != "predict":
                record_tool_call(
                    ToolCall(
                        name=name or "",
                        args=args,
                        kwargs={k: v for k, v in kwargs.items() if k != "pydantic_schemas"},
                        result=None,
                        error=str(exc),
                        duration_ms=ms_since(call_start),
                    )
                )
            self._log_lifecycle(
                "sbx.tool_call.error",
                tool=name,
                request_id=request_id,
                duration_ms=ms_since(call_start),
                error_type=type(exc).__name__,
                status="error",
            )
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32000, "message": str(exc)},
                "id": request_id,
            }
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

    async def _abuild_tool_response(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = request.get("id")
        params = request.get("params", {})
        name = params.get("name")
        temp_dir: str | None = None
        call_start = time.perf_counter()
        args: list[Any] = []
        kwargs: dict[str, Any] = {}
        try:
            if name not in self.tools:
                raise CodeInterpreterError(f"Unknown tool: {name}")
            tool = self.tools[name]
            args = list(params.get("args", []))
            kwargs = dict(params.get("kwargs", {}))
            args, kwargs, synced_entries, temp_dir = (
                await self._aprepare_synced_file_tool_args(tool, args, kwargs)
            )
            with (
                self._execution_gate.async_tool_callback(),
                host_sync_worker_policy(self, detach_on_cancel=True),
            ):
                if inspect.iscoroutinefunction(tool):
                    result = await tool(*args, **kwargs)
                else:
                    result = await invoke_host_callable(tool, *args, **kwargs)
                    if inspect.isawaitable(result):
                        result = await result
            for sandbox_path, host_path, writeback in synced_entries:
                if writeback and os.path.isfile(host_path):
                    await self.amount_file_at(host_path, sandbox_path)
            result = to_plain_data(result)
            is_json = result is None or isinstance(result, (dict, list, int, float, bool))
            response = {
                "jsonrpc": "2.0",
                "result": {
                    "value": json.dumps(result) if is_json else str(result or ""),
                    "type": "json" if is_json else "string",
                },
                "id": request_id,
            }
            if name != "predict":
                record_tool_call(
                    ToolCall(
                        name=name,
                        args=args,
                        kwargs={k: v for k, v in kwargs.items() if k != "pydantic_schemas"},
                        result=result,
                        duration_ms=ms_since(call_start),
                    )
                )
            self._log_lifecycle(
                "sbx.tool_call.ok",
                tool=name,
                request_id=request_id,
                duration_ms=ms_since(call_start),
            )
            return response
        except asyncio.CancelledError as exc:
            worker = getattr(exc, "sync_worker", None)
            deferred_dir = temp_dir
            if deferred_dir and self.defer_until_sync_workers_finish(
                lambda: shutil.rmtree(deferred_dir, ignore_errors=True),
                worker,
            ):
                temp_dir = None
            raise
        except Exception as exc:
            if name != "predict":
                record_tool_call(
                    ToolCall(
                        name=name or "",
                        args=args,
                        kwargs={k: v for k, v in kwargs.items() if k != "pydantic_schemas"},
                        result=None,
                        error=str(exc),
                        duration_ms=ms_since(call_start),
                    )
                )
            self._log_lifecycle(
                "sbx.tool_call.error",
                tool=name,
                request_id=request_id,
                duration_ms=ms_since(call_start),
                error_type=type(exc).__name__,
                status="error",
            )
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32000, "message": str(exc)},
                "id": request_id,
            }
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _prepare_synced_file_tool_args(
        self,
        tool: Callable[..., Any],
        args: list[Any],
        kwargs: dict[str, Any],
    ) -> tuple[list[Any], dict[str, Any], list[tuple[str, str, bool]], str | None]:
        synced_params = (
            {}
            if getattr(tool, "__predict_rlm_synced_file_operation__", False)
            else get_synced_file_params(tool)
        )
        temp_dir: str | None = None
        synced_entries: list[tuple[str, str, bool]] = []
        if not synced_params:
            return args, kwargs, synced_entries, temp_dir

        sig = inspect.signature(tool)
        param_names = list(sig.parameters.keys())
        for param_name, synced_file in synced_params.items():
            sandbox_path = kwargs.get(param_name)
            if sandbox_path is None and param_name in param_names:
                idx = param_names.index(param_name)
                if idx < len(args):
                    sandbox_path = args[idx]
            if not sandbox_path or not isinstance(sandbox_path, str):
                continue

            if synced_file.host_dir is not None:
                host_dir = synced_file.host_dir
                os.makedirs(host_dir, exist_ok=True)
            else:
                if temp_dir is None:
                    temp_dir = tempfile.mkdtemp(prefix="tool-file-sync-")
                host_dir = temp_dir

            host_path = os.path.join(host_dir, os.path.basename(sandbox_path))
            self.sync_file_to(sandbox_path, host_path)
            synced_entries.append((sandbox_path, host_path, synced_file.writeback))

            if param_name in kwargs:
                kwargs[param_name] = host_path
            elif param_name in param_names:
                idx = param_names.index(param_name)
                if idx < len(args):
                    args[idx] = host_path

        return args, kwargs, synced_entries, temp_dir

    async def _aprepare_synced_file_tool_args(
        self,
        tool: Callable[..., Any],
        args: list[Any],
        kwargs: dict[str, Any],
    ) -> tuple[list[Any], dict[str, Any], list[tuple[str, str, bool]], str | None]:
        synced_params = (
            {}
            if getattr(tool, "__predict_rlm_synced_file_operation__", False)
            else get_synced_file_params(tool)
        )
        temp_dir: str | None = None
        synced_entries: list[tuple[str, str, bool]] = []
        if not synced_params:
            return args, kwargs, synced_entries, temp_dir

        sig = inspect.signature(tool)
        param_names = list(sig.parameters.keys())
        for param_name, synced_file in synced_params.items():
            sandbox_path = kwargs.get(param_name)
            if sandbox_path is None and param_name in param_names:
                idx = param_names.index(param_name)
                if idx < len(args):
                    sandbox_path = args[idx]
            if not sandbox_path or not isinstance(sandbox_path, str):
                continue

            if synced_file.host_dir is not None:
                host_dir = synced_file.host_dir
                os.makedirs(host_dir, exist_ok=True)
            else:
                if temp_dir is None:
                    temp_dir = tempfile.mkdtemp(prefix="tool-file-sync-")
                host_dir = temp_dir

            host_path = os.path.join(host_dir, os.path.basename(sandbox_path))
            await self.async_file_to(sandbox_path, host_path)
            synced_entries.append((sandbox_path, host_path, synced_file.writeback))

            if param_name in kwargs:
                kwargs[param_name] = host_path
            elif param_name in param_names:
                idx = param_names.index(param_name)
                if idx < len(args):
                    args[idx] = host_path

        return args, kwargs, synced_entries, temp_dir

    def _write_tool_response(self, response: dict[str, Any]) -> None:
        if self._uses_websocket_transport():
            if self._ws is None:
                raise SandboxFatalError("Sbx WebSocket supervisor is not connected")
            self._ws.send(json.dumps(response))
            return
        assert self._proc is not None
        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(response) + "\n")
        self._proc.stdin.flush()

    async def _awrite_tool_response(self, response: dict[str, Any]) -> None:
        ws = self._require_async_websocket()
        await ws.send(json.dumps(response))

    def _send_request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict:
        if self._uses_websocket_transport():
            return self._send_websocket_request(method, params, timeout=timeout)
        return self._send_json_rpc_request(method, params, timeout=timeout)

    def _send_websocket_request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        recovered = self._recover_dead_supervisor_after_structured_execute(method)
        if recovered is not None:
            return recovered
        self._ensure_process_for_request(method)
        ws = self._require_websocket()
        params = params or {}
        request_timeout = self._request_timeout_seconds(method, params, timeout)
        request_id = self._next_request_id()
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": request_id,
        }
        try:
            ws.send(self._serialize_supervisor_message(payload))
        except Exception as exc:
            self._handle_supervisor_send_error(
                method,
                request_id,
                BrokenPipeError(str(exc)),
            )

        deadline = time.monotonic() + request_timeout
        request_start = time.perf_counter()
        stale_discards = 0
        stdout_tail: list[str] = []
        self._on_supervisor_request_start(
            method,
            params,
            request_id=request_id,
            request_timeout=request_timeout,
        )
        while True:
            self._drain_completed_supervisor_work()
            self._raise_if_websocket_supervisor_exited(
                method,
                request_id=request_id,
                request_start=request_start,
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return self._handle_supervisor_request_timeout(
                    method,
                    params,
                    self._websocket_process_for_diagnostics(),
                    request_id=request_id,
                    request_timeout=request_timeout,
                    request_start=request_start,
                    stdout_tail="".join(stdout_tail)[-4000:],
                )
            try:
                raw = ws.recv(timeout=min(remaining, 0.05))
            except TimeoutError:
                continue
            except ConnectionClosed as exc:
                raise SandboxFatalError(
                    f"Sbx WebSocket supervisor connection closed during {method}: {exc}"
                ) from exc
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            if not raw:
                continue
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                stdout_tail.append(str(raw))
                continue
            if not isinstance(message, dict):
                stdout_tail.append(str(raw))
                continue
            if self._handle_supervisor_control_message(message, deadline=deadline):
                continue
            if message.get("id") == request_id:
                self._record_supervisor_response(
                    method,
                    params,
                    request_id=request_id,
                    request_timeout=request_timeout,
                    response=message,
                )
                self._on_supervisor_request_response(
                    method,
                    request_id=request_id,
                    request_start=request_start,
                    response=message,
                )
                return message
            self._on_supervisor_stale_response(
                method,
                expected_request_id=request_id,
                stale_response=message,
                stale_discards=stale_discards + 1,
            )
            stale_discards += 1
            if stale_discards > self._stale_response_discard_limit:
                self._handle_stale_response_limit(
                    method,
                    request_id=request_id,
                    request_start=request_start,
                )

    async def _asend_websocket_request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        recovered = await self._arecover_dead_supervisor_after_structured_execute(method)
        if recovered is not None:
            return recovered
        await self._aensure_websocket_supervisor()
        ws = self._require_async_websocket()
        params = params or {}
        request_timeout = self._request_timeout_seconds(method, params, timeout)
        request_id = self._next_request_id()
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": request_id,
        }
        try:
            await ws.send(self._serialize_supervisor_message(payload))
        except Exception as exc:
            self._handle_supervisor_send_error(
                method,
                request_id,
                BrokenPipeError(str(exc)),
            )

        request_start = time.perf_counter()
        active_request = {
            "method": method,
            "params": params,
            "request_id": request_id,
            "request_timeout": request_timeout,
            "request_start": request_start,
        }
        self._active_async_request = active_request
        self._on_supervisor_request_start(
            method,
            params,
            request_id=request_id,
            request_timeout=request_timeout,
        )
        return await self._areceive_websocket_response(
            **active_request,
            deadline=time.monotonic() + request_timeout,
        )

    async def _areceive_websocket_response(
        self,
        *,
        method: str,
        params: dict[str, Any],
        request_id: int,
        request_timeout: float,
        request_start: float,
        deadline: float,
    ) -> dict[str, Any]:
        ws = self._require_async_websocket()
        stale_discards = 0
        stdout_tail: list[str] = []
        while True:
            await self._adrain_completed_tool_calls()
            await self._araise_if_websocket_supervisor_exited(
                method,
                request_id=request_id,
                request_start=request_start,
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._active_async_request = None
                return await self._ahandle_supervisor_request_timeout(
                    method,
                    params,
                    request_id=request_id,
                    request_timeout=request_timeout,
                    request_start=request_start,
                    stdout_tail="".join(stdout_tail)[-4000:],
                )
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 0.05))
            except TimeoutError:
                continue
            except ConnectionClosed as exc:
                self._active_async_request = None
                raise SandboxFatalError(
                    f"Sbx WebSocket supervisor connection closed during {method}: {exc}"
                ) from exc
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            if not raw:
                continue
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                stdout_tail.append(str(raw))
                continue
            if not isinstance(message, dict):
                stdout_tail.append(str(raw))
                continue
            if await self._ahandle_supervisor_control_message(message):
                continue
            if message.get("id") == request_id:
                self._record_supervisor_response(
                    method,
                    params,
                    request_id=request_id,
                    request_timeout=request_timeout,
                    response=message,
                )
                self._on_supervisor_request_response(
                    method,
                    request_id=request_id,
                    request_start=request_start,
                    response=message,
                )
                self._active_async_request = None
                return message
            self._on_supervisor_stale_response(
                method,
                expected_request_id=request_id,
                stale_response=message,
                stale_discards=stale_discards + 1,
            )
            stale_discards += 1
            if stale_discards > self._stale_response_discard_limit:
                self._active_async_request = None
                self._handle_stale_response_limit(
                    method,
                    request_id=request_id,
                    request_start=request_start,
                )

    def _require_websocket(self) -> ClientConnection:
        if self._ws is None:
            raise SandboxFatalError("Sbx WebSocket supervisor is not connected")
        return self._ws

    def _require_async_websocket(self) -> AsyncClientConnection:
        if self._async_ws is None:
            raise SandboxFatalError("Sbx WebSocket supervisor is not connected")
        return self._async_ws

    def _websocket_process_for_diagnostics(self) -> SupervisorProcess:
        process = self._proc
        if process is not None:
            return process
        return _DetachedWebSocketSupervisorProcess()

    def _raise_if_websocket_supervisor_exited(
        self,
        method: str,
        *,
        request_id: int,
        request_start: float,
    ) -> None:
        if self._proc is None or self._proc.poll() is None:
            return
        self._handle_supervisor_exit_during_request(
            method,
            request_id=request_id,
            request_start=request_start,
            process=self._proc,
        )

    async def _araise_if_websocket_supervisor_exited(
        self,
        method: str,
        *,
        request_id: int,
        request_start: float,
    ) -> None:
        if not self._async_process_exited():
            return
        stderr = await self._aread_active_process_stderr()
        self._log_lifecycle(
            "sbx.runner.exited",
            method=method,
            request_id=request_id,
            duration_ms=round((time.perf_counter() - request_start) * 1000),
            stderr_chars=len(stderr or ""),
            status="error",
        )
        self._active_async_request = None
        raise SandboxFatalError(f"Sbx supervisor exited unexpectedly: {stderr}")

    def _on_supervisor_request_start(
        self,
        method: str,
        params: dict[str, Any],
        *,
        request_id: int,
        request_timeout: float,
    ) -> None:
        timeout_seconds, timeout_failure_class = self._execution_timeout_metadata_from_params(params)
        if method == "execute" and timeout_failure_class == ITERATION_TIMEOUT_FAILURE_CLASS:
            self._active_execute_timeout_deadline = time.monotonic() + timeout_seconds
        elif method == "execute":
            self._active_execute_timeout_deadline = None
        self._log_lifecycle(
            "sbx.request.start",
            method=method,
            request_id=request_id,
            code_chars=(len(str((params or {}).get("code", ""))) if method == "execute" else None),
            timeout_seconds=timeout_seconds if method == "execute" else request_timeout,
            pending_tool_count=self._pending_tool_count(),
        )

    def _on_supervisor_request_response(
        self,
        method: str,
        *,
        request_id: int,
        request_start: float,
        response: dict[str, Any],
    ) -> None:
        if method == "execute":
            self._active_execute_timeout_deadline = None
            result = response.get("result")
            if isinstance(result, dict) and "timeout" in result:
                self._quarantine_sync_tool_calls()
                self._cancel_async_tool_calls()
        self._log_lifecycle(
            "sbx.request.ok",
            method=method,
            request_id=request_id,
            status="error_response" if "error" in response else "ok",
            duration_ms=round((time.perf_counter() - request_start) * 1000),
            pending_tool_count=self._pending_tool_count(),
        )

    def _pending_tool_count(self) -> int:
        sync_futures = {
            *getattr(self, "_pending_tool_calls", ()),
            *getattr(self, "_quarantined_tool_calls", ()),
            *(worker.future for worker in self._live_sync_workers()),
        }
        return sum(not future.done() for future in sync_futures) + sum(
            not task.done()
            for task in {
                *getattr(self, "_async_pending_tool_calls", ()),
                *getattr(self, "_quarantined_async_tool_calls", ()),
            }
        )

    def has_live_host_work(self) -> bool:
        return any(
            not future.done()
            for future in {
                *getattr(self, "_pending_tool_calls", ()),
                *getattr(self, "_quarantined_tool_calls", ()),
            }
        ) or any(
            not task.done()
            for task in {
                *getattr(self, "_async_pending_tool_calls", ()),
                *getattr(self, "_quarantined_async_tool_calls", ()),
            }
        ) or self.has_live_sync_workers()

    def _quarantine_sync_tool_calls(self) -> None:
        pending = getattr(self, "_pending_tool_calls", None)
        if not pending:
            return
        quarantine = getattr(self, "_quarantined_tool_calls", None)
        if quarantine is None:
            quarantine = set()
            self._quarantined_tool_calls = quarantine
        quarantine.difference_update(future for future in quarantine if future.done())
        for future in tuple(pending):
            quarantine.add(future)
        pending.clear()

    def retire_when_host_work_finishes(self) -> bool:
        self._quarantine_sync_tool_calls()
        if any(
            not task.done()
            for task in {
                *getattr(self, "_async_pending_tool_calls", ()),
                *getattr(self, "_quarantined_async_tool_calls", ()),
            }
        ):
            raise RuntimeError(
                "Async SBX host work must be retired with "
                "await aretire_when_host_work_finishes()"
            )
        return self.retire_when_sync_workers_finish()

    async def aretire_when_host_work_finishes(self) -> bool:
        self._quarantine_sync_tool_calls()
        self._cancel_async_tool_calls()
        if not self.has_live_host_work():
            return False

        retirement = getattr(self, "_host_work_retirement", None)
        if retirement is None:
            retirement = asyncio.create_task(self._aretire_host_work())
            self._host_work_retirement = retirement
        try:
            await asyncio.shield(retirement)
        except asyncio.CancelledError:
            await retirement
            raise
        return True

    async def _aretire_host_work(self) -> None:
        while tasks := tuple(
            task
            for task in getattr(self, "_quarantined_async_tool_calls", ())
            if not task.done()
        ):
            await asyncio.gather(*tasks, return_exceptions=True)
        while workers := self._live_sync_workers():
            await asyncio.gather(
                *(worker.wait() for worker in workers),
                return_exceptions=True,
            )
        await self.ashutdown()

    def _cancel_async_tool_calls(self) -> None:
        tasks = tuple(self._async_pending_tool_calls)
        quarantine = getattr(self, "_quarantined_async_tool_calls", None)
        if quarantine is None:
            quarantine = set()
            self._quarantined_async_tool_calls = quarantine
        for task in tasks:
            task.cancel()
            quarantine.add(task)
            task.add_done_callback(quarantine.discard)
        self._async_pending_tool_calls.clear()

    async def _acancel_async_tool_calls(self, *, timeout: float = 0.1) -> None:
        self._cancel_async_tool_calls()
        tasks = tuple(
            task
            for task in getattr(self, "_quarantined_async_tool_calls", ())
            if not task.done()
        )
        if tasks:
            await asyncio.wait(tasks, timeout=timeout)

    def _handle_supervisor_send_error(
        self,
        method: str,
        request_id: int,
        exc: BrokenPipeError,
    ) -> None:
        self._log_lifecycle(
            "sbx.request.broken_pipe",
            method=method,
            request_id=request_id,
            error_type=type(exc).__name__,
            status="error",
        )
        raise SandboxFatalError("Sbx supervisor pipe broke while sending request") from exc

    def _handle_supervisor_exit_during_request(
        self,
        method: str,
        *,
        request_id: int,
        request_start: float,
        process: SupervisorProcess,
    ) -> None:
        stderr = self._read_stderr_for_process(process)
        self._log_lifecycle(
            "sbx.runner.exited",
            method=method,
            request_id=request_id,
            duration_ms=round((time.perf_counter() - request_start) * 1000),
            stderr_chars=len(stderr or ""),
            status="error",
        )
        raise SandboxFatalError(f"Sbx supervisor exited unexpectedly: {stderr}")

    def _handle_supervisor_request_timeout(
        self,
        method: str,
        params: dict[str, Any],
        process: SupervisorProcess,
        *,
        request_id: int,
        request_timeout: float,
        request_start: float,
        stdout_tail: str,
    ) -> dict[str, Any]:
        timeout_seconds, timeout_failure_class = self._execution_timeout_metadata_from_params(
            params
        )
        self._active_execute_timeout_deadline = None
        self._log_lifecycle(
            "sbx.request.timeout",
            method=method,
            request_id=request_id,
            duration_ms=round((time.perf_counter() - request_start) * 1000),
            timeout_seconds=request_timeout,
            pending_tool_count=self._pending_tool_count(),
            failure_class=timeout_failure_class,
            status="error",
        )
        if timeout_failure_class != ITERATION_TIMEOUT_FAILURE_CLASS:
            self._fail_timed_out_request(
                timeout_seconds,
                request_timeout,
                timeout_failure_class,
            )
        if not self._pending_tool_calls:
            self._fail_timed_out_request(
                timeout_seconds,
                request_timeout,
                timeout_failure_class,
            )
        self._fail_timed_out_request(
            timeout_seconds,
            request_timeout,
            timeout_failure_class,
            fatal=False,
        )
        self._quarantine_sync_tool_calls()
        stderr = self._read_stderr_for_process(process)
        self._discard_supervisor_process()
        restart_error: BaseException | None = None
        try:
            self._ensure_process_for_request(method)
        except BaseException as exc:
            restart_error = exc
        if restart_error is not None:
            raise SandboxFatalError(
                "Sbx supervisor failed to recover from iteration timeout after "
                f"{timeout_seconds:g}s; waited {request_timeout:g}s before "
                "force-killing supervisor, and restart failed: "
                f"{restart_error}"
            ) from restart_error

        diagnostic = (
            "Sbx supervisor did not return a structured timeout before the host "
            f"watchdog expired after {request_timeout:g}s. The supervisor process "
            "was killed and restarted; Python globals from the timed-out supervisor "
            "were lost, while sandbox filesystem state is preserved. Re-run setup "
            "code before relying on in-memory variables."
        )
        if stderr:
            diagnostic = f"{diagnostic}\n[supervisor stderr before restart]\n{stderr.rstrip()}"
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "timeout": {"seconds": timeout_seconds},
                "stdout": stdout_tail,
                "stderr": diagnostic,
                "state": {
                    "preserved": False,
                    "source": "supervisor_restart",
                    "scope": "filesystem_only",
                    "reason": "supervisor restart after pending host tool timeout",
                },
            },
        }

    async def _ahandle_supervisor_request_timeout(
        self,
        method: str,
        params: dict[str, Any],
        *,
        request_id: int,
        request_timeout: float,
        request_start: float,
        stdout_tail: str,
    ) -> dict[str, Any]:
        timeout_seconds, timeout_failure_class = self._execution_timeout_metadata_from_params(
            params
        )
        self._active_execute_timeout_deadline = None
        self._log_lifecycle(
            "sbx.request.timeout",
            method=method,
            request_id=request_id,
            duration_ms=round((time.perf_counter() - request_start) * 1000),
            timeout_seconds=request_timeout,
            pending_tool_count=self._pending_tool_count(),
            failure_class=timeout_failure_class,
            status="error",
        )
        if timeout_failure_class != ITERATION_TIMEOUT_FAILURE_CLASS:
            await self._afail_timed_out_request(
                timeout_seconds,
                request_timeout,
                timeout_failure_class,
            )
        if not self._async_pending_tool_calls:
            await self._afail_timed_out_request(
                timeout_seconds,
                request_timeout,
                timeout_failure_class,
            )
        stderr = await self._aread_active_process_stderr(max_wait_seconds=0)
        await self._afail_timed_out_request(
            timeout_seconds,
            request_timeout,
            timeout_failure_class,
            fatal=False,
        )
        await self._acancel_async_tool_calls()
        restart_error: BaseException | None = None
        try:
            await self._aensure_websocket_supervisor()
        except BaseException as exc:
            restart_error = exc
        if restart_error is not None:
            raise SandboxFatalError(
                "Sbx supervisor failed to recover from iteration timeout after "
                f"{timeout_seconds:g}s; waited {request_timeout:g}s before "
                "force-killing supervisor, and restart failed: "
                f"{restart_error}"
            ) from restart_error

        diagnostic = (
            "Sbx supervisor did not return a structured timeout before the host "
            f"watchdog expired after {request_timeout:g}s. The supervisor process "
            "was killed and restarted; Python globals from the timed-out supervisor "
            "were lost, while sandbox filesystem state is preserved. Re-run setup "
            "code before relying on in-memory variables."
        )
        if stderr:
            diagnostic = f"{diagnostic}\n[supervisor stderr before restart]\n{stderr.rstrip()}"
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "timeout": {"seconds": timeout_seconds},
                "stdout": stdout_tail,
                "stderr": diagnostic,
                "state": {
                    "preserved": False,
                    "source": "supervisor_restart",
                    "scope": "filesystem_only",
                    "reason": "supervisor restart after pending host tool timeout",
                },
            },
        }

    async def _afail_timed_out_request(
        self,
        timeout_seconds: float,
        host_timeout_seconds: float,
        timeout_failure_class: str,
        *,
        fatal: bool = True,
    ) -> None:
        self._log_lifecycle(
            "sbx.request.timeout",
            timeout_seconds=self.config.exec_timeout,
            status="error",
        )
        await self._akill_websocket_supervisor()
        if not fatal:
            return
        if timeout_failure_class == ITERATION_TIMEOUT_FAILURE_CLASS:
            raise SandboxFatalError(
                "Sbx supervisor failed to recover from iteration timeout after "
                f"{timeout_seconds:g}s; waited {host_timeout_seconds:g}s before "
                "force-killing supervisor"
            )
        raise SandboxFatalError(
            f"Sbx supervisor request timed out after {host_timeout_seconds:g}s"
        )

    def _handle_stale_response_limit(
        self,
        method: str,
        *,
        request_id: int,
        request_start: float,
    ) -> None:
        self._log_lifecycle(
            "sbx.protocol.stale_response_limit",
            method=method,
            request_id=request_id,
            duration_ms=round((time.perf_counter() - request_start) * 1000),
            error_type="CodeInterpreterError",
            status="error",
        )
        raise CodeInterpreterError(
            "Too many stale top-level responses while resyncing "
            f"SBX request id={request_id}"
        )

    def _ensure_process_for_method(self, method: str) -> None:
        if method == "shutdown" and self._proc is not None:
            return
        self._ensure_process()

    def _unwrap_execute_response(self, response: dict) -> Any:
        if "error" in response:
            error = response["error"]
            error_data = error.get("data", {})
            error_type = error_data.get("type", "Sandbox Error")
            partial_output = error_data.get("output") or ""
            if partial_output:
                self._log_partial_output(partial_output, error_type=error_type)
            if interpreter_result_logging_enabled(self.verbose):
                if partial_output:
                    emit_trace_result({"output": partial_output})
                emit_trace_error(
                    error_type,
                    error.get("message") or error_data.get("args", []),
                )
            if error_type == "SyntaxError":
                raise SyntaxError(error.get("message", "Invalid Python syntax"))
            raise SandboxExecutionError(
                f"{error_type}: {error_data.get('args') or error.get('message', '')}",
                partial_output=partial_output,
            )

        result = response.get("result", {})
        if isinstance(result, dict) and "timeout" in result:
            return format_recoverable_timeout_result(result)
        if interpreter_result_logging_enabled(self.verbose):
            emit_trace_result(result)
        if "final" in result:
            return FinalOutput(result["final"])
        return result.get("output")

    def _ensure_process_for_request(self, method: str) -> None:
        self._ensure_process_for_method(method)

    def _teardown_failed_websocket_supervisor(self) -> None:
        # A connect/handshake failure otherwise leaves a half-started supervisor
        # alive with _websocket_url still set, which short-circuits relaunch so
        # the next prewarm/execute reconnects to the dead endpoint. Kill it and
        # reset transport state so the next attempt rebuilds from scratch.
        if self._proc is not None and self._proc.poll() is None:
            with contextlib.suppress(Exception):
                self._proc.kill()
                self._proc.wait(timeout=5)
        self._discard_supervisor_process()
        if self._websocket_supervisor_command is None:
            # The sbx path's URL came from `sbx ports --publish`; drop it so the
            # retry republishes instead of reusing the dead forward. The local
            # runner's externally supplied URL stays put.
            self._websocket_url = None
            self._published_websocket_url = None

    async def _ateardown_failed_websocket_supervisor(self) -> None:
        if self._async_ws is not None:
            with contextlib.suppress(Exception):
                await self._async_ws.close()
            self._async_ws = None
        if self._async_proc is not None and self._async_proc.returncode is None:
            self._async_proc.kill()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._async_proc.wait(), timeout=5)
        self._async_proc = None
        self._active_async_request = None
        await self._acancel_async_tool_calls()
        if self._websocket_supervisor_command is None:
            self._websocket_url = None
            self._published_websocket_url = None

    def _discard_supervisor_process(self) -> None:
        if self._ws is not None:
            with contextlib.suppress(Exception):
                self._ws.close()
            self._ws = None
        self._proc = None
        self._stdout_lines = queue.Queue()
        self._stdout_reader = None
        self._quarantine_sync_tool_calls()

    async def _adiscard_async_supervisor_process(self) -> None:
        if self._async_ws is not None:
            with contextlib.suppress(Exception):
                await self._async_ws.close()
            self._async_ws = None
        self._async_proc = None
        self._active_async_request = None
        await self._acancel_async_tool_calls()
        if self._websocket_supervisor_command is None:
            self._websocket_url = None
            self._published_websocket_url = None

    async def _arecover_dead_supervisor_after_structured_execute(
        self,
        method: str,
    ) -> dict[str, Any] | None:
        if method != "execute" or self._supervisor_exit_recovery_context is None:
            return None
        returncode = self._active_process_returncode()
        if returncode is None:
            return None

        stderr = await self._aread_active_process_stderr()
        context = self._supervisor_exit_recovery_context
        self._supervisor_exit_recovery_context = None
        await self._adiscard_async_supervisor_process()
        restart_error: BaseException | None = None
        try:
            await self._aensure_websocket_supervisor()
        except BaseException as exc:
            restart_error = exc
        if restart_error is not None:
            raise SandboxFatalError(
                f"{self._persistent_supervisor_name} exited after the previous execute "
                "response and could not be restarted: "
                f"{restart_error}. {self._format_supervisor_exit_evidence(returncode, context)}"
            ) from restart_error
        return {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "result": {
                "output": self._format_supervisor_restart_diagnostic(
                    returncode,
                    context,
                    stderr=stderr,
                )
            },
        }

    def _async_process_running(self) -> bool:
        if self._async_proc is not None:
            return self._async_proc.returncode is None
        return self._proc is not None and self._proc.poll() is None

    def _async_process_exited(self) -> bool:
        if self._async_proc is not None:
            return self._async_proc.returncode is not None
        return self._proc is not None and self._proc.poll() is not None

    def _active_process_returncode(self) -> int | None:
        if self._async_proc is not None:
            return self._async_proc.returncode
        if self._proc is not None:
            return self._proc.poll()
        return None

    def _async_process_pid(self) -> int | None:
        if self._async_proc is not None:
            return self._async_proc.pid
        return getattr(self._proc, "pid", None)

    async def _aread_active_process_stderr(
        self,
        *,
        max_wait_seconds: float = 0.5,
    ) -> str:
        if self._async_proc is None:
            if self._proc is None:
                return ""
            return self._read_stderr_for_process(
                self._proc,
                max_wait_seconds=max_wait_seconds,
            )
        stderr = self._async_proc.stderr
        if stderr is None or max_wait_seconds <= 0:
            return ""
        try:
            data = await asyncio.wait_for(stderr.read(), timeout=max_wait_seconds)
        except TimeoutError:
            return ""
        return data.decode("utf-8", errors="replace")

    def _read_stderr_for_process(
        self,
        process: SupervisorProcess,
        *,
        max_wait_seconds: float = 0.5,
    ) -> str:
        # Best-effort diagnostic drain. A blocking ``stderr.read()`` to EOF can
        # hang forever during timeout recovery: the force-killed supervisor may
        # have a forked kernel child that ``setsid()``'d into its own session
        # (so it survives ``self._proc.kill()``) and keeps the stderr pipe's
        # write end open. Drain only what is already buffered, bounded by a
        # short deadline, so recovery never blocks on a stray descendant.
        stderr = process.stderr
        if stderr is None:
            return ""
        try:
            fd = stderr.fileno()
        except (ValueError, OSError):
            return ""
        deadline = time.monotonic() + max_wait_seconds
        chunks: list[bytes] = []
        try:
            os.set_blocking(fd, False)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                ready, _, _ = select.select([fd], [], [], remaining)
                if not ready:
                    break
                try:
                    chunk = os.read(fd, 65536)
                except (BlockingIOError, OSError):
                    break
                if not chunk:  # EOF: write end fully closed
                    break
                chunks.append(chunk)
        except Exception:
            pass
        return b"".join(chunks).decode("utf-8", errors="replace")

    def _format_supervisor_restart_diagnostic(
        self,
        returncode: int | None,
        context: dict[str, Any],
        *,
        stderr: str,
    ) -> str:
        diagnostic = (
            "Sbx supervisor exited after the previous execute response. "
            "The supervisor process was restarted; Python globals from the "
            "prior supervisor were lost, while sandbox filesystem state is "
            "preserved. Re-run setup code before relying on in-memory variables."
            "\n"
            f"[supervisor lifecycle] {self._format_supervisor_exit_evidence(returncode, context)}"
        )
        if stderr:
            diagnostic = f"{diagnostic}\n[supervisor stderr before restart]\n{stderr.rstrip()}"
        return diagnostic

    def _raise_execute_error(self, response: dict[str, Any]) -> None:
        self._unwrap_execute_response(response)
