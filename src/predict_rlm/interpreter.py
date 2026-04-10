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
import functools
import inspect
import json
import logging
import os
import re
import select
import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput
from dspy.primitives.python_interpreter import PythonInterpreter
from pydantic import BaseModel

if TYPE_CHECKING:
    from os import PathLike
    from typing import Callable


logger = logging.getLogger(__name__)


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
RUNNER_PATH = Path(__file__).parent / "sandbox" / "runner.js"


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


class JspiInterpreter(PythonInterpreter):
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
        interpreter = JspiInterpreter(
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
        # Advanced options (passed through to PythonInterpreter)
        deno_command: list[str] | None = None,
        enable_read_paths: list[PathLike | str] | None = None,
        enable_write_paths: list[PathLike | str] | None = None,
        extra_read_paths: list[PathLike | str] | None = None,
        extra_write_paths: list[PathLike | str] | None = None,
        enable_env_vars: list[str] | None = None,
        sync_files: bool = True,
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
            debug: If True, print REPL code, output, and errors to stderr
                  in real-time for debugging. Defaults to False.
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
        """
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

        # Merge extra paths into Deno permissions (but NOT into parent's
        # enable_read_paths/enable_write_paths which trigger auto-mount/sync)
        all_read_paths = list(enable_read_paths or []) + list(extra_read_paths or [])
        all_write_paths = list(enable_write_paths or []) + list(extra_write_paths or [])

        # Scan tools for SyncedFile annotations with custom host_dir paths
        # and add them to Deno permissions so the runner can write there.
        if tools:
            from predict_rlm.files import get_synced_file_params

            for tool_fn in tools.values():
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

    def shutdown(self) -> None:
        """Shut down the Deno subprocess with timeout guards.

        Overrides PythonInterpreter.shutdown() which has no timeout on
        stdin.flush() or .wait(), causing hangs when Deno is unresponsive.
        """
        import subprocess

        if self.deno_process and self.deno_process.poll() is None:
            try:
                self._write_stdin(
                    json.dumps({"jsonrpc": "2.0", "method": "shutdown"}) + "\n"
                )
                self.deno_process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                self.deno_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.deno_process.kill()
                self.deno_process.wait()
        self.deno_process = None
        self._owner_thread = None
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)

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

    def _send_request(self, method: str, params: dict, context: str) -> dict:
        """Send a JSON-RPC request without blocking the OS pipe."""
        self._request_id += 1
        request_id = self._request_id
        msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
                "id": request_id,
            }
        )
        self._write_stdin(msg + "\n")

        response_line = self._read_with_timeout(timeout=None)
        if not response_line:
            exit_code = self.deno_process.poll()
            if exit_code is not None:
                stderr = self.deno_process.stderr.read() if self.deno_process.stderr else ""
                raise CodeInterpreterError(
                    f"Deno exited (code {exit_code}) {context}: {stderr}"
                )
            raise CodeInterpreterError(f"No response {context}")

        response = json.loads(response_line)
        if response.get("id") != request_id:
            raise CodeInterpreterError(
                f"Response ID mismatch {context}: expected {request_id}, got {response.get('id')}"
            )
        if "error" in response:
            raise CodeInterpreterError(
                f"Error {context}: {response['error'].get('message', 'Unknown error')}"
            )
        return response

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

    def mount_file_at(self, host_path: str, virtual_path: str) -> None:
        """Mount a host file at a specific virtual path in the sandbox."""
        self._ensure_deno_process()
        self._send_request(
            "mount_file",
            {"host_path": host_path, "virtual_path": virtual_path},
            f"mounting {host_path} at {virtual_path}",
        )

    def mkdir_p(self, virtual_path: str) -> None:
        """Create a directory (and parents) in the sandbox MEMFS."""
        self._ensure_deno_process()
        self._send_request("mkdir_p", {"path": virtual_path}, f"creating dir {virtual_path}")

    def list_dir(self, virtual_path: str) -> list[str]:
        """Recursively list all files under a virtual directory."""
        self._ensure_deno_process()
        response = self._send_request(
            "list_dir", {"path": virtual_path}, f"listing {virtual_path}"
        )
        return response.get("result", {}).get("files", [])

    def sync_file_to(self, virtual_path: str, host_path: str) -> None:
        """Sync a single file from the sandbox MEMFS back to the host."""
        self._ensure_deno_process()
        # sync_file is a notification (no response expected), but we use
        # _send_request for the JSON-RPC request pattern with a response.
        # Use direct stdin write like the parent's _sync_files does.
        sync_msg = json.dumps({
            "jsonrpc": "2.0",
            "method": "sync_file",
            "params": {"virtual_path": virtual_path, "host_path": host_path},
        })
        self.deno_process.stdin.write(sync_msg + "\n")
        self.deno_process.stdin.flush()

    def _strip_code_fences(self, code: str) -> str:
        """Extract code from ```repl fences.

        Uses a specific ```repl tag (like the original RLM) to avoid ambiguity.
        The closing ``` must be on its own line (^```$) to handle:
        1. Code containing inline ``` (like in strings) - not on own line, won't match
        2. Double fences from model (```...```\\n```) - stops at first proper close
        Falls back to generic fence matching for backwards compatibility.

        Supports multiple ```repl blocks - all blocks are concatenated with newlines.
        """
        # Primary: look for ```repl blocks
        # Use MULTILINE so ^ matches start of line - closing ``` must be alone on a line
        # Non-greedy .*? stops at FIRST ``` on its own line
        # findall to get ALL blocks, not just the first
        matches = re.findall(r"```repl\s*\n(.*?)^```\s*$", code, re.DOTALL | re.MULTILINE)
        if matches:
            # Join all blocks with double newlines to ensure separation
            return "\n\n".join(block.rstrip() for block in matches)

        # Fallback: try generic ```python or ``` blocks for backwards compatibility
        matches = re.findall(
            r"```(?:python|py)?\s*\n(.*?)^```\s*$", code, re.DOTALL | re.MULTILINE
        )
        if matches:
            return "\n\n".join(block.rstrip() for block in matches)

        # No fences found, return as-is
        return code

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
    ) -> Any:
        """Async execute — runs on the current event loop without blocking.

        Unlike execute(), this directly awaits _execute_async so other
        coroutines in an asyncio.gather can make progress concurrently.
        """
        variables = variables or {}
        code = self._strip_code_fences(code)
        code = self._inject_variables(code, variables)

        # Limit concurrent Deno sandboxes to prevent OOM.
        # Acquired per-execute, released when done — allows other
        # interpreters to run between iterations.
        await self._get_semaphore().acquire()
        try:
            return await self._aexecute_inner(code, variables)
        finally:
            self._get_semaphore().release()

    async def _aexecute_inner(self, code: str, variables: dict[str, Any] | None) -> Any:
        """Inner async execute — runs within the sandbox semaphore."""
        self._ensure_deno_process()
        self._mount_files()
        self._register_tools()

        self._request_id += 1
        execute_request_id = self._request_id
        input_data = json.dumps({
            "jsonrpc": "2.0",
            "method": "execute",
            "params": {"code": code},
            "id": execute_request_id,
        })
        try:
            await self._write_stdin_async(input_data + "\n")
        except BrokenPipeError:
            self._tools_registered = False
            self._mounted_files = False
            self._ensure_deno_process()
            self._mount_files()
            self._register_tools()
            self._request_id += 1
            execute_request_id = self._request_id
            input_data = json.dumps({
                "jsonrpc": "2.0",
                "method": "execute",
                "params": {"code": code},
                "id": execute_request_id,
            })
            await self._write_stdin_async(input_data + "\n")

        return await self._execute_async(execute_request_id)

    def execute(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
    ) -> Any:
        """Execute code with concurrent async tool support.

        All tools are async. Multiple tool calls via asyncio.gather()
        are executed concurrently on the host side.
        """
        variables = variables or {}

        # Strip markdown code fences that models often add
        code = self._strip_code_fences(code)

        code = self._inject_variables(code, variables)
        self._ensure_deno_process()
        self._mount_files()
        self._register_tools()

        # Send the code as JSON-RPC 2.0 request
        self._request_id += 1
        execute_request_id = self._request_id
        input_data = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "execute",
                "params": {"code": code},
                "id": execute_request_id,
            }
        )
        try:
            self._write_stdin(input_data + "\n")
        except BrokenPipeError:
            # Process died - restart it
            self._tools_registered = False
            self._mounted_files = False
            self._ensure_deno_process()
            self._mount_files()
            self._register_tools()
            self._request_id += 1
            execute_request_id = self._request_id
            input_data = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "execute",
                    "params": {"code": code},
                    "id": execute_request_id,
                }
            )
            self._write_stdin(input_data + "\n")

        # Run the async execute loop. Use nest_asyncio to allow
        # run_until_complete even inside an already-running loop (e.g. marimo).
        import nest_asyncio

        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._execute_async(execute_request_id))

    async def _execute_async(self, execute_request_id: int) -> Any:
        """Read messages and handle tool calls concurrently using asyncio."""
        pending_tasks: dict[str, asyncio.Task] = {}  # request_id -> Task

        while True:
            # Check for completed tool calls and send responses
            await self._send_completed_responses(pending_tasks)

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
                err_output = self.deno_process.stderr.read()
                raise CodeInterpreterError(
                    f"No output from Deno subprocess. Stderr: {err_output}"
                )

            # Skip non-JSON lines (e.g., Pyodide package loading messages)
            if not output_line.startswith("{"):
                logger.debug(f"Skipping non-JSON output: {output_line}")
                continue

            try:
                result = json.loads(output_line)
            except json.JSONDecodeError:
                logger.info(f"Skipping malformed JSON: {output_line[:100]}")
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
                    request_id = result["id"]
                    params = result.get("params", {})
                    task = asyncio.create_task(self._execute_tool_async(params["name"], params))
                    pending_tasks[request_id] = task
                    continue

            # Before returning, ensure all pending tool calls complete
            await self._wait_and_send_all_responses(pending_tasks)

            # JSON-RPC success response
            if "result" in result:
                if result.get("id") != execute_request_id:
                    raise CodeInterpreterError(
                        f"Response ID mismatch: expected {execute_request_id}, got {result.get('id')}"
                    )
                res = result["result"]
                self._sync_files()
                if self._debug:
                    import sys

                    if "final" in res:
                        print("\n\033[32m── SUBMIT ──\033[0m", file=sys.stderr)
                        print(
                            json.dumps(res["final"], indent=2, default=str)[:2000],
                            file=sys.stderr,
                        )
                        print("\033[32m────────────\033[0m", file=sys.stderr)
                    else:
                        output = res.get("output", "")
                        print("\n\033[32m── Output ──\033[0m", file=sys.stderr)
                        print(str(output)[:2000] if output else "(no output)", file=sys.stderr)
                        print("\033[32m────────────\033[0m", file=sys.stderr)
                if "final" in res:
                    return FinalOutput(res["final"])
                return res.get("output", None)

            # JSON-RPC error response
            if "error" in result:
                if result.get("id") is not None and result.get("id") != execute_request_id:
                    raise CodeInterpreterError(
                        f"Response ID mismatch: expected {execute_request_id}, got {result.get('id')}"
                    )
                error = result["error"]
                error_data = error.get("data", {})
                error_type = error_data.get("type", "Sandbox Error")
                error_args = error_data.get("args", [])
                error_msg = error.get("message", "")

                if self._debug:
                    import sys

                    print(f"\n\033[31m── Error ({error_type}) ──\033[0m", file=sys.stderr)
                    print(error_msg or error_args, file=sys.stderr)
                    print("\033[31m─────────────────────────\033[0m", file=sys.stderr)

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
                    raise CodeInterpreterError(f"{error_type}: {error_args or error_msg}")

            raise CodeInterpreterError(f"Unexpected message format from sandbox: {result}")

    def _read_with_timeout(self, timeout: float | None) -> str | None:
        """Sync read — used by _send_request (registration, mount, etc.)."""
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
        return self._read_line_raw()

    async def _read_with_timeout_async(self, timeout: float | None) -> str | None:
        """Async read using event loop fd watching — zero threads.

        Uses ``loop.add_reader()`` to watch the raw stdout fd, then reads
        with ``os.read()`` when data arrives.  Fully non-blocking: the
        event loop stays free for other coroutines while waiting.
        """
        # Check residual buffer first (instant, no I/O)
        if "\n" in self._read_buf:
            line, self._read_buf = self._read_buf.split("\n", 1)
            return line.strip()

        if timeout is not None:
            loop = asyncio.get_running_loop()
            ready = asyncio.Event()
            loop.add_reader(self._stdout_fd, ready.set)
            try:
                await asyncio.wait_for(ready.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                return None
            finally:
                loop.remove_reader(self._stdout_fd)

        return self._read_line_raw()

    def _read_line_raw(self) -> str:
        """Read one line from the raw stdout fd, accumulating in _read_buf."""
        stdout = getattr(self.deno_process, "stdout", None)
        if self._stdout_fd < 0 or stdout is None:
            return (stdout.readline() if stdout else "").strip()

        while "\n" not in self._read_buf:
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
        msg = json.dumps({
            "jsonrpc": "2.0", "method": "sync_file",
            "params": {"virtual_path": virtual_path, "host_path": host_path},
            "id": req_id,
        })
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
        msg = json.dumps({
            "jsonrpc": "2.0", "method": "mount_file",
            "params": {"host_path": host_path, "virtual_path": virtual_path},
            "id": req_id,
        })
        await self._write_stdin_async(msg + "\n")
        result = await future
        if "error" in result:
            raise CodeInterpreterError(
                f"mount_file failed: {result['error'].get('message', result['error'])}"
            )

    async def _execute_tool_async(self, tool_name: str, call_args: dict) -> dict:
        """Execute a tool asynchronously and return the response dict."""
        from .trace import ToolCall, ms_since, record_tool_call

        if self._debug:
            import sys

            kwargs_preview = json.dumps(call_args.get("kwargs", {}), default=str)[:200]
            print(
                f"\n\033[33m── Tool: {tool_name}({kwargs_preview}) ──\033[0m", file=sys.stderr
            )

        call_start = time.perf_counter()
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

            synced_params = get_synced_file_params(tool_fn)
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

            # Check if tool is async or sync
            if asyncio.iscoroutinefunction(tool_fn):
                result = await tool_fn(*args, **kwargs)
            else:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    self._executor, functools.partial(tool_fn, *args, **kwargs)
                )

            # Mount modified files back into the sandbox (only for writeback params)
            if synced_entries:
                for sandbox_path, host_path, writeback in synced_entries:
                    if writeback and os.path.isfile(host_path):
                        await self._mount_file_during_tool(host_path, sandbox_path)
                if temp_dir:
                    shutil.rmtree(temp_dir, ignore_errors=True)

            is_json = isinstance(result, (list, dict))
            response = {
                "value": json.dumps(result) if is_json else str(result or ""),
                "type": "json" if is_json else "string",
            }

            # Record non-predict tool calls (predict records itself with richer detail)
            if tool_name != "predict":
                record_tool_call(ToolCall(
                    name=tool_name,
                    args=args,
                    kwargs={k: v for k, v in kwargs.items() if k != "pydantic_schemas"},
                    result=result,
                    duration_ms=ms_since(call_start),
                ))

            return response
        except Exception as e:
            # Clean up any SyncedFile temp dir before returning
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
            if tool_name != "predict":
                record_tool_call(ToolCall(
                    name=tool_name,
                    args=args,
                    kwargs={k: v for k, v in kwargs.items() if k != "pydantic_schemas"},
                    result=None,
                    error=str(e),
                    duration_ms=ms_since(call_start),
                ))
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
