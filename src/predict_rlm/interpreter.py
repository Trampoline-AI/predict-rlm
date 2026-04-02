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
import json
import logging
import os
import re
import select
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

        args.append("--no-prompt")
        args.append(str(RUNNER_PATH))

        if env_vars:
            args.append(",".join(env_vars))

        return args

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
            self.deno_process.stdin.write(input_data + "\n")
            self.deno_process.stdin.flush()
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
            self.deno_process.stdin.write(input_data + "\n")
            self.deno_process.stdin.flush()

        # Handle messages with concurrent async tool execution
        # Check if we're already in an event loop (e.g., marimo, jupyter)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # Already in event loop - run in a separate thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self._execute_async(execute_request_id))
                return future.result()
        else:
            # No event loop - use asyncio.run directly
            return asyncio.run(self._execute_async(execute_request_id))

    async def _execute_async(self, execute_request_id: int) -> Any:
        """Read messages and handle tool calls concurrently using asyncio."""
        # Expand the default thread pool so many concurrent sync tool calls
        # (each blocking on an LLM HTTP round-trip) don't starve the read loop.
        # The default pool is min(32, cpu+4) which is too small for 100+ calls.
        import concurrent.futures

        loop = asyncio.get_running_loop()
        loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=256))

        pending_tasks: dict[str, asyncio.Task] = {}  # request_id -> Task

        while True:
            # Check for completed tool calls and send responses
            await self._send_completed_responses(pending_tasks)

            # Read next message - use thread to avoid blocking event loop
            if pending_tasks:
                # Non-blocking check with short timeout
                output_line = await asyncio.to_thread(self._read_with_timeout, 0.01)
                if output_line is None:
                    # Timeout - give other tasks a chance to run
                    await asyncio.sleep(0)
                    continue
            else:
                # No pending tasks, blocking read is ok
                output_line = await asyncio.to_thread(self._read_with_timeout, None)

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
                    raise SyntaxError(f"Invalid Python syntax. message: {error_msg}")
                else:
                    raise CodeInterpreterError(f"{error_type}: {error_args or error_msg}")

            raise CodeInterpreterError(f"Unexpected message format from sandbox: {result}")

    def _read_with_timeout(self, timeout: float | None) -> str | None:
        """Read a line from stdout with optional timeout.

        Returns None on timeout, empty string on EOF, or the line content.
        """
        if timeout is not None:
            # Use select to check if data is available
            ready, _, _ = select.select([self.deno_process.stdout], [], [], timeout)
            if not ready:
                return None

        return self.deno_process.stdout.readline().strip()

    async def _execute_tool_async(self, tool_name: str, call_args: dict) -> dict:
        """Execute a tool asynchronously and return the response dict."""
        if self._debug:
            import sys

            kwargs_preview = json.dumps(call_args.get("kwargs", {}), default=str)[:200]
            print(
                f"\n\033[33m── Tool: {tool_name}({kwargs_preview}) ──\033[0m", file=sys.stderr
            )
        try:
            if tool_name not in self.tools:
                raise CodeInterpreterError(f"Unknown tool: {tool_name}")

            tool_fn = self.tools[tool_name]
            args = call_args.get("args", [])
            kwargs = call_args.get("kwargs", {})

            # Pass pydantic_schemas through to predict tool if present
            pydantic_schemas = call_args.get("pydantic_schemas")
            if pydantic_schemas and tool_name == "predict":
                kwargs["pydantic_schemas"] = pydantic_schemas

            # Check if tool is async or sync
            if asyncio.iscoroutinefunction(tool_fn):
                result = await tool_fn(*args, **kwargs)
            else:
                # Run sync function in thread pool to not block event loop
                result = await asyncio.to_thread(tool_fn, *args, **kwargs)

            is_json = isinstance(result, (list, dict))
            return {
                "value": json.dumps(result) if is_json else str(result or ""),
                "type": "json" if is_json else "string",
            }
        except Exception as e:
            return {"error": str(e)}

    async def _send_completed_responses(self, pending_tasks: dict[str, asyncio.Task]) -> None:
        """Send JSON-RPC responses for any completed tool calls."""
        completed = [request_id for request_id, task in pending_tasks.items() if task.done()]

        for request_id in completed:
            task = pending_tasks.pop(request_id)
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
            self.deno_process.stdin.write(response + "\n")
            self.deno_process.stdin.flush()

    async def _wait_and_send_all_responses(
        self, pending_tasks: dict[str, asyncio.Task]
    ) -> None:
        """Wait for all pending tool calls and send their JSON-RPC responses."""
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
            self.deno_process.stdin.write(response + "\n")
            self.deno_process.stdin.flush()

        pending_tasks.clear()
