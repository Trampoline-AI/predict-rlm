from __future__ import annotations

import asyncio
import contextlib
import contextvars
import multiprocessing
import os
import queue as queue_module
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

from .backends import RuntimeHandle


def test_host_tool_result_shapes(runtime: RuntimeHandle) -> None:
    result = runtime.execute(
        "items = await shape_tool('list')\n"
        "mapping = await shape_tool('dict')\n"
        "none_value = await shape_tool('none')\n"
        "text = await shape_tool('text')\n"
        "print(items, mapping['ok'], none_value is None, text)"
    )
    assert runtime.output(result) == "[1, 2] True True hello\n"


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync-tool", "async-tool"])
def test_host_tools_run_concurrently(runtime: RuntimeHandle, asynchronous: bool) -> None:
    runtime.require("concurrent_tools")
    barrier = threading.Barrier(2)

    def sync_tool(value):
        barrier.wait(timeout=2)
        return value * 2

    async def async_tool(value):
        # Sync SBX dispatch gives each callback its own worker event loop.
        await asyncio.to_thread(barrier.wait, timeout=2)
        return value * 2

    runtime.configure(tools={"double": async_tool if asynchronous else sync_tool})
    result = runtime.execute(
        "import asyncio\nprint(await asyncio.gather(double(3), double(7)))"
    )
    assert runtime.output(result) == "[6, 14]\n"


def test_burst_of_small_concurrent_host_tool_calls_completes(runtime: RuntimeHandle) -> None:
    # Small requests arrive together, so several can share one pipe read.
    def echo(value):
        time.sleep(0.2)
        return value

    runtime.configure(tools={"echo": echo})
    result = runtime.execute(
        "import asyncio\nprint(await asyncio.gather(*[echo(i) for i in range(6)]))"
    )
    assert runtime.output(result) == "[0, 1, 2, 3, 4, 5]\n"


_CALLER_VALUE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "caller_value", default="unset"
)


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync-tool", "async-tool"])
def test_host_tools_see_the_callers_context(runtime: RuntimeHandle, asynchronous: bool) -> None:
    def sync_probe():
        return _CALLER_VALUE.get()

    async def async_probe():
        return _CALLER_VALUE.get()

    runtime.configure(tools={"probe": async_probe if asynchronous else sync_probe})

    async def execute_with_caller_context():
        _CALLER_VALUE.set("caller")
        return await runtime.aexecute("print(await probe())")

    result = asyncio.run(execute_with_caller_context())
    assert runtime.output(result) == "caller\n"


def test_async_host_tools_run_on_the_callers_event_loop(runtime: RuntimeHandle) -> None:
    tool_loops = []

    async def record_loop():
        tool_loops.append(asyncio.get_running_loop())
        return "ok"

    runtime.configure(tools={"record_loop": record_loop})

    async def execute_and_return_loop():
        await runtime.aexecute(
            "import asyncio\nprint(await asyncio.gather(record_loop(), record_loop()))"
        )
        return asyncio.get_running_loop()

    caller_loop = asyncio.run(execute_and_return_loop())
    assert tool_loops == [caller_loop, caller_loop]


# Large pipe messages have intermittently stalled on loaded CI; retain the
# boundary regression as local rather than multiplying the same request.
@pytest.mark.local
def test_large_host_tool_request_round_trips(runtime: RuntimeHandle) -> None:
    def inspect_payload(payload):
        assert payload == "x" * 950_000
        return len(payload)

    runtime.configure(tools={"inspect_payload": inspect_payload})
    result = runtime.execute("print(await inspect_payload('x' * 950000))")
    assert runtime.output(result) == "950000\n"


def test_tool_exception_allows_later_tool_use(runtime: RuntimeHandle) -> None:
    result = runtime.execute(
        "try:\n"
        "    await failing_tool()\n"
        "except Exception as exc:\n"
        "    print('host tool failed' in str(exc))"
    )
    assert runtime.output(result) == "True\n"
    followup = runtime.execute("print((await predict('question -> answer'))['answer'])")
    assert runtime.output(followup) == "4\n"


def _slow_tool() -> str:
    time.sleep(5)
    return "slow"


def _run_timeout_repro(runtime_name: str, staging: str, result_queue) -> None:
    from .backends import runtime_specs

    spec = next(spec for spec in runtime_specs() if spec.name == runtime_name)
    runtime = None
    try:
        runtime = spec.make(Path(staging), spec)
        runtime.configure(tools={"slow_tool": _slow_tool})
        runtime.execute("pass")
        result_queue.put(("ready",))
        result = runtime.execute(
            "import asyncio\nawait asyncio.gather(slow_tool(), slow_tool())",
            timeout=0.1,
        )
        followup = runtime.execute("print('still alive')")
        result_queue.put(("ok", result.timeout_seconds, runtime.output(followup)))
    except pytest.skip.Exception as exc:
        result_queue.put(("skip", str(exc)))
    except BaseException as exc:
        result_queue.put(("error", type(exc).__name__, str(exc)))
    finally:
        if runtime is not None:
            runtime.shutdown()


def _get_message(process, result_queue, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return result_queue.get(timeout=0.05)
        except queue_module.Empty:
            if not process.is_alive():
                break
    pytest.fail("runtime stalled during concurrent host-tool timeout recovery")


def test_timeout_during_concurrent_host_tools_is_recoverable(
    runtime: RuntimeHandle,
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_run_timeout_repro,
        args=(runtime.spec.name, str(tmp_path / "staging"), result_queue),
    )
    process.start()
    try:
        # Spawn imports the Python dependencies before the backend's own startup.
        status, *payload = _get_message(process, result_queue, 60)
        if status == "skip":
            pytest.skip(payload[0])
        assert status == "ready", payload
        status, *payload = _get_message(process, result_queue, 8)
        assert status == "ok", payload
        assert payload == [0.1, "still alive\n"]
        process.join(timeout=20)
        assert not process.is_alive(), "runtime cleanup stalled after recovery"
    finally:
        if process.is_alive():
            process.kill()
        process.join(timeout=2)
        result_queue.close()
        result_queue.join_thread()
        if (
            runtime.spec.name == "sbx"
            and os.environ.get("PREDICT_RLM_RUN_SBX_TESTS") == "1"
            and shutil.which("sbx") is not None
        ):
            subprocess.run(
                ["sbx", "rm", "-f", "runtime-contract-sbx"],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )


def test_iteration_timeout_cancels_pending_async_host_tools(runtime: RuntimeHandle) -> None:
    cancelled = []

    async def wait_forever():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    runtime.configure(tools={"wait_forever": wait_forever})

    async def time_out_then_settle():
        result = await runtime.aexecute(
            "import asyncio\nawait asyncio.gather(wait_forever(), wait_forever())",
            timeout=0.2,
        )
        for _ in range(100):
            if len(cancelled) == 2:
                break
            await asyncio.sleep(0.01)
        # Snapshot before asyncio.run() cancels any task still left on the loop.
        return result, list(cancelled)

    result, cancelled_before_loop_exit = asyncio.run(time_out_then_settle())
    assert result.timeout_seconds == 0.2
    assert cancelled_before_loop_exit == [True, True]


def test_cancelling_execute_cancels_pending_async_host_tools(runtime: RuntimeHandle) -> None:
    started = []
    cancelled = []

    async def wait_forever():
        started.append(True)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    runtime.configure(tools={"wait_forever": wait_forever})

    async def cancel_while_tool_waits():
        execution = asyncio.create_task(runtime.aexecute("await wait_forever()"))
        for _ in range(500):
            if started or execution.done():
                break
            await asyncio.sleep(0.01)
        execution.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await execution
        for _ in range(100):
            if cancelled:
                break
            await asyncio.sleep(0.01)
        return list(started), list(cancelled)

    assert asyncio.run(cancel_while_tool_waits()) == ([True], [True])
