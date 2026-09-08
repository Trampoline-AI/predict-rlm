from __future__ import annotations

import asyncio
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
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_run_timeout_repro,
        args=(runtime.spec.name, str(tmp_path / "staging"), result_queue),
    )
    process.start()
    try:
        status, *payload = _get_message(process, result_queue, 30)
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
