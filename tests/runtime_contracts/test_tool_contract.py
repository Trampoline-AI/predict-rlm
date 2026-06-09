from __future__ import annotations

import multiprocessing
import time
from pathlib import Path

import pytest

from .backends import RuntimeHandle


def _slow_tool() -> str:
    time.sleep(5)
    return "slow"


def _run_concurrent_host_tool_timeout_repro(
    runtime_name: str,
    staging: str,
    queue: multiprocessing.Queue,
) -> None:
    from .backends import runtime_specs

    spec = next(spec for spec in runtime_specs() if spec.name == runtime_name)
    runtime = None
    try:
        runtime = spec.make(Path(staging), spec)
        runtime.require("host_tools")
        runtime.require("recoverable_iteration_timeout")
        runtime.configure(tools={"slow_tool": _slow_tool})
        timeout_result = runtime.execute(
            "import asyncio\n"
            "await asyncio.gather(slow_tool(), slow_tool())\n",
            timeout=0.1,
        )
        output = runtime.execute("print('still alive')")
        queue.put(("ok", str(timeout_result), runtime.output(output)))
    except pytest.skip.Exception as exc:
        queue.put(("skip", str(exc)))
    except BaseException as exc:
        queue.put(("error", type(exc).__name__, str(exc)))
    finally:
        if runtime is not None:
            runtime.shutdown()


def test_host_tool_round_trip(runtime: RuntimeHandle) -> None:
    runtime.require("host_tools")

    result = runtime.execute(
        "result = await predict('question -> answer', question='2+2?')\n"
        "print(result['answer'])"
    )

    assert runtime.output(result) == "4\n"


def test_basic_host_tool_result_shapes(runtime: RuntimeHandle) -> None:
    runtime.require("host_tools")

    result = runtime.execute(
        "items = await shape_tool('list')\n"
        "mapping = await shape_tool('dict')\n"
        "none_value = await shape_tool('none')\n"
        "text = await shape_tool('text')\n"
        "print(items)\n"
        "print(mapping['ok'])\n"
        "print(none_value is None)\n"
        "print(text)"
    )

    assert runtime.output(result) == "[1, 2]\nTrue\nTrue\nhello\n"


def test_recoverable_tool_exception_allows_later_tool_use(runtime: RuntimeHandle) -> None:
    runtime.require("host_tools")
    runtime.require("recoverable_errors")

    result = runtime.execute(
        "try:\n"
        "    await failing_tool()\n"
        "except Exception as exc:\n"
        "    print(type(exc).__name__)\n"
        "print((await predict('question -> answer', question='2+2?'))['answer'])"
    )

    assert runtime.output(result).endswith("4\n")


def test_timeout_during_concurrent_host_tool_calls_is_recoverable(
    runtime: RuntimeHandle,
    tmp_path: Path,
) -> None:
    runtime.require("host_tools")
    runtime.require("recoverable_iteration_timeout")
    queue: multiprocessing.Queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_run_concurrent_host_tool_timeout_repro,
        args=(runtime.spec.name, str(tmp_path / "staging"), queue),
    )
    process.start()
    process.join(timeout=6)
    if process.is_alive():
        process.kill()
        process.join(timeout=2)
        pytest.fail(
            "SBX supervisor hung after iteration timeout while awaiting "
            "concurrent host tool calls"
        )

    assert not queue.empty()
    status, *payload = queue.get_nowait()
    if status == "skip":
        pytest.skip(payload[0])
    assert status == "ok", payload
    timeout_result, output = payload
    assert "[Timeout] Iteration execution timed out after 0.1s" in timeout_result
    assert output.strip() == "still alive"
