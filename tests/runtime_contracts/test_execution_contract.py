from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput

from .backends import RuntimeHandle


def test_execute_preserves_state_until_reset(runtime: RuntimeHandle) -> None:
    assert runtime.output(runtime.execute("counter = 40\nprint('ready')")) == "ready\n"
    assert runtime.output(runtime.execute("counter += 2\nprint(counter)")) == "42\n"
    runtime.reset()
    assert runtime.output(runtime.execute("print('counter' in globals())")) == "False\n"


def test_concurrent_execute_requests_are_serialized(runtime: RuntimeHandle) -> None:
    runtime.execute("counter = 0")
    started = threading.Barrier(2)

    def increment():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            started.wait(timeout=5)
            return runtime.output(
                runtime.execute(
                    "import time\n"
                    "previous = counter\n"
                    "time.sleep(0.05)\n"
                    "counter = previous + 1\n"
                    "print(counter)"
                )
            )
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(increment)
        second = executor.submit(increment)
        assert sorted([first.result(timeout=20), second.result(timeout=20)]) == ["1\n", "2\n"]


@pytest.mark.parametrize("fence", ["python", "repl", ""])
def test_code_fence_normalization(runtime: RuntimeHandle, fence: str) -> None:
    assert runtime.output(runtime.execute(f"```{fence}\nprint(42)\n```")) == "42\n"


def test_user_exception_reports_error_and_allows_recovery(
    runtime: RuntimeHandle,
) -> None:
    with pytest.raises(CodeInterpreterError) as raised:
        runtime.execute("print('before failure')\nraise ValueError('ordinary failure')")
    if "partial_error_output" not in runtime.spec.unsupported:
        assert raised.value.partial_output == "before failure\n"
    assert "ordinary failure" in str(raised.value)
    assert runtime.output(runtime.execute("print('recovered')")) == "recovered\n"


def test_syntax_error_allows_later_execute(runtime: RuntimeHandle) -> None:
    with pytest.raises(SyntaxError):
        runtime.execute("for")
    assert runtime.output(runtime.execute("print('after syntax')")) == "after syntax\n"


def test_submit_returns_final_output(runtime: RuntimeHandle) -> None:
    runtime.configure(output_fields=[{"name": "answer", "annotation": "str", "type": "str"}])
    result = runtime.execute("SUBMIT(answer='done')")
    assert isinstance(result, FinalOutput)
    assert result.output == {"answer": "done"}


def test_deferred_submit_keeps_runtime_alive_until_confirmed(runtime: RuntimeHandle) -> None:
    runtime.require("deferred_submit")
    runtime.configure(output_fields=[{"name": "answer", "annotation": "str"}])
    runtime.defer_next_submit_finalization()
    deferred = runtime.execute("SUBMIT(answer='draft')")
    probe = runtime.execute("print('alive after deferred submit')")
    final = runtime.execute("SUBMIT(answer='confirmed')")
    assert isinstance(deferred, FinalOutput)
    assert deferred.output == {"answer": "draft"}
    assert runtime.output(probe) == "alive after deferred submit\n"
    assert isinstance(final, FinalOutput)
    assert final.output == {"answer": "confirmed"}


def test_file_operations_round_trip(runtime: RuntimeHandle, tmp_path: Path) -> None:
    source = tmp_path / "input.txt"
    target = tmp_path / "output.txt"
    source.write_text("hello", encoding="utf-8")
    runtime.mount_file_at(str(source), "/sandbox/input.txt")
    runtime.mkdir_p("/sandbox/out")
    runtime.execute(
        "text = open('/sandbox/input.txt').read()\n"
        "open('/sandbox/out/result.txt', 'w').write(text + ' world')"
    )
    assert "/sandbox/out/result.txt" in runtime.list_dir("/sandbox/out")
    runtime.sync_file_to("/sandbox/out/result.txt", str(target))
    assert target.read_text(encoding="utf-8") == "hello world"


def test_recoverable_timeout_preserves_output_and_recovers(runtime: RuntimeHandle) -> None:
    result = runtime.execute(
        "import sys, time\n"
        "print('before timeout')\n"
        "print('stderr before timeout', file=sys.stderr)\n"
        "sys.stdout.flush(); sys.stderr.flush()\n"
        "while True:\n"
        "    time.sleep(0.05)\n",
        timeout=0.2,
    )
    followup = runtime.execute("print('after timeout')")
    timeout = runtime.timeout_observation(result)
    assert timeout["seconds"] == 0.2
    assert timeout["stdout"] == "before timeout\n"
    assert timeout["stderr"].startswith("stderr before timeout\n")
    assert runtime.output(followup) == "after timeout\n"
