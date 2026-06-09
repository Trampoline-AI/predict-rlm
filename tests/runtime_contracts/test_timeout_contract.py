from __future__ import annotations

from .backends import RuntimeHandle


def test_recoverable_iteration_timeout_preserves_output_and_recovers(
    runtime: RuntimeHandle,
) -> None:
    runtime.require("recoverable_iteration_timeout")

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
