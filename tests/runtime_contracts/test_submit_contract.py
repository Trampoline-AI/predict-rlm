from __future__ import annotations

from dspy.primitives.code_interpreter import FinalOutput

from .backends import RuntimeHandle


def test_submit_returns_final_output(runtime: RuntimeHandle) -> None:
    runtime.require("submit")
    runtime.configure(output_fields=[{"name": "answer", "annotation": "str"}])

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
