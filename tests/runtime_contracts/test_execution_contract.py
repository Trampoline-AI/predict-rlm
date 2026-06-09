from __future__ import annotations

import pytest

from .backends import RuntimeHandle


def test_execute_preserves_state_until_reset(runtime: RuntimeHandle) -> None:
    runtime.require("execute")
    runtime.require("state")
    runtime.require("reset")

    assert runtime.output(runtime.execute("counter = 40\nprint('ready')")) == "ready\n"
    assert runtime.output(runtime.execute("counter += 2\nprint(counter)")) == "42\n"

    runtime.reset()

    with pytest.raises((NameError, SyntaxError, RuntimeError, Exception)) as exc_info:
        runtime.execute("print(counter)")
    assert "counter" in str(exc_info.value)
    assert runtime.output(runtime.execute("print('after reset')")) == "after reset\n"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("```python\nprint('python')\n```", "python\n"),
        ("```repl\nprint('repl')\n```", "repl\n"),
        ("```\nprint('bare')\n```", "bare\n"),
    ],
)
def test_code_fence_normalization(runtime: RuntimeHandle, source: str, expected: str) -> None:
    runtime.require("code_fences")

    assert runtime.output(runtime.execute(source)) == expected
