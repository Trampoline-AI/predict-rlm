from __future__ import annotations

import pytest
from dspy.primitives.code_interpreter import CodeInterpreterError

from .backends import RuntimeHandle


def test_recoverable_user_exception_allows_later_execute(runtime: RuntimeHandle) -> None:
    runtime.require("recoverable_errors")

    with pytest.raises((CodeInterpreterError, NameError)) as exc_info:
        runtime.execute("raise ValueError('ordinary failure')")

    assert "ordinary failure" in str(exc_info.value)
    assert runtime.output(runtime.execute("print('recovered')")) == "recovered\n"


def test_syntax_error_allows_later_execute(runtime: RuntimeHandle) -> None:
    runtime.require("recoverable_errors")

    with pytest.raises(SyntaxError):
        runtime.execute("for")

    assert runtime.output(runtime.execute("print('after syntax')")) == "after syntax\n"
