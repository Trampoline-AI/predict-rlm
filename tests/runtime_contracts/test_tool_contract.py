from __future__ import annotations

from .backends import RuntimeHandle


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
