"""Tests for SyncedFile type annotations and host-side file sync during tool calls.

Unit tests verify type detection via get_synced_file_params. Integration tests
verify the full flow: sandbox code calls a SyncedFile-annotated tool -> framework
syncs the file from sandbox MEMFS to the host -> tool runs on host -> framework
mounts the modified file back into the sandbox.
"""

import os
from pathlib import Path
from typing import Annotated

import pytest

from predict_rlm.files import SyncedFile, get_synced_file_params
from predict_rlm.interpreter import JspiInterpreter

# ─── Unit tests ───────────────────────────────────────────────────────────────


class TestGetSyncedFileParams:
    """Unit tests for get_synced_file_params introspection."""

    def test_no_annotations_returns_empty(self):
        def my_tool(x: str) -> str:
            return x

        assert get_synced_file_params(my_tool) == {}

    def test_single_synced_param(self):
        def my_tool(file_path: Annotated[Path, SyncedFile()]) -> str:
            return "ok"

        result = get_synced_file_params(my_tool)
        assert "file_path" in result
        assert result["file_path"] == SyncedFile()

    def test_multiple_synced_params(self):
        def my_tool(
            input_path: Annotated[Path, SyncedFile(writeback=False)],
            output_path: Annotated[Path, SyncedFile()],
        ) -> str:
            return "ok"

        result = get_synced_file_params(my_tool)
        assert len(result) == 2
        assert result["input_path"] == SyncedFile(writeback=False)
        assert result["output_path"] == SyncedFile()

    def test_writeback_false_preserved(self):
        def my_tool(ref: Annotated[Path, SyncedFile(writeback=False)]) -> str:
            return "ok"

        assert get_synced_file_params(my_tool)["ref"].writeback is False

    def test_host_dir_preserved(self):
        def my_tool(
            f: Annotated[Path, SyncedFile(host_dir="/tmp/custom")],
        ) -> str:
            return "ok"

        assert get_synced_file_params(my_tool)["f"].host_dir == "/tmp/custom"

    def test_mixed_annotated_and_plain(self):
        def my_tool(
            synced: Annotated[Path, SyncedFile()],
            plain: str,
            count: int = 5,
        ) -> str:
            return "ok"

        result = get_synced_file_params(my_tool)
        assert list(result.keys()) == ["synced"]

    def test_function_without_type_hints(self):
        def my_tool(x, y):
            return x

        assert get_synced_file_params(my_tool) == {}

    def test_frozen_dataclass(self):
        sf = SyncedFile()
        with pytest.raises(AttributeError):
            sf.writeback = False  # type: ignore[misc]


class TestToolDocFormatting:
    """Test that SyncedFile-annotated tools render correctly in tool docs."""

    def test_annotated_path_renders_as_str(self):
        from predict_rlm._shared import format_tool_docs_full

        def my_tool(
            workbook: Annotated[Path, SyncedFile()],
            name: str,
        ) -> str:
            """Process a workbook."""
            return "ok"

        docs = format_tool_docs_full({"my_tool": my_tool})
        assert "workbook: str" in docs
        assert "name: str" in docs
        assert "SyncedFile" not in docs
        assert "Annotated" not in docs


# ─── Integration tests ───────────────────────────────────────────────────────


@pytest.mark.integration
class TestSyncedFileIntegration:
    """Integration tests for SyncedFile-annotated tools running through the interpreter.

    These tests verify the full flow: sandbox writes a file, calls a
    SyncedFile-annotated tool, the framework syncs the file to the host, the tool
    modifies it, and the modified file is mounted back into the sandbox.
    """

    def test_tool_receives_host_path_and_file_is_synced_back(self):
        """A SyncedFile tool gets a real host path with the sandbox file's content,
        and the modified file is mounted back into the sandbox."""
        received_paths = []

        def modify_file(
            file_path: Annotated[Path, SyncedFile()],
        ) -> str:
            received_paths.append(file_path)
            with open(file_path, "r") as f:
                content = f.read()
            assert content == "hello from sandbox"
            with open(file_path, "w") as f:
                f.write("modified by host tool")
            return "ok"

        interpreter = JspiInterpreter(
            preinstall_packages=False, tools={"modify_file": modify_file}
        )
        try:
            output = interpreter.execute("""
with open("/tmp/test_file.txt", "w") as f:
    f.write("hello from sandbox")

result = await modify_file(file_path="/tmp/test_file.txt")
print(f"tool returned: {result}")

with open("/tmp/test_file.txt", "r") as f:
    content = f.read()
print(f"content after tool: {content}")
""")
            assert "tool returned: ok" in str(output)
            assert "content after tool: modified by host tool" in str(output)
            assert len(received_paths) == 1
            assert received_paths[0] != "/tmp/test_file.txt"
            assert os.path.basename(received_paths[0]) == "test_file.txt"
        finally:
            interpreter.shutdown()

    def test_tool_with_positional_arg(self):
        """SyncedFile works when the sandbox passes the path as a positional arg."""
        def read_file(file_path: Annotated[Path, SyncedFile(writeback=False)]) -> str:
            with open(file_path, "r") as f:
                return f.read().strip()

        interpreter = JspiInterpreter(
            preinstall_packages=False, tools={"read_file": read_file}
        )
        try:
            output = interpreter.execute("""
with open("/tmp/pos_test.txt", "w") as f:
    f.write("positional arg test")

result = await read_file("/tmp/pos_test.txt")
print(f"got: {result}")
""")
            assert "got: positional arg test" in str(output)
        finally:
            interpreter.shutdown()

    def test_binary_file_roundtrip(self):
        """SyncedFile handles binary files correctly."""
        def flip_bytes(file_path: Annotated[Path, SyncedFile()]) -> str:
            with open(file_path, "rb") as f:
                data = f.read()
            with open(file_path, "wb") as f:
                f.write(bytes(b ^ 0xFF for b in data))
            return "ok"

        interpreter = JspiInterpreter(
            preinstall_packages=False, tools={"flip_bytes": flip_bytes}
        )
        try:
            output = interpreter.execute("""
data = bytes([0, 1, 2, 3, 255])
with open("/tmp/binary_test.bin", "wb") as f:
    f.write(data)

result = await flip_bytes(file_path="/tmp/binary_test.bin")

with open("/tmp/binary_test.bin", "rb") as f:
    result_data = f.read()

expected = bytes([255, 254, 253, 252, 0])
assert result_data == expected, f"Expected {list(expected)}, got {list(result_data)}"
print("binary roundtrip ok")
""")
            assert "binary roundtrip ok" in str(output)
        finally:
            interpreter.shutdown()

    def test_tool_without_synced_file_unchanged(self):
        """Tools without SyncedFile annotations are unaffected."""
        def synced_tool(
            file_path: Annotated[Path, SyncedFile()],
        ) -> str:
            return "synced"

        def plain_tool(msg: str) -> str:
            return f"echo: {msg}"

        interpreter = JspiInterpreter(
            preinstall_packages=False,
            tools={"synced_tool": synced_tool, "plain_tool": plain_tool},
        )
        try:
            output = interpreter.execute("""
result = await plain_tool(msg="hello")
print(result)
""")
            assert "echo: hello" in str(output)
        finally:
            interpreter.shutdown()

    def test_synced_file_with_nonexistent_file(self):
        """Tool call with a nonexistent sandbox file produces an error."""
        def read_it(file_path: Annotated[Path, SyncedFile(writeback=False)]) -> str:
            with open(file_path) as f:
                return f.read()

        interpreter = JspiInterpreter(
            preinstall_packages=False, tools={"read_it": read_it}
        )
        try:
            output = interpreter.execute("""
try:
    result = await read_it(file_path="/tmp/nonexistent_file.txt")
    print(f"unexpected: {result}")
except Exception as e:
    print(f"error: {e}")
""")
            assert "error:" in str(output)
        finally:
            interpreter.shutdown()

    def test_tool_can_grow_file(self):
        """SyncedFile handles a tool that makes the file larger."""
        def append_data(file_path: Annotated[Path, SyncedFile()]) -> str:
            with open(file_path, "a") as f:
                f.write("\nextra line from host")
            return "ok"

        interpreter = JspiInterpreter(
            preinstall_packages=False, tools={"append_data": append_data}
        )
        try:
            output = interpreter.execute("""
with open("/tmp/grow_test.txt", "w") as f:
    f.write("original line")

await append_data(file_path="/tmp/grow_test.txt")

with open("/tmp/grow_test.txt", "r") as f:
    lines = f.readlines()
print(f"line count: {len(lines)}")
print(f"last line: {lines[-1].strip()}")
""")
            assert "line count: 2" in str(output)
            assert "last line: extra line from host" in str(output)
        finally:
            interpreter.shutdown()

    def test_multiple_synced_file_calls(self):
        """Multiple SyncedFile tool calls in the same execution work correctly."""
        call_count = [0]

        def increment_file(file_path: Annotated[Path, SyncedFile()]) -> str:
            call_count[0] += 1
            with open(file_path, "r") as f:
                val = int(f.read().strip())
            with open(file_path, "w") as f:
                f.write(str(val + 1))
            return str(val + 1)

        interpreter = JspiInterpreter(
            preinstall_packages=False, tools={"increment_file": increment_file}
        )
        try:
            output = interpreter.execute("""
with open("/tmp/counter.txt", "w") as f:
    f.write("0")

r1 = await increment_file(file_path="/tmp/counter.txt")
r2 = await increment_file(file_path="/tmp/counter.txt")
r3 = await increment_file(file_path="/tmp/counter.txt")

with open("/tmp/counter.txt", "r") as f:
    final = f.read().strip()
print(f"results: {r1}, {r2}, {r3}")
print(f"final: {final}")
""")
            assert "results: 1, 2, 3" in str(output)
            assert "final: 3" in str(output)
            assert call_count[0] == 3
        finally:
            interpreter.shutdown()

    def test_synced_file_tool_error_still_works(self):
        """If a SyncedFile tool raises an error, subsequent calls still work."""
        def maybe_fail(
            file_path: Annotated[Path, SyncedFile()],
            should_fail: bool = False,
        ) -> str:
            if should_fail:
                raise ValueError("intentional failure")
            with open(file_path, "r") as f:
                return f.read().strip()

        interpreter = JspiInterpreter(
            preinstall_packages=False, tools={"maybe_fail": maybe_fail}
        )
        try:
            output = interpreter.execute("""
with open("/tmp/err_test.txt", "w") as f:
    f.write("test data")

try:
    await maybe_fail(file_path="/tmp/err_test.txt", should_fail=True)
except Exception as e:
    print(f"caught: {e}")

result = await maybe_fail(file_path="/tmp/err_test.txt", should_fail=False)
print(f"after error: {result}")
""")
            assert "caught:" in str(output)
            assert "after error: test data" in str(output)
        finally:
            interpreter.shutdown()

    def test_mixed_sync_and_synced_file_tools(self):
        """SyncedFile and plain tools work together in the same interpreter."""
        def transform_file(file_path: Annotated[Path, SyncedFile()]) -> str:
            with open(file_path, "r") as f:
                content = f.read()
            with open(file_path, "w") as f:
                f.write(content.upper())
            return "ok"

        def compute(x: int, y: int) -> int:
            return x + y

        interpreter = JspiInterpreter(
            preinstall_packages=False,
            tools={"transform_file": transform_file, "compute": compute},
        )
        try:
            output = interpreter.execute("""
total = await compute(x=10, y=20)
print(f"total: {total}")

with open("/tmp/mixed_test.txt", "w") as f:
    f.write("hello world")

await transform_file(file_path="/tmp/mixed_test.txt")

with open("/tmp/mixed_test.txt", "r") as f:
    result = f.read()
print(f"transformed: {result}")
""")
            assert "total: 30" in str(output)
            assert "transformed: HELLO WORLD" in str(output)
        finally:
            interpreter.shutdown()

    def test_synced_file_with_nested_directory(self):
        """SyncedFile handles files in nested sandbox directories."""
        def stamp_file(file_path: Annotated[Path, SyncedFile()]) -> str:
            with open(file_path, "a") as f:
                f.write(" [stamped]")
            return "ok"

        interpreter = JspiInterpreter(
            preinstall_packages=False, tools={"stamp_file": stamp_file}
        )
        try:
            output = interpreter.execute("""
import os
os.makedirs("/tmp/deep/nested/dir", exist_ok=True)

with open("/tmp/deep/nested/dir/data.txt", "w") as f:
    f.write("deep file")

await stamp_file(file_path="/tmp/deep/nested/dir/data.txt")

with open("/tmp/deep/nested/dir/data.txt", "r") as f:
    result = f.read()
print(f"result: {result}")
""")
            assert "result: deep file [stamped]" in str(output)
        finally:
            interpreter.shutdown()

    def test_async_synced_file_tool(self):
        """SyncedFile works with async tool functions too."""
        import asyncio

        async def async_transform(
            file_path: Annotated[Path, SyncedFile()],
        ) -> str:
            await asyncio.sleep(0.01)
            with open(file_path, "r") as f:
                content = f.read()
            with open(file_path, "w") as f:
                f.write(content[::-1])
            return "ok"

        interpreter = JspiInterpreter(
            preinstall_packages=False, tools={"async_transform": async_transform}
        )
        try:
            output = interpreter.execute("""
with open("/tmp/async_test.txt", "w") as f:
    f.write("abcdef")

await async_transform(file_path="/tmp/async_test.txt")

with open("/tmp/async_test.txt", "r") as f:
    result = f.read()
print(f"reversed: {result}")
""")
            assert "reversed: fedcba" in str(output)
        finally:
            interpreter.shutdown()

    def test_temp_dir_cleanup(self):
        """Temp directories created for file sync are cleaned up."""
        temp_dirs = []

        def capture_dir(file_path: Annotated[Path, SyncedFile()]) -> str:
            temp_dirs.append(os.path.dirname(file_path))
            return "ok"

        interpreter = JspiInterpreter(
            preinstall_packages=False, tools={"capture_dir": capture_dir}
        )
        try:
            interpreter.execute("""
with open("/tmp/cleanup_test.txt", "w") as f:
    f.write("data")
await capture_dir(file_path="/tmp/cleanup_test.txt")
""")
            assert len(temp_dirs) == 1
            assert not os.path.exists(temp_dirs[0])
        finally:
            interpreter.shutdown()

    def test_synced_file_preserves_tool_return_value(self):
        """The tool's return value is correctly passed back to the sandbox."""
        def analyze_file(
            file_path: Annotated[Path, SyncedFile(writeback=False)],
        ) -> dict:
            with open(file_path, "r") as f:
                content = f.read()
            return {"length": len(content), "lines": content.count("\n") + 1}

        interpreter = JspiInterpreter(
            preinstall_packages=False, tools={"analyze_file": analyze_file}
        )
        try:
            output = interpreter.execute("""
with open("/tmp/analyze_test.txt", "w") as f:
    f.write("line1\\nline2\\nline3")

result = await analyze_file(file_path="/tmp/analyze_test.txt")
print(f"length: {result['length']}, lines: {result['lines']}")
""")
            assert "length: 17, lines: 3" in str(output)
        finally:
            interpreter.shutdown()

    def test_writeback_false_skips_mount_after(self):
        """With writeback=False, the tool can modify the host file but the sandbox
        file remains unchanged."""
        def modify_but_readonly(
            file_path: Annotated[Path, SyncedFile(writeback=False)],
        ) -> str:
            with open(file_path, "w") as f:
                f.write("modified on host")
            return "ok"

        interpreter = JspiInterpreter(
            preinstall_packages=False,
            tools={"modify_but_readonly": modify_but_readonly},
        )
        try:
            output = interpreter.execute("""
with open("/tmp/readonly_test.txt", "w") as f:
    f.write("original content")

await modify_but_readonly(file_path="/tmp/readonly_test.txt")

with open("/tmp/readonly_test.txt", "r") as f:
    content = f.read()
print(f"content: {content}")
""")
            # Sandbox file should still have original content
            assert "content: original content" in str(output)
        finally:
            interpreter.shutdown()

    def test_host_dir_uses_specified_directory(self, tmp_path):
        """When host_dir is specified, the file is synced there instead of temp."""
        custom_dir = str(tmp_path / "custom_host")
        received_paths = []

        def check_dir(
            file_path: Annotated[Path, SyncedFile(host_dir=custom_dir)],
        ) -> str:
            received_paths.append(str(file_path))
            return "ok"

        interpreter = JspiInterpreter(
            preinstall_packages=False, tools={"check_dir": check_dir}
        )
        try:
            interpreter.execute("""
with open("/tmp/hostdir_test.txt", "w") as f:
    f.write("data")
await check_dir(file_path="/tmp/hostdir_test.txt")
""")
            assert len(received_paths) == 1
            assert received_paths[0].startswith(custom_dir)
            # Custom dir should NOT be cleaned up
            assert os.path.exists(custom_dir)
        finally:
            interpreter.shutdown()

    def test_non_string_param_skipped(self):
        """If a SyncedFile param receives a non-string value, it's skipped."""
        def flexible_tool(
            file_path: Annotated[Path, SyncedFile()] = None,
            data: str = "default",
        ) -> str:
            return f"path={file_path}, data={data}"

        interpreter = JspiInterpreter(
            preinstall_packages=False, tools={"flexible_tool": flexible_tool}
        )
        try:
            output = interpreter.execute("""
result = await flexible_tool(data="hello")
print(result)
""")
            assert "path=None, data=hello" in str(output)
        finally:
            interpreter.shutdown()
