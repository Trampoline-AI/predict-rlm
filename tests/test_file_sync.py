"""SyncedFile transfer, writeback, ownership, and recovery in a real sandbox."""

import shutil
from pathlib import Path
from typing import Annotated

import pytest
from dspy.utils.dummies import DummyLM

from predict_rlm import PredictRLM
from predict_rlm.backends import JspiBackend
from predict_rlm.files import SyncedFile

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("deno") is None, reason="requires Deno"),
]


def test_binary_writeback_is_visible_to_later_calls_and_cleans_temporary_files():
    received = []

    def invert(path: Annotated[Path, SyncedFile()]) -> int:
        path = Path(path)
        received.append(path)
        data = path.read_bytes()
        path.write_bytes(bytes(byte ^ 0xFF for byte in data))
        return len(data)

    code = """
from pathlib import Path
path = Path('/tmp/nested/data.bin')
path.parent.mkdir(parents=True)
data = bytes(range(256))
path.write_bytes(data)
assert await invert(str(path)) == 256
assert path.read_bytes() == bytes(byte ^ 0xFF for byte in data)
assert await invert(path=str(path)) == 256
assert path.read_bytes() == data
SUBMIT(answer='roundtrip complete')
"""
    rlm = PredictRLM(
        "query -> answer",
        tools={"invert": invert},
        lm=DummyLM([{"reasoning": "round trip", "code": code}]),
        max_iterations=1,
    )
    prediction = rlm(query="invert twice")
    assert prediction.answer == "roundtrip complete"
    assert len(received) == 2
    assert all(not path.parent.exists() for path in received)


def test_async_tool_failure_cleans_staging_and_allows_later_writeback():
    received = []

    async def mutate(path: Annotated[Path, SyncedFile()], fail: bool) -> str:
        path = Path(path)
        received.append(path)
        assert path.read_text(encoding="utf-8") == "original"
        path.write_text("changed", encoding="utf-8")
        if fail:
            raise ValueError("intentional failure")
        return "updated"

    interpreter = JspiBackend(preinstall_packages=False, tools={"mutate": mutate})
    try:
        output = interpreter.execute("""
from pathlib import Path
path = Path('/tmp/recovery.txt')
path.write_text('original')
try:
    await mutate(str(path), fail=True)
except Exception as exc:
    assert 'intentional failure' in str(exc)
else:
    raise AssertionError('tool failure was swallowed')
assert path.read_text() == 'original'
assert await mutate(str(path), fail=False) == 'updated'
assert path.read_text() == 'changed'
print('recovered')
""")
        assert output == "recovered\n"
        assert len(received) == 2
        assert all(not path.parent.exists() for path in received)
    finally:
        interpreter.shutdown()


def test_read_only_sync_retains_custom_host_directory_without_changing_sandbox(tmp_path):
    host_dir = tmp_path / "host"

    def mutate(
        path: Annotated[Path, SyncedFile(writeback=False, host_dir=str(host_dir))],
    ) -> str:
        path = Path(path)
        assert path.parent == host_dir
        assert path.read_text(encoding="utf-8") == "original"
        path.write_text("host-only", encoding="utf-8")
        return "inspected"

    interpreter = JspiBackend(preinstall_packages=False, tools={"mutate": mutate})
    try:
        output = interpreter.execute("""
from pathlib import Path
path = Path('/tmp/readonly.txt')
path.write_text('original')
assert await mutate(str(path)) == 'inspected'
assert path.read_text() == 'original'
print('sandbox unchanged')
""")
        assert output == "sandbox unchanged\n"
        assert (host_dir / "readonly.txt").read_text(encoding="utf-8") == "host-only"
    finally:
        interpreter.shutdown()


def test_missing_synced_file_is_reported_as_an_error():
    def read_file(path: Annotated[Path, SyncedFile(writeback=False)]) -> str:
        return Path(path).read_text(encoding="utf-8")

    interpreter = JspiBackend(preinstall_packages=False, tools={"read_file": read_file})
    try:
        output = interpreter.execute("""
try:
    await read_file('/tmp/nonexistent.txt')
except Exception:
    print('missing file rejected')
else:
    raise AssertionError('missing file was accepted')
""")
        assert output == "missing file rejected\n"
    finally:
        interpreter.shutdown()
