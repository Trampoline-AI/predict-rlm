from __future__ import annotations

from pathlib import Path

from .backends import RuntimeHandle


def test_basic_file_operations_round_trip(
    runtime: RuntimeHandle,
    tmp_path: Path,
) -> None:
    runtime.require("files")

    source = tmp_path / "input.txt"
    target = tmp_path / "output.txt"
    source.write_text("hello", encoding="utf-8")

    runtime.mount_file_at(str(source), "/sandbox/input.txt")
    runtime.mkdir_p("/sandbox/out")
    result = runtime.execute(
        "text = open('/sandbox/input.txt').read()\n"
        "open('/sandbox/out/result.txt', 'w').write(text + ' world')\n"
        "print(text)"
    )

    assert runtime.output(result) == "hello\n"
    assert "/sandbox/out/result.txt" in runtime.list_dir("/sandbox/out")

    runtime.sync_file_to("/sandbox/out/result.txt", str(target))

    assert target.read_text(encoding="utf-8") == "hello world"
