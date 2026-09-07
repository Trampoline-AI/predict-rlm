"""Declarative file I/O through PredictRLM and a real sandbox."""

from __future__ import annotations

import shutil
from pathlib import Path

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from predict_rlm import File, PredictRLM

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("deno") is None, reason="requires Deno"),
]


def test_file_roundtrip_preserves_binary_data_and_hides_host_destination(tmp_path):
    class Signature(dspy.Signature):
        source: File = dspy.InputField()
        result: File = dspy.OutputField()

    source = tmp_path / "source.bin"
    contents = bytes(range(256))
    source.write_bytes(contents)
    destination = tmp_path / "destination"
    rlm = PredictRLM(
        Signature,
        lm=DummyLM(
            [
                {
                    "reasoning": "transform the mounted file without seeing a host output path",
                    "code": (
                        "from pathlib import Path\n"
                        "assert 'result' not in globals()\n"
                        "data = Path(source).read_bytes()\n"
                        "output = Path('/sandbox/output/result/transformed.bin')\n"
                        "output.write_bytes(data[::-1])\n"
                        "SUBMIT(result=str(output))"
                    ),
                }
            ]
        ),
        max_iterations=1,
    )

    prediction = rlm(source=File(path=str(source)), result=File(path=str(destination)))

    assert Path(prediction.result.path) == destination / "transformed.bin"
    assert Path(prediction.result.path).read_bytes() == contents[::-1]
    assert source.read_bytes() == contents


def test_file_list_collects_generated_outputs_without_stale_host_files(tmp_path):
    class Signature(dspy.Signature):
        source: list[File] = dspy.InputField()
        results: list[File] = dspy.OutputField()

    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "first.txt").write_text("first", encoding="utf-8")
    (source / "nested" / "second.txt").write_text("second", encoding="utf-8")
    output_dir = tmp_path / "outputs"
    (output_dir / "results").mkdir(parents=True)
    (output_dir / "results" / "stale.txt").write_text("stale", encoding="utf-8")
    rlm = PredictRLM(
        Signature,
        lm=DummyLM(
            [
                {
                    "reasoning": "retain nested paths and discover an unsubmitted output",
                    "code": (
                        "from pathlib import Path\n"
                        "inputs = {Path(path).name: Path(path) for path in source}\n"
                        "root = Path('/sandbox/output/results')\n"
                        "for name, input_name in [('first', 'first.txt'), ('second', 'second.txt')]:\n"
                        "    output = root / name / 'result.txt'\n"
                        "    output.parent.mkdir(parents=True, exist_ok=True)\n"
                        "    output.write_text(inputs[input_name].read_text().upper())\n"
                        "SUBMIT(results=[str(root / 'first/result.txt')])"
                    ),
                }
            ]
        ),
        output_dir=output_dir,
        max_iterations=1,
    )

    prediction = rlm(source=File.from_dir(str(source)))

    assert {
        Path(item.path).relative_to(output_dir / "results").as_posix(): Path(
            item.path
        ).read_text(encoding="utf-8")
        for item in prediction.results
    } == {"first/result.txt": "FIRST", "second/result.txt": "SECOND"}
