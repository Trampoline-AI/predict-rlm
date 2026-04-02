"""Tests for declarative file I/O types."""

from __future__ import annotations

import os
import tempfile
from typing import Optional

import dspy
import pytest

from predict_rlm.files import (
    File,
    build_file_instructions,
    build_file_plan,
    is_file_type,
    scan_file_fields,
)

# -- Model creation tests --


class TestFile:
    def test_create_with_path(self):
        f = File(path="/tmp/test.pdf")
        assert f.path == "/tmp/test.pdf"

    def test_default_path_is_none(self):
        f = File()
        assert f.path is None

    def test_create_with_explicit_none(self):
        f = File(path=None)
        assert f.path is None


class TestFileFromDir:
    def test_from_dir_walks_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "a.txt"), "w") as f:
                f.write("a")
            subdir = os.path.join(tmpdir, "sub")
            os.makedirs(subdir)
            with open(os.path.join(subdir, "b.txt"), "w") as f:
                f.write("b")

            files = File.from_dir(tmpdir)
            assert len(files) == 2
            paths = {f.path for f in files}
            assert os.path.join(tmpdir, "a.txt") in paths
            assert os.path.join(subdir, "b.txt") in paths

    def test_from_dir_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = File.from_dir(tmpdir)
            assert files == []

    def test_from_dir_returns_file_instances(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "test.txt"), "w") as f:
                f.write("test")
            files = File.from_dir(tmpdir)
            assert all(isinstance(f, File) for f in files)


class TestDeprecatedAliases:
    def test_local_file_is_file(self):
        from predict_rlm.files import LocalFile
        assert LocalFile is File

    def test_local_dir_is_file(self):
        from predict_rlm.files import LocalDir
        assert LocalDir is File

    def test_output_file_is_file(self):
        from predict_rlm.files import OutputFile
        assert OutputFile is File

    def test_output_dir_is_file(self):
        from predict_rlm.files import OutputDir
        assert OutputDir is File


# -- Type detection tests --


class TestIsFileType:
    def test_file(self):
        assert is_file_type(File) is True

    def test_str_is_not_file(self):
        assert is_file_type(str) is False

    def test_optional_file(self):
        assert is_file_type(Optional[File]) is True

    def test_list_file(self):
        assert is_file_type(list[File]) is True


# -- scan_file_fields tests --


class TestScanFileFields:
    def test_no_file_fields(self):
        class Sig(dspy.Signature):
            query: str = dspy.InputField()
            answer: str = dspy.OutputField()

        inputs, outputs = scan_file_fields(Sig)
        assert inputs == {}
        assert outputs == {}

    def test_input_file_field(self):
        class Sig(dspy.Signature):
            source: File = dspy.InputField()
            answer: str = dspy.OutputField()

        inputs, outputs = scan_file_fields(Sig)
        assert inputs == {"source": "file"}
        assert outputs == {}

    def test_output_file_field(self):
        class Sig(dspy.Signature):
            query: str = dspy.InputField()
            result: File = dspy.OutputField()

        inputs, outputs = scan_file_fields(Sig)
        assert inputs == {}
        assert outputs == {"result": "file"}

    def test_list_file_input(self):
        class Sig(dspy.Signature):
            documents: list[File] = dspy.InputField()
            answer: str = dspy.OutputField()

        inputs, outputs = scan_file_fields(Sig)
        assert inputs == {"documents": "list_file"}

    def test_list_file_output(self):
        class Sig(dspy.Signature):
            query: str = dspy.InputField()
            results: list[File] = dspy.OutputField()

        inputs, outputs = scan_file_fields(Sig)
        assert outputs == {"results": "list_file"}

    def test_mixed_file_fields(self):
        class Sig(dspy.Signature):
            source: File = dspy.InputField()
            docs: list[File] = dspy.InputField()
            excel: File = dspy.OutputField()
            pdfs: list[File] = dspy.OutputField()

        inputs, outputs = scan_file_fields(Sig)
        assert inputs == {"source": "file", "docs": "list_file"}
        assert outputs == {"excel": "file", "pdfs": "list_file"}


# -- build_file_instructions tests --


class TestBuildFileInstructions:
    def test_input_only(self):
        result = build_file_instructions(
            input_mounts={"source": "/sandbox/input/source/report.pdf"},
            output_dirs={},
        )
        assert "source" in result
        assert "/sandbox/input/source/report.pdf" in result
        assert "Output" not in result

    def test_output_only(self):
        result = build_file_instructions(
            input_mounts={},
            output_dirs={"result": "/sandbox/output/result/"},
        )
        assert "result" in result
        assert "/sandbox/output/result/" in result
        assert "Input" not in result

    def test_both(self):
        result = build_file_instructions(
            input_mounts={"source": "/sandbox/input/source/report.pdf"},
            output_dirs={"result": "/sandbox/output/result/"},
        )
        assert "source" in result
        assert "result" in result

    def test_list_input(self):
        result = build_file_instructions(
            input_mounts={
                "docs": [
                    "/sandbox/input/docs/file1.pdf",
                    "/sandbox/input/docs/file2.pdf",
                ]
            },
            output_dirs={},
        )
        assert "docs" in result
        assert "file1.pdf" in result


# -- build_file_plan tests --


class TestBuildFilePlan:
    def test_returns_none_when_no_file_fields(self):
        result = build_file_plan({}, {}, {})
        assert result is None

    def test_input_file_plan(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"fake pdf")
            tmp_path = f.name

        try:
            plan = build_file_plan(
                input_args={"source": File(path=tmp_path)},
                input_file_fields={"source": "file"},
                output_file_fields={},
            )
            assert plan is not None
            assert len(plan["mounts"]) == 1
            host_path, virtual_path = plan["mounts"][0]
            assert host_path == tmp_path
            assert virtual_path.startswith("/sandbox/input/source/")
            assert tmp_path in plan["read_paths"]
        finally:
            os.unlink(tmp_path)

    def test_output_file_plan(self):
        plan = build_file_plan(
            input_args={},
            input_file_fields={},
            output_file_fields={"result": "file"},
        )
        assert plan is not None
        assert "/sandbox/output/result" in plan["output_dirs"]
        assert "result" in plan["output_field_map"]
        assert plan["output_field_map"]["result"]["kind"] == "file"

    def test_output_with_custom_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = build_file_plan(
                input_args={},
                input_file_fields={},
                output_file_fields={"result": "file"},
                output_dir=tmpdir,
            )
            assert plan["write_dir"] == tmpdir
            assert plan["output_field_map"]["result"]["host_dir"] == os.path.join(
                tmpdir, "result"
            )

    def test_instructions_generated(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"fake pdf")
            tmp_path = f.name

        try:
            plan = build_file_plan(
                input_args={"source": File(path=tmp_path)},
                input_file_fields={"source": "file"},
                output_file_fields={"result": "file"},
            )
            assert "## Files" in plan["instructions"]
            assert "source" in plan["instructions"]
            assert "result" in plan["instructions"]
        finally:
            os.unlink(tmp_path)

    def test_list_file_input_plan(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f1:
            f1.write(b"a")
            p1 = f1.name
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f2:
            f2.write(b"b")
            p2 = f2.name

        try:
            plan = build_file_plan(
                input_args={"docs": [File(path=p1), File(path=p2)]},
                input_file_fields={"docs": "list_file"},
                output_file_fields={},
            )
            assert plan is not None
            assert len(plan["mounts"]) == 2
            virtual_paths = [vp for _, vp in plan["mounts"]]
            assert all("/sandbox/input/docs/" in vp for vp in virtual_paths)
        finally:
            os.unlink(p1)
            os.unlink(p2)


# -- PredictRLM-level unit tests --


class TestPrepareFileIO:
    """Tests for PredictRLM._prepare_file_io."""

    def _make_rlm(self, sig, **kwargs):
        from unittest.mock import MagicMock

        from predict_rlm import PredictRLM

        return PredictRLM(sig, sub_lm=MagicMock(), max_iterations=1, **kwargs)

    def test_no_file_fields_returns_none(self):
        class Sig(dspy.Signature):
            query: str = dspy.InputField()
            answer: str = dspy.OutputField()

        rlm = self._make_rlm(Sig)
        plan, args = rlm._prepare_file_io({"query": "hello"})
        assert plan is None
        assert args == {"query": "hello"}

    def test_input_file_transformed_to_path_string(self):
        class Sig(dspy.Signature):
            source: File = dspy.InputField()
            answer: str = dspy.OutputField()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"data")
            tmp = f.name

        try:
            rlm = self._make_rlm(Sig)
            plan, args = rlm._prepare_file_io(
                {"source": File(path=tmp)}
            )
            assert plan is not None
            basename = os.path.basename(tmp)
            assert args["source"] == f"/sandbox/input/source/{basename}"
        finally:
            os.unlink(tmp)

    def test_output_fields_removed_from_args(self):
        class Sig(dspy.Signature):
            query: str = dspy.InputField()
            result: File = dspy.OutputField()

        rlm = self._make_rlm(Sig)
        plan, args = rlm._prepare_file_io(
            {"query": "hello", "result": File()}
        )
        assert "result" not in args
        assert args == {"query": "hello"}

    def test_non_file_fields_preserved(self):
        class Sig(dspy.Signature):
            source: File = dspy.InputField()
            query: str = dspy.InputField()
            result: File = dspy.OutputField()
            summary: str = dspy.OutputField()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"data")
            tmp = f.name

        try:
            rlm = self._make_rlm(Sig)
            plan, args = rlm._prepare_file_io({
                "source": File(path=tmp),
                "query": "summarize",
                "result": File(),
            })
            assert "query" in args
            assert args["query"] == "summarize"
            assert "result" not in args
            assert args["source"].startswith("/sandbox/input/")
        finally:
            os.unlink(tmp)

    def test_list_file_transformed_to_list_of_paths(self):
        class Sig(dspy.Signature):
            documents: list[File] = dspy.InputField()
            answer: str = dspy.OutputField()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f1:
            f1.write(b"a")
            p1 = f1.name
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f2:
            f2.write(b"b")
            p2 = f2.name

        try:
            rlm = self._make_rlm(Sig)
            plan, args = rlm._prepare_file_io({
                "documents": [File(path=p1), File(path=p2)]
            })
            assert plan is not None
            assert isinstance(args["documents"], list)
            assert len(args["documents"]) == 2
            assert all(p.startswith("/sandbox/input/documents/") for p in args["documents"])
            assert len(plan["mounts"]) == 2
        finally:
            os.unlink(p1)
            os.unlink(p2)


class TestBuildSignaturesWithFiles:
    """Tests for PredictRLM._build_signatures_with_files."""

    def _make_rlm(self, sig):
        from unittest.mock import MagicMock

        from predict_rlm import PredictRLM

        return PredictRLM(sig, sub_lm=MagicMock(), max_iterations=1)

    def test_output_file_type_replaced_with_str(self):
        class Sig(dspy.Signature):
            query: str = dspy.InputField()
            result: File = dspy.OutputField(desc="Generated file")

        rlm = self._make_rlm(Sig)
        action, extract = rlm._build_signatures_with_files("## Files\ntest")

        assert "result" in action.signature.instructions
        assert "SUBMIT(result)" in action.signature.instructions

    def test_file_instructions_in_action_signature(self):
        class Sig(dspy.Signature):
            query: str = dspy.InputField()
            answer: str = dspy.OutputField()

        rlm = self._make_rlm(Sig)
        file_instr = "## Files\n\n- source: /sandbox/input/source/test.pdf"
        action, _ = rlm._build_signatures_with_files(file_instr)
        assert "/sandbox/input/source/test.pdf" in action.signature.instructions

    def test_input_file_type_replaced_with_str(self):
        class Sig(dspy.Signature):
            source: File = dspy.InputField(desc="Input PDF")
            answer: str = dspy.OutputField()

        rlm = self._make_rlm(Sig)
        action, _ = rlm._build_signatures_with_files("## Files\ntest")
        assert "`source`" in action.signature.instructions


class TestSyncOutputFiles:
    """Tests for PredictRLM._sync_output_files."""

    def _make_rlm(self, sig):
        from unittest.mock import MagicMock

        from predict_rlm import PredictRLM

        return PredictRLM(sig, sub_lm=MagicMock(), max_iterations=1)

    def test_sync_with_valid_sandbox_path(self):
        from unittest.mock import MagicMock

        class Sig(dspy.Signature):
            query: str = dspy.InputField()
            result: File = dspy.OutputField()

        rlm = self._make_rlm(Sig)

        mock_repl = MagicMock()
        prediction = dspy.Prediction(
            result="/sandbox/output/result/out.xlsx", query="test"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_plan = {
                "output_field_map": {
                    "result": {
                        "virtual_dir": "/sandbox/output/result",
                        "host_dir": os.path.join(tmpdir, "result"),
                        "kind": "file",
                    }
                }
            }

            rlm._sync_output_files(
                mock_repl, prediction, {"result": "file"}, file_plan
            )

            mock_repl.sync_file_to.assert_called_once_with(
                "/sandbox/output/result/out.xlsx",
                os.path.join(tmpdir, "result", "out.xlsx"),
            )
            assert isinstance(prediction.result, File)
            assert prediction.result.path == os.path.join(
                tmpdir, "result", "out.xlsx"
            )

    def test_sync_fallback_when_no_valid_path(self):
        from unittest.mock import MagicMock

        class Sig(dspy.Signature):
            query: str = dspy.InputField()
            result: File = dspy.OutputField()

        rlm = self._make_rlm(Sig)

        mock_repl = MagicMock()
        mock_repl.list_dir.return_value = ["/sandbox/output/result/file.csv"]

        prediction = dspy.Prediction(
            result="some random text", query="test"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_plan = {
                "output_field_map": {
                    "result": {
                        "virtual_dir": "/sandbox/output/result",
                        "host_dir": os.path.join(tmpdir, "result"),
                        "kind": "file",
                    }
                }
            }

            rlm._sync_output_files(
                mock_repl, prediction, {"result": "file"}, file_plan
            )

            mock_repl.list_dir.assert_called_once_with("/sandbox/output/result")
            mock_repl.sync_file_to.assert_called_once()
            assert isinstance(prediction.result, File)

    def test_sync_list_file_output(self):
        from unittest.mock import MagicMock

        class Sig(dspy.Signature):
            query: str = dspy.InputField()
            outfiles: list[File] = dspy.OutputField()

        rlm = self._make_rlm(Sig)

        mock_repl = MagicMock()
        mock_repl.list_dir.return_value = [
            "/sandbox/output/outfiles/a.txt",
            "/sandbox/output/outfiles/sub/b.txt",
        ]

        prediction = dspy.Prediction(
            outfiles="/sandbox/output/outfiles", query="test"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_plan = {
                "output_field_map": {
                    "outfiles": {
                        "virtual_dir": "/sandbox/output/outfiles",
                        "host_dir": os.path.join(tmpdir, "outfiles"),
                        "kind": "list_file",
                    }
                }
            }

            rlm._sync_output_files(
                mock_repl, prediction, {"outfiles": "list_file"}, file_plan
            )

            assert mock_repl.sync_file_to.call_count == 2
            assert isinstance(prediction.outfiles, list)
            assert len(prediction.outfiles) == 2
            assert all(isinstance(f, File) for f in prediction.outfiles)

    def test_sync_no_files_written(self):
        from unittest.mock import MagicMock

        class Sig(dspy.Signature):
            query: str = dspy.InputField()
            result: File = dspy.OutputField()

        rlm = self._make_rlm(Sig)

        mock_repl = MagicMock()
        mock_repl.list_dir.return_value = []

        prediction = dspy.Prediction(result="", query="test")

        file_plan = {
            "output_field_map": {
                "result": {
                    "virtual_dir": "/sandbox/output/result",
                    "host_dir": "/tmp/test-result",
                    "kind": "file",
                }
            }
        }

        rlm._sync_output_files(
            mock_repl, prediction, {"result": "file"}, file_plan
        )

        mock_repl.sync_file_to.assert_not_called()


class TestOutputFieldsInfo:
    """Tests for _get_output_fields_info with File-typed fields."""

    def test_file_output_field_gets_str_type(self):
        """File output fields should appear as 'str' in SUBMIT signature."""
        from unittest.mock import MagicMock

        from predict_rlm import PredictRLM

        class Sig(dspy.Signature):
            query: str = dspy.InputField()
            workbook: File = dspy.OutputField(desc="output file")
            result: str = dspy.OutputField(desc="summary")

        rlm = PredictRLM(Sig, sub_lm=MagicMock(), max_iterations=1)
        info = rlm._get_output_fields_info()

        assert info == [
            {"name": "workbook", "type": "str"},
            {"name": "result", "type": "str"},
        ]

    def test_list_file_output_field_gets_list_type(self):
        """list[File] output fields should appear as 'list' in SUBMIT signature."""
        from unittest.mock import MagicMock

        from predict_rlm import PredictRLM

        class Sig(dspy.Signature):
            docs: list[File] = dspy.InputField()
            redacted: list[File] = dspy.OutputField(desc="redacted files")
            result: dict = dspy.OutputField(desc="summary")

        rlm = PredictRLM(Sig, sub_lm=MagicMock(), max_iterations=1)
        info = rlm._get_output_fields_info()

        assert info == [
            {"name": "redacted", "type": "list"},
            {"name": "result", "type": "dict"},
        ]


class TestProcessFinalOutput:
    """Tests for _process_final_output coercing strings to File for file fields."""

    def test_string_path_coerced_to_file(self):
        """A plain string for a File output field should be wrapped as {"path": str}."""
        from unittest.mock import MagicMock

        from dspy.primitives.code_interpreter import FinalOutput

        from predict_rlm import PredictRLM

        class Sig(dspy.Signature):
            query: str = dspy.InputField()
            workbook: File = dspy.OutputField(desc="output file")
            result: str = dspy.OutputField(desc="summary")

        rlm = PredictRLM(Sig, sub_lm=MagicMock(), max_iterations=1)
        final = FinalOutput({
            "workbook": "/sandbox/output/workbook/result.xlsx",
            "result": "done",
        })
        parsed, error = rlm._process_final_output(final, ["workbook", "result"])
        assert error is None, f"Unexpected error: {error}"
        assert isinstance(parsed["workbook"], File)
        assert parsed["workbook"].path == "/sandbox/output/workbook/result.xlsx"
        assert parsed["result"] == "done"

    def test_list_string_paths_coerced_to_files(self):
        """A list of strings for a list[File] field should each be wrapped."""
        from unittest.mock import MagicMock

        from dspy.primitives.code_interpreter import FinalOutput

        from predict_rlm import PredictRLM

        class Sig(dspy.Signature):
            docs: list[File] = dspy.InputField()
            redacted: list[File] = dspy.OutputField(desc="redacted files")

        rlm = PredictRLM(Sig, sub_lm=MagicMock(), max_iterations=1)
        final = FinalOutput({
            "redacted": ["/sandbox/output/redacted/a.pdf", "/sandbox/output/redacted/b.pdf"],
        })
        parsed, error = rlm._process_final_output(final, ["redacted"])
        assert error is None, f"Unexpected error: {error}"
        assert len(parsed["redacted"]) == 2
        assert all(isinstance(f, File) for f in parsed["redacted"])
        assert parsed["redacted"][0].path == "/sandbox/output/redacted/a.pdf"

    def test_dict_path_still_works(self):
        """A dict {"path": ...} for a File field should still work (no double-wrapping)."""
        from unittest.mock import MagicMock

        from dspy.primitives.code_interpreter import FinalOutput

        from predict_rlm import PredictRLM

        class Sig(dspy.Signature):
            query: str = dspy.InputField()
            workbook: File = dspy.OutputField(desc="output file")

        rlm = PredictRLM(Sig, sub_lm=MagicMock(), max_iterations=1)
        final = FinalOutput({
            "workbook": {"path": "/sandbox/output/workbook/result.xlsx"},
        })
        parsed, error = rlm._process_final_output(final, ["workbook"])
        assert error is None, f"Unexpected error: {error}"
        assert isinstance(parsed["workbook"], File)
        assert parsed["workbook"].path == "/sandbox/output/workbook/result.xlsx"


class TestOutputDirParameter:
    """Tests for output_dir parameter on PredictRLM."""

    def test_output_dir_none_by_default(self):
        from unittest.mock import MagicMock

        from predict_rlm import PredictRLM

        rlm = PredictRLM("query -> answer", sub_lm=MagicMock(), max_iterations=1)
        assert rlm._output_dir is None

    def test_output_dir_stored_as_string(self):
        from pathlib import Path
        from unittest.mock import MagicMock

        from predict_rlm import PredictRLM

        rlm = PredictRLM(
            "query -> answer",
            sub_lm=MagicMock(),
            max_iterations=1,
            output_dir=Path("/tmp/test"),
        )
        assert rlm._output_dir == "/tmp/test"
        assert isinstance(rlm._output_dir, str)

    def test_output_dir_flows_to_file_plan(self):
        class Sig(dspy.Signature):
            query: str = dspy.InputField()
            result: File = dspy.OutputField()

        from unittest.mock import MagicMock

        from predict_rlm import PredictRLM

        with tempfile.TemporaryDirectory() as tmpdir:
            rlm = PredictRLM(
                Sig,
                sub_lm=MagicMock(),
                max_iterations=1,
                output_dir=tmpdir,
            )
            plan, _ = rlm._prepare_file_io({"query": "test"})
            assert plan is not None
            assert plan["write_dir"] == tmpdir
            assert plan["output_field_map"]["result"]["host_dir"] == os.path.join(
                tmpdir, "result"
            )


# -- Integration tests (require Deno + WASM sandbox) --


@pytest.mark.integration
class TestFileIOIntegration:
    """Full round-trip tests: mount file → sandbox reads it → sandbox writes output → sync back."""

    def test_mount_and_read_file(self):
        """Mount a host file and read its contents inside the sandbox."""
        from predict_rlm.interpreter import JspiInterpreter

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("hello from host")
            tmp_path = f.name

        try:
            interpreter = JspiInterpreter(
                preinstall_packages=False,
                extra_read_paths=[tmp_path],
            )
            try:
                basename = os.path.basename(tmp_path)
                interpreter._ensure_deno_process()
                interpreter.mount_file_at(tmp_path, f"/sandbox/input/source/{basename}")

                result = interpreter.execute(f"""
content = open("/sandbox/input/source/{basename}").read()
print(content)
""")
                assert "hello from host" in str(result)
            finally:
                interpreter.shutdown()
        finally:
            os.unlink(tmp_path)

    def test_write_and_sync_output_file(self):
        """RLM writes a file in sandbox, sync it back to host."""
        from predict_rlm.interpreter import JspiInterpreter

        with tempfile.TemporaryDirectory() as tmpdir:
            interpreter = JspiInterpreter(
                preinstall_packages=False,
                extra_write_paths=[tmpdir],
            )
            try:
                interpreter._ensure_deno_process()
                interpreter.mkdir_p("/sandbox/output/result")

                interpreter.execute("""
with open("/sandbox/output/result/output.txt", "w") as f:
    f.write("generated by sandbox")
print("done")
""")

                # List files in output dir
                files = interpreter.list_dir("/sandbox/output/result")
                assert len(files) == 1
                assert files[0] == "/sandbox/output/result/output.txt"

                # Sync back to host
                host_path = os.path.join(tmpdir, "output.txt")
                interpreter.sync_file_to("/sandbox/output/result/output.txt", host_path)

                # Give sync_file a moment (it's a notification, not request-response)
                import time
                time.sleep(0.1)

                assert os.path.exists(host_path)
                with open(host_path) as f:
                    assert f.read() == "generated by sandbox"
            finally:
                interpreter.shutdown()

    def test_full_roundtrip_mount_read_write_sync(self):
        """Full round-trip: mount input → read in sandbox → transform → write output → sync."""
        from predict_rlm.interpreter import JspiInterpreter

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("name,age\nAlice,30\nBob,25\n")
            input_path = f.name

        with tempfile.TemporaryDirectory() as output_dir:
            try:
                interpreter = JspiInterpreter(
                    preinstall_packages=False,
                    extra_read_paths=[input_path],
                    extra_write_paths=[output_dir],
                )
                try:
                    basename = os.path.basename(input_path)
                    interpreter._ensure_deno_process()
                    interpreter.mount_file_at(
                        input_path, f"/sandbox/input/data/{basename}"
                    )
                    interpreter.mkdir_p("/sandbox/output/result")

                    # Read CSV, transform, write output
                    interpreter.execute(f"""
import csv
import json

with open("/sandbox/input/data/{basename}") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Transform: uppercase names
for row in rows:
    row["name"] = row["name"].upper()

with open("/sandbox/output/result/transformed.json", "w") as f:
    json.dump(rows, f)

print(f"Wrote {{len(rows)}} rows")
""")

                    # List and sync
                    files = interpreter.list_dir("/sandbox/output/result")
                    assert len(files) == 1

                    host_output = os.path.join(output_dir, "transformed.json")
                    interpreter.sync_file_to(files[0], host_output)

                    import time
                    time.sleep(0.1)

                    # Verify
                    import json

                    with open(host_output) as f:
                        data = json.load(f)
                    assert len(data) == 2
                    assert data[0]["name"] == "ALICE"
                    assert data[1]["name"] == "BOB"
                finally:
                    interpreter.shutdown()
            finally:
                os.unlink(input_path)

    def test_mkdir_p_creates_nested_dirs(self):
        """mkdir_p creates deeply nested directories in MEMFS."""
        from predict_rlm.interpreter import JspiInterpreter

        interpreter = JspiInterpreter(preinstall_packages=False)
        try:
            interpreter._ensure_deno_process()
            interpreter.mkdir_p("/sandbox/output/deep/nested/dir")

            result = interpreter.execute("""
import os
exists = os.path.isdir("/sandbox/output/deep/nested/dir")
print(exists)
""")
            assert "True" in str(result)
        finally:
            interpreter.shutdown()

    def test_list_dir_empty(self):
        """list_dir on empty directory returns empty list."""
        from predict_rlm.interpreter import JspiInterpreter

        interpreter = JspiInterpreter(preinstall_packages=False)
        try:
            interpreter._ensure_deno_process()
            interpreter.mkdir_p("/sandbox/output/empty")
            files = interpreter.list_dir("/sandbox/output/empty")
            assert files == []
        finally:
            interpreter.shutdown()

    def test_list_dir_nonexistent(self):
        """list_dir on nonexistent directory returns empty list."""
        from predict_rlm.interpreter import JspiInterpreter

        interpreter = JspiInterpreter(preinstall_packages=False)
        try:
            interpreter._ensure_deno_process()
            files = interpreter.list_dir("/sandbox/output/doesnotexist")
            assert files == []
        finally:
            interpreter.shutdown()

    def test_mount_binary_file_roundtrip(self):
        """Binary files survive the mount → read → write → sync round-trip."""
        from predict_rlm.interpreter import JspiInterpreter

        # Create a binary file with known bytes
        binary_content = bytes(range(256))
        with tempfile.NamedTemporaryFile(
            suffix=".bin", delete=False
        ) as f:
            f.write(binary_content)
            input_path = f.name

        with tempfile.TemporaryDirectory() as output_dir:
            try:
                interpreter = JspiInterpreter(
                    preinstall_packages=False,
                    extra_read_paths=[input_path],
                    extra_write_paths=[output_dir],
                )
                try:
                    basename = os.path.basename(input_path)
                    interpreter._ensure_deno_process()
                    interpreter.mount_file_at(
                        input_path, f"/sandbox/input/bin/{basename}"
                    )
                    interpreter.mkdir_p("/sandbox/output/bin")

                    # Copy binary file inside sandbox
                    interpreter.execute(f"""
data = open("/sandbox/input/bin/{basename}", "rb").read()
print(f"Read {{len(data)}} bytes")
with open("/sandbox/output/bin/copy.bin", "wb") as f:
    f.write(data)
print("Written")
""")

                    host_output = os.path.join(output_dir, "copy.bin")
                    interpreter.sync_file_to(
                        "/sandbox/output/bin/copy.bin", host_output
                    )

                    import time
                    time.sleep(0.1)

                    with open(host_output, "rb") as f:
                        output_content = f.read()
                    assert output_content == binary_content
                finally:
                    interpreter.shutdown()
            finally:
                os.unlink(input_path)
