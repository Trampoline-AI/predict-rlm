"""Unit tests for interpreter helpers (no Deno required)."""

import subprocess
import tempfile
from unittest.mock import patch

from predict_rlm.interpreter import RUNNER_PATH, JspiInterpreter, _needs_jspi_flag


class TestNeedsJspiFlag:
    @patch.object(subprocess, "check_output")
    def test_old_v8_needs_flag(self, mock_check):
        mock_check.return_value = "deno 2.0.0\nv8 12.9.245.12-rusty\ntypescript 5.6.2"
        assert _needs_jspi_flag() is True

    @patch.object(subprocess, "check_output")
    def test_v8_13_6_needs_flag(self, mock_check):
        mock_check.return_value = "deno 2.1.0\nv8 13.6.100.0\ntypescript 5.6.2"
        assert _needs_jspi_flag() is True

    @patch.object(subprocess, "check_output")
    def test_v8_13_7_no_flag(self, mock_check):
        mock_check.return_value = "deno 2.2.0\nv8 13.7.0.0\ntypescript 5.6.2"
        assert _needs_jspi_flag() is False

    @patch.object(subprocess, "check_output")
    def test_v8_14_0_no_flag(self, mock_check):
        mock_check.return_value = "deno 3.0.0\nv8 14.0.0.0\ntypescript 5.6.2"
        assert _needs_jspi_flag() is False

    @patch.object(subprocess, "check_output")
    def test_deno_not_found_returns_true(self, mock_check):
        mock_check.side_effect = FileNotFoundError("deno not found")
        assert _needs_jspi_flag() is True

    @patch.object(subprocess, "check_output")
    def test_unexpected_output_returns_true(self, mock_check):
        mock_check.return_value = "some garbage output"
        assert _needs_jspi_flag() is True


def _make_interpreter():
    """Create a JspiInterpreter without running __init__ (no Deno subprocess)."""
    return JspiInterpreter.__new__(JspiInterpreter)


class TestBuildDenoCommand:
    @patch("predict_rlm.interpreter._needs_jspi_flag", return_value=True)
    def test_includes_jspi_flag_when_needed(self, _):
        interp = _make_interpreter()
        with patch.object(interp, "_get_deno_dir", return_value=[]):
            cmd = interp._build_deno_command([], [], [], [])
        assert "--v8-flags=--experimental-wasm-jspi" in cmd

    @patch("predict_rlm.interpreter._needs_jspi_flag", return_value=False)
    def test_excludes_jspi_flag_when_not_needed(self, _):
        interp = _make_interpreter()
        with patch.object(interp, "_get_deno_dir", return_value=[]):
            cmd = interp._build_deno_command([], [], [], [])
        assert "--v8-flags=--experimental-wasm-jspi" not in cmd

    @patch("predict_rlm.interpreter._needs_jspi_flag", return_value=False)
    def test_runner_path_in_command(self, _):
        interp = _make_interpreter()
        with patch.object(interp, "_get_deno_dir", return_value=[]):
            cmd = interp._build_deno_command([], [], [], [])
        assert str(RUNNER_PATH) in cmd

    @patch("predict_rlm.interpreter._needs_jspi_flag", return_value=False)
    def test_allow_read_includes_runner_and_user_paths(self, _):
        interp = _make_interpreter()
        with patch.object(interp, "_get_deno_dir", return_value=[]):
            cmd = interp._build_deno_command(["/data/input"], [], [], [])
        read_arg = [a for a in cmd if a.startswith("--allow-read=")][0]
        read_paths = read_arg.split("=", 1)[1].split(",")
        assert str(RUNNER_PATH) in read_paths
        assert "/data/input" in read_paths

    @patch("predict_rlm.interpreter._needs_jspi_flag", return_value=False)
    def test_write_paths_also_in_allow_read(self, _):
        interp = _make_interpreter()
        with patch.object(interp, "_get_deno_dir", return_value=[]):
            cmd = interp._build_deno_command([], ["/data/output"], [], [])
        read_arg = [a for a in cmd if a.startswith("--allow-read=")][0]
        assert "/data/output" in read_arg

    @patch("predict_rlm.interpreter._needs_jspi_flag", return_value=False)
    def test_allow_write_includes_tempdir_and_user_paths(self, _):
        interp = _make_interpreter()
        with patch.object(interp, "_get_deno_dir", return_value=[]):
            cmd = interp._build_deno_command([], ["/data/output"], [], [])
        write_arg = [a for a in cmd if a.startswith("--allow-write=")][0]
        write_paths = write_arg.split("=", 1)[1].split(",")
        assert "/data/output" in write_paths
        assert tempfile.gettempdir() in write_paths
        assert "/tmp" in write_paths

    @patch("predict_rlm.interpreter._needs_jspi_flag", return_value=False)
    def test_allow_net_with_domains(self, _):
        interp = _make_interpreter()
        with patch.object(interp, "_get_deno_dir", return_value=[]):
            cmd = interp._build_deno_command(
                [], [], ["pypi.org", "api.example.com"], []
            )
        assert "--allow-net=pypi.org,api.example.com" in cmd

    @patch("predict_rlm.interpreter._needs_jspi_flag", return_value=False)
    def test_no_allow_net_when_empty(self, _):
        interp = _make_interpreter()
        with patch.object(interp, "_get_deno_dir", return_value=[]):
            cmd = interp._build_deno_command([], [], [], [])
        assert not any(a.startswith("--allow-net") for a in cmd)

    @patch("predict_rlm.interpreter._needs_jspi_flag", return_value=False)
    def test_always_includes_allow_env_and_no_prompt(self, _):
        interp = _make_interpreter()
        with patch.object(interp, "_get_deno_dir", return_value=[]):
            cmd = interp._build_deno_command([], [], [], [])
        assert "--allow-env" in cmd
        assert "--no-prompt" in cmd

    @patch("predict_rlm.interpreter._needs_jspi_flag", return_value=False)
    def test_env_vars_as_final_arg(self, _):
        interp = _make_interpreter()
        with patch.object(interp, "_get_deno_dir", return_value=[]):
            cmd = interp._build_deno_command(
                [], [], [], ["PYODIDE_PREINSTALL", "SKILL_PACKAGES"]
            )
        assert cmd[-1] == "PYODIDE_PREINSTALL,SKILL_PACKAGES"

    @patch("predict_rlm.interpreter._needs_jspi_flag", return_value=False)
    def test_no_env_vars_runner_is_last(self, _):
        interp = _make_interpreter()
        with patch.object(interp, "_get_deno_dir", return_value=[]):
            cmd = interp._build_deno_command([], [], [], [])
        assert cmd[-1] == str(RUNNER_PATH)


class TestGetDenoDir:
    def test_includes_home_cache_paths(self):
        interp = _make_interpreter()
        with patch.dict("os.environ", {"HOME": "/home/test"}, clear=False):
            dirs = interp._get_deno_dir()
        assert "/home/test/.cache/deno" in dirs
        assert "/home/test/Library/Caches/deno" in dirs

    def test_includes_deno_dir_env(self):
        interp = _make_interpreter()
        with patch.dict(
            "os.environ", {"DENO_DIR": "/custom/deno"}, clear=False
        ):
            dirs = interp._get_deno_dir()
        assert "/custom/deno" in dirs
