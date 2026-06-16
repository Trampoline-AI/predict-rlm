"""Tests for mutable Workspace inputs and sync-back behavior."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import dspy
import pytest
from dspy.primitives.code_interpreter import CodeInterpreterError

from predict_rlm import File, PredictRLM, Workspace, WorkspaceMode

if TYPE_CHECKING:
    from predict_rlm.backends.sbx import SbxBackend
from predict_rlm.files import (
    build_file_instructions,
    build_file_plan,
    is_workspace_type,
    scan_workspace_fields,
)
from predict_rlm.workspace import (
    DirectWorkspaceMount,
    WorkspaceFileInfo,
    WorkspaceSyncConflictError,
    WorkspaceSyncState,
)

RUNNER_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "predict_rlm"
    / "backends"
    / "supervisor"
    / "_payload.py"
)


class TestWorkspace:
    def test_exported_from_package(self):
        from predict_rlm import Workspace as ExportedWorkspace

        assert ExportedWorkspace is Workspace

    def test_create_with_defaults(self):
        workspace = Workspace(path="/tmp/repo")
        assert workspace.path == "/tmp/repo"
        assert workspace.mount_path == "/sandbox/workspace"
        assert workspace.mode is WorkspaceMode.MIRROR
        assert workspace.sync_back is True
        assert ".git" in workspace.exclude
        assert "node_modules" in workspace.exclude
        assert workspace.max_file_bytes == 5_000_000

    def test_create_direct_mode(self):
        workspace = Workspace(path="/tmp/repo", mount_path="/workspace", mode="direct")
        assert workspace.mode is WorkspaceMode.DIRECT
        assert workspace.mount_path == "/workspace"


class TestIsWorkspaceType:
    def test_workspace(self):
        assert is_workspace_type(Workspace) is True

    def test_list_workspace(self):
        assert is_workspace_type(list[Workspace]) is True

    def test_pep604_optional_workspace(self):
        assert is_workspace_type(Workspace | None) is True

    def test_file_is_not_workspace(self):
        assert is_workspace_type(File) is False


class TestScanWorkspaceFields:
    def test_input_workspace_field(self):
        class Sig(dspy.Signature):
            workspace: Workspace = dspy.InputField()
            answer: str = dspy.OutputField()

        assert scan_workspace_fields(Sig) == {"workspace": "workspace"}

    def test_pep604_optional_list_workspace_field(self):
        class Sig(dspy.Signature):
            workspaces: list[Workspace] | None = dspy.InputField()
            answer: str = dspy.OutputField()

        assert scan_workspace_fields(Sig) == {"workspaces": "list_workspace"}

    def test_output_workspace_rejected(self):
        class Sig(dspy.Signature):
            query: str = dspy.InputField()
            workspace: Workspace = dspy.OutputField()

        with pytest.raises(TypeError, match=r"Workspace fields are input-only.*File/list\[File\]"):
            scan_workspace_fields(Sig)


class TestWorkspaceInstructions:
    def test_workspace_instructions(self):
        result = build_file_instructions(
            input_mounts={},
            output_dirs={},
            workspace_mounts={"workspace": "/sandbox/workspace"},
        )
        assert "Workspace directories" in result
        assert "/sandbox/workspace" in result
        assert "Mirror-mode workspace changes sync back" in result
        assert "Direct SBX workspaces update host files immediately" in result


class TestWorkspaceFilePlan:
    def test_workspace_plan_excludes_safe_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, ".git"))
            os.makedirs(os.path.join(tmpdir, ".venv"))
            os.makedirs(os.path.join(tmpdir, "node_modules"))
            with open(os.path.join(tmpdir, "keep.txt"), "w") as f:
                f.write("keep")
            with open(os.path.join(tmpdir, ".git", "config"), "w") as f:
                f.write("git")
            with open(os.path.join(tmpdir, ".venv", "pyvenv.cfg"), "w") as f:
                f.write("venv")
            with open(os.path.join(tmpdir, "node_modules", "pkg.js"), "w") as f:
                f.write("pkg")

            workspace = Workspace(path=tmpdir)
            plan = build_file_plan(
                input_args={"workspace": workspace},
                input_file_fields={},
                output_file_fields={},
                input_workspace_fields={"workspace": "workspace"},
            )

            assert plan is not None
            assert os.path.abspath(tmpdir) in plan["write_paths"]
            state = plan["workspace_states"][0]
            mounts = state.iter_mounts()
            virtual_paths = [virtual for _, virtual in mounts]
            assert virtual_paths == ["/sandbox/workspace/keep.txt"]

    def test_direct_workspace_plan_does_not_create_sync_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Workspace(
                path=tmpdir,
                mount_path="/workspace",
                mode=WorkspaceMode.DIRECT,
            )

            plan = build_file_plan(
                input_args={"workspace": workspace},
                input_file_fields={},
                output_file_fields={},
                input_workspace_fields={"workspace": "workspace"},
            )

            assert plan is not None
            assert plan["workspace_states"] == []
            assert len(plan["direct_workspace_mounts"]) == 1
            assert plan["direct_workspace_mounts"][0].host_path == os.path.abspath(tmpdir)
            assert plan["direct_workspace_mounts"][0].sandbox_path == "/workspace"
            assert plan["workspace_mounts_for_instructions"] == {
                "workspace": "/workspace"
            }

    def test_direct_workspace_default_mount_path_uses_host_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Workspace(path=tmpdir, mode=WorkspaceMode.DIRECT)

            plan = build_file_plan(
                input_args={"workspace": workspace},
                input_file_fields={},
                output_file_fields={},
                input_workspace_fields={"workspace": "workspace"},
            )

            assert plan is not None
            assert plan["direct_workspace_mounts"][0].sandbox_path == os.path.abspath(
                tmpdir
            )
            assert plan["workspace_mounts_for_instructions"] == {
                "workspace": os.path.abspath(tmpdir)
            }

    def test_direct_workspace_rejects_sandbox_mount_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="must not be under /sandbox"):
                build_file_plan(
                    input_args={
                        "workspace": Workspace(
                            path=tmpdir,
                            mount_path="/sandbox/workspace",
                            mode=WorkspaceMode.DIRECT,
                        )
                    },
                    input_file_fields={},
                    output_file_fields={},
                    input_workspace_fields={"workspace": "workspace"},
                )

    def test_duplicate_workspace_mount_paths_rejected_for_list(self):
        with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
            workspaces = [
                Workspace(path=one, mount_path="/sandbox/workspace"),
                Workspace(path=two, mount_path="/sandbox/workspace"),
            ]

            with pytest.raises(ValueError, match="Duplicate Workspace.mount_path"):
                build_file_plan(
                    input_args={"workspaces": workspaces},
                    input_file_fields={},
                    output_file_fields={},
                    input_workspace_fields={"workspaces": "list_workspace"},
                )

    def test_duplicate_workspace_mount_paths_rejected_across_fields(self):
        with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
            with pytest.raises(ValueError, match="Duplicate Workspace.mount_path"):
                build_file_plan(
                    input_args={
                        "workspace": Workspace(path=one, mount_path="/sandbox/workspace"),
                        "other": Workspace(path=two, mount_path="/sandbox/workspace"),
                    },
                    input_file_fields={},
                    output_file_fields={},
                    input_workspace_fields={
                        "workspace": "workspace",
                        "other": "workspace",
                    },
                )


class TestWorkspaceSyncState:
    def _info(self, text: str) -> WorkspaceFileInfo:
        data = text.encode()
        return WorkspaceFileInfo(
            type="file",
            sha256=hashlib.sha256(data).hexdigest(),
            size=len(data),
        )

    def test_skipped_large_host_file_conflicts_instead_of_clobbering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            host_path = os.path.join(tmpdir, "large.txt")
            with open(host_path, "w") as f:
                f.write("too large")

            state = WorkspaceSyncState(Workspace(path=tmpdir, max_file_bytes=1))
            assert state.iter_mounts() == []

            repl = MagicMock()
            repl.workspace_manifest.return_value = {"large.txt": self._info("x")}

            with pytest.raises(WorkspaceSyncConflictError, match="large.txt"):
                state.sync_from_sandbox(repl)

            repl.sync_file_to.assert_not_called()
            with open(host_path) as f:
                assert f.read() == "too large"

    def test_oversized_sandbox_rewrite_conflicts_instead_of_deleting_host_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            host_path = os.path.join(tmpdir, "small.txt")
            with open(host_path, "w") as f:
                f.write("small")

            state = WorkspaceSyncState(Workspace(path=tmpdir, max_file_bytes=5))
            state.iter_mounts()

            repl = MagicMock()
            repl.workspace_manifest.return_value = {"small.txt": self._info("too large")}

            with pytest.raises(WorkspaceSyncConflictError, match="small.txt"):
                state.sync_from_sandbox(repl)

            repl.sync_file_to.assert_not_called()
            with open(host_path) as f:
                assert f.read() == "small"

    def test_host_symlink_skipped_from_manifest_and_conflicts_on_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "target.txt")
            link = os.path.join(tmpdir, "link.txt")
            with open(target, "w") as f:
                f.write("target")
            os.symlink(target, link)

            state = WorkspaceSyncState(Workspace(path=tmpdir))
            mounts = state.iter_mounts()

            assert (link, "/sandbox/workspace/link.txt") not in mounts

            repl = MagicMock()
            repl.workspace_manifest.return_value = {"link.txt": self._info("sandbox")}

            with pytest.raises(WorkspaceSyncConflictError, match="link.txt"):
                state.sync_from_sandbox(repl)

            repl.sync_file_to.assert_not_called()
            with open(target) as f:
                assert f.read() == "target"

    def test_workspace_root_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "target")
            link = os.path.join(tmpdir, "link")
            os.mkdir(target)
            os.symlink(target, link)

            with pytest.raises(ValueError, match="Workspace path cannot be a symlink"):
                WorkspaceSyncState(Workspace(path=link))

    def test_conflict_detection_is_atomic_before_any_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            clean = os.path.join(tmpdir, "clean.txt")
            conflict = os.path.join(tmpdir, "conflict.txt")
            with open(clean, "w") as f:
                f.write("base clean")
            with open(conflict, "w") as f:
                f.write("base conflict")

            state = WorkspaceSyncState(Workspace(path=tmpdir))
            state.iter_mounts()
            with open(conflict, "w") as f:
                f.write("host concurrent change")

            repl = MagicMock()
            repl.workspace_manifest.return_value = {
                "clean.txt": self._info("sandbox clean"),
                "conflict.txt": self._info("sandbox conflict"),
            }

            with pytest.raises(WorkspaceSyncConflictError, match="conflict.txt"):
                state.sync_from_sandbox(repl)

            repl.sync_file_to.assert_not_called()
            with open(clean) as f:
                assert f.read() == "base clean"

    def test_workspace_manifest_failure_conflicts_instead_of_deleting_host_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "README.md")
            with open(path, "w") as f:
                f.write("keep")

            state = WorkspaceSyncState(Workspace(path=tmpdir))
            state.iter_mounts()

            repl = MagicMock()
            repl.workspace_manifest.side_effect = FileNotFoundError("mount disappeared")

            with pytest.raises(WorkspaceSyncConflictError, match="mount disappeared"):
                state.sync_from_sandbox(repl)

            repl.sync_file_to.assert_not_called()
            with open(path) as f:
                assert f.read() == "keep"


class TestPredictRLMWorkspacePreparation:
    def _make_rlm(self, sig):
        return PredictRLM(sig, sub_lm=MagicMock(), max_iterations=1)

    def test_workspace_transformed_to_mount_path_string(self):
        class Sig(dspy.Signature):
            workspace: Workspace = dspy.InputField()
            answer: str = dspy.OutputField()

        with tempfile.TemporaryDirectory() as tmpdir:
            rlm = self._make_rlm(Sig)
            plan, args = rlm._prepare_file_io({
                "workspace": Workspace(path=tmpdir, mount_path="/sandbox/project")
            })

            assert plan is not None
            assert args["workspace"] == "/sandbox/project"
            assert len(plan["workspace_states"]) == 1

    def test_direct_workspace_transformed_to_effective_sandbox_path(self):
        class Sig(dspy.Signature):
            workspace: Workspace = dspy.InputField()
            answer: str = dspy.OutputField()

        with tempfile.TemporaryDirectory() as tmpdir:
            rlm = PredictRLM(
                Sig,
                sub_lm=MagicMock(),
                max_iterations=1,
                sandbox_backend="sbx",
            )
            plan, args = rlm._prepare_file_io({
                "workspace": Workspace(
                    path=tmpdir,
                    mount_path="/workspace",
                    mode=WorkspaceMode.DIRECT,
                )
            })

            assert plan is not None
            assert args["workspace"] == "/workspace"
            assert plan["workspace_states"] == []
            assert len(plan["direct_workspace_mounts"]) == 1

    def test_direct_workspace_rejects_default_jspi_backend(self):
        class Sig(dspy.Signature):
            workspace: Workspace = dspy.InputField()
            answer: str = dspy.OutputField()

        with tempfile.TemporaryDirectory() as tmpdir:
            rlm = self._make_rlm(Sig)
            with pytest.raises(ValueError, match="requires the SBX backend"):
                rlm._prepare_file_io({
                    "workspace": Workspace(
                        path=tmpdir,
                        mount_path="/workspace",
                        mode=WorkspaceMode.DIRECT,
                    )
                })

    def test_direct_workspace_rejects_sbx_pool(self):
        class Sig(dspy.Signature):
            workspace: Workspace = dspy.InputField()
            answer: str = dspy.OutputField()

        with tempfile.TemporaryDirectory() as tmpdir:
            pool = MagicMock()
            rlm = PredictRLM(
                Sig,
                sub_lm=MagicMock(),
                max_iterations=1,
                sandbox_backend="sbx",
                sbx_pool=pool,
            )
            with pytest.raises(ValueError, match="SbxPool"):
                rlm._prepare_file_io({
                    "workspace": Workspace(
                        path=tmpdir,
                        mount_path="/workspace",
                        mode=WorkspaceMode.DIRECT,
                    )
                })

    @pytest.mark.sbx
    def test_external_sbx_interpreter_reuses_direct_workspace_setup(self, tmp_path: Path):
        from predict_rlm.backends.sbx import SbxBackend, SbxConfig

        class Sig(dspy.Signature):
            workspace: Workspace = dspy.InputField()
            answer: str = dspy.OutputField()

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        mount = DirectWorkspaceMount(
            host_path=os.path.abspath(workspace),
            sandbox_path="/workspace",
        )
        interpreter = SbxBackend(
            config=SbxConfig(name="local-test"),
            preinstall_packages=False,
            direct_workspace_mounts=[mount],
            _runner_command=[sys.executable, "-u", str(RUNNER_PATH)],
            _staging_root=tmp_path / "staging",
        )
        rlm = PredictRLM(
            Sig,
            sub_lm=MagicMock(),
            max_iterations=1,
            interpreter=interpreter,
        )

        try:
            plan, _ = rlm._prepare_file_io({
                "workspace": Workspace(
                    path=str(workspace),
                    mount_path="/workspace",
                    mode=WorkspaceMode.DIRECT,
                )
            })
            assert plan is not None

            rlm._setup_sandbox_files(interpreter, plan)
            interpreter._proc = MagicMock()
            interpreter._proc.poll.return_value = None
            rlm._setup_sandbox_files(interpreter, plan)
        finally:
            interpreter._proc = None
            interpreter.shutdown()

    def test_missing_workspace_raises(self):
        class Sig(dspy.Signature):
            workspace: Workspace = dspy.InputField()
            answer: str = dspy.OutputField()

        rlm = self._make_rlm(Sig)
        with pytest.raises(FileNotFoundError, match="Workspace"):
            rlm._prepare_file_io({"workspace": Workspace(path="/no/such/workspace")})

    def test_input_workspace_type_replaced_with_str(self):
        class Sig(dspy.Signature):
            workspace: Workspace = dspy.InputField(desc="Mutable workspace")
            answer: str = dspy.OutputField()

        rlm = self._make_rlm(Sig)
        action, _ = rlm._build_signatures_with_files("## Files\ntest")
        assert "`workspace`" in action.signature.instructions


@pytest.mark.integration
class TestWorkspaceIOJspiIntegration:
    def _mount_workspace(self, interpreter, workspace):
        state = WorkspaceSyncState(workspace)
        interpreter._ensure_deno_process()
        interpreter.mkdir_p(workspace.mount_path)
        for host_path, virtual_path in state.iter_mounts():
            interpreter.mount_file_at(host_path, virtual_path)
        interpreter.add_post_execute_hook(state.sync_from_sandbox)
        return state

    @pytest.mark.asyncio
    async def test_sync_back_modifies_host_file_after_aexecute(self):
        from predict_rlm.backends.jspi import JspiBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "README.md")
            with open(path, "w") as f:
                f.write("before")

            interpreter = JspiBackend(preinstall_packages=False, extra_read_paths=[tmpdir])
            try:
                self._mount_workspace(interpreter, Workspace(path=tmpdir))
                await interpreter.aexecute("""
from pathlib import Path
Path("/sandbox/workspace/README.md").write_text("after")
print("changed")
""")
                with open(path) as f:
                    assert f.read() == "after"
            finally:
                interpreter.shutdown()

    @pytest.mark.asyncio
    async def test_sync_back_propagates_created_file_and_deletion(self):
        from predict_rlm.backends.jspi import JspiBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            deleted = os.path.join(tmpdir, "delete.txt")
            with open(deleted, "w") as f:
                f.write("delete me")

            interpreter = JspiBackend(preinstall_packages=False, extra_read_paths=[tmpdir])
            try:
                self._mount_workspace(interpreter, Workspace(path=tmpdir))
                await interpreter.aexecute("""
from pathlib import Path
Path("/sandbox/workspace/created.txt").write_text("created")
Path("/sandbox/workspace/delete.txt").unlink()
print("done")
""")
                with open(os.path.join(tmpdir, "created.txt")) as f:
                    assert f.read() == "created"
                assert not os.path.exists(deleted)
            finally:
                interpreter.shutdown()

    @pytest.mark.asyncio
    async def test_sync_back_runs_after_failed_code_block(self):
        from predict_rlm.backends.jspi import JspiBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "failed.txt")
            with open(path, "w") as f:
                f.write("before")

            interpreter = JspiBackend(preinstall_packages=False, extra_read_paths=[tmpdir])
            try:
                self._mount_workspace(interpreter, Workspace(path=tmpdir))
                with pytest.raises(CodeInterpreterError):
                    await interpreter.aexecute("""
from pathlib import Path
Path("/sandbox/workspace/failed.txt").write_text("after failure")
raise RuntimeError("boom")
""")
                with open(path) as f:
                    assert f.read() == "after failure"
            finally:
                interpreter.shutdown()

    @pytest.mark.asyncio
    async def test_conflict_detection_does_not_clobber_host_change(self):
        from predict_rlm.backends.jspi import JspiBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "conflict.txt")
            with open(path, "w") as f:
                f.write("base")

            interpreter = JspiBackend(preinstall_packages=False, extra_read_paths=[tmpdir])
            try:
                self._mount_workspace(interpreter, Workspace(path=tmpdir))
                with open(path, "w") as f:
                    f.write("host concurrent change")

                with pytest.raises(WorkspaceSyncConflictError):
                    await interpreter.aexecute("""
from pathlib import Path
Path("/sandbox/workspace/conflict.txt").write_text("sandbox change")
""")

                with open(path) as f:
                    assert f.read() == "host concurrent change"
            finally:
                interpreter.shutdown()


@pytest.mark.sbx
class TestWorkspaceIOSbxLocalRunner:
    def make_interpreter(self, tmp_path: Path) -> SbxBackend:
        from predict_rlm.backends.sbx import SbxBackend, SbxConfig

        return SbxBackend(
            config=SbxConfig(name="local-test"),
            preinstall_packages=False,
            _runner_command=[sys.executable, "-u", str(RUNNER_PATH)],
            _staging_root=tmp_path / "staging",
        )

    def mount_workspace(self, interpreter: SbxBackend, workspace: Workspace):
        state = WorkspaceSyncState(workspace)
        interpreter.mkdir_p(workspace.mount_path)
        for host_path, virtual_path in state.iter_mounts():
            interpreter.mount_file_at(host_path, virtual_path)
        interpreter.add_post_execute_hook(state.sync_from_sandbox)
        return state

    def test_sync_back_modifies_host_file(self, tmp_path: Path):
        workspace_dir = tmp_path / "workspace"
        workspace_dir.mkdir()
        readme = workspace_dir / "README.md"
        readme.write_text("before", encoding="utf-8")

        interpreter = self.make_interpreter(tmp_path)
        try:
            self.mount_workspace(interpreter, Workspace(path=str(workspace_dir)))
            interpreter.execute(
                "from pathlib import Path\n"
                "Path('/sandbox/workspace/README.md').write_text('after')"
            )
        finally:
            interpreter.shutdown()

        assert readme.read_text(encoding="utf-8") == "after"

    def test_sync_back_runs_after_failed_code_block(self, tmp_path: Path):
        workspace_dir = tmp_path / "workspace"
        workspace_dir.mkdir()
        path = workspace_dir / "failed.txt"
        path.write_text("before", encoding="utf-8")

        interpreter = self.make_interpreter(tmp_path)
        try:
            self.mount_workspace(interpreter, Workspace(path=str(workspace_dir)))
            with pytest.raises(CodeInterpreterError):
                interpreter.execute(
                    "from pathlib import Path\n"
                    "Path('/sandbox/workspace/failed.txt').write_text('after failure')\n"
                    "raise RuntimeError('boom')"
                )
        finally:
            interpreter.shutdown()

        assert path.read_text(encoding="utf-8") == "after failure"

    def test_sync_back_propagates_created_file_and_deletion(self, tmp_path: Path):
        workspace_dir = tmp_path / "workspace"
        workspace_dir.mkdir()
        deleted = workspace_dir / "delete.txt"
        deleted.write_text("delete me", encoding="utf-8")

        interpreter = self.make_interpreter(tmp_path)
        try:
            self.mount_workspace(interpreter, Workspace(path=str(workspace_dir)))
            interpreter.execute(
                "from pathlib import Path\n"
                "Path('/sandbox/workspace/created.txt').write_text('created')\n"
                "Path('/sandbox/workspace/delete.txt').unlink()"
            )
        finally:
            interpreter.shutdown()

        assert (workspace_dir / "created.txt").read_text(encoding="utf-8") == "created"
        assert not deleted.exists()

    def test_sync_back_does_not_delete_host_when_mount_root_disappears(
        self, tmp_path: Path
    ):
        workspace_dir = tmp_path / "workspace"
        workspace_dir.mkdir()
        readme = workspace_dir / "README.md"
        readme.write_text("keep me", encoding="utf-8")

        interpreter = self.make_interpreter(tmp_path)
        try:
            self.mount_workspace(interpreter, Workspace(path=str(workspace_dir)))
            with pytest.raises(WorkspaceSyncConflictError, match="workspace mount"):
                interpreter.execute(
                    "import shutil\n"
                    "from pathlib import Path\n"
                    "shutil.rmtree(Path('/sandbox/workspace'))"
                )
        finally:
            interpreter.shutdown()

        assert readme.read_text(encoding="utf-8") == "keep me"
