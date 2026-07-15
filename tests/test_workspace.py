"""Tests for mutable Workspace inputs and sync-back behavior."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import dspy
import pytest

from predict_rlm import PredictRLM, Workspace, WorkspaceMode
from predict_rlm.compatibility import WorkspaceInputAdapter
from predict_rlm.runtime import (
    ArtifactBinding,
    ArtifactFileInfo,
    ExecutionResult,
    FieldDescriptor,
    FileTransfer,
    HostDirectoryMount,
    PreparedInput,
    UnsupportedOperationError,
)
from predict_rlm.workspace import (
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


class _WorkspaceAdapterContext:
    def __init__(self) -> None:
        self.state = {}
        self.bindings = []

    def bind(self, binding: ArtifactBinding) -> None:
        self.bindings.append(binding)


class _WorkspaceTransportSession:
    name = "workspace-transport"

    def __init__(self) -> None:
        self.direct_mounts: list[HostDirectoryMount] = []
        self.direct_mount_path: str | None = None
        self.created_directories: list[str] = []
        self.transfers: list[FileTransfer] = []
        self.sandbox_files: dict[str, bytes] = {}
        self.inspect_calls: list[str] = []
        self.collect_calls: list[tuple[str, str]] = []

    async def transfer_file(self, transfer: FileTransfer) -> str:
        self.transfers.append(transfer)
        self.sandbox_files[transfer.sandbox_path] = Path(
            transfer.source_path
        ).read_bytes()
        return transfer.sandbox_path

    async def mount_host_directory(self, mount: HostDirectoryMount) -> str:
        self.direct_mounts.append(mount)
        return self.direct_mount_path or mount.sandbox_path

    async def create_directory(self, sandbox_path: str) -> None:
        self.created_directories.append(sandbox_path)

    async def inspect_directory(self, sandbox_path: str):
        self.inspect_calls.append(sandbox_path)
        prefix = sandbox_path.rstrip("/") + "/"
        return {
            path.removeprefix(prefix): ArtifactFileInfo(
                type="file",
                sha256=hashlib.sha256(contents).hexdigest(),
                size=len(contents),
            )
            for path, contents in self.sandbox_files.items()
            if path.startswith(prefix)
        }

    async def collect_file(self, sandbox_path: str, host_path: str) -> None:
        self.collect_calls.append((sandbox_path, host_path))
        destination = Path(host_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.sandbox_files[sandbox_path])


class _CopyInOnlyWorkspaceSession:
    name = "copy-in-only"

    def __init__(self) -> None:
        self.created_directories: list[str] = []
        self.transfers: list[FileTransfer] = []

    async def create_directory(self, sandbox_path: str) -> None:
        self.created_directories.append(sandbox_path)

    async def transfer_file(self, transfer: FileTransfer) -> str:
        self.transfers.append(transfer)
        return transfer.sandbox_path


class TestWorkspaceInputAdapterLifecycle:
    @pytest.mark.asyncio
    async def test_workspace_lifecycle_rejects_prepared_input_without_typed_state(self):
        with pytest.raises(TypeError, match="typed Workspace prepared state"):
            await WorkspaceInputAdapter().mount(
                FieldDescriptor("workspace", Workspace),
                PreparedInput(model_value="/sandbox/workspace"),
                _WorkspaceAdapterContext(),
                _WorkspaceTransportSession(),
            )

    @pytest.mark.asyncio
    async def test_copy_in_only_workspace_does_not_require_sync_back_capabilities(
        self,
        tmp_path: Path,
    ):
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        (workspace_root / "source.txt").write_text("before", encoding="utf-8")
        field = FieldDescriptor("workspace", Workspace)
        ctx = _WorkspaceAdapterContext()
        adapter = WorkspaceInputAdapter()
        prepared = await adapter.prepare(
            field,
            Workspace(path=str(workspace_root), sync_back=False),
            ctx,
        )
        session = _CopyInOnlyWorkspaceSession()

        mounted = await adapter.mount(field, prepared, ctx, session)
        await adapter.finalize(field, prepared, ctx, session, None)

        assert mounted.model_value == "/sandbox/workspace"
        assert session.created_directories == ["/sandbox/workspace"]
        assert [transfer.sandbox_path for transfer in session.transfers] == [
            "/sandbox/workspace/source.txt"
        ]

    @pytest.mark.asyncio
    async def test_prepare_uses_neutral_artifacts_and_typed_requirements(
        self,
        tmp_path: Path,
    ):
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        workspace = Workspace(
            path=str(workspace_root),
            mount_path="/workspace",
            mode=WorkspaceMode.DIRECT,
        )
        field = FieldDescriptor("workspace", Workspace)
        ctx = _WorkspaceAdapterContext()
        adapter = WorkspaceInputAdapter()

        prepared = await adapter.prepare(field, workspace, ctx)
        expected_mount = HostDirectoryMount(
            host_path=str(workspace_root.resolve()),
            sandbox_path="/workspace",
        )
        assert prepared.model_value == "/workspace"
        assert len(prepared.artifacts) == 1
        artifact = prepared.artifacts[0]
        assert artifact.kind == "compat.workspace"
        assert dict(artifact.metadata) == {"sandbox_path": "/workspace"}
        assert "workspace_binding" not in artifact.metadata
        assert "compat.workspace.direct" not in artifact.kind
        assert "compat.workspace.mirror" not in artifact.kind
        json.dumps(dict(artifact.metadata))
        assert prepared.host_directory_mounts == (expected_mount,)
        assert prepared.requirements.extra_read_paths == (str(workspace_root.resolve()),)
        assert prepared.requirements.extra_write_paths == (str(workspace_root.resolve()),)

    @pytest.mark.asyncio
    async def test_direct_mount_uses_exact_host_directory_primitive(self, tmp_path: Path):
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        field = FieldDescriptor("workspace", Workspace)
        ctx = _WorkspaceAdapterContext()
        adapter = WorkspaceInputAdapter()
        prepared = await adapter.prepare(
            field,
            Workspace(
                path=str(workspace_root),
                mount_path="/workspace",
                mode=WorkspaceMode.DIRECT,
            ),
            ctx,
        )
        session = _WorkspaceTransportSession()
        session.direct_mount_path = "/mounted/workspace"

        mounted = await adapter.mount(field, prepared, ctx, session)

        assert session.direct_mounts == [
            HostDirectoryMount(
                host_path=str(workspace_root.resolve()),
                sandbox_path="/workspace",
            )
        ]
        assert session.transfers == []
        assert mounted.model_value == "/mounted/workspace"
        assert [binding.path for binding in mounted.bindings] == [
            "/mounted/workspace"
        ]

    @pytest.mark.asyncio
    async def test_direct_mount_is_rejected_by_jspi(self, tmp_path: Path):
        from predict_rlm.backends.jspi import JspiExecutionBackend

        mount = HostDirectoryMount(str(tmp_path), "/workspace")
        ctx = MagicMock(spec=None)

        with pytest.raises(UnsupportedOperationError, match="JSPI"):
            await JspiExecutionBackend().validate_host_directory_mounts(
                (mount,),
                ctx,
            )

    @pytest.mark.sbx
    @pytest.mark.asyncio
    async def test_direct_mount_is_rejected_by_pooled_sbx(self, tmp_path: Path):
        from predict_rlm.backends.sbx import SbxPoolExecutionBackend

        mount = HostDirectoryMount(str(tmp_path), "/workspace")
        ctx = MagicMock(spec=None)

        with pytest.raises(UnsupportedOperationError, match="SbxPool"):
            await SbxPoolExecutionBackend(MagicMock()).validate_host_directory_mounts(
                (mount,),
                ctx,
            )

    @pytest.mark.asyncio
    async def test_mirror_mount_and_sync_use_generic_transport_after_success_and_failure(
        self,
        tmp_path: Path,
    ):
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        original = workspace_root / "original.txt"
        deleted = workspace_root / "deleted.txt"
        original.write_text("before", encoding="utf-8")
        deleted.write_text("delete", encoding="utf-8")
        field = FieldDescriptor("workspace", Workspace)
        ctx = _WorkspaceAdapterContext()
        adapter = WorkspaceInputAdapter()
        prepared = await adapter.prepare(field, Workspace(path=str(workspace_root)), ctx)
        session = _WorkspaceTransportSession()

        mounted = await adapter.mount(field, prepared, ctx, session)

        assert session.created_directories == ["/sandbox/workspace"]
        assert {transfer.sandbox_path for transfer in session.transfers} == {
            "/sandbox/workspace/deleted.txt",
            "/sandbox/workspace/original.txt",
        }
        assert mounted.model_value == "/sandbox/workspace"

        session.sandbox_files["/sandbox/workspace/original.txt"] = b"after success"
        session.sandbox_files["/sandbox/workspace/created.txt"] = b"created"
        del session.sandbox_files["/sandbox/workspace/deleted.txt"]
        await adapter.after_execution(
            field,
            prepared,
            ctx,
            session,
            ExecutionResult(value="ok"),
            None,
        )

        assert original.read_text(encoding="utf-8") == "after success"
        assert (workspace_root / "created.txt").read_text(encoding="utf-8") == "created"
        assert not deleted.exists()

        session.sandbox_files["/sandbox/workspace/original.txt"] = b"after failure"
        execution_error = RuntimeError("generated code failed")
        await adapter.after_execution(
            field,
            prepared,
            ctx,
            session,
            None,
            execution_error,
        )

        assert original.read_text(encoding="utf-8") == "after failure"
        assert session.inspect_calls == ["/sandbox/workspace", "/sandbox/workspace"]

    @pytest.mark.asyncio
    async def test_finalize_syncs_once_and_is_idempotent(self, tmp_path: Path):
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        source = workspace_root / "source.txt"
        source.write_text("before", encoding="utf-8")
        field = FieldDescriptor("workspace", Workspace)
        ctx = _WorkspaceAdapterContext()
        adapter = WorkspaceInputAdapter()
        prepared = await adapter.prepare(field, Workspace(path=str(workspace_root)), ctx)
        session = _WorkspaceTransportSession()
        await adapter.mount(field, prepared, ctx, session)
        session.sandbox_files["/sandbox/workspace/source.txt"] = b"final"
        await adapter.finalize(field, prepared, ctx, session, None)
        await adapter.finalize(field, prepared, ctx, session, None)

        assert source.read_text(encoding="utf-8") == "final"
        assert session.inspect_calls == ["/sandbox/workspace"]
        assert session.collect_calls == [
            ("/sandbox/workspace/source.txt", str(source.resolve()))
        ]

    @pytest.mark.asyncio
    async def test_finalize_continues_after_list_item_conflict(self, tmp_path: Path):
        first_root = tmp_path / "first"
        second_root = tmp_path / "second"
        first_root.mkdir()
        second_root.mkdir()
        first_source = first_root / "source.txt"
        second_source = second_root / "source.txt"
        first_source.write_text("base", encoding="utf-8")
        second_source.write_text("base", encoding="utf-8")
        field = FieldDescriptor("workspaces", list[Workspace])
        ctx = _WorkspaceAdapterContext()
        adapter = WorkspaceInputAdapter()
        prepared = await adapter.prepare(
            field,
            [
                Workspace(path=str(first_root), mount_path="/sandbox/first"),
                Workspace(path=str(second_root), mount_path="/sandbox/second"),
            ],
            ctx,
        )
        session = _WorkspaceTransportSession()
        await adapter.mount(field, prepared, ctx, session)
        first_source.write_text("host concurrent change", encoding="utf-8")
        session.sandbox_files["/sandbox/first/source.txt"] = b"sandbox change"
        session.sandbox_files["/sandbox/second/source.txt"] = b"synced"
        with pytest.raises(WorkspaceSyncConflictError, match="source.txt"):
            await adapter.after_execution(
                field,
                prepared,
                ctx,
                session,
                ExecutionResult(value="completed"),
                None,
            )

        with pytest.raises(WorkspaceSyncConflictError, match="source.txt"):
            await adapter.finalize(field, prepared, ctx, session, None)

        assert first_source.read_text(encoding="utf-8") == "host concurrent change"
        assert second_source.read_text(encoding="utf-8") == "synced"

    @pytest.mark.asyncio
    async def test_finalize_flushes_mirror_after_cancelled_execution(self, tmp_path: Path):
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        source = workspace_root / "source.txt"
        source.write_text("before", encoding="utf-8")
        field = FieldDescriptor("workspace", Workspace)
        ctx = _WorkspaceAdapterContext()
        adapter = WorkspaceInputAdapter()
        prepared = await adapter.prepare(field, Workspace(path=str(workspace_root)), ctx)
        session = _WorkspaceTransportSession()
        await adapter.mount(field, prepared, ctx, session)
        session.sandbox_files["/sandbox/workspace/source.txt"] = b"after cancellation"

        await adapter.finalize(
            field,
            prepared,
            ctx,
            session,
            asyncio.CancelledError(),
        )

        assert source.read_text(encoding="utf-8") == "after cancellation"
        assert session.inspect_calls == ["/sandbox/workspace"]

    @pytest.mark.asyncio
    async def test_mirror_conflict_through_generic_transport_preserves_host_change(
        self,
        tmp_path: Path,
    ):
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        source = workspace_root / "source.txt"
        source.write_text("base", encoding="utf-8")
        field = FieldDescriptor("workspace", Workspace)
        ctx = _WorkspaceAdapterContext()
        adapter = WorkspaceInputAdapter()
        prepared = await adapter.prepare(field, Workspace(path=str(workspace_root)), ctx)
        session = _WorkspaceTransportSession()
        await adapter.mount(field, prepared, ctx, session)
        source.write_text("host concurrent change", encoding="utf-8")
        session.sandbox_files["/sandbox/workspace/source.txt"] = b"sandbox change"

        with pytest.raises(WorkspaceSyncConflictError, match="source.txt"):
            await adapter.after_execution(
                field,
                prepared,
                ctx,
                session,
                None,
                RuntimeError("generated code failed"),
            )

        assert source.read_text(encoding="utf-8") == "host concurrent change"
        assert session.collect_calls == []

    @pytest.mark.asyncio
    async def test_missing_workspace_fails_during_adapter_preparation(self):
        with pytest.raises(FileNotFoundError, match="/no/such/workspace"):
            await WorkspaceInputAdapter().prepare(
                FieldDescriptor("workspace", Workspace),
                Workspace(path="/no/such/workspace"),
                _WorkspaceAdapterContext(),
            )


class _WorkspaceMutationActions:
    async def acall(self, *, iteration, **kwargs):
        del kwargs
        if iteration == "1/2":
            return dspy.Prediction(
                reasoning="persist a mutation from a failed generated attempt",
                code=(
                    "from pathlib import Path\n"
                    "root = Path(workspace)\n"
                    "(root / 'failed.txt').write_text('after failure')\n"
                    "raise RuntimeError('expected generated failure')"
                ),
            )
        return dspy.Prediction(
            reasoning="complete the workspace mutation",
            code=(
                "from pathlib import Path\n"
                "root = Path(workspace)\n"
                "(root / 'source.txt').write_text('after success')\n"
                "(root / 'created.txt').write_text('created')\n"
                "(root / 'deleted.txt').unlink()\n"
                "SUBMIT(answer='done')"
            ),
        )


class _WorkspaceSignature(dspy.Signature):
    workspace: Workspace = dspy.InputField()
    answer: str = dspy.OutputField()


async def _run_workspace_mutation(rlm: PredictRLM, workspace_root: Path) -> None:
    rlm.generate_action = _WorkspaceMutationActions()
    rlm._configure_run_predictors = MagicMock()

    result = await rlm.aforward(workspace=Workspace(path=str(workspace_root)))

    assert result.answer == "done"
    assert (workspace_root / "source.txt").read_text(encoding="utf-8") == "after success"
    assert (workspace_root / "failed.txt").read_text(encoding="utf-8") == "after failure"
    assert (workspace_root / "created.txt").read_text(encoding="utf-8") == "created"
    assert not (workspace_root / "deleted.txt").exists()


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("deno") is None, reason="requires Deno")
@pytest.mark.asyncio
async def test_workspace_lifecycle_through_maintained_jspi(tmp_path: Path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "source.txt").write_text("before", encoding="utf-8")
    (workspace_root / "deleted.txt").write_text("delete", encoding="utf-8")
    rlm = PredictRLM(
        _WorkspaceSignature,
        lm=MagicMock(history=[]),
        max_iterations=2,
        verbose=False,
    )

    await _run_workspace_mutation(rlm, workspace_root)


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("deno") is None, reason="requires Deno")
@pytest.mark.asyncio
async def test_workspace_conflict_through_maintained_jspi_preserves_host(tmp_path: Path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    source = workspace_root / "source.txt"
    source.write_text("base", encoding="utf-8")

    async def change_host() -> str:
        source.write_text("host concurrent change", encoding="utf-8")
        return "changed"

    rlm = PredictRLM(
        _WorkspaceSignature,
        lm=MagicMock(history=[]),
        tools={"change_host": change_host},
        max_iterations=1,
        verbose=False,
    )
    rlm.generate_action.acall = AsyncMock(
        return_value=dspy.Prediction(
            reasoning="create a host/sandbox conflict",
            code=(
                "from pathlib import Path\n"
                "await change_host()\n"
                "Path(workspace, 'source.txt').write_text('sandbox change')\n"
                "SUBMIT(answer='done')"
            ),
        )
    )
    rlm._configure_run_predictors = MagicMock()

    with pytest.raises(WorkspaceSyncConflictError):
        await rlm.aforward(workspace=Workspace(path=str(workspace_root)))

    assert source.read_text(encoding="utf-8") == "host concurrent change"


@pytest.mark.integration
@pytest.mark.sbx
@pytest.mark.asyncio
async def test_workspace_lifecycle_through_maintained_sbx_local_runner(
    tmp_path: Path,
):
    from predict_rlm.backends.sbx import SbxBackend, SbxConfig

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "source.txt").write_text("before", encoding="utf-8")
    (workspace_root / "deleted.txt").write_text("delete", encoding="utf-8")
    interpreter = SbxBackend(
        config=SbxConfig(name="workspace-lifecycle-local"),
        preinstall_packages=False,
        _runner_command=[sys.executable, "-u", str(RUNNER_PATH)],
        _staging_root=tmp_path / "staging",
    )
    rlm = PredictRLM(
        _WorkspaceSignature,
        lm=MagicMock(history=[]),
        interpreter=interpreter,
        max_iterations=2,
        verbose=False,
    )

    try:
        await _run_workspace_mutation(rlm, workspace_root)
    finally:
        interpreter.shutdown()


@pytest.mark.integration
@pytest.mark.sbx
@pytest.mark.skipif(
    os.environ.get("PREDICT_RLM_RUN_SBX_TESTS") != "1" or shutil.which("sbx") is None,
    reason="requires the real SBX service",
)
@pytest.mark.asyncio
async def test_workspace_lifecycle_through_owned_sbx(tmp_path: Path):
    from predict_rlm.backends.sbx import SbxConfig
    from predict_rlm.backends.sbx.execution import SbxExecutionBackend

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "source.txt").write_text("before", encoding="utf-8")
    (workspace_root / "deleted.txt").write_text("delete", encoding="utf-8")
    rlm = PredictRLM(
        _WorkspaceSignature,
        lm=MagicMock(history=[]),
        execution=SbxExecutionBackend(
            config=SbxConfig(name=f"workspace-lifecycle-{os.getpid()}")
        ),
        max_iterations=2,
        verbose=False,
    )

    await _run_workspace_mutation(rlm, workspace_root)
