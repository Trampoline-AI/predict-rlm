from __future__ import annotations

import asyncio
import json
import os
import shutil
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import dspy
import pytest

from predict_rlm import (
    DirectoryCreationSession,
    FileTransfer,
    FileTransferSession,
    HostDirectoryMount,
    HostDirectorySession,
    InputAdapter,
    MountedInput,
    MutableDirectorySession,
    PredictRLM,
    PreparedInput,
    SandboxRootReservation,
    SessionRequirements,
    UnsupportedOperationError,
)


@dataclass
class RepositoryLifecycle:
    staging_root: Path
    status_path: Path
    baseline: set[str]
    flushes: list[str] = field(default_factory=list)
    finalized: bool = False


@dataclass(frozen=True, kw_only=True)
class RepositoryPreparedInput(PreparedInput):
    state: RepositoryLifecycle


class MutableRepositoryAdapter(InputAdapter[str]):
    """Configuration-only adapter; every mutable value belongs to one prepared input."""

    name = "mutable-remote-repository"
    value_type = str

    async def prepare(self, field, value, ctx):
        del field, ctx
        descriptor = json.loads(value)
        staging_root = Path(descriptor["staging_root"])
        status_path = Path(descriptor["status_path"])
        files = descriptor["files"]
        staging_root.mkdir()
        for relative, contents in files.items():
            target = staging_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(contents, encoding="utf-8")
        return RepositoryPreparedInput(
            model_value="/repository",
            state=RepositoryLifecycle(staging_root, status_path, set(files)),
            sandbox_roots=(SandboxRootReservation("/repository"),),
            requirements=SessionRequirements(
                extra_read_paths=(str(staging_root),),
                extra_write_paths=(str(staging_root), str(status_path.parent)),
            ),
        )

    async def mount(self, field, prepared, ctx, session):
        del field, ctx
        prepared = self._prepared(prepared)
        if not isinstance(session, DirectoryCreationSession):
            raise TypeError("mutable repositories require directory creation")
        if not isinstance(session, FileTransferSession):
            raise TypeError("mutable repositories require file transfer")
        await session.create_directory("/repository")
        for source in sorted(
            path for path in prepared.state.staging_root.rglob("*") if path.is_file()
        ):
            relative = source.relative_to(prepared.state.staging_root).as_posix()
            await session.transfer_file(
                FileTransfer(str(source), f"/repository/{relative}")
            )
        return MountedInput(model_value="/repository")

    async def after_execution(self, field, prepared, ctx, session, result, error):
        del field, ctx, result
        prepared = self._prepared(prepared)
        await self._flush(prepared.state, session)
        prepared.state.flushes.append("error" if error is not None else "success")
        self._record(prepared.state)

    async def finalize(self, field, prepared, ctx, session, error):
        del field, ctx, error
        prepared = self._prepared(prepared)
        if prepared.state.finalized:
            return
        prepared.state.finalized = True
        if session is not None:
            await self._flush(prepared.state, session)
            prepared.state.flushes.append("final")
        self._record(prepared.state)

    @staticmethod
    def _prepared(prepared: PreparedInput) -> RepositoryPreparedInput:
        if not isinstance(prepared, RepositoryPreparedInput):
            raise TypeError("mutable repository state is missing")
        return prepared

    @staticmethod
    async def _flush(state: RepositoryLifecycle, session) -> None:
        if not isinstance(session, MutableDirectorySession):
            raise TypeError("mutable repositories require sync-back support")
        manifest = await session.inspect_directory("/repository")
        current = {
            relative for relative, info in manifest.items() if info.type == "file"
        }
        for relative in state.baseline - current:
            (state.staging_root / relative).unlink(missing_ok=True)
        for relative in current:
            destination = state.staging_root / relative
            await session.collect_file(f"/repository/{relative}", str(destination))
        state.baseline = current

    @staticmethod
    def _record(state: RepositoryLifecycle) -> None:
        state.status_path.write_text(
            json.dumps(
                {"flushes": state.flushes, "finalized": state.finalized},
                sort_keys=True,
            ),
            encoding="utf-8",
        )


class RepositoryActions:
    async def acall(self, *, iteration, **kwargs):
        del kwargs
        if iteration == "1/2":
            return dspy.Prediction(
                reasoning="mutate before a recoverable failure",
                code=(
                    "from pathlib import Path\n"
                    "root = Path(repository)\n"
                    "(root / 'failed.txt').write_text((root / 'id.txt').read_text())\n"
                    "raise RuntimeError('expected generated failure')"
                ),
            )
        return dspy.Prediction(
            reasoning="complete the repository update",
            code=(
                "from pathlib import Path\n"
                "root = Path(repository)\n"
                "repo_id = (root / 'id.txt').read_text()\n"
                "(root / 'success.txt').write_text(repo_id)\n"
                "(root / 'delete.txt').unlink()\n"
                "SUBMIT(answer=repo_id)"
            ),
        )


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("deno") is None, reason="requires Deno")
@pytest.mark.asyncio
async def test_one_stateless_adapter_handles_interleaved_real_jspi_runs(tmp_path: Path):
    adapter = MutableRepositoryAdapter()
    repositories = []
    for name in ("first", "second"):
        root = tmp_path / name
        status = tmp_path / f"{name}-lifecycle.json"
        repositories.append(
            (
                root,
                status,
                json.dumps(
                    {
                        "staging_root": str(root),
                        "status_path": str(status),
                        "files": {"id.txt": name, "delete.txt": "delete"},
                    }
                ),
            )
        )
    rlm = PredictRLM(
        "repository: str -> answer: str",
        lm=MagicMock(history=[]),
        adapters=[adapter],
        max_iterations=2,
        verbose=False,
    )
    rlm.generate_action = RepositoryActions()

    first_result, second_result = await asyncio.gather(
        *(rlm.aforward(repository=descriptor) for _, _, descriptor in repositories)
    )

    assert (first_result.answer, second_result.answer) == ("first", "second")
    for (root, status, _), expected in zip(
        repositories,
        ("first", "second"),
        strict=True,
    ):
        assert (root / "failed.txt").read_text() == expected
        assert (root / "success.txt").read_text() == expected
        assert not (root / "delete.txt").exists()
        assert json.loads(status.read_text()) == {
            "flushes": ["error", "success", "final"],
            "finalized": True,
        }
    assert vars(adapter) == {}


class ServiceAdapter(InputAdapter[str]):
    name = "ephemeral-service"
    value_type = str

    async def prepare(self, field, value, ctx):
        del field, ctx
        return PreparedInput(
            model_value=json.dumps(
                {"url": "https://snapshot.internal:8443", "snapshot": value}
            ),
            requirements=SessionRequirements(
                allowed_domains=("snapshot.internal:8443",)
            ),
        )


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("deno") is None, reason="requires Deno")
def test_service_plain_value_and_requirement_reach_real_jspi(monkeypatch):
    from predict_rlm.backends.jspi import execution as jspi_execution

    actual_backend = jspi_execution.JspiBackend
    captured = {}

    def build_backend(**kwargs):
        captured["allowed_domains"] = kwargs["allowed_domains"]
        return actual_backend(**kwargs)

    monkeypatch.setattr(jspi_execution, "JspiBackend", build_backend)
    rlm = PredictRLM(
        "service: str -> answer: str",
        lm=MagicMock(history=[]),
        adapters=[ServiceAdapter()],
        max_iterations=1,
        verbose=False,
    )
    rlm.generate_action.acall = AsyncMock(
        return_value=dspy.Prediction(
            reasoning="read the plain service descriptor",
            code="import json\nSUBMIT(answer=json.loads(service)['snapshot'])",
        )
    )

    result = rlm(service="snapshot-42")

    assert result.answer == "snapshot-42"
    assert captured["allowed_domains"] == ["snapshot.internal:8443"]


@pytest.mark.sbx
@pytest.mark.asyncio
async def test_service_requirement_is_rejected_before_reused_or_pooled_sbx_acquisition():
    from predict_rlm.backends.sbx import SbxConfig
    from predict_rlm.backends.sbx.execution import (
        SbxExecutionBackend,
        SbxPoolExecutionBackend,
    )

    named_rlm = PredictRLM(
        "service: str -> answer: str",
        lm=MagicMock(history=[]),
        execution=SbxExecutionBackend(
            config=SbxConfig(name="fixed-service-policy", reuse=True)
        ),
        adapters=[ServiceAdapter()],
        max_iterations=1,
    )
    with (
        patch(
            "predict_rlm.backends.sbx.execution.SbxBackend",
            side_effect=AssertionError("SBX construction must not occur"),
        ),
        pytest.raises(UnsupportedOperationError, match="Named/reused SBX sandboxes"),
    ):
        await named_rlm.aforward(service="snapshot-42")

    class NeverAcquiredPool:
        session_requirements = SessionRequirements()

        def __init__(self) -> None:
            self.acquired = False

        @asynccontextmanager
        async def alease(self, **kwargs):
            del kwargs
            self.acquired = True
            raise AssertionError("pool lease must not occur")
            yield

    pool = NeverAcquiredPool()
    pooled_rlm = PredictRLM(
        "service: str -> answer: str",
        lm=MagicMock(history=[]),
        execution=SbxPoolExecutionBackend(pool),
        adapters=[ServiceAdapter()],
        max_iterations=1,
    )
    with pytest.raises(UnsupportedOperationError, match="fixed policy"):
        await pooled_rlm.aforward(service="snapshot-42")
    assert not pool.acquired


@dataclass(frozen=True)
class CachedDataset:
    path: str


class ReadOnlyDatasetAdapter(InputAdapter[CachedDataset]):
    name = "read-only-dataset"
    value_type = CachedDataset

    async def prepare(self, field, value, ctx):
        del field, ctx
        mount = HostDirectoryMount(value.path, "/dataset", read_only=True)
        return PreparedInput(
            model_value=mount.sandbox_path,
            host_directory_mounts=(mount,),
            requirements=SessionRequirements(extra_read_paths=(value.path,)),
        )

    async def mount(self, field, prepared, ctx, session):
        del field, ctx
        if not isinstance(session, HostDirectorySession):
            raise TypeError("read-only datasets require host-directory mounts")
        return MountedInput(
            model_value=await session.mount_host_directory(
                prepared.host_directory_mounts[0]
            )
        )


class DatasetSignature(dspy.Signature):
    dataset: CachedDataset = dspy.InputField()
    answer: str = dspy.OutputField()


@pytest.mark.integration
@pytest.mark.sbx
@pytest.mark.skipif(
    os.environ.get("PREDICT_RLM_RUN_SBX_TESTS") != "1" or shutil.which("sbx") is None,
    reason="requires the real SBX service",
)
def test_owned_sbx_enforces_external_read_only_mount(tmp_path: Path):
    from predict_rlm.backends.sbx import SbxConfig
    from predict_rlm.backends.sbx.execution import SbxExecutionBackend

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    source = dataset / "rows.csv"
    source.write_text("before", encoding="utf-8")
    rlm = PredictRLM(
        DatasetSignature,
        lm=MagicMock(history=[]),
        execution=SbxExecutionBackend(
            config=SbxConfig(name=f"input-adapter-ro-{os.getpid()}")
        ),
        adapters=[ReadOnlyDatasetAdapter()],
        max_iterations=1,
        verbose=False,
    )
    rlm.generate_action.acall = AsyncMock(
        return_value=dspy.Prediction(
            reasoning="verify that the mounted dataset is immutable",
            code=(
                "from pathlib import Path\n"
                "try:\n"
                "    (Path(dataset) / 'rows.csv').write_text('changed')\n"
                "except OSError as exc:\n"
                "    SUBMIT(answer=type(exc).__name__)\n"
                "else:\n"
                "    SUBMIT(answer='writable')"
            ),
        )
    )

    result = rlm(dataset=CachedDataset(str(dataset)))

    assert result.answer != "writable"
    assert source.read_text(encoding="utf-8") == "before"
