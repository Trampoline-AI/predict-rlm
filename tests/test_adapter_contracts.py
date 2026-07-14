from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Annotated

import pytest

from predict_rlm import ExecutionSpec, File, HostDirectoryMount
from predict_rlm.compatibility import FileInputAdapter, FileOutputAdapter
from predict_rlm.runtime import (
    ArtifactBinding,
    FieldDescriptor,
    InputAdapter,
    OutputAdapter,
    PreparedInput,
)


@pytest.mark.parametrize(
    ("annotation", "is_list", "allows_none", "item_allows_none"),
    [
        (File, False, False, False),
        (Annotated[File, "metadata"], False, False, False),
        (File | None, False, True, False),
        (Annotated[File | None, "metadata"], False, True, False),
        (list[File], True, False, False),
        (list[Annotated[File, "metadata"]], True, False, False),
        (list[File] | None, True, True, False),
        (list[File | None], True, False, True),
    ],
)
def test_field_descriptor_normalizes_supported_annotation_shapes(
    annotation,
    is_list,
    allows_none,
    item_allows_none,
):
    field = FieldDescriptor("source", annotation)

    assert field.name == "source"
    assert field.matches(File)
    assert field.is_list is is_list
    assert field.allows_none is allows_none
    assert field.item_allows_none is item_allows_none

    replacement = field.replace_type(str)
    assert FieldDescriptor("replacement", replacement).is_list is is_list
    assert FieldDescriptor("replacement", replacement).allows_none is allows_none
    assert FieldDescriptor("replacement", replacement).item_allows_none is item_allows_none


def test_field_descriptor_does_not_flatten_arbitrary_unions_or_nested_lists():
    assert not FieldDescriptor("source", File | str).matches(File)
    assert not FieldDescriptor("source", list[list[File]]).matches(File)


def test_execution_spec_rejects_duplicate_host_mount_destinations(tmp_path):
    with pytest.raises(ValueError, match="Duplicate host-directory sandbox destination"):
        ExecutionSpec(
            host_directory_mounts=(
                HostDirectoryMount(str(tmp_path / "first"), "/dataset"),
                HostDirectoryMount(str(tmp_path / "second"), "/dataset"),
            )
        )


def test_execution_spec_rejects_conflicting_host_mount_access(tmp_path):
    with pytest.raises(ValueError, match="conflicting access modes"):
        ExecutionSpec(
            host_directory_mounts=(
                HostDirectoryMount(str(tmp_path), "/read", read_only=True),
                HostDirectoryMount(str(tmp_path), "/write"),
            )
        )


@pytest.mark.asyncio
async def test_typed_adapter_bases_own_matching_and_output_preparation():
    class StringInput(InputAdapter[str]):
        name = "string"
        value_type = str

        async def prepare(self, field, value, ctx):
            return PreparedInput(model_value=f"{field.name}:{value}")

    class StringOutput(OutputAdapter[str]):
        name = "string"
        value_type = str

        async def reserve(self, field, value, ctx, session):
            raise NotImplementedError

        async def materialize(self, reservation, submitted_value, ctx, session):
            raise NotImplementedError

    field = FieldDescriptor("message", str)
    input_adapter = StringInput()
    prepared = await input_adapter.prepare(field, "hello", object())

    assert input_adapter.supports(field, "hello")
    assert await input_adapter.prepare_session(field, prepared, object(), object()) is None
    mounted = await input_adapter.mount(field, prepared, object(), object())
    assert mounted.model_value == "message:hello"
    assert await input_adapter.after_execution(
        field,
        prepared,
        object(),
        object(),
        None,
        RuntimeError("failed"),
    ) is None
    assert await input_adapter.finalize(
        field,
        prepared,
        object(),
        object(),
        None,
    ) is None
    assert StringOutput().supports(field)
    assert await StringOutput().prepare_session(field, None, object()) is None


@pytest.mark.asyncio
async def test_file_input_adapter_preserves_nullable_list_items(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("source", encoding="utf-8")
    field = FieldDescriptor("sources", list[File | None])

    prepared = await FileInputAdapter().prepare(
        field,
        [File(path=str(source)), None],
        SimpleNamespace(state={}),
    )

    assert prepared.model_value == ["/sandbox/input/sources/source.txt", None]
    assert len(prepared.artifacts) == 1


@pytest.mark.asyncio
async def test_file_output_adapter_collects_every_generated_list_item(tmp_path):
    destination = tmp_path / "results"
    destination.mkdir()
    (destination / "stale.txt").write_text("stale", encoding="utf-8")

    class Context:
        state = {
            "output_host_dirs": {"results": str(destination)},
        }

        def bind(self, binding):
            return None

    class Session:
        async def mount(self, artifact):
            return ArtifactBinding(artifact.id, artifact.metadata["sandbox_path"])

        async def collect(self, artifact):
            target = Path(artifact.metadata["destination_path"])
            assert artifact.metadata["directory"] is True
            (target / "first").mkdir(parents=True, exist_ok=True)
            (target / "second").mkdir(parents=True, exist_ok=True)
            (target / "first" / "result.txt").write_text("first", encoding="utf-8")
            (target / "second" / "result.txt").write_text("second", encoding="utf-8")
            return str(target)

    adapter = FileOutputAdapter()
    field = FieldDescriptor("results", list[File])
    reservation = await adapter.reserve(field, None, Context(), Session())

    result = await adapter.materialize(
        reservation,
        [
            File(path="/sandbox/output/results/first/result.txt"),
        ],
        Context(),
        Session(),
    )

    assert [item.path for item in result] == [
        str(destination / "first" / "result.txt"),
        str(destination / "second" / "result.txt"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "submitted_value",
    [
        None,
        File(path=""),
        File(path="relative.txt"),
        File(path="/sandbox/output/result/missing.txt"),
    ],
)
async def test_file_output_adapter_scalar_falls_back_to_reserved_directory(
    tmp_path,
    submitted_value,
):
    destination = tmp_path / "result"

    class Session:
        async def collect(self, artifact):
            target = Path(artifact.metadata["destination_path"])
            if artifact.metadata.get("directory"):
                target.mkdir(parents=True, exist_ok=True)
                (target / "generated.txt").write_text("generated", encoding="utf-8")
                return str(target)
            raise FileNotFoundError(artifact.metadata["sandbox_path"])

    reservation = SimpleNamespace(
        field=FieldDescriptor("result", File),
        artifact=SimpleNamespace(
            id="output",
            kind="compat.output.directory",
            metadata={
                "sandbox_path": "/sandbox/output/result",
                "destination_path": str(destination),
            },
        ),
    )

    result = await FileOutputAdapter().materialize(
        reservation,
        submitted_value,
        SimpleNamespace(),
        Session(),
    )

    assert result == File(path=str(destination / "generated.txt"))


@pytest.mark.asyncio
async def test_file_output_adapter_scalar_fallback_returns_directory_for_multiple_files(
    tmp_path,
):
    destination = tmp_path / "result"

    class Session:
        async def collect(self, artifact):
            target = Path(artifact.metadata["destination_path"])
            if artifact.metadata.get("directory"):
                target.mkdir(parents=True, exist_ok=True)
                (target / "first.txt").write_text("first", encoding="utf-8")
                (target / "second.txt").write_text("second", encoding="utf-8")
                return str(target)
            raise FileNotFoundError(artifact.metadata["sandbox_path"])

    reservation = SimpleNamespace(
        field=FieldDescriptor("result", File),
        artifact=SimpleNamespace(
            id="output",
            kind="compat.output.directory",
            metadata={
                "sandbox_path": "/sandbox/output/result",
                "destination_path": str(destination),
            },
        ),
    )

    result = await FileOutputAdapter().materialize(
        reservation,
        File(path="/sandbox/output/result/missing.txt"),
        SimpleNamespace(),
        Session(),
    )

    assert result == File(path=str(destination))
    assert sorted(path.name for path in destination.iterdir()) == [
        "first.txt",
        "second.txt",
    ]


@pytest.mark.asyncio
async def test_file_output_adapter_does_not_infer_outputs_from_stale_files(tmp_path):
    destination = tmp_path / "results"
    destination.mkdir()
    (destination / "stale.txt").write_text("stale", encoding="utf-8")

    class Session:
        async def collect(self, artifact):
            assert artifact.metadata["directory"] is True
            Path(artifact.metadata["destination_path"]).mkdir(
                parents=True,
                exist_ok=True,
            )
            return artifact.metadata["destination_path"]

    reservation = SimpleNamespace(
        field=FieldDescriptor("result", File),
        artifact=SimpleNamespace(
            id="output",
            kind="compat.output.directory",
            metadata={
                "sandbox_path": "/sandbox/output/result",
                "destination_path": str(destination),
            },
        ),
    )

    result = await FileOutputAdapter().materialize(
        reservation,
        None,
        SimpleNamespace(),
        Session(),
    )

    assert result is None


@pytest.mark.asyncio
async def test_file_output_adapter_uses_remapped_reservation_root(tmp_path):
    destination = tmp_path / "results"

    class Context:
        state = {"output_host_dirs": {"results": str(destination)}}

        def bind(self, binding):
            return None

    class Session:
        async def mount(self, artifact):
            return ArtifactBinding(artifact.id, "/workspace/results")

        async def collect(self, artifact):
            target = Path(artifact.metadata["destination_path"])
            assert artifact.metadata["directory"] is True
            (target / "nested").mkdir(parents=True, exist_ok=True)
            (target / "nested" / "result.txt").write_text("result", encoding="utf-8")
            return str(target)

    adapter = FileOutputAdapter()
    reservation = await adapter.reserve(
        FieldDescriptor("results", list[File]),
        None,
        Context(),
        Session(),
    )

    result = await adapter.materialize(
        reservation,
        [File(path="/workspace/results/nested/result.txt")],
        Context(),
        Session(),
    )

    assert [item.path for item in result] == [str(destination / "nested" / "result.txt")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "submitted_path",
    [
        "/sandbox/input/source.txt",
        "/sandbox/output/results/../outside.txt",
    ],
)
async def test_file_output_adapter_rejects_paths_outside_reservation(
    tmp_path,
    submitted_path,
):
    artifact = SimpleNamespace(
        id="output",
        metadata={
            "sandbox_path": "/sandbox/output/results",
            "destination_path": str(tmp_path / "results"),
        },
    )
    reservation = SimpleNamespace(
        field=FieldDescriptor("results", list[File]),
        artifact=artifact,
    )

    with pytest.raises(ValueError, match="reserved output root"):
        await FileOutputAdapter().materialize(
            reservation,
            [File(path=submitted_path)],
            SimpleNamespace(),
            SimpleNamespace(),
        )
