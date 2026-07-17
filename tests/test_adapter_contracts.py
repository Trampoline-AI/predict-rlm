from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Annotated

import pytest
from pydantic import BaseModel

from predict_rlm import ExecutionSpec, File, HostDirectoryMount
from predict_rlm.compatibility import FileInputAdapter, FileOutputAdapter
from predict_rlm.runtime import (
    Artifact,
    ArtifactBinding,
    FieldDescriptor,
    InputAdapter,
    OutputAdapter,
    OutputReservation,
    PreparedInput,
    PreparedInputBinding,
    compile_prepared_input,
    resolve_input_adapter,
    validate_output_sandbox_root_reservation,
    validate_sandbox_root_reservations,
)


def test_concrete_input_adapter_wins_before_exact_fallback_specificity():
    class BaseValue:
        pass

    class SpecificValue(BaseValue):
        pass

    class ConcreteBaseAdapter(InputAdapter[BaseValue]):
        name = "concrete-base"
        value_type = BaseValue

        async def prepare(self, field, value, ctx):
            return PreparedInput(model_value=value)

    class ExactFallbackAdapter(InputAdapter[SpecificValue]):
        name = "exact-fallback"
        value_type = SpecificValue
        fallback = True

        async def prepare(self, field, value, ctx):
            return PreparedInput(model_value=value)

    concrete = ConcreteBaseAdapter()

    selected = resolve_input_adapter(
        [ExactFallbackAdapter(), concrete],
        FieldDescriptor("value", SpecificValue),
        SpecificValue(),
    )

    assert selected is concrete


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


def test_prepared_path_compiles_copy_without_adapter_plumbing(tmp_path):
    source = tmp_path / "report.txt"
    source.write_text("report", encoding="utf-8")

    prepared = compile_prepared_input(
        FieldDescriptor("source", str),
        PreparedInput.path(source),
    )

    assert prepared.model_value == "/sandbox/input/source/report.txt"
    assert [dict(artifact.metadata) for artifact in prepared.artifacts] == [
        {
            "source_path": str(source.resolve()),
            "sandbox_path": "/sandbox/input/source/report.txt",
        }
    ]
    assert prepared.requirements.extra_read_paths == (str(source.resolve()),)
    assert [reservation.path for reservation in prepared.sandbox_roots] == [
        "/sandbox/input/source/report.txt"
    ]


def test_prepared_path_and_paths_honor_relative_destinations(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")

    single = compile_prepared_input(
        FieldDescriptor("source", str),
        PreparedInput.path(first, at="reports/latest.csv"),
    )
    multiple = compile_prepared_input(
        FieldDescriptor("sources", list[str]),
        PreparedInput.paths([second, first], at="datasets/current"),
    )

    assert single.model_value == "/sandbox/reports/latest.csv"
    assert multiple.model_value == [
        "/sandbox/datasets/current/second.csv",
        "/sandbox/datasets/current/first.csv",
    ]

    with pytest.raises(ValueError, match="relative and traversal-free"):
        PreparedInput.path(first, at="../escape.csv")


def test_prepared_paths_reject_duplicate_destinations_within_one_field(tmp_path):
    first = tmp_path / "first" / "report.csv"
    second = tmp_path / "second" / "report.csv"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    field = FieldDescriptor("documents", list[str])
    prepared = compile_prepared_input(
        field,
        PreparedInput.paths([first, second], at="documents"),
    )

    with pytest.raises(ValueError, match="sandbox destinations overlap"):
        validate_sandbox_root_reservations({
            field.name: PreparedInputBinding(field, FileInputAdapter(), prepared)
        })


def test_output_reservation_rejects_overlap_with_prepared_input(tmp_path):
    source = tmp_path / "report.csv"
    source.write_text("report", encoding="utf-8")
    field = FieldDescriptor("source", str)
    prepared = compile_prepared_input(
        field,
        PreparedInput.path(source, at="output/report"),
    )
    input_bindings = {
        field.name: PreparedInputBinding(field, FileInputAdapter(), prepared)
    }
    output_field = FieldDescriptor("report", File)
    output = OutputReservation(
        field=output_field,
        artifact=Artifact(
            id="output",
            kind="test.output",
            metadata={"sandbox_path": "/sandbox/output/report"},
        ),
        model_value="/sandbox/output/report/",
    )

    with pytest.raises(ValueError, match="Input/output sandbox destinations overlap"):
        validate_output_sandbox_root_reservation(input_bindings, {}, output)


@pytest.mark.asyncio
async def test_pydantic_input_adapter_only_prepares_a_path(tmp_path):
    source = tmp_path / "object.json"
    source.write_text("{}", encoding="utf-8")

    class S3File(BaseModel):
        uri: str

    class S3FileAdapter(InputAdapter[S3File]):
        name = "s3-file"
        value_type = S3File

        async def prepare(self, field, value, ctx):
            return PreparedInput.path(source)

    class Session:
        async def mount(self, artifact):
            return ArtifactBinding(artifact.id, artifact.metadata["sandbox_path"])

    field = FieldDescriptor("document", S3File)
    adapter = S3FileAdapter()
    prepared = compile_prepared_input(
        field,
        await adapter.prepare(field, S3File(uri="s3://bucket/object.json"), object()),
    )
    bound = await adapter.bind(field, prepared, object(), Session())

    assert bound.model_value == "/sandbox/input/document/object.json"
    assert [binding.path for binding in bound.bindings] == [bound.model_value]


@pytest.mark.asyncio
async def test_prepared_directory_mount_uses_default_adapter_bind(tmp_path):
    source = tmp_path / "dataset"
    source.mkdir()

    class DatasetAdapter(InputAdapter[str]):
        name = "dataset"
        value_type = str

        async def prepare(self, field, value, ctx):
            return PreparedInput.path(value, mode="mount", read_only=True)

    class Session:
        async def mount_host_directory(self, mount):
            assert mount.host_path == str(source.resolve())
            assert mount.sandbox_path == "/sandbox/input/dataset"
            assert mount.read_only is True
            return mount.sandbox_path

    field = FieldDescriptor("dataset", str)
    adapter = DatasetAdapter()
    prepared = compile_prepared_input(
        field,
        await adapter.prepare(field, str(source), object()),
    )
    bound = await adapter.bind(field, prepared, object(), Session())
    bound_again = await adapter.bind(field, prepared, object(), Session())

    assert bound.model_value == "/sandbox/input/dataset"
    assert [binding.path for binding in bound.bindings] == [
        "/sandbox/input/dataset"
    ]
    assert bound.bindings[0].artifact_id != bound_again.bindings[0].artifact_id


def test_prepared_glob_is_sorted_filtered_and_preserves_relative_paths(tmp_path):
    root = tmp_path / "dataset"
    (root / "nested").mkdir(parents=True)
    (root / "archive").mkdir()
    (root / "z.csv").write_text("z", encoding="utf-8")
    (root / "nested" / "a.csv").write_text("a", encoding="utf-8")
    (root / "archive" / "old.csv").write_text("old", encoding="utf-8")
    (root / "ignored.txt").write_text("ignored", encoding="utf-8")

    prepared = compile_prepared_input(
        FieldDescriptor("files", list[str]),
        PreparedInput.glob(
            root,
            include="**/*.csv",
            exclude="archive/**",
            at="datasets/current",
        ),
    )

    assert prepared.model_value == [
        "/sandbox/datasets/current/nested/a.csv",
        "/sandbox/datasets/current/z.csv",
    ]
    assert [artifact.metadata["sandbox_path"] for artifact in prepared.artifacts] == (
        prepared.model_value
    )


def test_prepared_glob_rejects_empty_matches_by_default(tmp_path):
    with pytest.raises(ValueError, match="did not match any files"):
        PreparedInput.glob(tmp_path, include="**/*.csv")

    prepared = PreparedInput.glob(
        tmp_path,
        include="**/*.csv",
        allow_empty=True,
    )
    assert compile_prepared_input(
        FieldDescriptor("files", list[str]), prepared
    ).model_value == []


def test_prepared_glob_rejects_symlinks_outside_source_root(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / "outside.csv"
    outside.write_text("outside", encoding="utf-8")
    (root / "linked.csv").symlink_to(outside)

    with pytest.raises(ValueError, match="escapes source root"):
        PreparedInput.glob(root, include="*.csv")


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
    assert await input_adapter.open(field, prepared, object(), object()) is None
    bound = await input_adapter.bind(field, prepared, object(), object())
    assert bound.model_value == "message:hello"
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
