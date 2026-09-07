from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from predict_rlm import ExecutionSpec, File, HostDirectoryMount
from predict_rlm.compatibility import FileInputAdapter, FileOutputAdapter
from predict_rlm.runtime import (
    Artifact,
    FieldDescriptor,
    InputAdapter,
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
        validate_sandbox_root_reservations(
            {field.name: PreparedInputBinding(field, FileInputAdapter(), prepared)}
        )


def test_output_reservation_rejects_overlap_with_prepared_input(tmp_path):
    source = tmp_path / "report.csv"
    source.write_text("report", encoding="utf-8")
    field = FieldDescriptor("source", str)
    prepared = compile_prepared_input(
        field,
        PreparedInput.path(source, at="output/report"),
    )
    input_bindings = {field.name: PreparedInputBinding(field, FileInputAdapter(), prepared)}
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
    assert (
        compile_prepared_input(FieldDescriptor("files", list[str]), prepared).model_value == []
    )


def test_prepared_glob_rejects_symlinks_outside_source_root(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / "outside.csv"
    outside.write_text("outside", encoding="utf-8")
    (root / "linked.csv").symlink_to(outside)

    with pytest.raises(ValueError, match="escapes source root"):
        PreparedInput.glob(root, include="*.csv")


@pytest.mark.asyncio
async def test_file_output_adapter_scalar_falls_back_to_reserved_directory(tmp_path):
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
        File(path="/sandbox/output/result/missing.txt"),
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
async def test_file_output_adapter_rejects_paths_outside_reservation(tmp_path):
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
            [File(path="/sandbox/output/results/../outside.txt")],
            SimpleNamespace(),
            SimpleNamespace(),
        )
