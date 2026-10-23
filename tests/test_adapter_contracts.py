from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from predict_rlm import ExecutionSpec, File, HostDirectoryMount, Workspace
from predict_rlm.compatibility import (
    FileInputAdapter,
    FileOutputAdapter,
    WorkspaceInputAdapter,
    files,
)
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
@pytest.mark.parametrize("mode", ["copy", "mount"])
@pytest.mark.parametrize("nullable", [False, True])
async def test_file_input_rejects_duplicate_basenames(tmp_path, mode, nullable):
    sources = [tmp_path / directory / "same.bin" for directory in ("a", "b")]
    for source in sources:
        source.parent.mkdir()
        source.write_bytes(b"binary\x00")
    value = [File(path=str(source)) for source in sources]
    if nullable:
        value.insert(1, None)
    field = FieldDescriptor("documents", list[File | None] if nullable else list[File])
    adapter = FileInputAdapter() if mode == "copy" else FileInputAdapter(mode=mode)

    with pytest.raises(ValueError, match="(Duplicate|overlap)"):
        prepared = compile_prepared_input(
            field, await adapter.prepare(field, value, SimpleNamespace(state={}))
        )
        validate_sandbox_root_reservations(
            {field.name: PreparedInputBinding(field, adapter, prepared)}
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["scalar", "list", "nullable", "empty", "nulls"])
@pytest.mark.parametrize("mode", ["copy", "mount"])
async def test_file_input_modes_preserve_shape_and_mount_without_copy(tmp_path, shape, mode):
    source = tmp_path / "input.bin"
    source.write_bytes(b"binary\x00")
    item = File(path=str(source))
    value = {
        "scalar": item,
        "list": [item],
        "nullable": [None, item, None],
        "empty": [],
        "nulls": [None],
    }[shape]
    field = FieldDescriptor("documents", File if shape == "scalar" else list[File | None])
    adapter = FileInputAdapter(mode=mode)
    ctx = SimpleNamespace(state={})
    prepared = compile_prepared_input(field, await adapter.prepare(field, value, ctx))
    expected_path = "/sandbox/input/documents/input.bin"
    expected = {
        "scalar": expected_path,
        "list": [expected_path],
        "nullable": [None, expected_path, None],
        "empty": [],
        "nulls": [None],
    }[shape]
    assert prepared.model_value == expected
    validate_sandbox_root_reservations(
        {field.name: PreparedInputBinding(field, adapter, prepared)}
    )
    has_file = shape not in ("empty", "nulls")
    if mode == "copy":
        assert len(prepared.artifacts) == int(has_file)
        assert prepared.host_directory_mounts == ()
        return

    assert prepared.artifacts == ()
    assert prepared.requirements.extra_write_paths == ()
    assert prepared.host_directory_mounts == (
        (HostDirectoryMount(str(source.resolve()), expected_path, read_only=True),)
        if has_file
        else ()
    )

    class Session:
        async def mount(self, artifact):
            pytest.fail("mount mode must not copy artifacts")

        async def mount_host_directory(self, mount):
            assert mount.read_only
            return mount.sandbox_path

    bound = await adapter.bind(field, prepared, ctx, Session())
    assert bound.model_value == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("read_only", [False, True])
async def test_direct_workspace_mount_honors_read_only(tmp_path, read_only):
    field = FieldDescriptor("workspace", Workspace)
    workspace = Workspace(path=str(tmp_path), mode="direct", read_only=read_only)
    prepared = await WorkspaceInputAdapter().prepare(
        field, workspace, SimpleNamespace(state={})
    )
    assert prepared.host_directory_mounts == (
        HostDirectoryMount(str(tmp_path), str(tmp_path), read_only=read_only),
    )
    assert prepared.requirements.extra_write_paths == (() if read_only else (str(tmp_path),))
    assert Workspace(path=str(tmp_path), mode="direct").read_only is False


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["file", "files_module", "workspace", "workspace_alias"])
async def test_direct_python_mounts_pass_through_host_paths_without_copy(
    tmp_path, monkeypatch, kind
):
    from predict_rlm import DirectPythonBackend, PredictRLM

    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "input.bin"
    source.write_bytes(b"binary\x00")
    alias = tmp_path / "unused-alias"
    if kind.startswith("workspace"):
        adapter = WorkspaceInputAdapter()
        field = FieldDescriptor("document", Workspace)
        options = {"mount_path": str(alias)} if kind == "workspace_alias" else {}
        value = Workspace(path=str(source_root), mode="direct", read_only=True, **options)
        expected_path = str(source_root)
    else:
        adapter = FileInputAdapter(mode="mount")
        field = FieldDescriptor("document", File)
        value = File(path=str(source))
        expected_path = str(source.resolve())
    interpreter = DirectPythonBackend(
        runner_path=str(tmp_path / "runner.py"), workdir=str(tmp_path), exec_timeout=10
    )
    if kind == "files_module":
        agent = PredictRLM(
            "document -> answer", interpreter=interpreter, modules=[files(input_mode="mount")]
        )
    else:
        agent = PredictRLM("document -> answer", interpreter=interpreter, adapters=[adapter])
    adapter = next(
        item for item in agent.runtime_spec.input_adapters if item.name == adapter.name
    )
    ctx = SimpleNamespace(state={})
    prepared = compile_prepared_input(field, await adapter.prepare(field, value, ctx))
    assert prepared.host_directory_mounts[0].read_only
    assert not any("source_path" in artifact.metadata for artifact in prepared.artifacts)

    def reject_copy(*args, **kwargs):
        pytest.fail("DirectPython mounts must not copy host inputs")

    monkeypatch.setattr(interpreter.adapter, "copy_to", reject_copy)
    spec = ExecutionSpec(host_directory_mounts=prepared.host_directory_mounts)
    try:
        async with agent.runtime_spec.execution.start(spec, ctx) as session:
            bound = await adapter.bind(field, prepared, ctx, session)
            assert bound.model_value == expected_path
            assert bound.bindings[0].path == expected_path
            runtime_path = (
                str(Path(bound.model_value) / source.name)
                if kind.startswith("workspace")
                else bound.model_value
            )
            code = (
                "from pathlib import Path\n"
                f"print(Path({runtime_path!r}).read_bytes())\n"
                f"print(Path({runtime_path!r}).stat().st_ino)"
            )
            result = await session.run_code(code)
            assert result.value == f"{source.read_bytes()!r}\n{source.stat().st_ino}\n"
            source.write_bytes(b"live update")
            result = await session.run_code(code)
            assert result.value == f"{b'live update'!r}\n{source.stat().st_ino}\n"
    finally:
        interpreter.shutdown()
    assert not (interpreter.adapter.sandbox_root / "input").exists()
    assert not alias.exists()
    assert list(source_root.iterdir()) == [source]


@pytest.mark.asyncio
async def test_file_mount_mode_is_rejected_by_jspi_before_acquisition(tmp_path):
    from predict_rlm.backends.jspi.execution import JspiExecutionBackend
    from predict_rlm.runtime import UnsupportedOperationError

    source = tmp_path / "input.bin"
    source.write_bytes(b"input")
    field = FieldDescriptor("document", File)
    adapter = FileInputAdapter(mode="mount")
    ctx = SimpleNamespace(state={})
    prepared = compile_prepared_input(
        field, await adapter.prepare(field, File(path=str(source)), ctx)
    )
    with pytest.raises(UnsupportedOperationError, match="JSPI.*does not support"):
        await JspiExecutionBackend().validate_host_directory_mounts(
            prepared.host_directory_mounts, ctx
        )


@pytest.mark.asyncio
@pytest.mark.sbx
async def test_file_mount_mode_sbx_capabilities(tmp_path):
    from predict_rlm.backends.sbx.config import SbxConfig
    from predict_rlm.backends.sbx.execution import SbxExecutionBackend, SbxPoolExecutionBackend
    from predict_rlm.runtime import UnsupportedOperationError

    source = tmp_path / "input.bin"
    source.write_bytes(b"input")
    field = FieldDescriptor("document", File)
    ctx = SimpleNamespace(state={})
    prepared = compile_prepared_input(
        field, await FileInputAdapter(mode="mount").prepare(field, File(path=str(source)), ctx)
    )
    mounts = prepared.host_directory_mounts
    await SbxExecutionBackend().validate_host_directory_mounts(mounts, ctx)
    with pytest.raises(UnsupportedOperationError, match="Named/reused"):
        await SbxExecutionBackend(
            config=SbxConfig(name="fileref-contract", reuse=True)
        ).validate_host_directory_mounts(mounts, ctx)
    with pytest.raises(UnsupportedOperationError, match="prewarmed"):
        await SbxPoolExecutionBackend(SimpleNamespace()).validate_host_directory_mounts(
            mounts, ctx
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("is_list", [False, True])
async def test_file_output_caller_root_collects_under_field(tmp_path, is_list):
    import shutil

    import dspy

    from predict_rlm import PredictRLM

    signature = dspy.Signature(
        {"result": (list[File] if is_list else File, dspy.OutputField())}
    )

    root = tmp_path / ".outputs" / "step" / "attempt"
    program = PredictRLM(signature, output_dir=root)
    adapter = next(a for a in program.runtime_spec.output_adapters if a.name == "file")
    field = FieldDescriptor("result", list[File] if is_list else File)
    ctx = SimpleNamespace(state={}, bind=lambda binding: None)
    requirements = await adapter.prepare_session(field, None, ctx)
    assert requirements.extra_write_paths == (str(root / "result"),)
    generated = tmp_path / "sandbox-generated"
    (generated / "nested").mkdir(parents=True)
    (generated / "nested" / "result.bin").write_bytes(b"result\x00")

    class Session:
        async def mount(self, artifact):
            from predict_rlm.runtime import ArtifactBinding

            return ArtifactBinding(artifact.id, artifact.metadata["sandbox_path"])

        async def collect(self, artifact):
            target = Path(artifact.metadata["destination_path"])
            if artifact.metadata.get("directory"):
                shutil.copytree(generated, target, dirs_exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(generated / "nested" / "result.bin", target)
            return str(target)

    session = Session()
    reservation = await adapter.reserve(field, None, ctx, session)
    submitted = File(path=f"{reservation.model_value}nested/result.bin")
    result = await adapter.materialize(
        reservation, [submitted] if is_list else submitted, ctx, session
    )
    expected = File(path=str(root / "result" / "nested" / "result.bin"))
    assert result == ([expected] if is_list else expected)
    assert Path(expected.path).read_bytes() == b"result\x00"


@pytest.mark.asyncio
@pytest.mark.parametrize("is_list", [False, True])
@pytest.mark.parametrize("submitted_form", ["host", "virtual"])
async def test_direct_python_file_outputs_use_reserved_host_directory(
    tmp_path, monkeypatch, is_list, submitted_form
):
    import dspy

    from predict_rlm import DirectPythonBackend, PredictRLM

    root = tmp_path / ".outputs" / "step" / "attempt" / "invocation"
    interpreter = DirectPythonBackend(
        runner_path=str(tmp_path / "runner.py"), workdir=str(tmp_path), exec_timeout=10
    )
    annotation = list[File] if is_list else File
    program = PredictRLM(
        dspy.Signature({"result": (annotation, dspy.OutputField())}),
        interpreter=interpreter,
        output_dir=root,
    )
    adapter = next(a for a in program.runtime_spec.output_adapters if a.name == "file")
    field = FieldDescriptor("result", annotation)
    ctx = SimpleNamespace(state={}, bind=lambda binding: None)
    requirements = await adapter.prepare_session(field, None, ctx)
    expected = root / "result" / "nested" / "result.bin"
    virtual = "/sandbox/output/result/nested/result.bin"

    def reject_copy(*args, **kwargs):
        pytest.fail("DirectPython outputs already in place must not be copied")

    monkeypatch.setattr(interpreter.adapter, "_copy_path", reject_copy)
    try:
        async with program.runtime_spec.execution.start(
            ExecutionSpec(extra_write_paths=requirements.extra_write_paths), ctx
        ) as session:
            reservation = await adapter.reserve(field, None, ctx, session)
            assert reservation.model_value == f"{root / 'result'}/"
            runtime_path = str(expected) if submitted_form == "host" else virtual
            await session.run_code(
                "from pathlib import Path\n"
                f"p = Path({str(expected)!r})\n"
                "p.parent.mkdir(parents=True, exist_ok=True)\n"
                "p.write_bytes(b'result\\x00')"
            )
            inode = expected.stat().st_ino
            submitted = File(path=runtime_path)
            result = await adapter.materialize(
                reservation, [submitted] if is_list else submitted, ctx, session
            )
            expected_file = File(path=str(expected))
            assert result == ([expected_file] if is_list else expected_file)
            assert expected.read_bytes() == b"result\x00"
            assert expected.stat().st_ino == inode

            outside = root / "outside.bin"
            outside.write_bytes(b"outside")
            (root / "result" / "escape.bin").symlink_to(outside)
            for escaped in (
                str(outside),
                str(root / "result-other" / "result.bin"),
                str(root / "result" / ".." / "outside.bin"),
                str(root / "result" / "escape.bin"),
                "/sandbox/output/result/../outside.bin",
                "/sandbox/output/result/escape.bin",
            ):
                value = File(path=escaped)
                with pytest.raises(ValueError, match="reserved output root"):
                    await adapter.materialize(
                        reservation, [value] if is_list else value, ctx, session
                    )
    finally:
        interpreter.shutdown()


@pytest.mark.asyncio
async def test_files_contribution_mount_mode(tmp_path):
    contribution = files(input_mode="mount", output_dir=str(tmp_path))
    adapter = next(a for a in contribution.adapters if isinstance(a, FileInputAdapter))
    assert adapter.mode == "mount"
    with pytest.raises(ValueError):
        FileInputAdapter(mode="invalid")


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
