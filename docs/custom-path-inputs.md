# Custom path inputs

Use `PreparedInput` path factories to make a file, directory, workspace, or
selected file set available to generated code. The runtime handles sandbox
destinations and transfer mechanics.

Use an input adapter when a signature type owns runtime behavior such as fetching
an object or selecting host paths. DSPy already supports Pydantic models; a custom
model does not need an adapter merely to cross the DSPy boundary.

## Materialize, then declare paths

The common adapter has one responsibility: turn the input into local host paths
and return a `PreparedInput` declaration. The kernel validates destinations,
derives backend requirements, and copies or mounts the paths.

This schematic adapter fetches an object before copying it into the sandbox:

```python
from pydantic import BaseModel, Field

from predict_rlm import FieldDescriptor, InputAdapter, PreparedInput, RunContext


class S3File(BaseModel):
    uri: str = Field(description="S3 URI of the object to expose")


class S3FileAdapter(InputAdapter[S3File]):
    name = "s3-file"
    value_type = S3File

    async def prepare(
        self,
        field: FieldDescriptor,
        value: S3File,
        ctx: RunContext,
    ) -> PreparedInput:
        local_path = await materialize_s3(value.uri, ctx=ctx)
        return PreparedInput.path(
            local_path,
            instructions=(f"{field.name} is available at its sandbox path.",),
        )
```

`materialize_s3(...)` represents provider-specific download and cleanup plumbing.
If it creates temporary resources that outlive `prepare`, register cleanup on the
run context or use the advanced adapter lifecycle described in
[Custom adapters and the runtime kernel](custom-adapters.md).

Register the adapter directly or package it in a runtime module:

```python
rlm = PredictRLM(MySignature, adapters=[S3FileAdapter()])
```

Callers continue to pass the Pydantic value:

```python
result = rlm(source=S3File(uri="s3://example-bucket/report.pdf"))
```

The model receives a plain sandbox path, not the host path or provider metadata.

## One file or directory

`PreparedInput.path()` accepts a host file or directory. Copy is the portable
default:

```python
return PreparedInput.path(local_path)
```

The default model path is `/sandbox/input/{field_name}` for a directory and
`/sandbox/input/{field_name}/{filename}` for a file. Use `at=` to choose a path
relative to `/sandbox`:

```python
return PreparedInput.path(local_path, at="datasets/current")
```

`at=` is an exact destination for `path()`. It is a destination root for
`paths()` and `glob()`.

## An explicit file list

Use `paths()` when provider logic already selected several files:

```python
return PreparedInput.paths(local_paths, at="documents")
```

The model receives a `list[str]`. Each source keeps its basename below the shared
destination root. Duplicate or overlapping destinations fail before backend
acquisition.

## A filtered file tree

Use `glob()` when selection is naturally expressed relative to one host root:

```python
return PreparedInput.glob(
    extracted_root,
    include=("**/*.csv", "**/*.json"),
    exclude=("archive/**", "**/*.tmp"),
    at="datasets/current",
)
```

Glob expansion happens on the host before acquisition. It:

- selects files only;
- preserves paths relative to the source root;
- sorts results deterministically;
- fails on an empty result unless `allow_empty=True`; and
- rejects traversal and symlink escapes from the source root.

`glob()` is a copy operation. If data must remain live, mount the complete
directory and select files inside the sandbox.

## Live workspaces

Use a live mount only when host and sandbox must observe the same directory:

```python
return PreparedInput.path(
    workspace_directory,
    mode="mount",
    read_only=False,
)
```

Copy and mount are intentionally different contracts:

- `copy` creates an isolated snapshot and works across maintained backends;
- `mount` exposes a live host directory and fails when the backend cannot provide
  that capability; it never silently falls back to copy.

Per-call SBX sessions support live directory mounts. Reused or named SBX
sessions, `SbxPool`, and the current `DirectPythonBackend` reject them. Use copy
mode when backend portability matters.

## What the kernel owns

For `path()`, `paths()`, and `glob()`, adapters do not implement `mount()` or
inspect backend capabilities. The kernel owns:

1. sandbox destination normalization;
2. overlap detection across inputs and outputs;
3. host read/write requirements;
4. copy or mount lowering for the selected backend; and
5. replacement of planned paths with actual session paths.

Override the adapter lifecycle only when the boundary cannot be represented as
host paths—for example, a provider lease, a mutable remote volume, or a custom
session capability. See [Custom adapters and the runtime kernel](custom-adapters.md).
