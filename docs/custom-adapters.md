# Custom adapters and the runtime kernel

Custom adapters extend signature types with runtime behavior that cannot be
expressed by copying, mounting, or globbing host paths. This guide explains the
kernel lifecycle, ownership boundaries, and extension interfaces.

For ordinary file, directory, workspace, and glob inputs, use the smaller
[custom path input](custom-path-inputs.md) contract. Do not add an adapter merely
because a signature uses a custom Pydantic model.

## Kernel structure

`PredictRLM` compiles configured adapters, tools, packages, event sinks, and one
execution backend into an immutable `RuntimeSpec`. Every invocation creates a
fresh `RunContext`; mutable provider state belongs there, not on the shared
adapter instance.

```text
construction
  runtime modules + explicit configuration
      -> RuntimeSpec

invocation
  prepare inputs
      -> open adapter-owned resources
      -> validate requirements and acquire one ExecutionSession
      -> bind inputs and reserve outputs
      -> run generated-code attempts
           -> after_execution hooks
      -> materialize outputs
      -> finalize adapters in reverse order
      -> finalize and release the execution session
```

This ordering is the extension contract. Preparation that affects backend policy
must finish before acquisition. Session operations happen only after acquisition.
Adapter finalization does not replace framework-owned session finalization.

## Choose a binding pattern

### Pattern 1: The backend opens it from a description

Use this when an ID, URI, or host path contains everything the backend needs.
`prepare()` records that description as an `Artifact`. Once the execution environment
is ready, the default `bind()` gives the artifact to the backend through
`session.mount(artifact)`. The backend opens the resource and returns its sandbox
path. The adapter does not override `open()` or `bind()`.

### Pattern 2: The adapter opens it first

Use this when opening the resource requires a provider client, login, lease, or other
live handle. `prepare()` records the configuration, `open()` obtains the handle and
stores it in `ctx.state`, and a custom `bind()` passes both to a backend-specific
method on `session`. That method attaches the resource and returns its sandbox path.

## The adapter contracts

An `InputAdapter` selects fields through `value_type` and returns a
`PreparedInput`. Use its hooks to:

- `prepare`: choose the value the RLM receives and declare files or directories to
  copy or mount;
- `open`: open a client, lease, or synchronization handle needed for the
  whole call;
- `bind`: attach a resource that cannot be represented as a host path by using a
  custom execution-session capability, then return the value the RLM should
  receive;
- `after_execution`: persist changes after each completed generated-code block; and
- `finalize`: perform the final save and release adapter-owned resources.

The reserved `ctx_str` name may replace the built-in adapter only through a
`CtxStrInputAdapter` instance or subclass. An independent adapter must use a
different name; it cannot replace the prompt-injecting behavior implicitly.

Backend and session access follows the lifecycle:

- `prepare()` receives neither; it returns declarative requirements;
- `open()` receives the selected `ExecutionBackend`, but no session exists yet;
- `bind()` and `after_execution()` receive the active `ExecutionSession`; and
- `finalize()` receives that session when acquisition succeeded, otherwise `None`.

Adapters may use these objects to check compatibility or call supported session
capabilities, but they never acquire, finalize, or release the execution session.
See [Input adapter lifecycle](api.md#input-adapter-lifecycle) for the exact signatures
and parameter roles.

An `OutputAdapter` contributes pre-acquisition requirements, reserves a
session-bound destination, and materializes the final value after execution.
Host/provider cleanup that can outlive a session belongs on `RunContext`.

The framework guarantees `finalize` after `open` is entered, including
setup failure, backend acquisition failure, execution failure, and cancellation.
Hooks must still clean up resources acquired before that point if
`open` itself never begins.

## Advanced input example

The example below implements Pattern 2. It is schematic: provider and backend-specific
operations are ellipsed, while the kernel hooks and ownership boundaries are real.

```python
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from predict_rlm import (
    ArtifactBinding,
    FieldDescriptor,
    BoundInput,
    InputAdapter,
    PreparedInput,
    RunContext,
    SandboxRootReservation,
)


class RemoteWorkspace(BaseModel):
    uri: str = Field(description="Provider URI of the workspace")
    writable: bool = Field(
        default=False,
        description="Whether generated code may persist workspace changes",
    )


@runtime_checkable
class RemoteWorkspaceSession(Protocol):
    async def mount_remote_workspace(
        self,
        *,
        uri: str,
        lease: object,
        sandbox_path: str,
        writable: bool,
    ) -> ArtifactBinding: ...


class RemoteWorkspaceAdapter(InputAdapter[RemoteWorkspace]):
    name = "remote-workspace"
    value_type = RemoteWorkspace

    def __init__(self, provider):
        self.provider = provider

    def _state_key(self, field: FieldDescriptor) -> str:
        return f"{self.name}:{field.name}"

    async def prepare(
        self,
        field: FieldDescriptor,
        value: RemoteWorkspace,
        ctx: RunContext,
    ) -> PreparedInput:
        sandbox_path = f"/sandbox/input/{field.name}"
        return PreparedInput(
            model_value=sandbox_path,
            metadata={"uri": value.uri, "writable": value.writable},
            sandbox_roots=(SandboxRootReservation(sandbox_path),),
        )

    async def open(self, field, prepared, ctx, backend) -> None:
        lease = await self.provider.acquire(prepared.metadata["uri"])
        ctx.state[self._state_key(field)] = lease

    async def bind(self, field, prepared, ctx, session) -> BoundInput:
        if not isinstance(session, RemoteWorkspaceSession):
            raise TypeError("The execution session cannot mount remote workspaces")
        binding = await session.mount_remote_workspace(
            uri=prepared.metadata["uri"],
            lease=ctx.state[self._state_key(field)],
            sandbox_path=prepared.model_value,
            writable=prepared.metadata["writable"],
        )
        return BoundInput(model_value=binding.path, bindings=(binding,))

    async def after_execution(
        self,
        field,
        prepared,
        ctx,
        session,
        result,
        error,
    ) -> None:
        if prepared.metadata["writable"]:
            await self.provider.flush(ctx.state[self._state_key(field)])

    async def finalize(self, field, prepared, ctx, session, error) -> None:
        lease = ctx.state.pop(self._state_key(field), None)
        if lease is not None:
            await self.provider.release(lease)
```

The Pydantic value contains caller-facing configuration. The prepared value
contains only model-visible data and non-secret metadata. The provider lease stays
in the invocation-local context. `bind()` uses the kernel-owned execution session
to attach that leased workspace to the execution environment; it does not expose
the session to the RLM. The returned `BoundInput.model_value` becomes the
signature field's value before generated code starts. In this example that value is
the mounted sandbox path. The adapter never owns or finalizes the session.

## Output adapters

Output adapters reserve a destination before generated code runs and materialize
the submitted value while the session is active. They implement
`prepare_session()`, `reserve()`, and `materialize()`; they do not use the input
adapter's `finalize()` hook.

When an output owns a sandbox path, record it as `artifact.metadata["sandbox_path"]`
in its `OutputReservation`. The kernel uses that claim to reject overlaps with
prepared inputs and previously reserved outputs before generated code runs.

## Runtime modules

Package related extensions as a construction-time module:

```python
from predict_rlm import RuntimeContribution


def remote_workspace_module() -> RuntimeContribution:
    return RuntimeContribution(
        adapters=(RemoteWorkspaceAdapter(provider),),
        tools=(...),
        packages=(...),
    )


rlm = PredictRLM(MySignature, modules=[remote_workspace_module])
```

`PredictRLM(modules=...)` contributes host-side runtime behavior. It is distinct
from `Skill.modules`, which copies Python modules into the sandbox for generated
code to import.

## Design checklist

A new adapter is ready when:

- construction-time objects are immutable or safe to share across calls;
- mutable state is invocation-local;
- all acquisition requirements are known before backend acquisition;
- sandbox destinations cannot overlap another input or output;
- partial setup, failure, cancellation, and success all release owned resources;
- `after_execution` handles changes made before a generated-code error; and
- tests cover concurrent calls when the extension owns external state.

For the component and process architecture beneath this lifecycle, see
[predict-rlm Architecture](../ARCHITECTURE.md).
