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
      -> prepare adapter-owned sessions
      -> validate requirements and acquire one ExecutionSession
      -> mount inputs and reserve outputs
      -> run generated-code attempts
           -> after_execution hooks
      -> materialize outputs
      -> finalize adapters in reverse order
      -> finalize and release the execution session
```

This ordering is the extension contract. Preparation that affects backend policy
must finish before acquisition. Session operations happen only after acquisition.
Adapter finalization does not replace framework-owned session finalization.

## The adapter contracts

An `InputAdapter` selects fields through `value_type` and returns a
`PreparedInput`. Its hooks are:

- `prepare`: describe the model-visible value, artifacts, sandbox destinations,
  and requirements without relying on an acquired execution session;
- `prepare_session`: acquire adapter-owned provider state before backend
  acquisition;
- `mount`: bind the prepared value into the acquired session;
- `after_execution`: durably observe changes after each completed generated-code
  attempt; and
- `finalize`: flush and release adapter-owned state exactly once.

An `OutputAdapter` contributes pre-acquisition requirements, reserves a
session-bound destination, and materializes the final value after execution.
Host/provider cleanup that can outlive a session belongs on `RunContext`.

The framework guarantees `finalize` after `prepare_session` is entered, including
setup failure, backend acquisition failure, execution failure, and cancellation.
Hooks must still clean up resources acquired before that point if
`prepare_session` itself never begins.

## Advanced input example

The example below is schematic: provider and backend-specific operations are
ellipsed, while the kernel hooks and ownership boundaries are real.

```python
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from predict_rlm import (
    ArtifactBinding,
    FieldDescriptor,
    InputAdapter,
    MountedInput,
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

    async def prepare_session(self, field, prepared, ctx, backend) -> None:
        lease = await self.provider.acquire(prepared.metadata["uri"])
        ctx.state[self._state_key(field)] = lease

    async def mount(self, field, prepared, ctx, session) -> MountedInput:
        if not isinstance(session, RemoteWorkspaceSession):
            raise TypeError("The execution session cannot mount remote workspaces")
        binding = await session.mount_remote_workspace(
            uri=prepared.metadata["uri"],
            lease=ctx.state[self._state_key(field)],
            sandbox_path=prepared.model_value,
            writable=prepared.metadata["writable"],
        )
        return MountedInput(model_value=binding.path, bindings=(binding,))

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
in the invocation-local context. The execution backend implements the custom
session capability; the adapter never owns or finalizes that session.

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
