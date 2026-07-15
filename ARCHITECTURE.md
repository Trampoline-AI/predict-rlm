# predict-rlm Architecture

This document describes the implemented execution architecture for PredictRLM
execution backends.

For extension workflows, see [Custom path inputs](docs/custom-path-inputs.md),
[Custom adapters and the runtime kernel](docs/custom-adapters.md), and
[Runtime observability](docs/observability.md). The interpreter-shaped protocol
in `backends/base.py` is retained behind the compatibility bridge; it is not the
root `execution=` contract.

## Execution backend feature matrix

| Mode | Boundary | Payload | Transport | State owner |
| --- | --- | --- | --- | --- |
| JSPI | Deno proc | `jspi/payload.js` | stdio JSON-RPC | Pyodide VM |
| Direct | local proc | `_payload.py` | stdio | CPython kernel |
| SBX | `sbx` box | copied `_payload.py` | WS | CPython kernel |

Implementation names:

- **JSPI** is `JspiBackend`.
- **Direct** is `DirectPythonBackend`.
- **SBX** is `SbxBackend`.

`PythonSupervisor` is not a backend. It is shared host-side implementation for
native CPython backends such as Direct. It owns common JSON-RPC client behavior
for `_payload.py`-based runtimes.

Tests also use an injected local supervisor command to exercise the shared
native protocol without real SBX. That seam is not a production backend.

## Key components and classes

Code map:

```text
src/predict_rlm/
├── predict_rlm.py             PredictRLM
├── runtime.py                 runtime contributions, adapters, root execution
│                              backend and session contracts
├── _shared.py                 build_rlm_signatures()
├── rlm_skills.py              Skill, merge_skills()
├── files.py                   File, SyncedFile
├── trace.py                   IterationStep, RunTrace
├── backends/
│   ├── base.py                BackendName, legacy interpreter backend
│   │                          SupervisorClient, SupervisorProcess
│   ├── jspi/
│   │   ├── __init__.py        public JSPI reexports
│   │   ├── backend.py         JspiBackend
│   │   └── payload.js         JSPI / Pyodide supervisor bridge
│   ├── supervisor/
│   │   ├── __init__.py        public native supervisor reexports
│   │   ├── runner.py          PythonSupervisor, DirectPythonBackend
│   │   │                      SupervisorTransport
│   │   └── _payload.py        native Python supervisor payload
│   └── sbx/
│       ├── __init__.py        public SBX reexports
│       ├── config.py          DEFAULT_SBX_TEMPLATE, SbxConfig
│       ├── backend.py         SbxBackend
│       ├── pool.py            SbxPool
│       └── logging.py         SBX logging helpers
```

### Orchestration

- `PredictRLM` (`predict_rlm.py`) is the main DSPy module. It builds the RLM
  action loop, exposes `predict()` to generated code, creates or leases the
  configured execution backend, and records trace output.
- `build_rlm_signatures()` (`_shared.py`) builds the action/extract signatures
  shown to the outer LM, including tool docs and skill instructions.
- `Skill` and `merge_skills()` (`rlm_skills.py`) bundle reusable instructions,
  PyPI packages, Python modules, and tools for the sandbox.

### Backend contract and selection

- `BackendName` (`backends/base.py`) is the public runtime selector used by
  `sandbox_backend`; currently `jspi` and `sbx`.
- The root `ExecutionBackend` (`runtime.py`) starts one invocation-scoped
  `ExecutionSession`; this is the protocol accepted by `execution=`.
- `ExecutionBackend` (`backends/base.py`) is the legacy interpreter-shaped
  execute/file/shutdown protocol adapted through `interpreter=`.
- `SbxConfig` (`backends/sbx/config.py`) carries Docker Sandboxes
  configuration such as name, resources, template, persistence, and WebSocket
  settings.

### Shared native supervisor

- `_payload.py` (`backends/supervisor/_payload.py`) is the shared Python
  JSON-RPC payload copied into native CPython backends.
- It owns common runtime behavior: `execute`, `reset`, `shutdown`, stdout/stderr
  capture, host-tool callbacks, `SUBMIT`, virtual-file helpers, and timeout
  response metadata.
- Successful native executions run through a persistent CPython kernel process
  owned by the supervisor. That kernel owns user globals and REPL state.
- The split lets the supervisor interrupt or restart stuck user code while
  preserving the transport when possible.

### Backend implementations

- `JspiBackend` (`backends/jspi/backend.py`) runs the default Deno/Pyodide
  backend via `backends/jspi/payload.js`. It does not use the native
  supervisor.
- `PythonSupervisor` (`backends/supervisor/runner.py`) is shared host-side
  JSON-RPC implementation for native supervisor backends. It is not a runtime
  mode by itself.
- `DirectPythonBackend` (`backends/supervisor/runner.py`) launches the native
  supervisor as a local Python subprocess.
- `SbxBackend` (`backends/sbx/backend.py`) creates or reuses an `sbx`
  sandbox, copies `_payload.py`, launches it, publishes the WebSocket
  port, and sends repeated JSON-RPC requests over WebSocket.
- `SupervisorTransport` (`backends/supervisor/runner.py`) is the low-level
  process/file transport protocol used by native supervisor backends.
- `SupervisorClient` (`backends/base.py`) owns shared request-id framing and
  response handling for long-lived supervisors.
- `SbxPool` (`backends/sbx/pool.py`) leases prewarmed `SbxBackend`
  instances. It is a pooling wrapper, not a separate backend mode.

### Data, files, and traces

- `File` and `SyncedFile` (`files.py`) describe host/sandbox file movement for
  RLM inputs, outputs, and host-tool file writeback.
- `IterationStep` and `RunTrace` (`trace.py`) are the structured record of an
  RLM run: generated code, observations, tool calls, timing, and usage metadata.

## Invocation lifecycle and ownership

`RuntimeSpec` is immutable construction-time configuration. Every call creates a
fresh `RunContext`, which owns invocation state, prepared inputs, mounted
bindings, output reservations, cleanup callbacks, and evidence status.

```text
prepare inputs
    -> prepare adapter sessions
    -> validate requirements and acquire one execution session
    -> mount inputs and reserve outputs
    -> execute the RLM loop
    -> materialize outputs
    -> finalize adapters
    -> finalize and release the execution session
```

The ordering preserves four boundaries:

- adapters declare policy and sandbox destinations before acquisition;
- adapters may bind values after acquisition but do not own the session;
- adapter-owned resources finalize in reverse preparation order; and
- framework-owned session finalization and release still run after adapter
  failure or cancellation.

Path declarations are compiled by the kernel before acquisition. Input overlaps
are rejected at that boundary; input/output overlaps are checked when outputs are
reserved, before generated code runs. Portable copies or explicit live mounts are
lowered to backend operations. This keeps path-transfer mechanics out of ordinary
input adapters while leaving custom session capabilities available to advanced
adapters.

## Default JSPI / Deno / Pyodide path

```text
┌──────────────────────────────┐  stdio JSON-RPC  ┌────────────────────────────┐
│ host process                 │ ───────────────▶ │ Deno subprocess            │
│ - JspiBackend                │                  │ - JS supervisor / bridge   │
│ - host tool implementations  │                  │ - file: jspi/payload.js    │
│ - file sync bookkeeping      │                  │ - live Pyodide VM          │
│                              │                  │ - user code in Pyodide     │
└──────────────────────────────┘                  └────────────────────────────┘
```

`JspiBackend` starts Deno, sends `execute` requests over stdio, and handles
host-tool callbacks. The live Python state is the Pyodide VM inside the Deno
process. If Deno dies, the interpreter is fatal because the live VM and mounted
state are gone.

## Shared native Python supervisor shape

```text
┌──────────────────────────────┐  backend RPC     ┌────────────────────────────┐
│ host process                 │ ───────────────▶ │ runtime boundary           │
│ - execution backend          │                  │ - local proc / sbx / ctr   │
│ - host tool implementations  │                  │ - supervisor process       │
│ - request/response handling  │                  │ - file: _payload.py        │
│ - backend lifecycle hooks    │                  │ - JSON-RPC server          │
│                              │                  │ - host-tool bridge         │
│                              │                  │ - optional kernel process  │
└──────────────────────────────┘                  └────────────────────────────┘
```

The native path has two layers inside the runtime boundary. The supervisor is
the protocol endpoint. The persistent kernel process owns successful-iteration
REPL state. This split lets the supervisor interrupt or restart a stuck kernel
while keeping the transport alive when possible.

That process boundary is the recovery contract: user code runs in the
kernel/runner process, and the supervisor is expected to survive kernel exits,
hard-killed timeouts, and native-code aborts in the runner. A kernel failure is
returned as an observation with explicit state metadata so the RLM can continue.
If the supervisor process itself dies, the backend treats that as fatal and
raises `SandboxFatalError`, because the JSON-RPC endpoint, host-tool bridge,
and live kernel handle are gone.

## Direct local Python backend

```text
┌──────────────────────────────┐  stdio JSON-RPC  ┌────────────────────────────┐
│ host process                 │ ───────────────▶ │ local Python subprocess    │
│ - DirectPythonBackend        │                  │ - supervisor process       │
│ - local copy/start/sync      │                  │ - file: _payload.py        │
│ - host tool implementations  │                  │ - JSON-RPC server          │
│                              │                  │ - optional kernel process  │
└──────────────────────────────┘                  └────────────────────────────┘
```

This is the simplest native supervisor path: the backend boundary is just a
local Python subprocess. It is useful for local execution and for testing the
native supervisor contract without Deno or SBX.

## SBX backend

```text
┌──────────────────────────────┐  sbx lifecycle   ┌────────────────────────────┐
│ host process                 │ ───────────────▶ │ SBX sandbox                │
│ - SbxBackend                 │                  │ - copied supervisor        │
│ - create/reuse sandbox       │  WS JSON-RPC     │ - .predict_rlm_supervisor  │
│ - copy supervisor payload    │ ───────────────▶ │ - WebSocket server         │
│ - launch via `sbx exec -d`   │                  │ - JSON-RPC server          │
│ - publish WebSocket port     │                  │ - optional kernel process  │
└──────────────────────────────┘                  └────────────────────────────┘
```

SBX startup uses the `sbx` CLI to prepare the runtime before any `execute`
request is sent:

1. create or reuse the SBX sandbox;
2. copy `_payload.py` under `.predict_rlm_supervisor/`;
3. launch the supervisor with `sbx exec -d`;
4. publish the supervisor WebSocket port to localhost.

After startup, `SbxBackend` sends repeated `execute` requests to the supervisor
over WebSocket JSON-RPC.

`SbxPool` is a pooling wrapper around prewarmed `SbxBackend` instances. It does
not change the SBX runtime shape.

Tests cover `SbxPool` as pool behavior and as a `PredictRLM` SBX integration
path, but it is not part of the runtime-contract backend matrix.

## State and timeout behavior

Each RLM action may set `execution_timeout_seconds`. A timeout should return an
observation to the RLM, not silently erase state or masquerade as a normal
Python error.

### JSPI / Deno / Pyodide

- Successful execute: same live Pyodide VM; full globals persist.
- Cooperative timeout: Pyodide is interrupted by trace deadline and JS interrupt
  buffer; stdout/stderr are returned and the VM remains live.
- Hard timeout or crash: the host watchdog kills Deno and raises
  `SandboxFatalError`; the interpreter is dead and no state is recovered.

### Native supervisor backends

- Successful execute: same live CPython kernel; full globals persist.
- Cooperative timeout: host sends `SIGINT`; if the kernel returns a structured
  timeout, full live state persists and `state.preserved=true`.
- Hard timeout or kernel/runner crash: the supervisor kills/restarts the
  kernel, restores the pre-timeout safe globals snapshot, and reports
  `state.preserved=false`, restored names, and lost globals/imports. The
  supervisor should survive this class of failure.
- Supervisor crash: the backend raises `SandboxFatalError`. This is the native
  fatal boundary; recovery is only guaranteed below the supervisor, not after
  the protocol endpoint itself exits.

Snapshot restore is intentionally a downgrade from REPL persistence. It can
restore only values that existed before timed execution and can be copied
without invoking arbitrary pickle or native-extension hooks. It cannot restore
arbitrary live objects such as modules, function definitions, classes,
open handles, imported modules, native-extension objects, or mutations made
during the killed execution.

The safe snapshot currently preserves:

- scalar values: `None`, `bool`, `int`, `float`, `str`, and `bytes`;
- `pathlib` paths;
- plain `list`, `tuple`, `set`, and `frozenset` containers whose contents are
  also safe;
- `collections.abc.Mapping` instances, restored as plain `dict` values when
  all keys and values are safe;
- dataclass instances, restored as plain `dict` values when all fields are
  safe.

Everything else is marked lost. In particular, native/extension objects from
packages such as MuJoCo, Torch, or OpenCV are not pickled or reduced during
snapshotting; they are recorded under `lost_globals` instead.

## Shared contracts

Native supervisor backends share:

- the copied Python supervisor payload (`_payload.py`);
- JSON-RPC response shapes for normal output, errors, `SUBMIT`, and timeouts;
- the persistent-kernel REPL contract;
- cooperative-interrupt then hard-kill timeout policy;
- safe-snapshot fallback metadata, including restored and lost globals.

Backend implementations own only substrate-specific operations: starting Deno,
Docker Sandboxes, local processes, or containers; copying/mounting files;
low-level process signaling; and resource cleanup.

## Naming conventions

- **Execution backend**: host-side controller implementing the PredictRLM
  runtime contract.
- **Runtime boundary**: the process/container/sandbox that contains executing
  user code.
- **Supervisor**: long-lived protocol endpoint inside a runtime boundary. It
  receives JSON-RPC, dispatches execution, and bridges host tool calls.
- **JSPI supervisor**: `src/predict_rlm/backends/jspi/payload.js`; hosts
  Pyodide and runs user code in the Pyodide VM.
- **Native supervisor**: `src/predict_rlm/backends/supervisor/_payload.py`; may
  spawn a persistent CPython kernel process for user code and globals.
- **Kernel process**: persistent CPython process that owns user globals for
  native supervisor backends.
