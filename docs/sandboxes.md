# Sandbox architecture and recovery

PredictRLM runs each RLM iteration as Python in a sandboxed REPL. The core
contract is that successful iterations share one live interpreter state: imports,
variables, functions, classes, model instances, and tool results created in one
iteration should be available to later iterations.

The project currently has two execution families:

- **JSPI/Deno/Pyodide** — the default browser-WASM-style Python VM.
- **Native container runners** — CPython copied into a sandbox/container, used by
  Docker Sandboxes (`sbx`) and Terminal-Bench/Harbor-style adapters.

## Feature matrix

| Capability | JSPI / Deno / Pyodide | SBX / Docker Sandbox | Harbor / Terminal-Bench-style native runner |
| --- | --- | --- | --- |
| Python runtime | Pyodide in one Deno process | Native CPython in sandbox/container | Native CPython in benchmark container |
| Live state owner | Pyodide VM | Persistent Python kernel process | Persistent Python kernel process |
| Successful execute state | Full REPL state persists | Full REPL state persists | Full REPL state persists |
| Imports/functions/classes persist | Yes | Yes | Yes |
| Pydantic/tool-result objects persist | Yes, within Pyodide limits | Yes, while kernel survives | Yes, while kernel survives |
| Host filesystem exposure | Virtualized file inputs and outputs | Per-run `.predict_rlm_sbx/` staging plus explicit extra workspaces | Backend/container filesystem and mounted task workspace |
| Host-tool callbacks | JSON-RPC bridge through Deno runner | JSON-RPC bridge through copied Python supervisor | JSON-RPC bridge through copied Python supervisor |
| Cooperative timeout | Python trace deadline plus JS interrupt buffer | `SIGINT` to live kernel | `SIGINT` to live kernel |
| State after cooperative timeout | Full VM state remains live | Full kernel state remains live | Full kernel state remains live |
| Hard timeout / crash recovery | Deno is killed; interpreter is fatal | Kernel restarted; pre-timeout pickleable snapshot restored | Kernel restarted; pre-timeout pickleable snapshot restored |
| Hard-timeout state warning | Fatal sandbox error | Reports restored globals and lost globals / imports | Reports restored globals and lost globals / imports |
| Shared implementation | JS runner and JSPI-specific interpreter | Native runner payload plus shared persistent JSON-RPC client | Native runner payload plus shared persistent JSON-RPC client |
| Backend-specific pieces | Deno/Pyodide startup and interrupt mechanics | `sbx` lifecycle, copy/mount/exec/kill/cleanup | Harbor/container lifecycle, mounts, exec/kill/cleanup |
| Contract coverage | `test_iteration_execution_timeout.py` | `test_runner_contracts.py`, `test_sbx_interpreter.py` | `test_runner_contracts.py`, Terminal-Bench container-runner tests |

## Layering

### JSPI/Deno/Pyodide

```text
host `JspiClientAdapter`
  -> persistent Deno process running `sandbox/runner.js`
       -> one live Pyodide Python VM
```

The Deno process is both the supervisor and the Python runtime host. There is no
fork/copyback boundary between successful `execute` calls: Python globals live in
the same Pyodide VM until the interpreter is shut down or the Deno process dies.

### Native container runners

```text
host interpreter / adapter
  -> persistent supervisor process inside the sandbox/container
       -> persistent Python kernel process
```

The host layer owns backend-specific lifecycle and transport concerns:

- creating or attaching to the sandbox/container;
- copying or mounting the runner payload;
- sending JSON-RPC requests;
- enforcing host-side request deadlines;
- mapping `/sandbox/...` paths to the backend filesystem;
- deciding when a dead supervisor can be restarted vs when the whole run is
  unrecoverable.

The supervisor is the copied Python runner payload
(`src/predict_rlm/sandbox/python_runner.py`). It owns shared runtime behavior:

- one persistent Python kernel for successful `execute` requests;
- tool registration and host-tool callback bridging;
- stdout/stderr capture;
- typed `SUBMIT` handling;
- virtual-file/path helpers;
- per-iteration timeout handling;
- structured JSON-RPC responses.

The kernel process is the canonical live Python REPL state. It is separated from
the supervisor so a stuck iteration can be interrupted or killed without losing
the supervisor transport itself when possible.

## State and timeout behavior

Each RLM action may set `execution_timeout_seconds`. This is an iteration-level
contract: a timeout should return an observation to the RLM, not silently erase
state or masquerade as a normal Python error.

**JSPI/Deno/Pyodide**

- Successful execute: same live Pyodide VM; full globals persist.
- Cooperative timeout: Pyodide is interrupted by Python trace deadline and JS
  interrupt buffer; stdout/stderr are returned; the VM remains live, so full
  globals persist.
- Hard timeout / crash: the host watchdog kills Deno and raises
  `SandboxFatalError`; the interpreter is dead and no state is recovered.

**Native container runner (`sbx`, Harbor payload)**

- Successful execute: same live CPython kernel; full globals persist.
- Cooperative timeout: host sends `SIGINT`; if the kernel returns a structured
  timeout, full live state persists and `state.preserved=true`.
- Hard timeout / crash: supervisor kills/restarts the kernel, restores the
  pre-timeout pickleable globals snapshot, and reports `state.preserved=false`,
  `source=pickle_snapshot`, restored names, and lost globals / imports.

A native hard-timeout snapshot is intentionally a downgrade, not a replacement
for REPL persistence. It can restore simple pickleable values that existed before
the timed execution, but it cannot restore arbitrary live objects such as modules,
function/class definitions, dynamic instances, open handles, or mutations made
during the killed execution. The RLM-facing timeout message names both sides:

```text
[state]
Full live Python state was not preserved.
Reason: kernel did not respond to SIGINT before hard kill.
Restored pickleable globals: data, mapping, x.
Lost globals / imports: C, f, json, obj.
```

## What is shared today

Shared across native container backends:

- the copied Python runner payload (`python_runner.py`);
- JSON-RPC response shapes for normal output, structured errors, `SUBMIT`, and
  structured timeouts;
- the persistent-kernel REPL contract;
- cooperative-interrupt then hard-kill timeout policy;
- pickle-snapshot fallback metadata;
- shared contract tests in `tests/test_runner_contracts.py` for local runner,
  SBX, optional real SBX, and adapter-style surfaces.

Partially shared:

- `PersistentJsonRpcRunnerClient` centralizes request IDs, framing, stale
  response handling, and response unwrapping for persistent supervisors.
- Backend adapters still own substantial filesystem, process, timeout, and
  tool-callback plumbing.

Not shared:

- JSPI's Pyodide runtime implementation. It is a different VM and should not use
  pickle snapshots.
- Backend-specific sandbox/container creation, copy, exec, kill, and cleanup.
- The exact interruption mechanism: Pyodide uses trace/interrupt-buffer
  deadlines; native runners use POSIX process signals.

## Sharing direction

The desired architecture is to share the maximum amount of semantics while
leaving only substrate-specific operations in backend adapters.

Target split:

```text
PredictRLM host
  -> shared interpreter/client contract
       -> backend adapter: copy/start/exec/kill/sync
            -> shared runner payload when backend is native CPython
```

The shared layer should own:

- request/response schema;
- timeout metadata schema;
- RLM-facing timeout formatting;
- stale-response and restart diagnostics;
- contract tests for successful REPL state, cooperative timeout preservation,
  hard-timeout snapshot recovery, stdout/stderr capture, tool callbacks, typed
  submit, and file sync semantics.

Backend adapters should own only irreducibly backend-specific operations:

- starting a Deno process, Docker Sandbox, local process, or Harbor container;
- copying/mounting runner files and `/sandbox` workspaces;
- low-level process signaling or provider kill calls;
- cleaning up backend resources.

When a new sandbox backend is added, it should first implement the shared
contract matrix. If it cannot preserve full live state across successful
iterations, it should be treated as a different execution mode and documented as
such rather than quietly weakening the RLM REPL contract.

## Regression coverage

Key coverage points:

- `tests/test_iteration_execution_timeout.py` — JSPI timeout recovery and host
  watchdog behavior.
- `tests/test_runner_contracts.py` — shared REPL semantics across runner
  backends, including imports, functions/classes, Pydantic instances,
  host-tool result objects, cooperative timeouts, and hard-timeout snapshot
  recovery.
- `tests/test_sbx_interpreter.py` — SBX/local supervisor protocol and
  Docker-Sandbox-specific behavior.
- `examples/terminal_bench/tests/test_container_runner.py` — copied-runner
  behavior used by Terminal-Bench/Harbor-style adapters.
