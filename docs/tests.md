# Test suites and coverage inventory

This document maps the repository's tests to the features and failure boundaries
that they protect. It distinguishes package regressions, example-specific tests,
local subprocess contracts, and tests requiring external software or services.

The inventory describes **implemented test coverage**, not everything an API
might support and not a line- or branch-coverage guarantee.

## Contents

- [Discovery and inventory totals](#discovery-and-inventory-totals)
- [Execution tiers and commands](#execution-tiers-and-commands)
- [Shared backend contract matrix](#shared-backend-contract-matrix)
- [Package suite inventory](#package-suite-inventory)
- [Example suite inventory](#example-suite-inventory)
- [Cross-cutting feature coverage](#cross-cutting-feature-coverage)
- [Fixtures, isolation, and external prerequisites](#fixtures-isolation-and-external-prerequisites)
- [CI and interpreting results](#ci-and-interpreting-results)
- [Coverage limits](#coverage-limits)
- [Maintaining the suite and this inventory](#maintaining-the-suite-and-this-inventory)

## Discovery and inventory totals

The root [pytest configuration](../pyproject.toml) sets `testpaths = ["tests"]`.
Consequently, running `pytest` or `make test` without explicit paths does **not**
collect the example-local test directories.

| Collection root | Test modules | Test functions | Collected cases | Purpose |
| --- | ---: | ---: | ---: | --- |
| `tests/` | 45 | 315 | 388 | Default package regression suite: PredictRLM, execution backends, Codex LM, and RLM-GEPA. |
| `examples/terminal_bench/tests/` | 8 | 136 | 136 | Terminal-Bench agents, controller adapters, local runner, scoring, and harness integration seams. |
| `examples/spreadbench/tests/` | 6 | 42 | 43 | Spreadsheet recalculation/rendering and example evaluation configuration. |
| `examples/appworld/tests/` | 1 | 47 | 47 | AppWorld task/session integration, evaluation, scoring, and fixture-backed project behavior. |

These are inventory snapshots collected with the root project's **all-extras**
environment. Parametrization expands functions into cases. Counts include cases
that subsequently skip because a capability, binary, or service is unavailable.
They are not minimum-count acceptance criteria. Support files such as `conftest.py`
are not included in the test-module count.

All paths and commands below are relative to the repository root unless stated
otherwise. The per-module tables list **collected cases**, not function counts.

## Execution tiers and commands

### Setup and ordinary runs

The root project requires Python 3.11+ and uses `uv`. The Terminal-Bench example
project declares Python 3.12+.

```bash
uv sync --all-extras

# Complete default package suite, including locally enabled integration cases.
uv run --all-extras pytest

# Compact output without the configured testdox/pass-output verbosity.
uv run --all-extras pytest -o addopts= -q

# Inventory only: imports and collects tests without executing them.
uv run --all-extras pytest --collect-only -q -o addopts=
```

Deno v2 is a default Python dependency, but JSPI execution still requires the
`deno` executable to be available. Initial Pyodide and sandbox-package setup can
require network access. **No LM API credentials are required for the package
regressions**: model responses are scripted or transport endpoints are replaced
with local/fake implementations.

### Marker model

Markers are independent axes, not a directory hierarchy:

| Marker | Meaning | Important distinction |
| --- | --- | --- |
| `integration` | Deno/Pyodide execution or the real Docker Sandboxes service in the package suite. | Local CPython subprocesses are not automatically integration tests. Example suites also use this marker for their own integrations, such as LibreOffice. |
| `sbx` | SBX/supervisor subsystem selection; the `sbx` extra supplies `websockets`. | Most non-integration SBX cases use local supervisors or fakes, not a real sandbox service. |
| `gepa` | RLM-GEPA subsystem selection. | Covers deterministic optimization/evaluation mechanics, not a live optimization experiment. |
| `codex_lm` | Codex LM subsystem selection. | Uses simulated provider streams and local servers, not authenticated Codex calls. |
| `local` | Timing-sensitive or large-payload cases excluded from CI. | Local Make targets include these unless explicitly deselected. |

Do not infer markers from a filename. For example, `test_sbx_pool.py` contains
both unmarked and SBX-marked tests, while `test_lm_config.py` is GEPA-marked.
The inventory tables show the actual routing observed during collection.

### Make targets

The [root Makefile](../Makefile) supplies these selections:

| Target | Pytest selection | Dependency/runtime scope |
| --- | --- | --- |
| `make test` | No marker filter | Runs with the environment selected by `uv run`; does not request all extras. |
| `make test-unit` | `not integration` | Requests all extras. Includes host logic, local processes, and local WebSocket seams. |
| `make test-integration` | `integration` | Requests all extras; combines JSPI and opt-in real SBX cases. |
| `make test-core` | `not integration and not sbx and not gepa and not codex_lm` | Requests no optional extras; includes local CPython processes. |
| `make test-sbx` | `sbx and not integration` | Requests the `sbx` extra; local supervisor/pool coverage. |
| `make test-gepa` | `gepa and not integration` | Requests the `gepa` extra. |
| `make test-codex-lm` | `codex_lm and not integration` | Requests the `codex-lm` extra. |
| `make test-integration-jspi` | `integration and not sbx` | Real JSPI/Deno package cases. |
| `make test-integration-sbx` | `integration and sbx` | Requests the `sbx` extra and sets `PREDICT_RLM_RUN_SBX_TESTS=1`; requires CLI/login/service access. |
| `make test-local` | `local` | Requests all extras; explicitly selects the local-only cases. |

With all extras installed, the package cases partition as follows. The CI column
also excludes `local`; prerequisite and capability skips can still reduce the
number actually executed.

| Selection | Collected locally | Selected by current CI |
| --- | ---: | ---: |
| Core | 167 | 166 |
| SBX, non-integration | 63 | 60 |
| GEPA, non-integration | 43 | 43 |
| Codex LM, non-integration | 46 | 46 |
| JSPI integration | 47 | 46 |
| Real SBX integration | 22 | No standing job |

**Bootstrap exception:** the five Docker bootstrap cases are currently unmarked.
They appear in core/non-integration collections but skip unless
`PREDICT_RLM_RUN_BOOTSTRAP_DOCKER_TESTS=1`. If that environment variable is set,
`make test-unit` can launch Docker builds; `not integration` is not a universal
"no external binaries" guarantee.

### Focused runs

Use pytest directly for paths, node IDs, and expression filters:

```bash
uv run --all-extras pytest tests/test_predict_rlm.py -q
uv run --all-extras pytest tests/test_small_kernel.py -k cancellation -q
uv run --all-extras pytest tests/test_files.py tests/test_file_sync.py -q
uv run --all-extras pytest tests/runtime_contracts -m 'not integration' -q
uv run --all-extras pytest tests/codex_lm -q
uv run --all-extras pytest tests/test_rlm_gepa.py -k patch -q
```

Selecting a path does not remove its integration requirements. For example, the
file transfer tests above execute a real JSPI sandbox.

## Shared backend contract matrix

The canonical shared runtime suite is [tests/runtime_contracts](../tests/runtime_contracts/).
Its [backend factories](../tests/runtime_contracts/backends.py) and
[fixture](../tests/runtime_contracts/conftest.py) run the same contracts through
four concrete configurations:

| Matrix ID | Actual boundary | Selection | Prerequisites |
| --- | --- | --- | --- |
| `jspi` | Deno subprocess and Pyodide/WASM | JSPI integration | Deno v2; sandbox initialization/package availability. |
| `python-runner/direct-process` | Real local CPython supervisor/runner | Core | Local Python; no SBX service or WebSocket transport. |
| `sbx/local-websocket` | Real local supervisor payload over a loopback WebSocket, driven through `SbxBackend` | SBX | `websockets`; no Docker Sandboxes service. |
| `sbx` | Actual Docker Sandboxes environment and WebSocket supervisor | Real SBX integration | `sbx` CLI, login/service access, and explicit opt-in. |

`RuntimeHandle` normalizes result shapes and lifecycle operations, not behavior.
For JSPI, the matrix's reset operation shuts down the interpreter; the next
execution recreates its state. Other matrix backends use their reset API.
The fixture shuts down each handle after its case.

### Covered shared features

| Contract | JSPI | Direct | Local SBX WebSocket | Real SBX |
| --- | --- | --- | --- | --- |
| Execute code, retain state, reset state | Covered | Covered | Covered | Opt-in |
| Serialize concurrent top-level execute requests | Covered | Covered | Covered | Opt-in |
| Normalize Python/REPL/unlabelled code fences | Covered | Covered | Covered | Opt-in |
| Report user/syntax errors and execute again | Covered | Covered | Covered | Opt-in |
| Preserve stdout on a user-code error | Covered | Not asserted: exception has no partial-output contract here | Covered | Opt-in |
| Return structured `SUBMIT` output | Covered | Covered | Covered | Opt-in |
| Defer submission finalization through the interpreter API | Unsupported/skipped | Covered | Unsupported/skipped | Unsupported/skipped |
| Mount, create directories, list, write, and collect files | Covered | Covered | Covered | Opt-in |
| Preserve timeout output and recover for another execution | Covered | Covered | Covered | Opt-in |
| Round-trip list/dict/null/text host-tool results | Covered | Covered | Covered | Opt-in |
| Execute synchronous and asynchronous host tools concurrently | Covered | Unsupported/skipped: callbacks are serial | Covered | Opt-in |
| Recover from host-tool exceptions and tool deadlines | Covered | Covered | Covered | Opt-in |
| Round-trip a roughly 950 KB host-tool request | Local-only | Local-only | Local-only | Local-only and opt-in |

"Covered" identifies an implemented assertion, not a promise that every
configuration has been exercised on a given machine. Unsupported capabilities
are declared in `RuntimeSpec.unsupported`; they are not successful test coverage.
The Direct-only deferred-submit row concerns the **interpreter API**, not the
higher-level PredictRLM submit-confirmation loop.

| Module | Cases | Core feature and covered behavior |
| --- | ---: | --- |
| [test_execution_contract.py](../tests/runtime_contracts/test_execution_contract.py) | 44 | Execution/state/reset; concurrent request serialization; code fences; user and syntax error recovery; partial error output where supported; `SUBMIT` and deferred finalization; file roundtrips; recoverable execution deadlines with stdout/stderr. |
| [test_tool_contract.py](../tests/runtime_contracts/test_tool_contract.py) | 24 | Host-tool value shapes; genuine overlap of sync/async callbacks where supported; large-payload transport; recovery after tool errors; bounded timeout/recovery while multiple tools are pending. The deadline reproduction runs in a child process so a stalled backend cannot hang the test indefinitely. |

## Package suite inventory

Routing labels below refer to the marker selections above. "Mixed" tests can
combine deterministic fakes with genuine filesystem or subprocess behavior;
it does not mean they contact a live model provider.

### RLM execution, inputs, callbacks, and skills

| Module | Cases | Routing | Core feature and covered behavior |
| --- | ---: | --- | --- |
| [test_predict_rlm.py](../tests/test_predict_rlm.py) | 12 | Core + JSPI | Action-loop execution and recovery; nested sandbox-defined Pydantic schemas and an `items` output-name collision; defaulted collection nullability; invalid custom-predictor null output; missing LM errors; concurrent LM-context isolation/restoration; submit confirmation and invalidation by intervening work; sync/async fatal error propagation; no double-charging iteration usage. Real sandbox cases use scripted `DummyLM` responses. |
| [test_empty_code_retry.py](../tests/test_empty_code_retry.py) | 3 | Core | Invalid actions recover through the chat-to-JSON adapter fallback, persistently invalid responses exhaust recovery, and empty custom-predictor code is rejected before execution. |
| [test_in_context.py](../tests/test_in_context.py) | 12 | Core | `CtxStr` adapter precedence and ambiguity; delimiter collisions; prepared versus final bound values; concurrent and sequential run-local predictor/signature isolation; invalid prompt-hook returns; rejection of non-string, optional, and output `CtxStr` declarations. These are host-side prompt construction contracts, not LM comprehension tests. |
| [test_callbacks.py](../tests/test_callbacks.py) | 7 | Core + JSPI | Paired iteration events and shared call IDs; end events on execution failure; awaiting asynchronous handlers; handling async callbacks on the sync path; handler failure isolation; delivery of actual sandbox output to callbacks. |
| [test_rlm_skills.py](../tests/test_rlm_skills.py) | 4 | Core + JSPI | Multiple skills execute together in an RLM run; duplicate tool/module ownership is rejected; a skill cannot silently replace a user tool. Package installation is covered separately below. |

### Runtime kernel, adapters, files, and workspaces

| Module | Cases | Routing | Core feature and covered behavior |
| --- | ---: | --- | --- |
| [test_small_kernel.py](../tests/test_small_kernel.py) | 32 | Core + SBX | Invocation and resource ownership: acquisition failure/retry, cancellation and worker draining, adapter/session finalization order, preservation of primary errors, release failure reporting, fixed mount declarations, and SyncedFile temporary-file lifetime. Strict evidence cannot publish success before required finalization or after emit/flush/close failure. Cancellation produces paired terminal evidence. Most cases use purpose-built sessions/backends and real synchronization primitives rather than a sandbox process. |
| [test_adapter_contracts.py](../tests/test_adapter_contracts.py) | 13 | Core | Adapter specificity; duplicate/conflicting host mounts; explicit relative destinations; duplicate destination and input/output overlap rejection; sorted/filtered glob preparation; empty-glob and escaping-symlink rejection; output reservation fallback; exclusion of stale files and out-of-reservation output paths. |
| [test_external_input_adapter_contracts.py](../tests/test_external_input_adapter_contracts.py) | 3 | JSPI + SBX + real SBX | A stateless custom input adapter handles interleaved real JSPI invocations; incompatible service requirements are rejected before reused/pool acquisition; an opt-in owned SBX case enforces a read-only external mount. |
| [test_files.py](../tests/test_files.py) | 2 | JSPI | PredictRLM input/output `File` handling with actual bytes: binary transformation and source preservation; hiding host output destinations from sandbox inputs; directory enumeration via `File.from_dir()` into `list[File]`; discovery of generated output files omitted from the submitted list; exclusion of stale host outputs. A scalar input `File` is not used as a directory mount. |
| [test_file_sync.py](../tests/test_file_sync.py) | 4 | JSPI | Owned PredictRLM SyncedFile operation performs binary writeback visible to later calls and removes temporary files. Backend-local cases cover async tool failure/recovery, no-writeback mode with a retained custom host directory, and missing files. This deliberately distinguishes the portable tool-operation path from backend-local SyncedFile handling. |
| [test_workspace.py](../tests/test_workspace.py) | 11 | Core + JSPI + real SBX | Mirror lifecycle and synchronization through maintained backends; conflict detection before writes; preservation of host changes; oversized/skipped file safety; host/root symlink rejection; manifest failures must not delete host files; one workspace item's conflict must not prevent other items from finalizing. Includes a local supervisor seam and opt-in owned SBX execution. |

The kernel's safety cases protect several separate transitions: failure while
opening an adapter, acquiring a session, installing/binding runtime inputs,
running generated code, finalizing resources, and releasing the backend. A
successful execution test is not a replacement for these ownership boundaries.

### Backend-specific execution and transport

These complement the shared matrix; they are not separate copies of its basic
execute/state/tool/file scenarios.

| Module | Cases | Routing | Core feature and covered behavior |
| --- | ---: | --- | --- |
| [test_interpreter.py](../tests/test_interpreter.py) | 15 | JSPI | Submit defaults/errors; multi-block fence extraction and inline backticks; nested Pydantic/tool serialization, null values, and string fallback for otherwise unserializable values; nested signature schemas; serialization-error recovery; ignoring late tool responses; reconstructed model extra fields and `items` access; fatal process deadline; workspace flush before cancellation shutdown. Uses real Deno/Pyodide. |
| [test_direct_python_backend.py](../tests/test_direct_python_backend.py) | 3 | Core | Real local runner path virtualization: sandbox paths do not poison subsequent executions; directory contents are copied into the sandbox namespace; a regular `Path` global survives timeout recovery with usable path operations. |
| [test_sbx_interpreter.py](../tests/test_sbx_interpreter.py) | 40 | SBX + real SBX | Native supervisor snapshot/recovery behavior, subprocess stdin isolation, child stdout/stderr attribution, runner exit survival, timeout enforcement despite user exception handlers, tool-reader handoff, late-call quarantine, staging-root ownership, reset, SyncedFile writeback, WebSocket authentication, independent reusable supervisors, Pydantic reconstruction/validation, async interruption, post-hook failure precedence, fixed invocation policy, and cancellation-safe worker cleanup. Most cases use local payload processes; the persistent/re-attach/destroy lifecycle is an opt-in real SBX case. |
| [test_sbx_pool.py](../tests/test_sbx_pool.py) | 20 | Core + SBX | Exclusive leasing/reset; startup and shutdown races; failed reset/replacement/prewarm must not lose capacity or requeue retired interpreters; cancellation-safe retirement and loop migration; waiter cancellation; release after configuration failure; fixed session-policy validation; restart after shutdown. Mixes fake-interpreter synchronization contracts and local supervisor-backed pool cases, not real SBX provisioning. |
| [test_jspi_async_operations.py](../tests/test_jspi_async_operations.py) | 7 | Core | Synthetic async JSPI seams: interrupt until execution quiesces, host-task ownership through recoverable timeout/cancellation, quarantine before the next iteration, and post-hook behavior after fatal/cancelled/failed execution. The filename does not make these Deno integration tests. |
| [test_interpreter_io.py](../tests/test_interpreter_io.py) | 4 | Core | Real OS pipes around a synthetic backend: sync/async UTF-8 backpressure without dropped bytes; partial stdout cannot defeat a request deadline; buffered bytes survive that deadline; stdout EOF must not block draining a still-live stderr pipe. |
| [test_supervisor_client.py](../tests/test_supervisor_client.py) | 2 | Core | Synthetic supervisor frames: discard stale responses/errors until the correct request ID arrives; fail cleanly when the resynchronization limit is exhausted. |
| [test_response_id_resync.py](../tests/test_response_id_resync.py) | 6 | Core | JSPI response multiplexing: stale response rejection in sync/async paths, bounded resynchronization, file-operation response routing, and distinguishing tool calls from stale top-level responses. Uses controlled transport frames, not a Deno process. |
| [test_iteration_execution_timeout.py](../tests/test_iteration_execution_timeout.py) | 7 | Core + JSPI | Bounded failure of silent JSPI timeout recovery; a real PredictRLM deadline preserves state/history/output and permits later `predict()` use; suspended execution and CPU-bound child coroutines from `exec`, imported modules, and async generators recover without losing state or the previous SIGINT handler; non-finite LM-selected deadlines fail before executing code. |
| [test_tool_call_timeout.py](../tests/test_tool_call_timeout.py) | 3 | Core | Hung asynchronous tools return bounded errors; a timed-out synchronous worker does not poison later executor work; evidence-wrapped sync tool deadlines return without waiting for the still-live worker. Uses real tasks/threads with controlled backend seams. |
| [test_runtime_hooks.py](../tests/test_runtime_hooks.py) | 3 | Core | Real local supervisor hook registration, before/after events for file/subprocess operations, clearing hooks, suppression of internal capture operations, and error events for failed user operations. |
| [test_skill_package_integration.py](../tests/test_skill_package_integration.py) | 2 | JSPI + real SBX | A skill installs `python-slugify`, imports it in generated code, and returns a result through PredictRLM. Covers JSPI and an opt-in real SBX pool; not a fake package-list assertion. |
| [test_bootstrap_controller.py](../tests/test_bootstrap_controller.py) | 5 | Core selection; Docker opt-in | Build fixture images to exercise Python/venv/controller bootstrap on Alpine, Python 3.13 slim, and Ubuntu without Python; reject unsupported BusyBox package management and a non-root environment needing privileged repair. Docker builds install dependencies and can require network access. |

### Traces, telemetry, and cancellation evidence

| Module | Cases | Routing | Core feature and covered behavior |
| --- | ---: | --- | --- |
| [test_trace.py](../tests/test_trace.py) | 9 | Core | Replacing an in-progress exported trace; base64 sanitization without mutating the full in-memory trace; proposer projection that keeps behavioral evidence but removes accounting/raw fields; strict-evidence projection; grouping predict calls by signature/instructions/model; nested predict/tool collector isolation; history-delta usage and cache-hit accounting; completion metadata/cache statistics. |
| [test_trace_on_cancellation.py](../tests/test_trace_on_cancellation.py) | 3 | Core | Real traced-loop handling with a controlled interpreter: `KeyboardInterrupt` and `CancelledError` preserve completed and pending iteration evidence; an error while building the trace must not replace the original cancellation. |
| [test_telemetry.py](../tests/test_telemetry.py) | 3 | Core | Persisted JSONL spans, timing and trace/parent correlation; redaction of known secret keys/values in telemetry and debug output; stable candidate hashing across equivalent dictionaries and distinct hashes for different candidates. |
| [test_telemetry_analyzer.py](../tests/test_telemetry_analyzer.py) | 5 | GEPA | Failure precedence across lifecycle, timeout, and model-output evidence; row/event precedence; joining task traces with telemetry artifacts; infrastructure-excluded scores; missing or unrelated evidence remains unknown rather than being confidently misclassified. |

### Optimization and evaluation

| Module | Cases | Routing | Core feature and covered behavior |
| --- | ---: | --- | --- |
| [test_rlm_gepa.py](../tests/test_rlm_gepa.py) | 35 | GEPA | Candidate acceptance/significance; group-aware sampling; project seed/component validation; merge eligibility, pair deduplication, and stronger-base selection; capped/balanced disagreement evidence plus shared-success guardrails; reject insufficient evidence or non-improving children; preserve other base components; sidecar resume; correct raw/logical spend; numerical reporting and checkpoint precedence/repair; evaluation timeout/cancellation; failure/structured trace artifacts; unique write-once artifact namespaces on resume; no-op patch preservation; proposer-visible evidence serialization. LM/proposer outputs are controlled rather than generated by a live optimization run. |
| [test_rlm_gepa_patch_merge_costs.py](../tests/test_rlm_gepa_patch_merge_costs.py) | 2 | GEPA | Exact patch text survives artifact persistence, including whitespace/Unicode; main and sub-LM proposer costs both reach reports; legacy patch roles sharing an operation ID must not collapse distinct charges. |
| [test_lm_config.py](../tests/test_lm_config.py) | 1 | GEPA | A constructed GEPA LM retries simulated LiteLLM rate-limit failures and eventually returns success. Environment validation and waiting are replaced; this does not contact a provider. |

The optimizer suite checks the mechanics of evaluation and instruction evolution.
It does not establish that an optimized instruction improves a real benchmark or
that a particular model follows the proposer prompt correctly.

### Codex LM transport, authentication, and usage

[tests/codex_lm/conftest.py](../tests/codex_lm/conftest.py) marks that directory as
`codex_lm`. It clears/disables disk caching for isolation, disables production
retry delays by default, isolates the auth home, and supplies controlled stream
events and LM instances. Retry tests opt back into their required retry behavior.
The root WebSocket module is marked separately.

| Module | Cases | Routing | Core feature and covered behavior |
| --- | ---: | --- | --- |
| [test_auth.py](../tests/codex_lm/test_auth.py) | 12 | Codex | Explicit legacy-auth opt-in; private credential persistence/removal; enable/disable state; profile-name validation and slug collisions; explicit/environment/active-profile precedence; rotation skips disabled accounts; long-lived LM refresh after account-state changes; redacted status metadata. Uses temporary profiles, not real accounts. |
| [test_auto_retry.py](../tests/codex_lm/test_auto_retry.py) | 2 | Codex | Retry a stalled stream using another enabled rotation profile; expose `CodexStreamError` after retry exhaustion. |
| [test_build_request.py](../tests/codex_lm/test_build_request.py) | 2 | Codex | Request conversion does not mutate reusable constructor reasoning configuration; request-scoped proxy environment changes are restored. |
| [test_cli.py](../tests/codex_lm/test_cli.py) | 8 | Codex | Interception of supported DSPy OpenAI construction; rejection of unsupported models; other providers pass through; child argv/exit status behavior; disabled-profile usage avoids live fetches; login uses an isolated Codex home and preserves the login exit code. CLI invocations are exercised with controlled credentials/login seams, not an actual login. |
| [test_concurrent_cache.py](../tests/codex_lm/test_concurrent_cache.py) | 1 | Codex | Concurrent requests keep their results and cache keys separate; repeated requests hit the correct cache entry instead of transport. Does not promise defensive copying of caller-mutated cached objects. |
| [test_forward.py](../tests/codex_lm/test_forward.py) | 3 | Codex | HTTP stream assembly; cached-input/output pricing; cached calls do not double-charge usage or alter fresh history accounting; asynchronous DSPy prediction exposes parsed answers and billable usage. |
| [test_stream_errors.py](../tests/codex_lm/test_stream_errors.py) | 4 | Codex | Failed, incomplete, explicit-error, and truncated streams surface errors rather than being returned as successful responses. |
| [test_stream_heartbeat.py](../tests/codex_lm/test_stream_heartbeat.py) | 3 | Codex | A silent async stream times out; async HTTP completion does not wait for connection closure; sync HTTP consumption stops at the completion event. Separate sync/async paths both have a terminal-boundary assertion. |
| [test_stream_redaction.py](../tests/codex_lm/test_stream_redaction.py) | 1 | Codex | Stream failure diagnostics do not expose the supplied credential values. |
| [test_usage.py](../tests/codex_lm/test_usage.py) | 3 | Codex | Derive remaining credit and nested model limits; keep live usage windows distinct from model-specific limits; preserve profile display names without leaking raw secret payloads. |
| [test_ws_lm.py](../tests/codex_lm/test_ws_lm.py) | 2 | Codex | Retry-scoped turn state does not leak to the next invocation; exhausted WebSocket transport falls back to HTTP and remains on HTTP for subsequent calls. Uses controlled transport seams, including a local HTTP response path. |
| [test_codex_ws_lm.py](../tests/test_codex_ws_lm.py) | 5 | Codex | Real loopback WebSocket server: prewarm before generation, preserve server error details, isolate turn state per forward, finish synchronously while the server keeps the connection open, and classify HTTP 401 as expired authentication. This is local protocol validation, not the provider service. |

## Example suite inventory

Example suites are collected only when selected explicitly. They retain some
source-text, prompt-wording, defaults, and wiring checks that are intentionally
absent from the reduced package suite. Inventorying them does not mean those
checks meet the package suite's preferred behavioral-test standard.

The following root-environment collection command was checked without running
example tests:

```bash
uv run --all-extras pytest --collect-only -q -o addopts= \
  examples/terminal_bench/tests \
  examples/spreadbench/tests \
  examples/appworld/tests
```

To execute a suite, remove `--collect-only` or select its path separately. Passing
collection is not proof that its execution-time binaries or optional packages are
available, nor that the example suite currently passes.

### Terminal-Bench: 136 cases in eight modules

```bash
uv run --all-extras pytest examples/terminal_bench/tests -q
```

The [example Makefile](../examples/terminal_bench/Makefile) also provides `make test`
when invoked from that example directory. Its `make setup` provisions a separate
Terminal-Bench environment, and `make smoke` runs a synthetic three-task scoring
scenario. These are not substitutes for a real benchmark run. See the
[setup script](../examples/terminal_bench/scripts/setup_terminal_bench.sh) for harness environment installation.

| Module | Cases | Core feature and covered behavior |
| --- | ---: | --- |
| [test_container_runner.py](../examples/terminal_bench/tests/test_container_runner.py) | 19 | Python supervisor request/reset/shutdown mapping; process-clear/shutdown races; code-fence and execution-timeout forwarding; real local runner stdout, timeout recovery, child exit, and stdin isolation; restart diagnostics; structured error mapping; backend-routed file operations; bounded host-tool timeout. Despite the filename, these use fakes and local Python subprocesses, not a container service. |
| [test_gepa_project.py](../examples/terminal_bench/tests/test_gepa_project.py) | 59 | Project/config/CLI contracts; task timeouts/resources; Harbor controller locality selection; supplied Daytona/controller requirements; local shell file transfer and simulated SSH/SBX/SDK adapters; source-bundle/bootstrap layout; remote-root preservation; tracked-file packaging and tar traversal rejection; selective transient setup retries; phase-event aggregation; official task-cache timeout lookup and bounded failure diagnostics; Harbor result/CTRF parsing; GEPA scoring, traces, and LM/agent argument propagation. Most remote/harness operations are fake; local filesystem, archive, and shell paths are exercised. |
| [test_harbor_agent.py](../examples/terminal_bench/tests/test_harbor_agent.py) | 20 | Remote agent payload/environment boundaries; answer sentinel parsing; status/trace persistence and download on success/cancellation; debug streaming and shutdown failure handling; confirmation callback reconstruction; bootstrap and opaque auth upload; Harbor context metadata; setup/agent phase events. Uses simulated remote environments. One context-model case skips if its optional Harbor import is unavailable. |
| [test_runner.py](../examples/terminal_bench/tests/test_runner.py) | 15 | Real local JSON-RPC runner: state/reset, predict/tool and image data URL roundtrips, path behavior, host errors, child-output attribution, runner exit recovery, non-swallowable/native-blocking deadlines, and termination of generated child processes. Also contains a source-text payload-sharing check. |
| [test_scoring.py](../examples/terminal_bench/tests/test_scoring.py) | 8 | Full/partial rewards and CTRF detail precedence; evaluator exception/timeout classification; verified pass evidence overrides timeout placeholders where appropriate; numeric GEPA objective scores. |
| [test_setup_wiring.py](../examples/terminal_bench/tests/test_setup_wiring.py) | 4 | Static package dependencies/entry points, setup-script content, and Make targets. These do not run installation or prove a provisioned environment works. |
| [test_smoke.py](../examples/terminal_bench/tests/test_smoke.py) | 1 | Runs a local synthetic scoring script and distinguishes all-pass, partial, and all-fail outcomes. Does not execute Terminal-Bench tasks against an LM. |
| [test_tbench_agent.py](../examples/terminal_bench/tests/test_tbench_agent.py) | 10 | Agent names/factory, mocked PredictRLM construction, custom signature and confirmation configuration, Codex installation order/error hints, trace export, and rejection of wrapper tools. These are adapter/wiring contracts, not live agent performance tests. |

### Spreadbench: 43 cases in six modules

```bash
uv run --all-extras pytest examples/spreadbench/tests -q
```

The root `examples` extra supplies workbook/PDF-related Python dependencies,
including `openpyxl`, `formulas`, and PyMuPDF. Real rendering additionally requires
**LibreOffice/`soffice` and Poppler's `pdftoppm` on PATH**. The render module skips
if either binary is absent. Recalculation has LibreOffice-dependent cases,
including an `integration`-marked rescue path; absence of that marker on another
case does not imply no external binaries.

| Module | Cases | Core feature and covered behavior |
| --- | ---: | --- |
| [test_eval_sbx_pool.py](../examples/spreadbench/tests/test_eval_sbx_pool.py) | 7 | Backend/pool/logging CLI configuration, invalid pool/backend combinations, and fake pool/PredictRLM argument and lifecycle wiring. Does not provision SBX or execute JSPI. |
| [test_gepa_telemetry.py](../examples/spreadbench/tests/test_gepa_telemetry.py) | 2 | Case start/end telemetry and preservation of a host-tool error span when best-effort recalculation catches an exception. Model execution and scoring are controlled. |
| [test_instruction_prompt.py](../examples/spreadbench/tests/test_instruction_prompt.py) | 1 | Static instruction framing: natural-language requests describe workbook edits, preserve existing values, and do not solicit a prose answer. Not an instruction-following evaluation. |
| [test_recalculate.py](../examples/spreadbench/tests/test_recalculate.py) | 15 | Actual generated workbooks and formula caches; target discovery/resolution counts; formula-library evaluation; preservation of non-formula content; additive/no-op behavior; missing/no-formula inputs; winner/tie precedence; missing-library/error/timeout fallback; LibreOffice rescue of incomplete results. Some failure/fallback seams are monkeypatched. |
| [test_recalculate_hang.py](../examples/spreadbench/tests/test_recalculate_hang.py) | 2 | A checked-in full-column-reference workbook is recalculated in a bounded child process to defend against hangs; no-formula workbook calls emit host-tool telemetry. The hang reproduction disables LibreOffice and bounds formula-worker termination. |
| [test_render.py](../examples/spreadbench/tests/test_render.py) | 16 | Real workbook-to-PNG rendering and data URIs; string/path inputs; cell ranges and sheet-qualified selection; missing files/bad sheets/empty ranges; wrapper error conversion and registration/forwarding checks. Entire module is gated on LibreOffice and Poppler. |

The [Spreadbench README](../examples/spreadbench/README.md) also describes live
evaluation prerequisites. Dataset downloads, provider keys, and live LM runs are
not implied by these example unit/tool tests.

### AppWorld: 47 cases in one module

```bash
uv run --all-extras pytest examples/appworld/tests -q
```

[conftest.py](../examples/appworld/tests/conftest.py) adds the example package to
`sys.path`. Tests use tiny fixture datasets, generated task assets, fake workers,
and controlled RLM results; they do not launch the real AppWorld runtime.

| Module | Cases | Core feature and covered behavior |
| --- | ---: | --- |
| [test_appworld_smoke.py](../examples/appworld/tests/test_appworld_smoke.py) | 47 | Service/project construction; official ICL manifest loading and runtime demo adaptation; dataset/spec loading and deterministic group-disjoint splits; evaluator/count-derived scoring; worker path/JSON conversion; isolated runtime discovery; session JSON argument validation and EOF/stderr deadlock prevention; hiding model-facing completion APIs and internal fields; persistence before evaluation; completion from several answer shapes without legacy fallback/double completion; task-bound host tools; harness-side scoring; evaluation artifacts/LM construction; Codex CLI setup and error hints. Also includes static prompt/default/wiring assertions. |

Live AppWorld execution uses a separate `.appworld-venv` because its dependency
stack includes Pydantic v1. The normal PredictRLM environment remains separate.
Follow the [AppWorld README](../examples/appworld/README.md) for runtime/data
setup; those installations and provider credentials are not needed for the
fixture-backed suite described here.

## Cross-cutting feature coverage

Use this map when deciding where a new regression belongs. The feature's owner
is more useful than the particular bug report or backend that exposed it.

| Feature or invariant | Primary coverage locations |
| --- | --- |
| Generated code produces the final typed result | `test_predict_rlm.py`; shared execution contracts; `test_files.py`. |
| Invalid model actions do not reach code execution | `test_empty_code_retry.py`; `test_iteration_execution_timeout.py`; predict-output validation in `test_predict_rlm.py`. |
| Invocation-local prompts, LM contexts, and collector state | `test_in_context.py`; `test_predict_rlm.py`; `test_trace.py`; external adapter interleaving. |
| Adapter specificity, declared path ownership, and output reservations | `test_adapter_contracts.py`; `test_small_kernel.py`; `test_external_input_adapter_contracts.py`. |
| Host changes and files survive failure without silent clobbering/deletion | `test_workspace.py`; `test_files.py`; `test_file_sync.py`; workspace cancellation in `test_interpreter.py`. |
| Cancellation does not release a lease or delete staging while work is live | `test_small_kernel.py`; `test_jspi_async_operations.py`; `test_sbx_pool.py`; `test_sbx_interpreter.py`. |
| Primary exceptions survive failures in finalization, callbacks, or trace building | `test_small_kernel.py`; `test_callbacks.py`; `test_trace_on_cancellation.py`; backend post-hook cases. |
| Stale responses and partial pipe data cannot corrupt/hang the protocol | `test_response_id_resync.py`; `test_supervisor_client.py`; `test_interpreter_io.py`; native supervisor handoff cases. |
| Timeouts are bounded and recoverable where specified | Shared execution/tool contracts; iteration/tool timeout modules; backend-specific native recovery cases. |
| Credentials and raw image/accounting payloads stay out of restricted outputs | Codex auth/redaction/usage tests; `test_telemetry.py`; `test_trace.py`; proposer artifact tests. |
| Cached or resumed work does not inflate or erase spend | Codex forward/cache tests; `test_trace.py`; GEPA cost/reporting and patch-merge cost tests. |
| Instruction patches require evidence and preserve unrelated solved behavior | GEPA acceptance, balanced/shared-success evidence, improvement gate, and base-component preservation tests. |
| Benchmark scores are based on evaluator evidence, not model claims | Example Terminal-Bench scoring and project tests; AppWorld harness-side evaluation tests. |

## Fixtures, isolation, and external prerequisites

### Shared support and lifecycle ownership

- The shared backend fixture always shuts down the runtime it creates. Its
  factories separate local Direct, local WebSocket, and real SBX environments.
- Kernel tests use explicit fake sessions/backends and synchronization events to
  make cleanup order, acquisition failures, and live-worker ownership observable.
  A fake session proving ordering is not a real filesystem-transfer test.
- Files/workspaces use temporary host directories; the owned SyncedFile test
  additionally verifies staging cleanup after actual roundtrips.
- Codex fixtures isolate cache/auth state. CLI fixtures restore DSPy LM symbols,
  `sys.argv`, and `sys.path` after script execution. HTTP/WebSocket tests use
  loopback servers or simulated event streams.
- Example AppWorld uses checked-in tiny datasets and generated task assets.
  Spreadbench creates workbooks in temporary directories and retains a captured
  huge-range workbook for the bounded-hang reproduction.
- The bootstrap image fixtures live under
  [tests/fixtures/bootstrap_controller](../tests/fixtures/bootstrap_controller/).
  They are Docker build scenarios, not Python test modules.

### Explicit external runs

Real SBX tests create and remove sandbox resources. The persistent lifecycle case
also reattaches to a named sandbox before destroying it. Use an appropriate test
account/environment, not a sandbox containing unrelated work.

```bash
# Requires an installed sbx CLI and a valid login/service connection.
make test-integration-sbx

# Requires Docker CLI and a reachable daemon; builds five fixture images.
PREDICT_RLM_RUN_BOOTSTRAP_DOCKER_TESTS=1 \
  uv run --all-extras pytest tests/test_bootstrap_controller.py -q
```

Local WebSocket tests need loopback networking but no remote sandbox account.
Deno/Pyodide initialization, skill package installation, and bootstrap Docker
builds can download runtime/package artifacts even though no LM is contacted.

## CI and interpreting results

The [test workflow](../.github/workflows/tests.yml) runs:

- Core tests on Ubuntu with Python 3.11, 3.12, and 3.13.
- SBX, GEPA, Codex LM, and JSPI selections in separate Python 3.12 jobs.
- `and not local` in every pytest job.
- Ruff over `src/` and `tests/`.

The workflow does not explicitly run the example-local suites, real SBX, or
opt-in bootstrap Docker scenarios. The six collected `local` cases comprise the
four backend variants of the large host-tool payload contract and two native
SBX/supervisor timing/interrupt cases. One of those four payload variants also
requires real SBX opt-in.

Current CI pytest invocations request `--cov=predict_rlm` and upload coverage
artifacts. Test counts and those artifacts should not be read as a separate
coverage guarantee for `rlm_gepa`, `dspy_codex_lm`, or example packages.

A recent full package run in the all-extras environment completed with
**358 passed, 30 skipped**. The skips were 21 real-SBX prerequisite cases, five
bootstrap Docker cases, and four unsupported shared-matrix capability cases.
That run also reported DSPy deprecation warnings and two async-stream `aclose`
cleanup warnings. They were not suppressed. This historical execution result is
not an execution claim for the example suites, which were collected separately
for this inventory.

When reading a result:

1. Check the selected paths and markers. A green `test-core` run is not a full
   package or example run.
2. Use `-rs` to inspect skip reasons. A capability skip is different from a
   missing binary, missing optional dependency, or disabled real-service test.
3. Distinguish controlled model responses from live provider behavior and local
   SBX supervisor seams from real SBX provisioning.
4. Treat unexpected warnings and incomplete evidence as diagnostics, not as
   functionality proven merely because pytest exited successfully.

## Coverage limits

- These tests do not measure real model quality, benchmark accuracy, or live
  provider authentication/API compatibility. Scripted LMs and local protocol
  servers make the relevant software contracts deterministic.
- The backend matrix does not assert uniform capabilities. Direct serial
  callbacks, partial-error-output differences, and deferred-submit support are
  explicit limitations of the tested interfaces.
- Nested model reconstruction is exercised for sandbox-global model definitions;
  this is not a guarantee of reconstruction for arbitrary function-local classes.
- The successful portable SyncedFile roundtrip uses an **owned** PredictRLM JSPI
  session. It is not proof that every injected legacy interpreter has the same
  reentrant artifact-transfer behavior.
- Workspace/path and credential-redaction tests cover specified safety
  boundaries; they are not a comprehensive sandbox security audit or proof that
  every possible secret representation is redacted.
- Timeout and overlap assertions defend boundedness/ownership, not throughput
  targets. This suite is not a performance benchmark.
- Example source/default/prompt checks prove strings or configuration shape, not
  installation success, semantic instruction-following, or a live benchmark run.
- External service/image coverage is absent from a run unless its prerequisites
  are enabled. Tests named for a platform-specific path pattern do not establish
  native execution on that platform; current CI runs on Ubuntu.

## Maintaining the suite and this inventory

### Where to add a test

1. Find the owning feature in the cross-cutting map and existing module inventory.
2. For shared execution behavior, extend `tests/runtime_contracts/` rather than
   copying the scenario into each backend module.
3. Keep backend-specific process, protocol, cancellation, or policy failures in
   their owning backend module. Declare actual capability differences explicitly.
4. Keep example harness/config/scoring behavior under the relevant example.
5. Add a permanent case only when a plausible regression would violate an
   observable contract. Avoid export/default/docstring/source-text assertions,
   mock echoes, and multiple variants exercising the same branch.
6. Preserve genuinely distinct failure transitions: cleanup, primary-error
   precedence, data loss, account isolation, and terminal stream completion are
   not interchangeable happy-path checks.
7. Bound waits and make teardown own every process, server, thread, lease, or
   temporary artifact created by the scenario. Do not inherit a collected test
   class just to reuse its fixtures and accidentally execute all its tests again.

### Updating and checking this document

When adding, removing, moving, or remarking tests, update the relevant table row,
feature map, prerequisites, and collection snapshot. Do not add tests that pin
this Markdown or require its counts to remain constant.

```bash
# Recompute the default suite's node IDs and count.
uv run --all-extras pytest --collect-only -q -o addopts=

# Inspect an exact CI-like selection without running it.
uv run --all-extras pytest --collect-only -q -o addopts= \
  -m 'integration and not sbx and not local'

# Include skip reasons when executing a selected suite.
uv run --all-extras pytest tests/runtime_contracts -q -rs

# Run the repository's Python lint check.
uv run ruff check src/ tests/
```

For long-running local benchmark/evaluation work beyond these regression tests,
follow [the local-run runbook](runbooks/long-running-local-runs.md). For execution
ownership and backend boundaries, consult [the architecture](../ARCHITECTURE.md);
for custom inputs and sessions, see [custom adapters](custom-adapters.md) and
[custom path inputs](custom-path-inputs.md).
