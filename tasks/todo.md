# PredictRLM Small-Kernel Refactor

## Adapter Contract Cleanup

- [x] Reproduce the missing normalized adapter contract with focused failing tests.
- [x] Add a named field descriptor that normalizes `Annotated`, optional, and list shapes once.
- [x] Replace structural adapter protocols with explicit typed adapter base classes.
- [x] Move File and Workspace adapters onto the typed contract.
- [x] Make output pre-session preparation a declared lifecycle method.
- [x] Remove duplicate annotation parsing from maintained adapter paths.
- [x] Run focused tests, the full unit suite, integration coverage, Ruff, and final review.

### Review

- Focused adapter/kernel/File/Workspace unit slice: 125 passed, 11 deselected.
- `make test-unit`: 1114 passed, 28 skipped, 149 deselected.
- Live File/Workspace integration slice: 11 passed, 84 deselected.
- `uv run ruff check src/ tests/`: passed.
- `git diff --check`: passed.
- Adapter field names and normalized type shapes no longer travel through mutable context state.
- Output destination values are consumed before sandbox variable injection, and output mounts start
  empty so files from a prior run are not exposed to the model.

## Async-Native Migration Correction

- [x] Audit File and Workspace paths for legacy small-kernel bypasses.
- [x] Audit sync-native implementations adapted with `asyncio.to_thread()`.
- [x] Make compatibility adapters the sole planners for File and Workspace inputs/outputs.
- [x] Remove `file_plan` and fabricated compatibility bindings from the async kernel path.
- [x] Execute generated code only through `ExecutionSession.run_code()`.
- [x] Implement File and Workspace mount, sync-back, conflict, and output collection through
  session artifact operations for owned, injected, and pooled lifecycles.
- [x] Give maintained owned backends async-native session implementations; retain a sync bridge
  only for explicitly caller-supplied legacy interpreters.
- [x] Add red/green tests for File input/output and Workspace mirror/direct behavior through the
  final session lifecycle, including failure and cancellation paths.
- [x] Re-run focused tests, the full unit suite, Ruff, and final architecture review.

- [x] Read the operating plan and supporting architecture.
- [x] Inventory current constructor, execution, backend, artifact, Workspace, and trace paths.
- [x] Add final immutable runtime contracts and deterministic contribution normalization.
- [x] Add fresh per-invocation run context and strict evidence lifecycle.
- [x] Add async execution backend/session contracts and legacy interpreter, owned, injected, and pool adapters.
- [x] Add opaque file/directory artifact bindings and input/output adapter contracts.
- [x] Compile current files, Workspace, skills, tools, and backend options into compatibility contributions.
- [x] Route async PredictRLM execution through the kernel lifecycle while retaining legacy parity oracles.
- [x] Preserve the sync API as a wrapper over the async implementation.
- [x] Add focused contract, adapter, evidence, artifact, and concurrency tests.
- [x] Run focused unit tests, the full unit suite, and Ruff.
- [x] Review the final diff for public compatibility and document verification below.

## Constraints

- Preserve current `predict_rlm.Workspace` mirror/direct behavior as input-only compatibility.
- Keep artifact/session contracts directory-capable and provider-agnostic.
- Do not add `ava.Workspace`, durable snapshot protocols, Delta, lakeFS, JuiceFS, or S3 Files.
- Do not modify upstream DSPy or depend on upstream acceptance.
- Do not remove a legacy parity path before its replacement passes.
- Maintained JSPI, SBX, pool, and kernel paths must be async-native. Thread bridges are allowed
  only for caller-supplied synchronous tools and interpreters.

## Review

- Previous review was reopened after discovering that File/Workspace and maintained backend
  execution still bypassed the final async session contract.
- `uv run pytest tests/test_small_kernel.py -q`: 23 passed.
- `make test-unit`: 1097 passed, 28 skipped, 149 deselected.
- `make test-integration`: 137 passed, 12 skipped, 1124 deselected.
- Live File/Workspace integration slice after final lifecycle fixes: 11 passed.
- `uv run ruff check src/ tests/`: passed.
- `git diff --check`: passed.
- Final targeted review found no regressions after restoring the legacy backend import contract
  and fixing cancellation, context-manager, evidence, and adapter-resolution lifecycles.
- Maintained JSPI, websocket SBX, and normal async pools use native async control paths. Thread
  bridges remain only for caller-supplied synchronous tools/interpreters and the private legacy
  `_supervisor_command` test transport.
- Residual risk: regular-file traversal/copy uses cooperative chunking because Python has no
  native async filesystem API; real Docker SBX integration remains environment-gated.
- No upstream DSPy files, durable Workspace provider protocols, commits, or pushes were added.

## Accepted Review Repair — 2026-07-13

- [x] Rebuild baseline action/extract predictors from finalized module contributions.
- [x] Make strict evidence transactional, cancellation-paired, release-aware, and visible to
  `RunTrace`/RLM-GEPA.
- [x] Reject unsupported File/Workspace annotations and Workspace outputs before backend startup.
- [x] Prevent injected direct-Workspace state from leaking or being pinned by failed acquisition.
- [x] Skip JSPI workspace post-hooks after cancellation/fatal failure and preserve primary errors.
- [x] Make synchronous and asynchronous SBX pool replacement transactional and fail waiting leases.
- [x] Hold session leases and SyncedFile temporary storage until cancelled sync workers finish.
- [x] Implement SyncedFile as a portable tool operation for custom final backends while preserving
  direct legacy-backend compatibility.
- [x] Select maintained JSPI/SBX/pooled implementations through the final backend/session seam.

### Second Repair Pass

- [x] Enforce JSPI sync-tool deadlines while quarantining live workers and preserving sync-leaf
  classification through SyncedFile and evidence wrappers.
- [x] Keep cancelled SBX host tasks and workers quarantined; retire busy owned/pooled backends
  instead of resetting, requeueing, or shutting them down early.
- [x] Make session finalization task-owned and cancellation-safe, and record incomplete evidence
  without replacing a primary failure.
- [x] Project proposer evidence without raw inputs, base64 payloads, accounting fields, or full
  iteration dumps.
- [x] Keep SyncedFile temporary storage alive through cancelled legacy and maintained sync I/O.
- [x] Recursively reject unsupported File/Workspace generic containers and all Workspace outputs.
- [x] Restore scalar and list File output directory fallback with root/traversal validation.

### Third and Final Repair Pass

- [x] Preserve JSPI ownership of cancelled host-tool tasks and synchronous leaves until they
  finish; fence later iterations and cancellation-safe release without changing the 0.1-second
  iteration deadline or the accepted 10-second parent harness grace.
- [x] Replace detached loop-local SBX retirement with cancellation-shielded awaited retirement
  for owned and async-pooled interpreters while retaining loop-independent sync-pool retirement.
- [x] Match JSPI post-execute terminal classification in SBX: skip hooks after cancellation and
  fatal failure, and attach hook failures without replacing the primary execution error.
- [x] Add red-first regressions for cancellation-suppressing async tools, sync-tool quarantine,
  owned release, pooled replacement, cancellation during retirement, and sync-forward loop
  teardown.
- [x] Audit maintained JSPI, local/owned SBX, pooled SBX, injected legacy interpreters, and custom
  final execution backends for ownership and release behavior.

### Final Review Results

- Focused final JSPI/SBX lifecycle regressions: 41 passed, 1 environment-gated skip.
- `make test-unit`: 1194 passed, 28 skipped, 149 deselected.
- `make test-integration`: 137 passed, 12 skipped, 1222 deselected.
- `uv run ruff check src/ tests/`: passed.
- `git diff --check`: passed.
- Real Docker SBX integration remains environment-gated; maintained local websocket SBX lifecycle
  tests and the real JSPI/Deno integration suite passed.
- Durable ava/Delta Workspace storage remains intentionally out of scope.
