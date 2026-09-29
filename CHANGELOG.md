# Changelog

All notable changes to `predict-rlm` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking Changes

- `RunTrace` and its proposer/exported forms no longer contain `evidence`.
  Successful calls return separate `prediction.trace` and `prediction.evidence`
  objects; failures and cancellation expose `exception.evidence` independently
  of any `exception.trace`, including failures before a trace exists.
- Import `RunEvidence` and `RunEvidenceEvent` from `predict_rlm` or
  `predict_rlm.evidence`, not `predict_rlm.trace`. Export the lifecycle log with
  `prediction.evidence.to_exportable_json()` and read its `run_id`, `complete`,
  and `terminal_outcome` directly. Event sink payloads and delivery are unchanged.
- The signature output name `evidence` is reserved for the runtime evidence
  object. Rename application output fields that used this name.
- GEPA evaluation results carry lifecycle logs in a separate
  `RLMGepaExampleResult.evidence` list. Evaluators must pass captured evidence
  alongside traces to preserve completeness validation and lifecycle records.
  Consumers such as Avalanche and Delta must obtain evidence metadata from the
  separate result or exception object rather than `trace.evidence`.

## [0.8.1] - 2026-09-25

### Changed

- Reduced the package regression suite to behavioral contracts; consolidated
  shared backend execution tests and removed duplicate inherited cases,
  static/presentation checks, and one-off benchmark/example checks.
- Direct CPython contracts now run in the core tier without the SBX extra.

### Fixed

- JSPI timeout cleanup now waits for the interrupt worker to disarm before
  entering Python, and preserves the timeout response if tracing interrupts
  cleanup instead of leaving the host waiting for an uncorrelated error.
- JSPI deadlines no longer interrupt asyncio scheduler callbacks; suspended
  executions are cancelled without orphaning their result promises, and the
  previous Python SIGINT handler is restored after bounded execution.
- JSPI deadlines interrupt CPU-bound child coroutines defined by `exec` or
  imported modules without killing the sandbox or losing execution state.
- Workspace atomicity regression coverage now performs real host writes, so it
  detects a transfer attempted before all conflicts have been checked.

### Breaking Changes

- `SbxBackend.shutdown()` can no longer be called from the event loop that owns
  an active asynchronous SBX transport. Async callers must use
  `await SbxBackend.ashutdown()` instead. Synchronous callers outside the owning
  event loop can continue using `SbxBackend.shutdown()`. This change accompanies
  the move to native asynchronous SBX execution and prevents synchronous
  shutdown from blocking its own event loop.

[Unreleased]: https://github.com/Trampoline-AI/predict-rlm/compare/v0.8.1...HEAD
[0.8.1]: https://github.com/Trampoline-AI/predict-rlm/compare/v0.8.0...v0.8.1
