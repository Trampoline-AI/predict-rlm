# Changelog

All notable changes to `predict-rlm` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.0] - 2026-10-02

### Added

- `PredictRLM(run_export_root=...)` saves `trace.json` and `evidence.json` under an
  invocation run ID after success, failure, or cancellation. A run without
  a captured trace writes JSON `null` to `trace.json`.
- All examples, including notebook and benchmark/GEPA paths, use
  `<example>/.run/<run_id>/` for these artifacts. Example `.run` directories are
  ignored by Git; generated documents and aggregate reports keep their existing
  locations.

### Changed

- Run IDs now use filename-safe UTC creation timestamps with nanosecond precision
  (`YYYYMMDDTHHMMSS.nnnnnnnnnZ`) instead of random UUIDs.
- Runnable examples and notebooks now default to Codex LM authentication instead
  of API-key-backed model calls: `gpt-5.6-terra` for all main/executor, GEPA proposer,
  and sub-LM roles. Benchmark CLIs default Codex routing on and retain an explicit
  `--no-codex-lm` opt-out.

### Fixed

- Preserve AppWorld evaluator errors when retaining a completed agent's trace and
  evidence, so GEPA reports harness failures as errors rather than completed runs.
- Publish canonical traces only after successful writes, preserving Terminal-Bench
  live snapshots when artifact export fails.

### Breaking Changes

- `RunTrace` and its proposer/exported forms no longer contain `evidence`.
  Successful calls always return separate `prediction.trace` and `prediction.evidence`
  objects; internal consumers require both. Failures and cancellation expose `exception.evidence` independently
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
- Example artifact readers must use `.run/<run_id>/trace.json` and
  `evidence.json`: invoice processing no longer writes `output/<timestamp>/run_trace.json`,
  and Terminal-Bench no longer writes final `predict_rlm_trace*.json` or
  `predict_rlm_evidence*.json` sidecars in harness logs.

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

[Unreleased]: https://github.com/Trampoline-AI/predict-rlm/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/Trampoline-AI/predict-rlm/compare/v0.8.1...v0.9.0
[0.8.1]: https://github.com/Trampoline-AI/predict-rlm/compare/v0.8.0...v0.8.1
