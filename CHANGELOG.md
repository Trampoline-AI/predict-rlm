# Changelog

All notable changes to `predict-rlm` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Reduced the package regression suite to behavioral contracts; consolidated
  shared backend execution tests and removed duplicate inherited cases,
  static/presentation checks, and one-off benchmark/example checks.
- Direct CPython contracts now run in the core tier without the SBX extra.

### Fixed

- JSPI timeout cleanup now waits for the interrupt worker to disarm before
  entering Python, and preserves the timeout response if tracing interrupts
  cleanup instead of leaving the host waiting for an uncorrelated error.

### Breaking Changes

- `SbxBackend.shutdown()` can no longer be called from the event loop that owns
  an active asynchronous SBX transport. Async callers must use
  `await SbxBackend.ashutdown()` instead. Synchronous callers outside the owning
  event loop can continue using `SbxBackend.shutdown()`. This change accompanies
  the move to native asynchronous SBX execution and prevents synchronous
  shutdown from blocking its own event loop.

[Unreleased]: https://github.com/Trampoline-AI/predict-rlm/compare/v0.7.2...HEAD
