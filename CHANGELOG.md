# Changelog

All notable changes to `predict-rlm` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **BREAKING:** `Skill` (bundles of instructions, packages, modules, and tools)
  is renamed to `Capability`, and `PredictRLM(skills=...)` for those bundles is
  now `PredictRLM(capabilities=...)`. The built-in `pdf`, `spreadsheet`, and
  `docx` bundles moved from `predict_rlm.skills` to `predict_rlm.capabilities`.
  Prompt sections are headed `## Capabilities` / `## Capability: <name>`.
  `predict_rlm.rlm_skills` is removed; import `Capability` and
  `merge_capabilities` from `predict_rlm.capabilities`. There is no shim:
  constructing the new `Skill` with `packages`, `modules`, or `tools` raises
  `TypeError` pointing to `Capability`.

  | Before                                    | After                                           |
  | ----------------------------------------- | ----------------------------------------------- |
  | `from predict_rlm import Skill`           | `from predict_rlm import Capability`            |
  | `Skill(name=..., instructions=..., ...)`  | `Capability(name=..., instructions=..., ...)`   |
  | `PredictRLM(..., skills=[...])`           | `PredictRLM(..., capabilities=[...])`           |
  | `from predict_rlm.skills import pdf`      | `from predict_rlm.capabilities import pdf`      |
  | `from predict_rlm.rlm_skills import ...`  | `from predict_rlm.capabilities import ...`      |

### Added

- `Skill` now follows the [Agent Skills](https://agentskills.io) format. Load a
  `SKILL.md` folder with `Skill.from_dir(path)` and pass it with
  `PredictRLM(skills=[...])`. Only each skill's name and description go in the
  prompt; the RLM calls the built-in `load_skill(name)` tool for the
  instructions and file list, and skill folders are mounted at
  `/sandbox/skills/<name>/`.
- `pyyaml` is now a direct dependency (used to parse `SKILL.md` frontmatter).

## [0.13.2] - 2026-10-30

### Fixed

- `DirectPythonBackend` (and other Python-supervisor backends) now run concurrent
  host tool calls in parallel, so `asyncio.gather(predict(...), ...)` takes about
  one call's time instead of the sum. The supervisor shares the SBX backend's
  dispatch: each call runs in its own worker and is answered when it finishes.
- A burst of three or more small concurrent tool calls no longer stalls a Direct
  execute until its request deadline.
- Direct host tools run in the caller's context, and async tools on the event loop
  that called `aexecute()`. `predict()` therefore uses the RLM's adapter (one LM
  call instead of a ChatAdapter attempt plus a JSON retry), and evidence and
  telemetry context reach the tool. Pending async tools are cancelled when an
  iteration times out or the execute is cancelled, as on JSPI and SBX.

### Changed

- Direct host tools no longer run under a per-call deadline thread. With no
  execution timeout, a tool still running at the request deadline ends the
  session with `SandboxFatalError`, as on SBX. In 0.13.1 the per-call error
  reached the sandbox only 50 ms before that deadline, too late to answer, so
  such runs already ended the same way. Pass an execution timeout to keep
  REPL state when a tool hangs.

## [0.13.1] - 2026-10-23

### Fixed

- File output directory creation through the Python supervisor no longer requires
  `mkdir` on PATH. Directory operations run with the configured Python executable
  inside the adapter boundary, including hardened DirectPython containers.
- Supervisor descendant cleanup and SBX startup, health checks, and timeout cleanup
  no longer require `ps`, `env`, `true`, or `pkill`; they use native process APIs
  and in-sandbox Python instead.

## [0.13.0] - 2026-10-23

### Added

- Opt-in read-only File input mounts with `FileInputAdapter(mode="mount")` or
  `compatibility.files(input_mode="mount")`. Inputs use live host-path mounts
  without copy artifacts; unsupported backends fail instead of copying. The
  default remains `"copy"`.
- `Workspace(read_only=True)` requests read-only direct-mode host mounts; existing
  writable direct-mode and mirror-mode defaults are unchanged.

### Fixed

- DirectPython File outputs expose the reserved host directory under
  `output_dir/<field>/` and accept both host and virtual submission paths.
  Files already in place are returned without copying; paths escaping the
  reserved field directory, including symlink escapes, are rejected.
- File input mount mode and direct Workspace mounts now work on
  `DirectPythonBackend` via host-path passthrough, without copies or sandbox-path
  aliases. Read-only is advisory on DirectPython; the host must provide a
  read-only path (for example, a Delta Mesa snapshot) to enforce protection.
  Host permissions and default copy behavior are unchanged. JSPI, pooled SBX,
  and named/reused SBX still reject mount mode.
- File lists containing nulls now reject duplicate basenames, just like other
  File lists, instead of silently overwriting an input in the sandbox.

### Documentation

- Documented per-step/attempt File output collection using the existing
  `PredictRLM(output_dir=...)` and `FileOutputAdapter(output_dir=...)` APIs,
  mount backend support, and direct versus mirror Workspace policy.

## [0.12.0] - 2026-10-23

### Changed

- The action signature's output field is now `plan` ("Your next action and why, in
  one or two sentences.") instead of `reasoning` ("Think step-by-step…"). A custom
  `generate_action` predictor or mock must return `plan`; returning only `reasoning`
  now fails validation. Trace steps, REPL history, `SubmitConfirmationContext`, and
  telemetry still call it `reasoning`.

### Fixed

- Claude Sonnet 5.5 no longer refuses every action call (`stop_reason: "refusal"`,
  category `reasoning_extraction`, surfaced as "The LM returned an empty or null
  response."). It refused requests for a `reasoning` output field and for long
  step-by-step reasoning (#69).

## [0.11.0] - 2026-10-23

### Changed

- The action, extract, and `predict()` calls now default to `ValidatingJSONAdapter`
  instead of `ValidatingChatAdapter`. Chat-format replies that failed to parse were
  silently re-sent in full through JSON, often on 20–35% of main-model calls. Pass
  `adapter=ValidatingChatAdapter()` to restore the previous format.

### Added

- `PredictRLM(adapter=...)` selects the DSPy adapter; `ValidatingJSONAdapter` and
  `ValidatingChatAdapter` are exported from `predict_rlm`.
- `IterationStep.adapter_fallbacks` and `RunTrace.adapter_fallbacks` count
  ChatAdapter → JSON re-sends. Each re-send also logs a warning and is recorded on
  the `rlm.action_generation.*` telemetry spans.

### Fixed

- `ValidatingChatAdapter` only falls back to JSON on parse failures; rate limits,
  timeouts, and other LM errors are no longer re-sent through a second request.
  The fallback keeps the native function-calling settings and DSPy's streaming
  guard.
- `ValidatingChatAdapter` parses replies whose `[[ ## field ## ]]` markers are inline
  or whose code marker is missing before a fenced code block, instead of re-sending.

## [0.10.0] - 2026-10-21

### Security

- Removed the direct LiteLLM dependency after the compromised `litellm` 1.107.0
  PyPI release. The `codex-lm` extra no longer requires `litellm`, and the
  project's own uv resolution excludes it via `[tool.uv] exclude-dependencies`.
  DSPy still declares `litellm` for pip installs on Python < 3.15; uv consumers
  should add `exclude-dependencies = ["litellm"]` to their root project.

### Changed

- Upgraded DSPy to `>=3.4.1,<3.5` and use its native `lm15` engines.
- `dspy_codex_lm` now streams the Codex Responses API through the OpenAI SDK
  instead of `litellm.responses`, and prices calls from a local cost table.
- Context-overflow and rate-limit handling use `dspy.ContextWindowExceededError`
  and `dspy.LMRateLimitError`.
- Routes without native DSPy support (for example `bedrock/`, `vertex_ai/`,
  `huggingface/`, `together_ai/`) require installing `dspy[litellm]` or declaring a
  native provider. The legacy `thinking_budget` option is rejected.

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

[Unreleased]: https://github.com/Trampoline-AI/predict-rlm/compare/v0.13.2...HEAD
[0.13.2]: https://github.com/Trampoline-AI/predict-rlm/compare/v0.13.1...v0.13.2
[0.13.1]: https://github.com/Trampoline-AI/predict-rlm/compare/v0.13.0...v0.13.1
[0.13.0]: https://github.com/Trampoline-AI/predict-rlm/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/Trampoline-AI/predict-rlm/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/Trampoline-AI/predict-rlm/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/Trampoline-AI/predict-rlm/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/Trampoline-AI/predict-rlm/compare/v0.8.1...v0.9.0
[0.8.1]: https://github.com/Trampoline-AI/predict-rlm/compare/v0.8.0...v0.8.1
