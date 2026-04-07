# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workflow Orchestration

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately -- don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 4. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes -- don't over-engineer
- Challenge your own work before presenting it

### 7. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests -- then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

### 8. Test-Driven Bug Fixing
- **Reproduce first**: Before touching any fix, write a test that reproduces the bug
- **Red before green**: The new test MUST fail before you change any production code -- if it passes, your test doesn't actually cover the bug
- **Minimal regression test**: Keep the test focused on the exact failure; don't boil the ocean
- **Fix, then verify**: Only after the test is red, implement the fix and confirm it goes green
- **Never skip this**: No matter how "obvious" the fix seems -- a bug without a test is a bug that will come back
- **Cover edge cases**: If the bug came from an edge case, add tests for adjacent edge cases too

### 9. Pull Request Descriptions
- Every PR description must start with a **Rationale** section explaining *why* the changes are being made (the problem, motivation, or user-facing impact)
- Follow with a Summary of *what* changed
- Assume the reader has no prior context -- don't reference earlier commits or conversations
- Include a Test Plan with checkable items

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.

## Commands

```bash
# Install
uv sync                        # core deps
uv sync --extra examples       # include example deps (pymupdf)

# Test
make test                      # all tests
make test-unit                 # unit only (no Deno required)
make test-integration          # integration only (requires Deno v2)
uv run pytest tests/test_predict_rlm.py::TestPredictTool::test_name -v  # single test

# Lint
uv run ruff check src/ tests/
```

Requires Python 3.11+ and Deno v2 (for integration tests / sandbox).

## Architecture

**predict-rlm** extends DSPy's RLM with a built-in `predict()` tool. It's a two-level system:

1. **Outer LLM** -- writes and executes Python in a Pyodide/WASM sandbox (via Deno). Orchestrates the task across iterations of code -> output -> reasoning -> next code.
2. **Sub-LM** (via `predict()`) -- handles perception and extraction. Each call gets its own context window. Supports `dspy.Image` type hints for multimodal work.

Large inputs are passed as **metadata** (file paths, page counts), not raw content. The RLM accesses content on-demand through tools, keeping its context small.

### Key modules

- **`predict_rlm.py`** -- `PredictRLM` class. Creates the `predict()` tool, builds action/extract signatures, manages LM contexts. The `_create_predict_tool()` method handles signature parsing, Pydantic model reconstruction from JSON schemas, and `dspy.Image` type coercion.
- **`interpreter.py`** -- `JspiInterpreter`. Manages the Deno subprocess running `sandbox/runner.js`. Handles JSPI flag detection, concurrent tool execution via `asyncio.gather()`, package installation, and network permissions.
- **`rlm_skills.py`** -- `Skill` dataclass and `merge_skills()`. Skills bundle instructions + PyPI packages + tools. Merging concatenates instructions, deduplicates packages, and raises on tool name conflicts.
- **`_shared.py`** -- `build_rlm_signatures()` builds the action/extract signatures with full tool docs and skill instructions. `format_tool_docs_full()` formats tool docstrings (unlike DSPy's default truncated format).
- **`sandbox/runner.js`** -- Deno script: Pyodide runtime, JSON-RPC 2.0 tool bridge, concurrent async execution, micropip package installation.

### Example structure

Each example follows: `schema.py` (Pydantic models) -> `signature.py` (DSPy Signature with strategy docstring) -> `tools.py` (host-side callables) -> `skills.py` (optional sandbox packages/instructions) -> `service.py` (DSPy Module wiring PredictRLM) -> `run.py` (CLI entry point).

## Code Style

- Ruff rules: E, F, W, I, N. Line length 96. E501 ignored.
- N806 ignored in tests (allows non-lowercase variable names).
- pytest markers: `integration` for tests requiring the WASM sandbox.

### General
- **No AI slop** - Remove "Note:" comments explaining removed code or changes. Don't add comments that just describe what code does. Don't add docstrings or type annotations to code you didn't modify.
- **Self-documenting code** - Use meaningful names over prefixes. Names should describe purpose, not structure.
- **Trust your libraries** - Don't add defensive checks on top of library validation. If a config field is required, declare it without a default and let pydantic raise `ValidationError`. Declare constraints in the schema, not imperatively.
- **No silent failures** - Never swallow errors with empty catch blocks or guard-away missing dependencies. If something is expected to exist, let it crash so the problem is surfaced immediately. Silent fallbacks hide bugs and waste debugging time.
- **No premature abstractions** - Avoid singletons, factories, and "reset for testing" patterns unless there's actual shared state to manage.
- **Validate at system boundaries** - Add explicit validation for edge cases that pass type checks but produce confusing errors downstream.
- **Document domain concepts, not code mechanics** - Complex domain logic deserves prose documentation. But don't add inline comments that just narrate what the code does.
- **No dead code** - When creating a new component that replaces an existing one, delete the old file. Check for unused exports before committing.
- **No code duplication** - If two components share the same logic, extract the shared logic into a single source of truth. Before adding a new component, check if a near-identical one already exists.

### Pydantic Models
- **Use Field descriptions** - All non-obvious fields should use `Field(description=...)` for self-documentation
- **Separate LLM vs internal models** - Models for LLM I/O should be lean (avoid context rot). Internal models can have richer fields.
- **No defensive programming burden** - If a field is always expected, make it required. Don't use `Optional` when consumers always need the value.

### LLM Integration
- **Context rot awareness** - Don't send unnecessary data to LLMs (e.g., internal IDs the model doesn't need for its task)
