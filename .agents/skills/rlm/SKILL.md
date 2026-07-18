---
name: rlm
description: >
  Plan, design, and build callable PredictRLM/RLM packages with typed inputs,
  structured outputs, skills, host-side tools, and smoke tests. Use when the user
  wants to create a new RLM, assess whether a workflow is a good RLM fit, or add
  normal PredictRLM usage code. Do not use for contributing to the predict-rlm
  repository itself or for RLM-GEPA optimization wiring; use
  predict-rlm-contributor or rlm-gepa for those tasks.
---

# Build An RLM

An RLM is a callable, pre-configured agent. It autonomously explores context,
writes and executes code in a sandboxed Python REPL, calls tools, inspects
results, and iterates until the task is done. Unlike a chat agent, an RLM is a
function: define its inputs, outputs, and tools, then call it from code. It
returns structured data, not chat messages.

Use this skill for new PredictRLM packages and application code. If the user is
modifying the `predict-rlm` repo, switch to `predict-rlm-contributor`. If the
user asks for GEPA, optimization, train/validation candidate selection, or
`AgentSpec` wiring, switch to `rlm-gepa`.

## Skill Freshness

When shell, network, and a writable cache are available, check at most once per
day whether this file differs from
`https://raw.githubusercontent.com/Trampoline-AI/predict-rlm/main/.agents/skills/rlm/SKILL.md`.
Store the last-check marker under
`${XDG_CACHE_HOME:-$HOME/.cache}/predict-rlm/skill-update-check.json`. If an
update exists, ask before installing it. Never update automatically. For Hermes,
use `hermes skills check` and `hermes skills update`; for Skills CLI installs,
use `npx skills add Trampoline-AI/predict-rlm`. Skip silently when the check
cannot run.

## Reference Map

Read only what the task needs:

- `references/project-layout.md`: generated package layout, smoke tests, and
  service wiring patterns.
- `references/predict-rlm-api.md`: `PredictRLM`, `File`, `Skill`, built-in
  skills, tools, `predict()`, and CodexLM usage.
- `references/sandbox-and-research.md`: feasibility research, Pyodide package
  compatibility, network allowlists, and host-side tool decisions.

## Workflow

### 1. Define The Goal

Ask what success looks like, what input material the RLM receives, and what
structured output or generated files it should return.

Validate RLM fit. An RLM is appropriate when the task needs selective
exploration of large inputs, multi-step tool use, stateful file transformations,
parallel sub-LM calls, or repeated callable workflows. If a simple script or
single LLM call is the better tool, say so and suggest that path.

### 2. Design Inputs

Define every input:

- name and type: `File`, `list[File]`, `str`, or a Pydantic model;
- description: what it contains and how the RLM uses it;
- source: user-provided file, API response, config, or generated data.

Use `File` references for large content such as PDFs, images, workbooks,
documents, datasets, audio, or video. Keep raw bulk content out of the LLM
schema.

### 3. Design Outputs

Define every output field with name, type, and description. Use Pydantic models
with `Field(description=...)` for structured outputs. Use `File` output fields
when the RLM must write artifacts back to the host.

Push vague outputs into concrete schemas before implementation. Ask what fields
the caller will inspect first, which derived fields matter, and whether generated
files are required.

### 4. Research Feasibility

Do autonomous research before writing the plan. Read
`references/sandbox-and-research.md` when package compatibility, network access,
or host-side tools are relevant.

Report feasibility clearly: package support, sandbox constraints,
`allowed_domains`, required host tools, likely iteration count, and blockers.

### 5. Design Skills And Tools

Choose built-in skills or custom `Skill(...)` definitions. Use built-in skills
for common document domains:

- `pdf`: PDF reading, rendering, manipulation, and redaction.
- `spreadsheet`: Excel workbook editing, formulas, and verification.
- `docx`: Word document reading, writing, tables, formatting, and styles.

Create custom skills only when the RLM needs reusable domain instructions,
sandbox packages, mounted modules, or bundled host-side tools.

Use host-side tools for authenticated APIs, database calls, native binaries,
heavy filesystem work, or anything that cannot run cleanly in the sandbox.

### 6. Choose Architecture

Write the signature docstring as the RLM playbook:

1. how to survey the inputs;
2. how to gather information with files, skills, tools, and `predict()`;
3. how to process and verify results;
4. what to return or save.

Use a single RLM when the task is one coherent workflow and can stay within a
reasonable iteration budget. Use chained RLMs when phases have different inputs,
skills, outputs, or budgets.

### 7. Confirm Delivery Scope

Default to an agent-only package unless the user asks for more:

- **Agent only**: callable RLM package, domain skills/tools, and fast smoke
  tests.
- **Agent + evals**: add project-local dataset loading and scoring when the user
  has examples, fixtures, labels, or deterministic metrics.

For GEPA optimization or candidate selection, stop using this skill and switch
to `rlm-gepa`.

### 8. Write The Plan

Produce a plan with:

1. Overview.
2. Delivery scope.
3. File manifest.
4. Input schemas.
5. Output schemas.
6. Signature code and strategy docstring.
7. Skills and host-side tools.
8. Service architecture or chained RLM DAG.
9. Feasibility notes.
10. Estimated complexity: iterations, sub-LM calls, runtime, and cost range.
11. Smoke tests and commands.

Get approval before building when the environment or calling surface has an
explicit plan mode. Otherwise proceed when the user has already asked for
implementation.

### 9. Build And Verify

Implement the approved plan using the package structure in
`references/project-layout.md`. Every generated RLM must include fast
no-network smoke tests that import the package, inspect the signature fields,
and construct the service without making LLM calls.
