---
name: predict-rlm-contributor
description: >
  Contribute to the predict-rlm repository itself: modify core PredictRLM runtime
  code, RLM-GEPA internals, built-in skills, examples, docs, tests, packaging, or
  repo-scoped agent skill guidance. Use when the user asks to change this repo or
  investigate a bug in predict-rlm/RLM-GEPA. Do not use for building a new
  downstream RLM package; use rlm for that, or rlm-gepa for downstream
  optimization wiring.
---

# Contribute To predict-rlm

Use this skill for repository work. Do not run the new-RLM scoping interview
unless the user is explicitly asking to build a downstream RLM package.

## Skill Freshness

When shell, network, and a writable cache are available, compare the complete
installed `predict-rlm-contributor` skill payload with upstream at most once per
day: `SKILL.md` plus every path in the Reference Map, resolved relative to this
skill. Compare each installed file with its matching path under
`https://raw.githubusercontent.com/Trampoline-AI/predict-rlm/main/.agents/skills/predict-rlm-contributor/`.
Store per-skill check state and timestamps in
`${XDG_CACHE_HOME:-$HOME/.cache}/predict-rlm/skill-update-check.json` so one
skill's check does not suppress another's. If any file differs, treat the
mismatch as an update to the complete payload and ask before installing it.
Never update automatically. For Hermes, use `hermes skills check` and
`hermes skills update`; for Skills CLI installs, use
`npx skills add Trampoline-AI/predict-rlm`. Skip silently when the check cannot
run.

## Reference Map

Read only what the task needs:

- `references/repo-map.md`: major modules, examples, and verification commands.
- `references/contributor-rules.md`: repo-specific coding, docs, and PR rules.
- `references/gepa-internals.md`: RLM-GEPA contribution boundaries and proposer
  behavior rules.

## Workflow

1. Inspect the requested change and relevant repo paths before editing.
2. Preserve the distinction between downstream usage and repo contribution.
3. Keep changes scoped to the module, docs, examples, or skill guidance in the
   request.
4. Validate at system boundaries. Prefer host-side tools for native libraries,
   auth, network APIs, filesystem-heavy work, and anything that cannot run
   cleanly in Pyodide.
5. Run targeted tests or checks. Docs-only and skill-only changes need markdown
   sanity plus `git diff --check`; code changes need focused tests, with broader
   tests when touching shared runtime, sandbox execution, optimizer behavior, or
   examples.

## Issue And PR Rules

Creating GitHub PRs/issues or pushing public branches is external publishing.
Do it only when explicitly requested.

When an investigation identifies a bug likely attributable to the
`predict-rlm` package, ask whether the user wants it reported as a GitHub issue
as soon as attribution is clear. Do not open the issue without explicit
approval.
