# Contributor Rules

- PredictRLM is for callable, repeatable, deep-context workflows, not open-ended
  interactive chat flows.
- Keep large inputs as `File` references or metadata. Use focused `predict()`
  calls and keep LLM-facing Pydantic schemas lean with `Field(description=...)`.
- Validate at system boundaries. Let library validation raise when schema fields
  are required; do not add silent fallbacks.
- Keep generic runtime behavior domain-neutral. Domain or benchmark specifics
  belong in examples, `AgentSpec`, seed/domain skills, runtime-grounding
  examples, or evaluator feedback.
- Persist experimental behavior in config, CLI options, or artifacts rather than
  hidden env-only switches.
- Use Conventional Commits. The allowed scopes are `rlm-gepa`, `predict-rlm`,
  and `examples/[example-name]`.
- PR descriptions must start with **Rationale**, followed by Summary and Test
  Plan.

## Skill Guidance Changes

Keep each repo skill focused on one job. Use short trigger descriptions with
clear boundaries. Put detailed API and workflow material in one-level
`references/` files linked from `SKILL.md`.

Do not put downstream RLM-building guidance and repository-contributor guidance
in the same `SKILL.md`.
