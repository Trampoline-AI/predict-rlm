# RLM-GEPA Internals

Use these rules when changing `src/rlm_gepa/`, tests, examples, or docs.

- Treat `AgentSpec`, evaluator feedback, and seed instructions as the
  optimization direction. Keep runtime and budget knobs separate.
- Derive signature and tool context from the constructed RLM with
  `agent_spec_from_rlm(...)` where possible.
- Avoid duplicating broad prose or exposing internal IDs unnecessarily.
- Keep generic proposer behavior domain-neutral. Domain or benchmark specifics
  belong in `AgentSpec`, seed/domain skills, runtime-grounding examples, or
  evaluator feedback.
- Patch-merge/crossover should be evidence-backed behavioral grafting from train
  disagreement traces, not broad synthesis, prompt concatenation, or source text
  import.
- GEPA project wiring should live in downstream `gepa/` packages. Generic
  optimizer orchestration belongs in `src/rlm_gepa/`.

For verification, run targeted RLM-GEPA tests when touching optimizer schemas,
runtime adapters, proposer behavior, reporting, or SpreadBench GEPA wiring.
