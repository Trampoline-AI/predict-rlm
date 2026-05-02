# RLM-GEPA Proposer Regression Postmortem

## Bottom line

The strongest evidence points to the May 2 **generic proposer SOP over-expansion** as the likely damaging change. The latest run's normal proposer traces visibly adopted the new retained-rule/audit shape (`trigger/action/boundary/preservation/verification`) on every material audit line, used slightly more helper calls, and produced broader early instruction edits. The patch-merge schema/audit changes are less directly implicated: the latest run had only one patch-merge attempt, it produced a narrow +424 char patch, and GEPA rejected it on subsample, so it did not become a surviving candidate.

Uncertainty: the latest run appears incomplete or interrupted: it has no `optimization_summary.json`, only 5 full validation evals / 1,360 metric calls, versus 19 full evals / 7,288 metric calls in the reference. Score comparison is therefore directionally useful, not a controlled final-run comparison.

| Metric | Previous/reference `20260501_183613` | Latest/suspect `20260502_103433` |
|---|---:|---:|
| Same optimize command/config | yes | yes |
| Completed summary artifact | yes | missing |
| Candidates in `gepa_state.bin` | 19 | 5 |
| GEPA iterations stored | 29 | 5 |
| Total metric calls | 7,288 | 1,360 |
| Full validation evals | 19 | 5 |
| Normal proposer traces | 26 | 6 |
| Patch-merge attempts | 11 | 1 |
| Best full-val mean | 0.7434, cand 17 | 0.7102, cand 4 |
| Best hard-pass count | 58, cand 10 | 50, cand 4 |
| Best-mean instruction length | 31,541 chars | 14,759 chars |
| Total cost logged | $342.94 | $67.77 |

## What changed

The normal proposer behavior changed in exactly the shape expected from `.hermes/codex-prompts/surgical_generic_proposer_sop_helper_role_20260502.txt`:

- Previous generic proposer audit lines were mostly `grounding/use_case/principle/counterfactual_1/counterfactual_2`; across 315 audit lines, `trigger=`, `action=`, `preservation=`, and `verification=` appeared 0 times.
- Latest generic proposer audit lines all carried the expanded retained-rule shape: across 62 audit lines, `trigger=`, `action=`, `preservation=`, and `verification=` each appeared 62 times; `boundary=` appeared 66 times.
- Latest audit lines were longer: average 939 chars versus 629 chars previously.
- Helper use increased, but not explosively: normal proposer attempts averaged 3.33 `predict()` calls in latest versus 2.50 previously. One latest attempt used 7 helper calls.
- The latest helpers were explicitly asked to return the new shape: helper instructions included “trigger, action, boundary, preservation, verification.” This looks like the generic SOP change propagating into helper briefs, not an independent helper-count problem.

Patch merge also changed structurally:

- Previous patch outputs used `imported_from_other`; latest used `behavioral_rules` plus `patch_merge_audit`.
- Latest patch merge selected one bounded capability, “Label-aligned array and matrix fills,” with supported/unsupported source-win IDs and guardrail fields.
- That patch was **subsample rejected** and did not become a candidate, so it is not direct evidence for the score drop.

## Why it likely failed

Direct evidence: the latest normal proposer overfit the expanded process contract. It spent output budget proving every retained/modified/new rule in a trigger/action/boundary/preservation/verification frame and then reflected that framing into helper prompts and final edit decisions. That is visible in traces before accepted candidates were evaluated.

The damage pattern was not mainly “helper takeover.” Helpers were used as evidence workers in both runs, and latest helper counts were only modestly higher. The failure mode looks more like **outer proposer over-constraining and over-appending based on helper-shaped outputs**: the outer proposer still owned the final splice, but the SOP made it treat many lessons as fully shaped retained rules requiring explicit boundaries and verification language.

Score artifacts showing damage or weak search quality:

- Latest best full-val mean was 0.7102, below previous best 0.7434.
- Latest best hard-pass count was 50; previous best-hard candidate reached 58 and previous best-mean candidate had 56.
- Latest candidate 2 was accepted on minibatch but regressed on full val versus candidate 1: 0.6775 -> 0.6532, while adding +3,778 chars.
- Comparing latest best cand 4 to previous best cand 17 on the 102 validation indices: 10 previous hard passes became non-passes, while 4 latest-only hard passes appeared.
- Latest normal proposer average instruction delta was larger: +3,909 chars per attempt versus +3,023 previously, despite the run being much shorter.

The final latest instructions were not longer than the previous best; in fact they were shorter. So “bloat” here is less about final absolute length and more about redundant process pressure: broad rule families, more structured audit obligations, and many boundary/preservation clauses generated earlier and more uniformly than in the reference.

## What to keep

Keep the optimized baseline ideas that survived the rollback:

- The ~10K instruction budget should remain **soft**, not a hard cap. Growth is acceptable when a trace-backed rule is high value; compression is useful only when it removes redundancy without deleting operational specificity.
- Keep suggested helper `predict()` use over bounded evidence packets, especially concurrent cluster analysis with `asyncio.gather(...)`.
- Keep the division of labor where helpers extract latent causes, exact spans, reusable know-how, and local wording alternatives, while the outer proposer owns final selection, deduplication, preservation of strong parent behavior, and the final splice.
- Keep patch-merge’s one-capability framing and support filtering for now. In the latest run it produced a narrow rejected patch, not an adopted harmful candidate.

## What not to reintroduce

Do not reintroduce these generic proposer prompt patterns:

- Requiring every retained or proposed rule to have the full `trigger/action/boundary/preservation/verification` shape.
- Expanding `generalization_check` from concise audit lines into full structured mini-specs for every kept/modified/new/removed rule.
- Prompting helpers to return fully shaped rule objects for every cluster, then asking the outer proposer to convert those into many retained-rule clauses.
- Turning “use helpers on bounded packets” into an evidence-to-diff pipeline that pressures the proposer to draft/splice broad rule families even when the right move is no edit, removal, or compression.
- Treating provenance and semantic dedup as more audit text rather than as a filter that reduces final instruction surface.

## Recommended next action

No immediate source edit is recommended beyond the generic rollback already done. The next small step should be an A/B validation run with the rolled-back generic SOP and the current patch-merge schema left in place, using the same command/seed, then compare the first 5-6 proposer attempts and full-val candidates against the suspect run.

If another prompt edit is needed after that, make it only a generic-proposer guardrail: keep `trigger/action/boundary/preservation/verification` as optional internal thinking for genuinely new or inverted rules, not a required audit/output shape for every material rule decision. Patch-merge schema changes should not be rolled back unless a run shows an accepted patch-merge candidate causing measurable regression.
