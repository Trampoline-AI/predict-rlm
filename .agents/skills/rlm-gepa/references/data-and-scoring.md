# Data And Scoring

Investigate the dataset before writing split or scoring code. Do not treat it
as an opaque list of rows.

Inspect enough examples to identify:

- task types and input sizes;
- label or reference-output shape;
- duplicate or near-duplicate examples;
- missing labels or ambiguous references;
- source grouping keys such as document, user, customer, or task family;
- failure buckets the scorer should expose.

## Split Semantics

Use split names consistently:

- **Train**: examples the optimizer/proposer may use to generate and gate edits.
- **Validation**: examples used for candidate selection and regression checks.
- **Test / held-out eval**: optional final reporting set.

Prefer deterministic splits. Put random seed, split ratio/counts, grouping key,
and sampling limits in `bench/config.py` or `gepa/config.py`. Split by group when
leakage is plausible. Never let near-identical examples from the same source
land in both train and validation without calling it out.

If the dataset is tiny, prefer explicit hand-authored train/validation files
over random splitting.

For benchmarks with official splits, preserve the benchmark's public semantics.
Use the official train split for optimization data, carve GEPA validation from
that train split when candidate selection needs a separate set, and reserve
official dev/test/challenge splits for reporting. Do not let optimizer feedback
from held-out splits influence seed instructions or candidate selection.

## Benchmark Integration Boundaries

Keep benchmark evaluators and oracle-style answer checkers harness-side. The RLM
may use environment-safe tools, documentation, state APIs, and session controls,
but it must not see evaluator feedback or hidden scoring APIs while solving an
example. After the attempt, the harness may pass score and feedback to GEPA as
the learning signal.

When benchmark packages conflict with PredictRLM, DSPy, Pyodide, or the main
project environment, run them in an isolated host-side process behind a typed
JSON boundary. Do not force incompatible benchmark dependencies into the RLM
sandbox or the generated package environment.

## Scoring Feedback

Each `evaluate_example()` should return a scalar score plus feedback that helps
the proposer make a targeted behavioral change.

Good feedback names concrete misses:

- missing fields;
- unsupported citations;
- extraction or parsing errors;
- wrong calculations;
- formatting or file-output failures;
- tool-use mistakes visible in traces.

Avoid feedback that only says "wrong" or restates the score. GEPA quality is
bounded by the evidence the metric returns.

## Overfitting Boundaries

State what counts as a transferable improvement versus a benchmark-specific
hack. Examples:

- preserve citation grounding instead of memorizing answer strings;
- improve table handling generally instead of keying on fixture names;
- preserve sandbox path conventions and tool APIs;
- prefer behavior that transfers across document lengths and layouts.
