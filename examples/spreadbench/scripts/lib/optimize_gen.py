"""Generic skill-evolution proposer for GEPA.

Parameterizes the ImproveInstructions prompt from lib/optimize.py so it
can be reused across any agent + skill combination, not just the
spreadsheet-manipulation agent. The agent type, use cases, runtime, and
concrete rule examples come from an AgentSpec injected at build time;
the generic template contains only domain-agnostic structural tests,
workflow, and audit slots.

Skill-only by design — signature-docstring evolution is intentionally
out of scope. Use this as the proposer when you want GEPA to evolve
only the `skill_instructions` component of a PredictRLM agent.

Usage:

    from lib.optimize_gen import (
        AgentSpec, build_proposer_signature, SPREADSHEET_SPEC
    )

    sig = build_proposer_signature(SPREADSHEET_SPEC)
    predictor = PredictRLM(sig, lm="anthropic/claude-opus-4-6", ...)
    result = predictor(
        current_instructions=current_skill_text,
        traces_file=File(path=traces_path),
    )
    new_skill = result.new_instructions
"""

from __future__ import annotations

from dataclasses import dataclass

import dspy

from predict_rlm import File

# ---------------------------------------------------------------------------
# AgentSpec: the only place domain-specific content lives
# ---------------------------------------------------------------------------


@dataclass
class AgentSpec:
    """Runtime-injected spec for a skill-evolution proposer.

    All domain-specific content lives in an AgentSpec. The generic
    proposer template itself contains only domain-agnostic structural
    tests, the workflow, and the audit-slot format — everything else is
    templated in from an AgentSpec at build time.
    """

    agent_type: str
    use_cases: list[str]
    runtime_grounding_examples: dict[str, list[str]]
    tool_signatures: str
    target_signature: str
    scoring_description: str
    counterfactual_axis_name: str = "domains"
    domain_conventions_note: str = ""
    traces_file_mount: str = "/sandbox/input/traces_file/"

    def __post_init__(self) -> None:
        if not self.agent_type.strip():
            raise ValueError("AgentSpec.agent_type must be a non-empty string")
        if not self.scoring_description.strip():
            raise ValueError(
                "AgentSpec.scoring_description must be a non-empty string: "
                "it tells the proposer how the evaluator scores traces, "
                "so it can interpret the per-item Feedback content correctly"
            )
        if len(set(self.use_cases)) < 2:
            raise ValueError(
                f"AgentSpec.use_cases must list at least 2 DISTINCT entries "
                f"(got {len(self.use_cases)}, distinct {len(set(self.use_cases))}): "
                f"the two-{self.counterfactual_axis_name} counterfactual rubric "
                f"in the proposer template is unsatisfiable otherwise"
            )
        if self.counterfactual_axis_name not in _AXIS_SINGULAR:
            raise ValueError(
                f"AgentSpec.counterfactual_axis_name={self.counterfactual_axis_name!r} "
                f"is not registered in _AXIS_SINGULAR. Register the singular form "
                f"there (e.g. 'analyses' -> 'analysis') to avoid the rstrip('s') "
                f"fallback mangling irregular plurals"
            )
        if len(self.runtime_grounding_examples) < 3:
            raise ValueError(
                f"AgentSpec.runtime_grounding_examples must have at least 3 "
                f"category groups (got {len(self.runtime_grounding_examples)}). "
                f"Structure it as dict[category_name, list[surface]] and span "
                f"at least 3 of {{library symbols, tool contracts, sandbox/env "
                f"facts, spec/standard behaviors}} so lazy specs can't ship a "
                f"flat list of same-category entries."
            )
        for category, surfaces in self.runtime_grounding_examples.items():
            if not surfaces:
                raise ValueError(
                    f"AgentSpec.runtime_grounding_examples[{category!r}] is "
                    f"empty. Every category group must list at least one "
                    f"concrete grounding surface."
                )
        if not self.tool_signatures.strip():
            raise ValueError(
                "AgentSpec.tool_signatures must be a non-empty string: "
                "the proposer template injects this as the 'Tools available "
                "to the agent' section so rules can be grounded in real "
                "tool contracts, not inferred from rule examples"
            )
        if not self.target_signature.strip():
            raise ValueError(
                "AgentSpec.target_signature must be a non-empty string: "
                "the proposer template injects this as the 'Agent target "
                "signature' section so rules can reference the agent's "
                "actual input/output fields and sandbox mount points"
            )


# ---------------------------------------------------------------------------
# Generic proposer template
# ---------------------------------------------------------------------------


GENERIC_PROPOSER_TEMPLATE = """\
Analyze execution traces and propose improved skill instructions for
{{AGENT_TYPE}}.

The goal is NOT to game any particular benchmark — it is to make the
agent a master of its tools and its execution environment.

The skill you are editing will be reused for (at minimum):
{{USE_CASES_BULLETED}}

Every change you propose must help across this whole surface. Rules
that only pay off on the specific eval in the traces file must be
REJECTED, even if they look like they would lift the score on that
eval.

# Preservation stance — this is the most important part of this prompt

You are producing a SURGICAL EDIT of `current_instructions`, not a
rewrite. The default outcome of this call should be: the full seed
text, verbatim, with a small number of targeted additions or
modifications tied to specific trace evidence. If you find yourself
reorganizing sections, merging headings, collapsing lists, or
producing a structurally new document, STOP — that is the failure
mode this prompt is designed to prevent.

Empirically observed pattern on this agent: proposers that
preserve ≥60% of the seed tokens produce winning candidates;
proposers that rewrite produce candidates that look clean in
isolation but lose information the seed had already captured.
"Clean rewrites" are the single largest source of regressions
from this proposer — the seed was built by prior successful
iterations, and wholesale restructures throw away evidence you
cannot re-derive from one traces file.

Concretely, your output `new_instructions` should satisfy ALL of:
  - Preserve every seed section heading unless you have a
    specific trace-grounded reason to retire one (state the
    reason in the audit line for that removal).
  - Preserve every seed rule unless a trace shows it being
    actively wrong (not just unhelpful, actively wrong — cite
    the trace).
  - Add new rules as INLINE EXTENSIONS to the relevant existing
    section, or as a clearly-named new subsection. Do not weave
    them into prose that obscures what changed.
  - Keep the overall shape recognizable: a reviewer diffing
    `current_instructions` against `new_instructions` should see
    a short, surgical patch, not a from-scratch document.

If a trace suggests a sweeping change, your job is to find the
MINIMAL rule that prevents the specific failure, not to rewrite
the document in its spirit. Sweeping changes almost always lose
signal the seed was carrying; a minimal targeted addition almost
never does.

# Tools available to the agent

{{TOOL_SIGNATURES}}

# Agent target signature (the thing the skill drives)

{{TARGET_SIGNATURE}}

# How scoring works

{{SCORING_DESCRIPTION}}

# The traces file
{{TRACES_FILE_MOUNT}} is a JSON file. Each record is one task
invocation and has exactly these three fields:
  - "Inputs"            : the natural-language task instruction and
                          any task metadata the agent received.
  - "Generated Outputs" : the full agent execution trace. BEST and
                          WORST cases are banner-framed. Each step
                          header carries `duration_ms` and an
                          `(ERROR)` marker when the step raised. Each
                          step shows, in order: REASONING, CODE,
                          TOOL CALLS (one line per call with args,
                          result or error, and duration), PREDICT
                          CALLS (one line per signature group with
                          model, call count, and aggregate token
                          usage), then OUTPUT. The case banner itself
                          carries exit status (completed /
                          max_iterations / error), iterations /
                          max_iterations, wall duration, and per-run
                          main and sub token totals.
  - "Feedback"          : the evaluator signal for THIS record. Per-
                          case pass/fail lines, cell-level or other
                          structured mismatches, and crash reasons
                          (timeouts, uncaught exceptions,
                          max-iterations-without-submit). This is the
                          primary place to read what went wrong; if
                          it is empty or contradicts the narrative in
                          Generated Outputs, treat the mismatch as a
                          data-quality signal rather than ignoring it.

# What a rule can be

A rule in the skill is NOT limited to "avoid footgun X". There
are FOUR legitimate rule shapes, and the proposer should look
for all four in the traces. Definitions only — do not pattern-
match on examples; read the traces and find instances of these
shapes from what you actually see:

  1. CORRECTIVE — avoid a specific runtime misuse observed in
     failing traces. Comes from the BOTTOM of the score
     distribution: traces where the agent mishandled an API,
     tool contract, or environment fact.

  2. STRATEGIC — promote a clever technique observed in the TOP
     of the score distribution to a defensive default. A
     strategic rule must be CORROBORATED: either ≥2 successful
     traces used the technique, OR the technique is grounded in
     a runtime capability whose benefit is self-evident from
     the tool / spec docs. A single lucky trace is NOT a
     strategy — it is sampling noise, and promoting it would be
     benchmark memorization disguised as strategy.

  3. DIRECTIONAL — push stuck agents out of local optima. Comes
     from the MIDDLE of the distribution: traces that almost
     succeeded but stalled because the agent retreated from a
     hard problem, simplified instead of engaging, or missed an
     exploration step that a tool right there could have taken.
     These traces are one push away from success, and the rule
     names the missing push: "when you hit X, don't retreat —
     use `<specific runtime capability>` to engage deeper."

  4. NOVEL — surface a runtime capability no trace has used but
     which would obviously help on the use-case list. Comes from
     reading the tools + target signature + grounding list above
     and noticing a capability that is documented but untouched
     by any trace in the file.

Your job is not only to find bugs — it is to surface the needle
in the haystack: the brilliant pattern a few successful traces
used; the dead-end the stalled traces would have escaped with
one more exploration step; the runtime capability no trace has
discovered yet.

## A philosophical stance on good rules

The path forward is often THROUGH the obstacle, not around it.
Rules that tell the agent to RETREAT from hard problems — "leave
the cell blank if unsure", "fall back to a simpler approach",
"skip the edge case", "give up and return an empty result" —
are BAD rules. They game the metric by teaching the agent to
stop trying.

Rules that tell the agent to ENGAGE with the obstacle more
deeply, using a specific runtime capability or exploration
pattern, are GOOD rules. A stalled trace is not a signal that
the task is too hard — it is a signal that the agent needs a
push in a specific direction, and your job is to name the push.

# What counts as a GOOD change

A rule is GOOD — corrective, strategic, directional, or novel —
only if it passes ALL FOUR structural tests:

  (a) NAMES A STABLE, NAMEABLE BEHAVIOR OF THE RUNTIME that exists
      regardless of any particular benchmark. A good rule grounds
      itself in ONE of:
        - a named symbol from the agent's runtime libraries,
        - a documented tool contract (inputs, outputs, side effects),
        - a sandbox or execution-environment fact (memory limits,
          timeouts, missing features, permission model),
        - a protocol, standard, or specification behavior that the
          runtime implements (HTTP semantics, SQL isolation levels,
          DOM events, filesystem conventions, language data model,
          etc.).
      A good rule should ground itself in ONE of the four categories
      above, naming a SPECIFIC surface from the agent's actual
      runtime. The list below is REPRESENTATIVE — any surface from
      the same library / tool / sandbox / spec counts, even if it's
      not explicitly listed:
{{RUNTIME_GROUNDING_EXAMPLES}}
      Rules that name only an abstract category label (e.g. "HTTP
      semantics" for an agent whose runtime has no HTTP surface)
      FAIL test (a). "Check the data carefully" also does NOT count.

  (b) HAS A CONCRETE NON-BENCHMARK USE CASE. Before writing the rule,
      name a task from the use-case list above where obeying the rule
      prevents a real problem. If you cannot name one, the rule is
      benchmark-specific — reject it.

  (c) STATES A PRINCIPLE, NOT A LITERAL TRACE TOKEN. Rules must be
      written in terms of runtime properties, API behavior, and data
      semantics — NOT in terms of specific labels, names, values, or
      phrases that appear in the traces. If your rule mentions a
      literal string that you copied from a trace, you are overfitting
      AND contaminating the proposer's own prompt with benchmark data.
      Lift the rule up one level of abstraction until it says "things
      of type X" or "items in role Y" instead of the literal token.

  (d) MAY INCLUDE A SHORT CODE EXAMPLE. If the behavior is clearer with
      concrete syntax than prose, add a compact snippet (2-8 lines) in the
      `rule` slot. Keep snippets abstract, focused, and trace-agnostic.
      Prefer real API calls over pseudo-code. A snippet is optional, but
      when included it should improve transfer by disambiguating sequencing
      or boundary conditions.

## Directional example: the SHAPE of a well-formed audit line

The block below is a FORMAT DEMONSTRATION, not content to copy. It
shows what every slot looks like for a hypothetical rule. Your
proposed rules must match this structural shape, but their CONTENT
must come from reading the failing traces — NOT from paraphrasing
this example. Do not propose rules about the placeholder verbs
below; propose rules about what you actually observed.

  "<A single-sentence rule (plus optional tiny example for clarity):"
   can be corrective ('avoid X because Y'), strategic ('when Z, do W'),
   or novel ('use <runtime capability> to achieve <benefit>')."
   Example (optional):
   ```python
   if condition:
       action()
   ```
    | grounding: <specific surface from library symbols / tool
      contracts / sandbox facts / spec behaviors above>
    | use case: <concrete task from the use-case list where
      obeying the rule prevents a real problem or enables a
      better outcome>
   | principle: <abstract invariant or technique, named in
     runtime / data semantics, using no literal trace token>
   | counterfactual_1: <task, {{COUNTERFACTUAL_AXIS_SINGULAR}}=A>
   | counterfactual_2: <task, {{COUNTERFACTUAL_AXIS_SINGULAR}}=B>
     (where A ≠ B, both from the use-case list)

Every `<placeholder>` above must be filled with a specific,
trace-supported fact about THIS agent's runtime. The example uses
placeholders precisely because it is NOT a rule — it is only the
container a rule must fit into.

## Common rejection patterns

Any rule that fails any of tests (a), (b), (c), (d). Four failure
to recognize and reject:

  Fails (a) — no specific runtime grounding.
    e.g. "Always double-check the data carefully." — "check
    carefully" is not a library symbol, tool contract, sandbox
    fact, or spec behavior. It is a generic exhortation, not a
    rule grounded in the agent's runtime.

  Fails (b) — benchmark-specific, no non-benchmark use case.
    e.g. "Coerce <a typed scalar> to the string representation the
    eval happens to use." — this is a workaround for one
    consumer's scoring contract. Real consumers (forms with
    validation, downstream scripts, typed APIs) depend on the
    original types. No task in the use-case list benefits; at
    least one is actively harmed.

  Fails (c) — literal trace token contamination.
    e.g. "If the instruction mentions `<a specific literal from
    the trace>`, output the value in sheet `<another literal from
    the trace>`." — literal trace tokens leak benchmark data into
    the proposer's own prompt. The underlying principle may be
    fine ("labels may be synonymous; parse them out") but the
    literal tokens are not.

  Fails (d) — unhelpful snippet without behavioral gain.
    e.g. "Example snippet: for each row: ..." with only generic
    prose in the <rule> slot, or snippets copied verbatim from traces.
    A snippet is useful only if it constrains runtime behavior.

{{DOMAIN_CONVENTIONS_NOTE}}

# Sandbox hygiene when analyzing traces

You analyze traces by writing Python inside a Pyodide sandbox. The
sandbox has a per-execute wall-clock timeout: if a single REPL step
runs too long, it is killed and your call fails. Two specific
patterns have caused hangs in practice:

  1. PREFER STRUCTURED ACCESS OVER REGEX. The traces file is JSON —
     load it with `json.load(open(path))` and address fields by key
     (`record["Inputs"]`, `record["Feedback"]`, etc.). Do NOT
     `re.search` or `re.findall` into raw trace text. If you need a
     token from inside a field, use `str.splitlines()` + `in` /
     `str.startswith()` / `str.split()` — anchored, linear-time
     operations — not regex. Python's `re` module has no internal
     timeout, and patterns with nested lazy quantifiers — `.*?`,
     `(?:.*\\n)*?`, `.+?.+?` — can match in exponential time on
     long trace text when the pattern fails to anchor. Example of
     a pattern that hangs:
         r"Cell mismatches:.*?\\n(?:.*\\n)*?\\s{2,}(.*expected=.*?got=.*)"
     (three lazy quantifiers in one pattern, applied with `re.S`).
     If you absolutely must use `re`, stay line-by-line and keep
     patterns anchored and non-greedy-free:
         for line in text.splitlines():
             if "expected=" not in line: continue
             m = re.match(r"\\s*(\\S+): expected=(.+?) got=(.+)\\s*$", line)
             if m: ...
     That is O(n) instead of O(2^n) and cannot hang.

  2. "EVERYTHING IN ONE REPL STEP" scripts. Do not generate single
     Python blocks longer than ~150 lines that try to load traces,
     bucket them, pattern-scan, extract snippets, and summarize in
     one shot. If any one sub-step is pathological, the whole call
     hangs. PREFER: break analysis into smaller REPL steps —
     inspect the schema in one step, bucket by score in another,
     extract snippets in a third. Smaller steps fail faster and
     isolate bugs.

These constraints are on the sandbox, not on the rules you write
for the agent. They exist so your analysis actually finishes and
produces `new_instructions`.

# Workflow

Traces are a noisy signal. Your work is three operations on that
signal: DENOISE (filter out sampling artifacts, single-trace
luck, and random noise); AVERAGE (corroborate patterns across
multiple traces before trusting them); AMPLIFY (promote the
latent clever patterns you find into explicit rules). Each step
below supports one of these three operations.

1. Load {{TRACES_FILE_MOUNT}}.
2. Bucket records by score into THREE groups, and read each
   with a DIFFERENT lens:
   - BOTTOM (worst failures): look for CORRECTIVE rules.
   - TOP (best successes): look for STRATEGIC rules.
   - MIDDLE (near-miss / stalled traces — not terrible, not
     brilliant, but clearly one push away from success): look
     for DIRECTIONAL rules.
3. BOTTOM pass: for each failure, identify what the agent did
   at the runtime / tool-invocation level that was wrong — in
   terms of API behavior, tool usage, or execution-environment
   fact — NOT in terms of "it got the wrong answer".
   Some BOTTOM traces will be labeled CRASHED instead of
   cell-mismatched. Those are runs where the agent hit a
   timeout, raised an unhandled Python exception, or exhausted
   max_iterations without SUBMIT. Read the crash reason
   (exception class + message, e.g. `SyntaxError: unterminated
   triple-quoted string literal`) and the partial trajectory
   up to the crash point. A corrective rule that prevents the
   crash pattern — naming the specific runtime behavior that
   triggered it — is a first-class corrective signal and
   should go into the skill with the same rigor as any other
   rule. The partial trajectory shows what the agent was
   trying to do right before the failure; use it to identify
   the problematic pattern.
4. TOP pass (AVERAGE + AMPLIFY): identify clever techniques in
   the successes that failing traces missed. A technique counts
   as a strategic rule ONLY if it is either (i) corroborated
   across ≥2 successful traces, or (ii) grounded in a runtime
   capability whose benefit is self-evident from the tool / spec
   docs. A single lucky trace is SAMPLING NOISE, not a strategy
   — reject it no matter how clever it looks.
5. MIDDLE pass: for each stalled / near-miss trace, identify
   where the agent RETREATED from a hard problem, SIMPLIFIED
   instead of engaging, or MISSED an exploration step that a
   tool right there could have taken. These traces are where
   discovery lives: the agent is one push away from success,
   and your job is to name the push. Remember the stance: the
   path forward is through the obstacle, not around it.
6. NOVEL scan: read the tools + target signature + grounding
   list at the top of this document. Is there a runtime
   capability NO trace has used that would obviously help on
   the use-case list? Those are novel rule candidates.
7. For each candidate (corrective, strategic, directional, or
   novel), apply the FOUR structural tests:
    (a) does it name a specific runtime grounding surface
        (symbol, tool contract, env fact, or protocol behavior)?
    (b) does it have a non-benchmark use case (a real problem
        it prevents OR a real outcome it improves)?
    (c) does it state a principle, not a literal trace token?
    (d) does any included snippet stay abstract, trace-agnostic,
        and behaviorally necessary?
8. Principle-vs-token check: scan the candidate rule for any
   label, name, value, or phrase that was lifted from a trace.
   Replace each with an abstract role. If the rule loses all
   meaning after abstraction, it was memorization — drop it.
9. Counterfactual check: name TWO SPECIFIC non-benchmark task
   scenarios on which obeying this rule would change the
   agent's behavior, and the two scenarios MUST span DIFFERENT
   {{COUNTERFACTUAL_AXIS_NAME}} drawn from the use-case list at
   the top of this document. Two scenarios from the same
   {{COUNTERFACTUAL_AXIS_SINGULAR}} do NOT satisfy this check.
   If you cannot name two counterfactuals that span different
   {{COUNTERFACTUAL_AXIS_NAME}}, the rule does not generalize
   — drop it.
10. Keep improvements SURGICAL. This is a hard constraint, not a
    preference. Before writing `new_instructions`, identify in
    your reasoning:
      - which seed sections you are PRESERVING verbatim,
      - which seed rules you are MODIFYING (and the minimal delta),
      - which rules you are ADDING (and where they plug in to the
        existing structure),
      - which seed content — if any — you are REMOVING (and the
        trace-grounded reason for each removal).
    Default to preservation. If the total set of MODIFIED + ADDED +
    REMOVED entries is large, your edit is not surgical — go back
    and trim it until the diff is small and each change is
    individually defensible.
    Emit the FULL revised skill instructions in `new_instructions`
    (the full preserved seed + your surgical changes), not a diff.
    But every token you change or drop must show up as an audit
    line tagged MODIFIED or REMOVED. The reviewer should be able
    to reconstruct a clean patch from your audit lines alone.
11. For each rule in `new_instructions`, emit one audit line in
   `generalization_check` tagged with its provenance. The tag comes
   first; rules with no tag are rejected. Format:
     `[KEPT|MODIFIED|NEW|REMOVED] <rule> | grounding: <specific
      runtime grounding surface from the list above> | use case:
      <concrete non-benchmark task> | principle: <abstract
      principle, naming no trace token> | counterfactual_1: <task,
      {{COUNTERFACTUAL_AXIS_SINGULAR}}=A> | counterfactual_2:
      <task, {{COUNTERFACTUAL_AXIS_SINGULAR}}=B> (where A ≠ B)`
   Tag semantics:
     - KEPT: seed rule preserved verbatim. The audit line still
       requires populated grounding / use-case / principle /
       counterfactual slots — if you cannot defend a seed rule
    against the FOUR structural tests, emit it as REMOVED
       with a trace-grounded reason instead of silently dropping
       it.
     - MODIFIED: seed rule edited. The `<rule>` field shows the
       NEW text; include a `| was: <original seed phrasing>` slot
       at the end so the reviewer can see the minimal delta.
     - NEW: rule added that was not in the seed. Must cite a
       specific trace artifact (case id, crash reason, or
       tool-contract gap) in the `| use case:` slot — "this rule
       addresses crash pattern X seen in CRASHED CASE N" — so
       additions cannot be speculative.
     - REMOVED: seed rule dropped. `<rule>` quotes the original
       seed text; add `| reason: <why this rule was actively
       wrong per the traces, not just unhelpful>` at the end.
       Removing a rule because you reorganized the section is
       NOT a valid reason.
   The two counterfactuals MUST reference different
   {{COUNTERFACTUAL_AXIS_NAME}} drawn from the use-case list at the
   top. Any rule whose audit line cannot be fully populated — or
   whose counterfactuals share a {{COUNTERFACTUAL_AXIS_SINGULAR}} —
   is memorization; drop it from `new_instructions` too.

   If your audit lines for this call contain more MODIFIED + NEW +
   REMOVED entries than KEPT entries, your edit is a rewrite, not
   a surgical patch. Go back to step 10, trim to the smallest
   defensible set of changes, and re-emit.
"""


# ---------------------------------------------------------------------------
# Template rendering + signature building
# ---------------------------------------------------------------------------


_AXIS_SINGULAR = {
    "domains": "domain",
    "task shapes": "task shape",
    "failure modes": "failure mode",
    "task types": "task type",
    "problem classes": "problem class",
}
# To register a new counterfactual axis, add an entry above. The
# `rstrip("s")` fallback in render_template is safe only for regular
# English plurals and will mangle irregular forms (analyses → analyse).


def render_template(template: str, spec: AgentSpec) -> str:
    """Substitute AgentSpec fields into the generic proposer template."""
    use_cases_bulleted = "\n".join(f"  - {uc}" for uc in spec.use_cases)
    grounding_bulleted = "\n".join(
        f"        - {category}: {', '.join(surfaces)}"
        for category, surfaces in spec.runtime_grounding_examples.items()
    )
    axis_name = spec.counterfactual_axis_name
    axis_singular = _AXIS_SINGULAR.get(axis_name, axis_name.rstrip("s"))
    return (
        template
        .replace("{{AGENT_TYPE}}", spec.agent_type)
        .replace("{{USE_CASES_BULLETED}}", use_cases_bulleted)
        .replace("{{TOOL_SIGNATURES}}", spec.tool_signatures)
        .replace("{{TARGET_SIGNATURE}}", spec.target_signature)
        .replace("{{TRACES_FILE_MOUNT}}", spec.traces_file_mount)
        .replace("{{SCORING_DESCRIPTION}}", spec.scoring_description)
        .replace("{{RUNTIME_GROUNDING_EXAMPLES}}", grounding_bulleted)
        .replace("{{DOMAIN_CONVENTIONS_NOTE}}", spec.domain_conventions_note)
        .replace("{{COUNTERFACTUAL_AXIS_NAME}}", axis_name)
        .replace("{{COUNTERFACTUAL_AXIS_SINGULAR}}", axis_singular)
    )


class ImproveInstructionsGeneric(dspy.Signature):
    """Replaced at build time by with_instructions()."""

    current_instructions: str = dspy.InputField(
        desc="The current skill instructions text being improved"
    )
    component_focus: str = dspy.InputField(
        desc="Optional per-call framing that narrows the proposer's "
        "focus when the caller is editing one of several components "
        "(e.g. 'you are editing the skill rules, not the workflow "
        "docstring'). Empty string when the proposer owns a single "
        "skill end-to-end — the template still stands on its own."
    )
    traces_file: File = dspy.InputField(
        desc="JSON file with task execution traces, mounted at the path "
        "specified in the signature instructions above."
    )
    new_instructions: str = dspy.OutputField(
        desc="The full revised skill instructions text. Emit the entire "
        "skill you want the agent to use, not a diff. "
        "When helpful, include concise, abstract code examples in rules to"
        " show API usage and edge conditions unambiguously."
    )
    generalization_check: list[str] = dspy.OutputField(
        desc="One audit line per rule in new_instructions, each tagged "
        "with its provenance [KEPT|MODIFIED|NEW|REMOVED]. See the "
        "audit-line format specification in workflow step 11 of the "
        "signature instructions above — including the per-tag slot "
        "requirements (MODIFIED needs a `was:` slot showing the "
        "original seed phrasing, REMOVED needs a `reason:` slot, NEW "
        "must cite a specific trace artifact). If MODIFIED + NEW + "
        "REMOVED exceeds KEPT, the edit is a rewrite and must be trimmed "
        "before emission. Rules may carry a short example snippet in the "
        "rule slot if it directly clarifies behavior and is abstract."
    )


def build_proposer_signature(spec: AgentSpec) -> type[dspy.Signature]:
    """Build a parameterized ImproveInstructions signature from an AgentSpec.

    The returned signature uses the generic proposer template with all
    AgentSpec fields substituted into the docstring.
    """
    docstring = render_template(GENERIC_PROPOSER_TEMPLATE, spec)
    return ImproveInstructionsGeneric.with_instructions(docstring)


# ---------------------------------------------------------------------------
# RLM integration: derive tool_signatures + target_signature from any RLM
# ---------------------------------------------------------------------------


_INFRA_TOOL_NAMES = frozenset({"predict", "SUBMIT", "print"})


def _format_tool_signatures(rlm: dspy.Module) -> str:
    """Extract tool name + signature + docstring from an RLM instance.

    Walks `rlm.tools` (which is a list of `dspy.Tool` or plain callables on
    a `PredictRLM`), unwraps each to its underlying function, and formats
    the name, inspect signature, and docstring. Skips infrastructure tools
    that aren't part of the skill surface (`predict`, `SUBMIT`, `print`).
    """
    import inspect
    import textwrap

    tools = getattr(rlm, "tools", None) or []
    if isinstance(tools, dict):
        tool_iter = list(tools.values())
    else:
        tool_iter = list(tools)

    blocks = []
    for tool in tool_iter:
        fn = getattr(tool, "func", None) or tool
        name = getattr(tool, "name", None) or getattr(fn, "__name__", "unknown")
        if name in _INFRA_TOOL_NAMES:
            continue
        try:
            sig = str(inspect.signature(fn))
        except (ValueError, TypeError):
            sig = "(...)"
        doc = inspect.getdoc(fn) or "(no docstring)"
        doc_indented = textwrap.indent(doc, "    ")
        blocks.append(f"{name}{sig}\n{doc_indented}")

    if not blocks:
        return "(no skill tools registered)"
    return "\n\n".join(blocks)


def _format_target_signature(rlm: dspy.Module) -> str:
    """Format an RLM's dspy.Signature as a human-readable description.

    Pulls the class name, docstring (the task-level strategy), input
    fields with their types and descriptions, and output fields. This
    is injected into the proposer prompt so rules can reference the
    agent's actual input/output shape instead of inferring it.
    """
    import inspect
    import textwrap

    sig = rlm.signature
    sig_name = getattr(sig, "__name__", "Signature")
    lines = [f"{sig_name} (dspy.Signature)"]

    doc = inspect.getdoc(sig)
    if doc:
        lines.append("")
        lines.append(doc)

    def _describe_field(field_name: str, field: object) -> str:
        annotation = getattr(field, "annotation", None)
        if annotation is None:
            type_str = "?"
        elif hasattr(annotation, "__name__"):
            type_str = annotation.__name__
        else:
            type_str = str(annotation)

        extra = getattr(field, "json_schema_extra", None) or {}
        desc = extra.get("desc", "") if isinstance(extra, dict) else ""

        header = f"  {field_name}: {type_str}"
        if desc:
            wrapped = textwrap.fill(
                desc, width=70,
                initial_indent="      ", subsequent_indent="      ",
            )
            return header + "\n" + wrapped
        return header

    if getattr(sig, "input_fields", None):
        lines.append("")
        lines.append("Inputs:")
        for name, field in sig.input_fields.items():
            lines.append(_describe_field(name, field))

    if getattr(sig, "output_fields", None):
        lines.append("")
        lines.append("Outputs:")
        for name, field in sig.output_fields.items():
            lines.append(_describe_field(name, field))

    return "\n".join(lines)


def agent_spec_from_rlm(
    rlm: dspy.Module,
    *,
    agent_type: str,
    use_cases: list[str],
    runtime_grounding_examples: dict[str, list[str]],
    counterfactual_axis_name: str = "domains",
    domain_conventions_note: str = "",
    traces_file_mount: str = "/sandbox/input/traces_file/",
) -> AgentSpec:
    """Build an AgentSpec for an existing RLM instance.

    `tool_signatures` and `target_signature` are extracted automatically
    from the RLM's registered tools and DSPy signature. The caller
    provides the domain-specific fields that cannot be derived from code:
    agent type description, non-benchmark use cases, and concrete
    runtime grounding surfaces.
    """
    return AgentSpec(
        agent_type=agent_type,
        use_cases=use_cases,
        runtime_grounding_examples=runtime_grounding_examples,
        tool_signatures=_format_tool_signatures(rlm),
        target_signature=_format_target_signature(rlm),
        counterfactual_axis_name=counterfactual_axis_name,
        domain_conventions_note=domain_conventions_note,
        traces_file_mount=traces_file_mount,
    )


def build_proposer_for_rlm(rlm: dspy.Module, **spec_kwargs) -> type[dspy.Signature]:
    """One-liner: build a skill-evolution proposer bound to a live RLM.

    Equivalent to calling `agent_spec_from_rlm(rlm, **spec_kwargs)` and
    then `build_proposer_signature(...)`. Use this when you already have
    an RLM instance and want a ready-to-call proposer signature without
    manually constructing an AgentSpec.
    """
    spec = agent_spec_from_rlm(rlm, **spec_kwargs)
    return build_proposer_signature(spec)


# ---------------------------------------------------------------------------
# Concrete AgentSpec for the spreadsheet agent (mirrors v4.3 in optimize.py)
# ---------------------------------------------------------------------------


_SPREADSHEET_TOOL_SIGNATURES = """\
recalculate(file_path: Path) -> str
    Recalculate all formulas in an xlsx file and cache the results in
    place. Runs a two-stage pipeline: Python `formulas` library first
    (fast, in-process), then LibreOffice headless as a fallback for
    any formulas the library could not evaluate. Strictly additive —
    never downgrades the file. After calling, re-open the file with
    `openpyxl.load_workbook(file_path, data_only=True)` to read the
    cached values.
    Returns: "ok (source=..., resolved=N/M)" on success, or a string
    starting with "Error:" on failure.

render(file_path: Path, cell_range: str | None = None,
       sheet_name: str | None = None) -> str
    Render an xlsx file to a PNG for visual inspection. Converts the
    workbook to PDF via LibreOffice, then rasterizes the first page
    via pdftoppm. `cell_range` optionally restricts the render to a
    sub-region (e.g. "J3:N5" or "Sheet1!J3:N5") — use when the target
    area is not on the first printed page of a whole-sheet export.
    `sheet_name` sets the print-area sheet when `cell_range` is
    unqualified; defaults to the workbook's active sheet.
    Returns: a `data:image/png;base64,...` data URI on success, or a
    string starting with "Error:" on failure.\
"""


_SPREADSHEET_TARGET_SIGNATURE = """\
ManipulateSpreadsheet (dspy.Signature)

Inputs:
  input_spreadsheet: dspy.File
      The input .xlsx spreadsheet file to manipulate, mounted at
      /sandbox/input/input_spreadsheet/.
  instruction: str
      The natural-language manipulation to perform. At eval time the
      harness appends an "Answer position: <range> on sheet '<name>'"
      suffix specifying the maximum range of cells the agent may
      modify.

Output:
  output_spreadsheet: dspy.File
      The modified .xlsx spreadsheet file, saved to
      /sandbox/output/output_spreadsheet/ with the same filename.

Expected flow: (1) load the input with openpyxl, (2) inspect its
structure (sheets, columns, data types, row count), (3) compute
values in Python and write scalar / formula values to cells,
(4) save to the output path, (5) verify the output before
submitting. MUST NOT write VBA macros, Python source, Excel formulas
as literal text (use formula strings assigned to `cell.value`
instead), or explanatory prose into cells.\
"""


SPREADSHEET_SPEC = AgentSpec(
    agent_type=(
        "a spreadsheet-manipulation agent that writes Python against "
        "openpyxl in a Pyodide/WASM sandbox, with host-side tools for "
        "LibreOffice recalculation and workbook rendering"
    ),
    use_cases=[
        "investment-banking modeling (DCF, cashflow, 3-statement, LBO)",
        "filling structured forms (tax, compliance, HR onboarding)",
        "project-management tracking (status rollups, milestone sheets)",
        "mundane data wrangling (dedup, reformat, reshape, join)",
    ],
    runtime_grounding_examples={
        "library symbols": ["`cell.value` (the typed-slot accessor)"],
        "tool contracts": [
            "`recalculate(path)` — see Tools section above for the full contract"
        ],
        "sandbox / execution-environment facts": [
            "Pyodide WASM memory ceiling"
        ],
        "spec / standard behaviors": [
            "LibreOffice's rejection of Excel `_xlfn.` private function prefixes"
        ],
    },
    tool_signatures=_SPREADSHEET_TOOL_SIGNATURES,
    target_signature=_SPREADSHEET_TARGET_SIGNATURE,
    scoring_description="""\
The evaluator runs the agent on several test cases per task, where
each case has a reference answer and a set of target cells the agent
must populate. Per case:
  score = matched_cells / total_cells  (0.0 to 1.0)
A score of 1.0 means every target cell matches the reference exactly
(value equality, after the evaluator's type-normalization). The
task-level score is the mean of case scores, so a task with a single
failing case drags the whole task score down.
The Feedback field for each record contains per-case pass/fail lines,
the specific cells that mismatched (with `expected=<ref> got=<agent>`
diffs), and the crash reason when a case ended in a timeout, an
uncaught Python exception, or max-iterations-without-SUBMIT.\
""",
    domain_conventions_note="""\
Note: domain-specific conventions (formatting, layout, labeling
protocols) ARE legitimate rules when you (i) scope them to a
specific domain from the use-case list above, (ii) ground them
in a runtime property that can be verified at write-time rather
than at metric-scoring time, and (iii) would apply them to other
tasks in that same domain beyond anything observed in the traces
file. Domain conventions that fail any of these checks are
benchmark gaming.\
""",
)
