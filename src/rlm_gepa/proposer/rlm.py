from __future__ import annotations

import asyncio
import inspect
import json
import tempfile
import textwrap
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import dspy
from pydantic import BaseModel, Field

from predict_rlm import File, PredictRLM
from predict_rlm.trace import extract_trace_from_exc

from ..reporting.cost import append_trace_cost_rows
from ..runtime.progress import install_rlm_log_stream, progress_write, restore_rlm_log_stream
from ..runtime.trace_rendering import trace_to_json
from ..runtime.utils import atomic_write_json, run_coro_sync
from ..schema import COUNTERFACTUAL_AXIS_SINGULAR, AgentSpec

GENERIC_PROPOSER_TEMPLATE = """\
Analyze execution traces and propose improved skill instructions for
{{AGENT_TYPE}}.

The goal is not to game a benchmark. Make the agent better across these
use cases:
{{USE_CASES_BULLETED}}

You are producing a surgical edit of `current_instructions`, not a rewrite.
Preserve the existing structure unless trace evidence shows a rule is actively
wrong. Add rules inline near the relevant existing section. Emit the full revised
instructions in `new_instructions`, not a diff.

# Tools Available To The Agent

{{TOOL_SIGNATURES}}

# Agent Target Signature

{{TARGET_SIGNATURE}}

# How Scoring Works

{{SCORING_DESCRIPTION}}

# Runtime Grounding Surfaces

Good rules must name a stable runtime behavior from one of these groups:
{{RUNTIME_GROUNDING_EXAMPLES}}

{{DOMAIN_CONVENTIONS_NOTE}}

# The Traces File

{{TRACES_FILE_MOUNT}} is a JSON file. Each record has:
- `Inputs`: task inputs and metadata.
- `Generated Outputs`: rendered PredictRLM trace with reasoning, code, tool
  calls, predict calls, output, timings, status, iterations, and token usage.
- `Feedback`: evaluator signal for that record, including scores, mismatches,
  crash reasons, and other domain-specific failure details.

# Valid Rule Shapes

Look for four rule shapes:
- CORRECTIVE: avoid a runtime/tool/API misuse observed in failing traces.
- STRATEGIC: promote a technique used by successful traces or clearly supported
  by the documented runtime.
- DIRECTIONAL: push near-miss traces through the hard part instead of retreating.
- NOVEL: use a documented capability that traces did not discover yet.

Every proposed rule must pass all structural tests:
- It names a specific runtime surface, tool contract, environment fact, library
  symbol, protocol behavior, or specification behavior.
- It has a concrete non-benchmark use case from the list above.
- It states a principle, not a literal trace token.
- Any code example is short, abstract, and trace-agnostic.

# Workflow

1. Load {{TRACES_FILE_MOUNT}} with JSON parsing and work from parsed records.
   Use regex only as a local extractor inside parsed string fields when helpful,
   not as the primary way to scan the raw trace file.
2. Bucket scored records into bottom failures, middle partials/near-misses, and
   top successes before choosing edits. Keep unscored/crashed records separate
   from score-ranked evidence.
3. Inspect bottom records for concrete failures, top records for strategies to
   preserve, and middle records for near-miss patterns. For large or diverse
   trace sets, use divide-and-conquer: run separate `predict()` calls with
   `asyncio.gather(...)` over focused trace subsets to extract non-lossy,
   evidence-grounded patterns tied to record ids/scores. Helper `predict()`
   calls may extract and reason about trace evidence; you choose the final
   rules or edits and emit the final full instructions.
4. Only add or modify rules grounded in trace evidence plus available tools,
   target signature, runtime behavior, or scoring contract.
5. Keep the edit surgical: identify kept, modified, added, and removed rules. If
   evidence shows an existing rule/default is causing the repeated failure,
   modify or replace that local text instead of appending a conflicting
   exception, and state the trigger and non-application boundary in the audit.
6. Emit one `generalization_check` audit line per material rule change. Each line
   starts with `[KEPT|MODIFIED|NEW|REMOVED]` and includes: grounding, use case,
   principle, counterfactual_1, and counterfactual_2. The counterfactuals must
   span different {{COUNTERFACTUAL_AXIS_NAME}}.
"""


PATCH_MERGE_PROPOSER_TEMPLATE = """\
Patch one base skill-instructions text for {{AGENT_TYPE}} by adding at most
one verified missing behavioral capability supported by a patch-source parent.

Start from `base_parent_instructions`. Preserve base behavior by default.
Compare the two parents as policies over the paired train traces. Your job is
to select the single best-supported bounded capability family from
patch-source wins that the base lacks, then graft that capability into the base
while preserving base wins and both-success behavior.

Do not copy, summarize, concatenate, average, or globally rewrite parent
instructions. Source text is evidence, not the unit of transfer. The unit of
transfer is a proposer-authored behavioral rule with a clear trigger, action,
non-application boundary, preservation boundary, and verification signal.

The selected capability must cite supporting task IDs in structured metadata,
not final skill prose. Do not broaden the capability to cover unrelated source
behavior. If evidence is mixed, narrow the trigger and non-application boundary
until the patch preserves base wins and both-success behavior.
If the base already contains an equivalent rule, do not graft it; either narrow
to a missing facet with concrete source-win support or leave the base
effectively unchanged. The final graft invariant is: "This patch changes
behavior only when T; outside T, base behavior remains unchanged."

The patched skill must remain a surgical edit and must work across these use
cases:
{{USE_CASES_BULLETED}}

# Tools Available To The Agent

{{TOOL_SIGNATURES}}

# Agent Target Signature

{{TARGET_SIGNATURE}}

# How Scoring Works

{{SCORING_DESCRIPTION}}

# Runtime Grounding Surfaces

{{RUNTIME_GROUNDING_EXAMPLES}}

{{DOMAIN_CONVENTIONS_NOTE}}

# Inputs

- `base_parent_id`: integer identifier for the base parent.
- `base_parent_instructions`: full base skill text.
- `patch_source_parent_id`: integer identifier for the patch-source parent.
- `patch_source_parent_instructions`: full source skill text.
- `paired_disagreement_traces_file`, mounted at {{PAIRED_TRACES_FILE_MOUNT}}:
  JSONL from train examples only. Each row contains `task_id`, `winner`,
  `evidence_role`, `abs_delta`, base-parent trace/feedback/score, and
  patch-source-parent trace/feedback/score. Rows with `winner="both_success"`
  are guardrails: both parents solved them, so preserve that behavior rather
  than using them as import evidence.

# Patch Method

1. Load the disagreement JSONL with structured parsing. Separate rows into
   source wins, base wins, and both-success guardrails.
2. Use one or more `predict()` helper calls over paired base and patch-source
   raw trace slices when the raw execution details matter. Give helpers paired
   base/source slices, source-win rows, base-win rows, both-success guardrails,
   and relevant parent instruction slices. Helpers are evidence, guardrail, and
   support-filter workers; you own final capability selection, deduplication,
   the local splice, and the audit.
   - Each helper output must include exactly one candidate capability or
     `none`, trigger, action, non-application boundary, preservation boundary,
     supported source-win IDs, unsupported source-win IDs, base-win hazards,
     both-success invariants, exact evidence spans or trace facts,
     verification signal, and whether the base already contains equivalent
     behavior.
   - If a helper proposes multiple capability families, broad rewrites, or
     copied source wording, split into smaller helper calls or run a focused
     support-filter call.
   - For local wording, pass only the selected capability brief and relevant
     base section. Ask for bounded wording alternatives, not a whole-instruction
     rewrite.
3. Compare the two parents as policies: identify what behavioral differences
   explain patch-source wins vs. base wins. Do not begin by selecting text to
   copy.
4. For each source-win pattern, describe the capability the source has that the
   base lacks. For each candidate capability, ask:
   - What source-win rows does this explain?
   - Which base-win rows could this harm if phrased too broadly?
   - Which both-success rows define invariants that must not change?
5. Select exactly one capability family. Choose the one with the cleanest
   source-win support and clearest preservation boundary. Multiple bullets are
   allowed only when they are facets of the same capability family.
6. Reject capabilities that are unsupported, redundant with the base, overly
   broad, or only explain base wins/failures. Before finalizing, support-filter
   every source-win row: classify it as supported or not-supported by the
   selected capability, and do not cite generally related rows as evidence.
7. Apply the capability as the smallest local edit to the base. When the base
   already has a related section, tighten or extend it rather than appending a
   new block. Do not add a tool, API, code, or protocol detail unless it is
   supported by a trace span, parent instruction, or documented runtime/tool
   contract. Prove preservation separately from source-win evidence using
   base-win and both-success guardrails. Do not copy or renumber source bullets:
   changed prose must be original, local to the nearest relevant base section,
   and no broader than the verified capability. The patched instructions should
   not be materially longer than the base.
8. Keep all task IDs inside `behavioral_rules.evidence_task_ids`,
   `selected_capability.evidence_task_ids`, and `patch_merge_audit` metadata.
   Keep task IDs and audit metadata out of `new_instructions`.
"""


_AXIS_SINGULAR = COUNTERFACTUAL_AXIS_SINGULAR


class BehavioralRule(BaseModel):
    behavioral_rule: str = Field(
        description=(
            "Proposer-authored behavioral rule backed by patch-source evidence; "
            "not copied source wording"
        )
    )
    evidence_task_ids: list[str] = Field(
        description="Train task IDs supporting this behavioral rule"
    )
    reason: str = Field(description="Why this behavioral rule is supported")


class PatchMergeAudit(BaseModel):
    helper_purpose_or_call_reference: str = Field(
        description="Compact description of helper call purpose or reference"
    )
    supported_source_win_ids: list[str] = Field(
        description="Source-win task IDs actually supported by the selected capability"
    )
    unsupported_source_win_ids: list[str] = Field(
        description="Source-win task IDs reviewed but not supported by the selected capability"
    )
    base_win_hazards: list[str] = Field(
        description="Base-win hazards the patch must avoid"
    )
    both_success_invariants: list[str] = Field(
        description="Both-success behavior that must remain unchanged"
    )
    base_duplicate_check: str = Field(
        description="Whether the base already has equivalent behavior and any missing facet"
    )
    guardrail_conflicts: list[str] = Field(
        description="Conflicts with base-win or both-success guardrails, if any"
    )
    verification_signal: str = Field(
        description="Trace-observable signal that the selected capability worked"
    )
    outer_proposer_narrowed_or_overrode_helper: str = Field(
        description="Whether and how the outer proposer narrowed or overrode helper output"
    )


class SelectedCapability(BaseModel):
    name: str = Field(description="Short name for the bounded capability family")
    evidence_task_ids: list[str] = Field(
        description="Train task IDs supporting this capability"
    )
    trigger: str = Field(description="Condition that activates this capability")
    action: str = Field(description="What the agent should do when triggered")
    non_application_boundary: str = Field(
        description="When this capability should NOT apply"
    )
    preservation_note: str = Field(
        description="How base wins and both-success behavior are preserved"
    )
    verification_signal: str = Field(
        description="Trace-observable signal that this capability applied correctly"
    )


def render_template(template: str, spec: AgentSpec) -> str:
    use_cases_bulleted = "\n".join(f"  - {use_case}" for use_case in spec.use_cases)
    grounding_bulleted = "\n".join(
        f"  - {category}: {', '.join(surfaces)}"
        for category, surfaces in spec.runtime_grounding_examples.items()
    )
    agent_type = spec.agent_type or "the target RLM"
    axis_name = spec.counterfactual_axis_name
    axis_singular = _AXIS_SINGULAR[axis_name]
    return (
        template.replace("{{AGENT_TYPE}}", agent_type)
        .replace("{{USE_CASES_BULLETED}}", use_cases_bulleted)
        .replace("{{TOOL_SIGNATURES}}", spec.tool_signatures)
        .replace("{{TARGET_SIGNATURE}}", spec.target_signature)
        .replace("{{SCORING_DESCRIPTION}}", spec.scoring_description)
        .replace("{{RUNTIME_GROUNDING_EXAMPLES}}", grounding_bulleted)
        .replace("{{DOMAIN_CONVENTIONS_NOTE}}", spec.domain_conventions_note)
        .replace("{{TRACES_FILE_MOUNT}}", spec.traces_file_mount)
        .replace("{{PAIRED_TRACES_FILE_MOUNT}}", spec.paired_traces_file_mount)
        .replace("{{COUNTERFACTUAL_AXIS_NAME}}", axis_name)
        .replace("{{COUNTERFACTUAL_AXIS_SINGULAR}}", axis_singular)
    )


class ImproveInstructionsGeneric(dspy.Signature):
    current_instructions: str = dspy.InputField(desc="Current skill instructions text")
    component_focus: str = dspy.InputField(desc="Optional per-component focus text")
    traces_file: File = dspy.InputField(desc="JSON file containing rendered task traces")
    new_instructions: str = dspy.OutputField(
        desc="Full revised skill instructions text. Do not include audit labels or task IDs."
    )
    generalization_check: list[str] = dspy.OutputField(
        desc=(
            "Audit lines for kept/modified/new/removed rules, including evidence "
            "and boundaries for removals/default inversions."
        )
    )



class PatchMergeInstructionsGeneric(dspy.Signature):
    base_parent_id: int = dspy.InputField(desc="Candidate ID for the base parent")
    base_parent_instructions: str = dspy.InputField(desc="Full base parent skill text")
    patch_source_parent_id: int = dspy.InputField(
        desc="Candidate ID for the parent providing patch-source evidence"
    )
    patch_source_parent_instructions: str = dspy.InputField(
        desc="Full patch-source parent skill text"
    )
    paired_disagreement_traces_file: File = dspy.InputField(
        desc="JSONL file carrying train disagreement traces for the two parents"
    )
    patch_summary: str = dspy.OutputField(desc="Short summary of the patch decision")
    selected_capability: SelectedCapability = dspy.OutputField(
        desc="The one behavioral capability grafted onto the base"
    )
    behavioral_rules: list[BehavioralRule] = dspy.OutputField(
        desc=(
            "Proposer-authored behavioral rules backed by patch-source evidence"
        )
    )
    patch_merge_audit: PatchMergeAudit = dspy.OutputField(
        desc=(
            "Compact audit of helper discipline, support filtering, duplicate "
            "check, guardrails, verification, and any outer-proposer narrowing"
        )
    )
    rejected_from_other: list[str] = dspy.OutputField(
        desc="Patch-source clauses or themes intentionally rejected"
    )
    new_instructions: str = dspy.OutputField(desc="Full patched skill instructions text")


def build_proposer_signature(spec: AgentSpec) -> type[dspy.Signature]:
    return ImproveInstructionsGeneric.with_instructions(
        render_template(GENERIC_PROPOSER_TEMPLATE, spec)
    )


def build_merge_signature(spec: AgentSpec) -> type[dspy.Signature]:
    return PatchMergeInstructionsGeneric.with_instructions(
        render_template(PATCH_MERGE_PROPOSER_TEMPLATE, spec)
    )


def build_patch_merge_signature(spec: AgentSpec) -> type[dspy.Signature]:
    return build_merge_signature(spec)


_INFRA_TOOL_NAMES = frozenset({"predict", "SUBMIT", "print"})


def agent_spec_from_rlm(
    rlm: dspy.Module,
    *,
    use_cases: list[str],
    runtime_grounding_examples: dict[str, list[str]],
    scoring_description: str,
    agent_type: str = "",
    counterfactual_axis_name: str = "domains",
    domain_conventions_note: str = "",
    traces_file_mount: str = "/sandbox/input/traces_file/",
) -> AgentSpec:
    return AgentSpec(
        agent_type=agent_type,
        use_cases=use_cases,
        runtime_grounding_examples=runtime_grounding_examples,
        tool_signatures=_format_tool_signatures(rlm),
        target_signature=_format_target_signature(rlm),
        scoring_description=scoring_description,
        counterfactual_axis_name=counterfactual_axis_name,
        domain_conventions_note=domain_conventions_note,
        traces_file_mount=traces_file_mount,
    )


def build_proposer_for_rlm(rlm: dspy.Module, **spec_kwargs: Any) -> type[dspy.Signature]:
    return build_proposer_signature(agent_spec_from_rlm(rlm, **spec_kwargs))


class RLMInstructionProposer:
    def __init__(
        self,
        *,
        spec: AgentSpec,
        lm: Any,
        sub_lm: Any,
        output_dir: str | Path,
        cost_log_path: str | Path | None = None,
        max_iterations: int = 20,
        timeout: int = 600,
        heartbeat_interval_seconds: float = 30.0,
        run_id: str = "run",
        component_focus: Callable[[str], str] | None = None,
    ):
        self.spec = spec
        self.lm = lm
        self.sub_lm = sub_lm
        self.output_dir = Path(output_dir)
        self.trace_dir = self.output_dir / "proposer_traces"
        self.cost_log_path = Path(cost_log_path) if cost_log_path is not None else None
        self.max_iterations = max_iterations
        self.timeout = max(1, int(timeout))
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.run_id = run_id
        self.component_focus = component_focus or (lambda _component: "")
        self._call_count = 0

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        new_texts: dict[str, str] = {}
        for component in components_to_update:
            if component not in candidate:
                continue
            records = list(reflective_dataset.get(component, []))
            new_texts[component] = self.propose_one_component(
                component,
                candidate[component],
                records,
            )
        return new_texts

    def propose_one_component(
        self,
        component_name: str,
        current_text: str,
        records: list[Mapping[str, Any]],
    ) -> str:
        self._call_count += 1
        call_idx = self._call_count
        event_id = f"{self.run_id}_proposer_attempt_{call_idx:04d}_component_{component_name}"
        operation_id = f"proposer_{component_name}_{call_idx:04d}"
        serializable = [
            {
                "Inputs": record.get("Inputs", ""),
                "Generated Outputs": record.get("Generated Outputs", ""),
                "Feedback": record.get("Feedback", ""),
            }
            for record in records
        ]

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix="_traces.json",
            prefix=f"gepa_proposer_{component_name}_",
            delete=False,
            encoding="utf-8",
        ) as f:
            json.dump(serializable, f, indent=2)
            traces_path = Path(f.name)

        try:
            signature = build_proposer_signature(self.spec)
            predictor = PredictRLM(
                signature,
                lm=self.lm,
                sub_lm=self.sub_lm,
                skills=[],
                max_iterations=self.max_iterations,
                verbose=True,
                debug=False,
            )
            kwargs: dict[str, Any] = {
                "current_instructions": current_text,
                "traces_file": File(path=str(traces_path)),
            }
            kwargs["component_focus"] = self.component_focus(component_name)

            try:
                progress_write("\n" + "=" * 80)
                progress_write(f"RLM PROPOSER STARTING ({component_name})")
                progress_write("=" * 80)
                log_stream = install_rlm_log_stream(f"PROPOSER {component_name}")
                try:
                    result = run_coro_sync(
                        _acall_with_heartbeat(
                            predictor.acall(**kwargs),
                            tag=f"PROPOSER {event_id}",
                            timeout=self.timeout,
                            heartbeat_interval_seconds=self.heartbeat_interval_seconds,
                        )
                    )
                finally:
                    restore_rlm_log_stream(log_stream)
                progress_write("=" * 80)
                progress_write(
                    f"RLM PROPOSER DONE ({component_name}, "
                    f"{len(getattr(result, 'trajectory', []))} iterations)"
                )
                progress_write("=" * 80 + "\n")
            except BaseException as exc:
                self._persist_error(
                    event_id=event_id,
                    operation_id=operation_id,
                    attempt_id=f"attempt_{call_idx:04d}",
                    component_name=component_name,
                    current_text=current_text,
                    exc=exc,
                )
                raise

            try:
                new_text = getattr(result, "new_instructions", None)
                if not isinstance(new_text, str) or not new_text.strip():
                    exc = RuntimeError(
                        f"RLM proposer returned empty new_instructions for {component_name}"
                    )
                    exc.trace = getattr(result, "trace", None)  # type: ignore[attr-defined]
                    raise exc
            except BaseException as exc:
                self._persist_error(
                    event_id=event_id,
                    operation_id=operation_id,
                    attempt_id=f"attempt_{call_idx:04d}",
                    component_name=component_name,
                    current_text=current_text,
                    exc=exc,
                )
                raise

            trace = getattr(result, "trace", None)
            self._write_trace_cost(
                event="proposer_call",
                event_id=event_id,
                operation_id=operation_id,
                attempt_id=f"attempt_{call_idx:04d}",
                main_role="proposer",
                sub_role="proposer_sub_lm",
                trace=trace,
            )
            self._write_success_artifact(
                event_id=event_id,
                operation_id=operation_id,
                component_name=component_name,
                current_text=current_text,
                new_text=new_text,
                result=result,
                serializable=serializable,
            )
            return new_text
        finally:
            traces_path.unlink(missing_ok=True)

    def _write_success_artifact(
        self,
        *,
        event_id: str,
        operation_id: str,
        component_name: str,
        current_text: str,
        new_text: str,
        result: Any,
        serializable: list[dict[str, Any]],
    ) -> None:
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "event_id": event_id,
            "operation_id": operation_id,
            "status": "completed",
            "component": component_name,
            "current_instructions": current_text,
            "new_instructions": new_text,
            "generalization_check": getattr(result, "generalization_check", None),
            "reflective_dataset": serializable,
            "rlm_trajectory": getattr(result, "trajectory", []),
            "run_trace": trace_to_json(getattr(result, "trace", None)),
            "error": None,
        }
        atomic_write_json(self.trace_dir / f"{event_id}_proposer_{component_name}.json", payload)

    def _persist_error(
        self,
        *,
        event_id: str,
        operation_id: str,
        attempt_id: str,
        component_name: str,
        current_text: str,
        exc: BaseException,
    ) -> None:
        trace = extract_trace_from_exc(exc)
        self._write_trace_cost(
            event="proposer_error",
            event_id=event_id,
            operation_id=operation_id,
            attempt_id=attempt_id,
            main_role="proposer",
            sub_role="proposer_sub_lm",
            trace=trace,
        )
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "event_id": event_id,
            "operation_id": operation_id,
            "status": "error",
            "component": component_name,
            "current_instructions": current_text,
            "new_instructions": None,
            "generalization_check": None,
            "run_trace": trace_to_json(trace),
            "error": str(exc),
            "error_type": type(exc).__name__,
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        }
        atomic_write_json(self.trace_dir / f"{event_id}_proposer_{component_name}_ERROR.json", payload)

    def _write_trace_cost(
        self,
        *,
        event: str,
        event_id: str,
        operation_id: str,
        attempt_id: str,
        main_role: str,
        sub_role: str,
        trace: Any,
    ) -> None:
        append_trace_cost_rows(
            self.cost_log_path,
            event=event,
            event_id=event_id,
            operation_id=operation_id,
            attempt_id=attempt_id,
            main_role=main_role,
            sub_role=sub_role,
            trace=trace,
            sum_traces=sum_traces,
        )


def sum_traces(traces: Sequence[Any]) -> tuple[Any, Any, str, str | None, int, int]:
    from predict_rlm.trace import TokenUsage

    main = TokenUsage()
    sub = TokenUsage()
    main_model = ""
    sub_model: str | None = None
    main_calls = 0
    sub_calls = 0
    for trace in traces:
        if trace is None:
            continue
        if not main_model:
            main_model = str(getattr(trace, "model", ""))
            sub_model = getattr(trace, "sub_model", None)
        usage = getattr(trace, "usage", None)
        if usage is not None:
            main += usage.main
            if usage.sub is not None:
                sub += usage.sub
        steps = getattr(trace, "steps", None) or []
        main_calls += len(steps)
        for step in steps:
            for group in getattr(step, "predict_calls", None) or []:
                sub_calls += len(getattr(group, "calls", None) or [])
    sub_usage = sub if sub.input_tokens or sub.output_tokens or sub.cache_hits else None
    return main, sub_usage, main_model, sub_model, main_calls, sub_calls


async def _acall_with_heartbeat(
    awaitable: Any,
    *,
    tag: str,
    timeout: int,
    heartbeat_interval_seconds: float,
) -> Any:
    start = time.monotonic()
    progress_write(f"[{tag}] DISPATCH (timeout={timeout}s)")

    async def heartbeat() -> None:
        while True:
            await asyncio.sleep(heartbeat_interval_seconds)
            elapsed = time.monotonic() - start
            progress_write(f"[{tag}] WAITING {elapsed:.0f}s / {timeout}s")

    task = asyncio.create_task(heartbeat())
    try:
        result = await asyncio.wait_for(awaitable, timeout=timeout)
        progress_write(f"[{tag}] RETURNED after {time.monotonic() - start:.1f}s")
        return result
    except asyncio.TimeoutError:
        progress_write(f"[{tag}] TIMED OUT after {time.monotonic() - start:.1f}s")
        raise
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def _format_tool_signatures(rlm: dspy.Module) -> str:
    tools = getattr(rlm, "tools", None) or []
    tool_iter = list(tools.values()) if isinstance(tools, dict) else list(tools)
    blocks: list[str] = []
    for tool in tool_iter:
        fn = getattr(tool, "func", None) or tool
        name = getattr(tool, "name", None) or getattr(fn, "__name__", "unknown")
        if name in _INFRA_TOOL_NAMES:
            continue
        try:
            signature = str(inspect.signature(fn))
        except (TypeError, ValueError):
            signature = "(...)"
        doc = inspect.getdoc(fn) or "(no docstring)"
        blocks.append(f"{name}{signature}\n{textwrap.indent(doc, '    ')}")
    return "\n\n".join(blocks) if blocks else "(no skill tools registered)"


def _format_target_signature(rlm: dspy.Module) -> str:
    signature = rlm.signature
    lines = [f"{getattr(signature, '__name__', 'Signature')} (dspy.Signature)"]
    doc = inspect.getdoc(signature)
    if doc:
        lines.extend(["", doc])

    def describe_field(name: str, field: object) -> str:
        annotation = getattr(field, "annotation", None)
        type_str = getattr(annotation, "__name__", str(annotation or "?"))
        extra = getattr(field, "json_schema_extra", None) or {}
        desc = extra.get("desc", "") if isinstance(extra, dict) else ""
        if desc:
            return f"  {name}: {type_str}\n" + textwrap.fill(
                desc,
                width=70,
                initial_indent="      ",
                subsequent_indent="      ",
            )
        return f"  {name}: {type_str}"

    if getattr(signature, "input_fields", None):
        lines.extend(["", "Inputs:"])
        lines.extend(describe_field(name, field) for name, field in signature.input_fields.items())
    if getattr(signature, "output_fields", None):
        lines.extend(["", "Outputs:"])
        lines.extend(describe_field(name, field) for name, field in signature.output_fields.items())
    return "\n".join(lines)
