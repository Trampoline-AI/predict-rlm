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

1. Load {{TRACES_FILE_MOUNT}} with JSON parsing, not broad regex over raw text.
2. Bucket records by score into bottom, middle, and top groups.
3. Read bottom records for corrective rules, top records for strategic rules,
   and middle records for directional rules.
4. Scan tools and target signature for novel capabilities that would help across
   the use-case list.
5. Keep the edit surgical: identify kept, modified, added, and removed rules.
6. Emit one `generalization_check` audit line per rule. Each line starts with
   `[KEPT|MODIFIED|NEW|REMOVED]` and includes: grounding, use case, principle,
   counterfactual_1, counterfactual_2. The counterfactuals must span different
   {{COUNTERFACTUAL_AXIS_NAME}}.
"""


MERGE_PROPOSER_TEMPLATE = """\
Synthesize one merged skill-instructions text for {{AGENT_TYPE}} from two
parents that evolved from a common ancestor.

The merged skill must inherit branch-specific strengths without concatenating
both parents. It must remain a surgical, length-disciplined edit and must work
across these use cases:
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

- `current_instructions_a` and `current_instructions_b`: full parent skill texts.
- `common_ancestor_instructions`: common ancestor text; use it to identify
  inherited text, branch-unique text, and differently rewritten text.
- `paired_traces_file`, mounted at {{PAIRED_TRACES_FILE_MOUNT}}: JSONL where each
  row contains the same task for both parents, both rendered traces, both
  feedback fields, both scores, and `winner` set to `a`, `b`, or `tie`.

# Merge Method

1. Identify verbatim text shared by A and B and confirm whether it came from the
   ancestor. This is `[COMMON]` text.
2. Identify text present in the ancestor but rewritten differently by A and B.
   Reconcile it as `[MERGED]` text using paired trace evidence.
3. Identify text unique to A (`[A_ONLY]`) and B (`[B_ONLY]`). Keep it only when
   the paired traces show that branch winning or avoiding a concrete failure.
4. For tasks where both parents fail, add `[NEW]` text only when the trace names
   a concrete actionable runtime fact. `[NEW]` audit lines must cite a `task_id`
   from the paired trace file.
5. Do not exceed `1.10 * max(len(A), len(B))` characters.

# Audit Contract

Every `generalization_check` line starts with exactly one tag:
`[COMMON]`, `[A_ONLY]`, `[B_ONLY]`, `[MERGED]`, or `[NEW]`.

After the tag, include grounding, use case, principle, counterfactual_1, and
counterfactual_2. Counterfactuals must span different {{COUNTERFACTUAL_AXIS_NAME}}.
"""


_AXIS_SINGULAR = COUNTERFACTUAL_AXIS_SINGULAR


def render_template(template: str, spec: AgentSpec) -> str:
    use_cases_bulleted = "\n".join(f"  - {use_case}" for use_case in spec.use_cases)
    grounding_bulleted = "\n".join(
        f"  - {category}: {', '.join(surfaces)}"
        for category, surfaces in spec.runtime_grounding_examples.items()
    )
    axis_name = spec.counterfactual_axis_name
    axis_singular = _AXIS_SINGULAR[axis_name]
    return (
        template.replace("{{AGENT_TYPE}}", spec.agent_type)
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
    new_instructions: str = dspy.OutputField(desc="Full revised skill instructions text")
    generalization_check: list[str] = dspy.OutputField(desc="Audit lines for every rule")


class MergeInstructionsGeneric(dspy.Signature):
    current_instructions_a: str = dspy.InputField(desc="Full skill instructions for parent A")
    current_instructions_b: str = dspy.InputField(desc="Full skill instructions for parent B")
    common_ancestor_instructions: str = dspy.InputField(
        desc="Full skill instructions for the common ancestor"
    )
    paired_traces_file: File = dspy.InputField(
        desc="JSONL file carrying both parents' trajectories side by side"
    )
    new_instructions: str = dspy.OutputField(desc="Full merged skill instructions text")
    generalization_check: list[str] = dspy.OutputField(desc="Merge audit lines")


def build_proposer_signature(spec: AgentSpec) -> type[dspy.Signature]:
    return ImproveInstructionsGeneric.with_instructions(
        render_template(GENERIC_PROPOSER_TEMPLATE, spec)
    )


def build_merge_signature(spec: AgentSpec) -> type[dspy.Signature]:
    return MergeInstructionsGeneric.with_instructions(render_template(MERGE_PROPOSER_TEMPLATE, spec))


_INFRA_TOOL_NAMES = frozenset({"predict", "SUBMIT", "print"})


def agent_spec_from_rlm(
    rlm: dspy.Module,
    *,
    agent_type: str,
    use_cases: list[str],
    runtime_grounding_examples: dict[str, list[str]],
    scoring_description: str,
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
