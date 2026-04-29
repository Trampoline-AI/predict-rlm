from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from predict_rlm.trace import RunTrace


def render_inputs(inputs: Mapping[str, Any]) -> str:
    if not inputs:
        return "(no inputs recorded)"
    lines: list[str] = []
    for key, value in inputs.items():
        lines.append(f"{key}: {_safe_value_summary(value)}")
    return "\n".join(lines)


def render_trace(trace: Any, tag: str = "TRACE") -> str:
    if isinstance(trace, RunTrace):
        return render_run_trace(trace, tag)
    if not trace:
        return f"({tag}: no trace captured)"

    parts: list[str] = []
    steps = list(trace) if isinstance(trace, list) else []
    for i, step in enumerate(steps):
        if not isinstance(step, Mapping):
            continue
        label = f"{tag}: step {i + 1} of {len(steps)}"
        parts.append(_banner(label, "*"))
        reasoning = step.get("reasoning", "")
        code = step.get("code", "")
        output = step.get("output", "")
        if reasoning:
            parts.append(f"REASONING:\n{reasoning}")
        if code:
            parts.append(f"CODE:\n{code}")
        if output:
            parts.append(f"OUTPUT:\n{output}")
    return "\n\n".join(parts) if parts else f"({tag}: no renderable trace captured)"


def render_run_trace(trace: RunTrace, tag: str = "TRACE") -> str:
    steps = list(getattr(trace, "steps", []) or [])
    if not steps:
        return f"({tag}: no steps captured; status={trace.status})"

    parts: list[str] = []
    for i, step in enumerate(steps):
        label = (
            f"{tag}: step {i + 1} of {len(steps)} - {step.duration_ms:,}ms"
            + (" (ERROR)" if step.error else "")
        )
        parts.append(_banner(label, "*"))
        if step.reasoning:
            parts.append(f"REASONING:\n{step.reasoning}")
        if step.code:
            parts.append(f"CODE:\n{step.code}")
        tool_lines = format_tool_calls(step.tool_calls)
        if tool_lines:
            parts.append("TOOL CALLS:\n" + tool_lines)
        predict_lines = format_predict_calls(step.predict_calls)
        if predict_lines:
            parts.append("PREDICT CALLS:\n" + predict_lines)
        if step.output:
            parts.append(f"OUTPUT:\n{step.output}")
    return "\n\n".join(parts)


def render_case_summary(score: float, feedback: str, trace: RunTrace | None = None) -> str:
    base = f"score={score:.3f}"
    if trace is None:
        return base
    extras = [
        f"status={trace.status}",
        f"{trace.iterations}/{trace.max_iterations} iters",
        f"{trace.duration_ms / 1000:.1f}s total ({trace.duration_ms:,}ms)",
    ]
    main = trace.usage.main
    if main.input_tokens or main.output_tokens:
        extras.append(f"main={main.input_tokens:,}/{main.output_tokens:,} tokens")
    sub = trace.usage.sub
    if sub.input_tokens or sub.output_tokens:
        extras.append(f"sub={sub.input_tokens:,}/{sub.output_tokens:,} tokens")
    if feedback:
        extras.append("feedback_present=yes")
    return base + " | " + " ".join(extras)


def format_tool_calls(tool_calls: list[Any]) -> str:
    if not tool_calls:
        return ""
    lines: list[str] = []
    for call in tool_calls:
        kwargs = getattr(call, "kwargs", {}) or {}
        args = getattr(call, "args", []) or []
        arg_strs = [repr(arg) for arg in args]
        arg_strs.extend(f"{key}={value!r}" for key, value in kwargs.items())
        name = getattr(call, "name", "unknown")
        error = getattr(call, "error", None)
        if error:
            outcome = f"ERROR: {error}"
        else:
            result = str(getattr(call, "result", "") or "")
            if len(result) > 200:
                result = result[:200] + f"... (truncated, {len(str(getattr(call, 'result', '')))} chars)"
            outcome = f"-> {result}"
        duration_ms = getattr(call, "duration_ms", 0)
        lines.append(f"  {name}({', '.join(arg_strs)}) {outcome} ({duration_ms:,}ms)")
    return "\n".join(lines)


def format_predict_calls(groups: list[Any]) -> str:
    if not groups:
        return ""
    lines: list[str] = []
    for group in groups:
        usage = getattr(group, "total_usage", None)
        calls = getattr(group, "calls", []) or []
        if usage is None:
            lines.append(f"  [{len(calls)}x] sig={getattr(group, 'signature', '')!r}")
            continue
        lines.append(
            f"  [{len(calls)}x] sig={group.signature!r} model={group.model} "
            f"total in={usage.input_tokens:,} out={usage.output_tokens:,} "
            f"cost=${usage.cost:.4f}"
        )
    return "\n".join(lines)


def trace_to_json(trace: Any) -> Any:
    if trace is None:
        return None
    if hasattr(trace, "to_exportable_json"):
        return json.loads(trace.to_exportable_json(indent=0))
    return trace


def _banner(text: str, char: str) -> str:
    line = f"** {text} **"
    rule = char * len(line)
    return f"{rule}\n{line}\n{rule}"


def _safe_value_summary(value: Any) -> str:
    path = getattr(value, "path", None)
    if path:
        return f"File(path={path!r})"
    if isinstance(value, bytes):
        return f"<bytes {len(value)} bytes>"
    text = repr(value)
    if len(text) > 500:
        return text[:500] + f"... (truncated, {len(text)} chars)"
    return text
