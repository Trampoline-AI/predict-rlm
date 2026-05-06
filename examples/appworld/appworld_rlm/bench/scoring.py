from __future__ import annotations

import json
from typing import Any

APPWORLD_EVALUATION_TOOLS = {"evaluate_appworld_task", "run_appworld_program"}


def extract_runner_payload(prediction: Any) -> dict[str, Any] | None:
    """Return the last scored AppWorld tool payload captured in a PredictRLM trace."""
    trace = getattr(prediction, "trace", None)
    for step in reversed(list(getattr(trace, "steps", []) or [])):
        for call in reversed(list(getattr(step, "tool_calls", []) or [])):
            name = getattr(call, "name", None)
            if name not in APPWORLD_EVALUATION_TOOLS:
                continue
            if getattr(call, "error", None):
                return {
                    "score": 0.0,
                    "success": False,
                    "feedback": f"{name} error: {call.error}",
                }
            result = getattr(call, "result", None)
            if isinstance(result, str):
                return json.loads(result)
            if isinstance(result, dict):
                return result
            return {
                "score": 0.0,
                "success": False,
                "feedback": f"{name} returned unsupported payload: {type(result).__name__}",
            }
    return None


def score_prediction_result(prediction: Any) -> tuple[float, str]:
    payload = extract_runner_payload(prediction)
    if payload is None:
        final_answer = str(getattr(prediction, "final_answer", "") or "")
        feedback = (
            "RLM did not call evaluate_appworld_task or run_appworld_program; "
            "no AppWorld evaluator score is available."
        )
        if final_answer:
            feedback = f"{feedback}\nFinal answer: {final_answer}"
        return 0.0, feedback
    return score_runner_result(payload)


def score_runner_result(payload: str | dict[str, Any]) -> tuple[float, str]:
    data = json.loads(payload) if isinstance(payload, str) else payload
    score = max(0.0, min(1.0, float(data.get("score", 0.0) or 0.0)))
    success = bool(data.get("success", score >= 1.0))
    feedback = str(data.get("feedback") or "")
    stderr = str(data.get("stderr") or "")
    if success and score >= 1.0:
        return 1.0, feedback or "AppWorld evaluator reported success"
    parts = [f"AppWorld score={score:.3f}"]
    if feedback:
        parts.append(feedback)
    if stderr:
        parts.append(stderr)
    return score, "\n".join(parts)
