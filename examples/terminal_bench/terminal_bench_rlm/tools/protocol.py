from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RunnerError:
    type: str
    message: str
    args: list[Any] | None = None

    @classmethod
    def from_exception(cls, exc: BaseException) -> RunnerError:
        return cls(
            type=type(exc).__name__,
            message=str(exc),
            args=list(getattr(exc, "args", ())),
        )

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> RunnerError:
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        return cls(
            type=str(data.get("type") or payload.get("type") or "RuntimeError"),
            message=str(payload.get("message") or ""),
            args=data.get("args") or payload.get("args"),
        )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": self.type, "message": self.message}
        if self.args is not None:
            payload["args"] = self.args
        return payload


def request(request_id: int, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}


def response_ok(request_id: Any, result: Any | None = None) -> dict[str, Any]:
    return {"id": request_id, "ok": True, "result": {} if result is None else result}


def response_error(request_id: Any, error: RunnerError | BaseException) -> dict[str, Any]:
    runner_error = error if isinstance(error, RunnerError) else RunnerError.from_exception(error)
    return {"id": request_id, "ok": False, "error": runner_error.to_payload()}


def dumps(message: dict[str, Any]) -> str:
    return json.dumps(message, default=str, separators=(",", ":"))


def loads(line: str) -> dict[str, Any]:
    message = json.loads(line)
    if not isinstance(message, dict):
        raise ValueError("Protocol line must decode to a JSON object")
    return message
