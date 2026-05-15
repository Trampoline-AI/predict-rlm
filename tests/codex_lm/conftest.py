import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import dspy
import pytest
from dspy_codex_lm.lm import CodexLM


@pytest.fixture(autouse=True)
def reset_dspy_cache():
    dspy.cache.memory_cache.clear()
    prev_disk = dspy.cache.enable_disk_cache
    dspy.cache.enable_disk_cache = False
    yield
    dspy.cache.memory_cache.clear()
    dspy.cache.enable_disk_cache = prev_disk


@pytest.fixture(autouse=True)
def _disable_codex_retries_in_tests(monkeypatch):
    """By default, tests see CodexLM behave like the pre-tenacity version:
    one attempt, no retry, no backoff. Stream-error tests rely on a single
    ``iter(events)`` side-effect that would be exhausted on retry. Tests
    that want to exercise retry behaviour (``tests/test_auto_retry.py``)
    re-raise this fixture's values to enable retries with zero backoff.
    """
    monkeypatch.setattr("dspy_codex_lm.lm.CODEX_STREAM_MAX_ATTEMPTS", 1)
    monkeypatch.setattr("dspy_codex_lm.lm.CODEX_STREAM_WAIT_MULTIPLIER", 0.0)
    monkeypatch.setattr("dspy_codex_lm.lm.CODEX_STREAM_WAIT_MAX", 0.0)


@pytest.fixture
def fake_auth_file(tmp_path: Path) -> Path:
    path = tmp_path / "auth.json"
    path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "fake-access-token",
                    "account_id": "fake-account-id",
                    "refresh_token": "fake-refresh",
                    "id_token": "fake-id",
                }
            }
        )
    )
    return path


@pytest.fixture
def lm() -> CodexLM:
    return CodexLM(
        model="gpt-5.3-codex",
        access_token="fake-access",
        account_id="fake-account",
    )


def make_text_delta(delta: str) -> SimpleNamespace:
    return SimpleNamespace(type="response.output_text.delta", delta=delta)


def make_completed(
    text: str = "",
    input_tokens: int = 10,
    output_tokens: int = 5,
    cached_tokens: int = 0,
    model: str = "gpt-5.3-codex",
    response_id: str = "resp_test",
) -> SimpleNamespace:
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
        output_tokens_details=SimpleNamespace(reasoning_tokens=0),
    )
    # make it model_dump-able like a pydantic object would be
    usage.model_dump = lambda: {  # type: ignore[attr-defined]
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_tokens_details": {"cached_tokens": cached_tokens},
        "output_tokens_details": {"reasoning_tokens": 0},
    }
    return SimpleNamespace(
        type="response.completed",
        response=SimpleNamespace(id=response_id, model=model, usage=usage),
    )


def build_stream_events(text: str, **usage_kwargs: Any) -> list[SimpleNamespace]:
    """Build a fake Codex SSE event list that yields `text` one chunk at a time."""
    events = [make_text_delta(c) for c in text]
    events.append(make_completed(text=text, **usage_kwargs))
    return events
