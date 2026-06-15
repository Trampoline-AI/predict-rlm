from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TERMINAL_BENCH_DIR = ROOT / "examples" / "terminal_bench"
if str(TERMINAL_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(TERMINAL_BENCH_DIR))

from terminal_bench_rlm.skills import (  # noqa: E402
    DEFAULT_TERMINAL_BENCH_SKILL_INSTRUCTIONS,
    build_terminal_bench_skill,
)
from terminal_bench_rlm.tools.tbench_agent import _REMOTE_CONTROLLER_ENV_KEYS  # noqa: E402
from terminal_bench_rlm.web_search import web_search  # noqa: E402


class DummySkill:
    def __init__(self, *, name, instructions, tools):
        self.name = name
        self.instructions = instructions
        self.tools = tools


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(
            {
                "choices": [{"message": {"content": "Answer text"}}],
                "citations": ["https://example.com/source"],
                "search_results": [
                    {
                        "title": "Source",
                        "url": "https://example.com/source",
                        "date": "2026-01-01",
                        "snippet": "Useful snippet",
                        "ignored": "field",
                    },
                    {"title": "Second", "url": "https://example.com/second"},
                ],
            }
        ).encode()


def test_terminal_bench_skill_registers_web_search_tool():
    skill = build_terminal_bench_skill(DummySkill)

    assert skill.tools == {"web_search": web_search}
    assert "await web_search(query, max_results=5)" in skill.instructions
    assert 'domains=["example.com"]' in skill.instructions
    assert "await web_search(query, max_results=5)" in DEFAULT_TERMINAL_BENCH_SKILL_INSTRUCTIONS
    assert 'domains=["example.com"]' in DEFAULT_TERMINAL_BENCH_SKILL_INSTRUCTIONS


def test_perplexity_key_is_forwarded_to_daytona_remote_controller():
    assert "PERPLEXITY_API_KEY" in _REMOTE_CONTROLLER_ENV_KEYS


def test_web_search_returns_cited_json(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode())
        captured["authorization"] = request.headers["Authorization"]
        return FakeResponse()

    monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    payload = json.loads(
        asyncio.run(
            web_search(
                "what is predict-rlm?",
                max_results=1,
                domains=[" example.com "],
                timeout=3,
            )
        )
    )

    assert captured["timeout"] == 3
    assert captured["authorization"] == "Bearer test-key"
    assert captured["body"]["model"] == "sonar"
    assert captured["body"]["messages"][-1]["content"] == "what is predict-rlm?"
    assert captured["body"]["search_domain_filter"] == ["example.com"]
    assert payload == {
        "query": "what is predict-rlm?",
        "domains": ["example.com"],
        "answer": "Answer text",
        "citations": ["https://example.com/source"],
        "search_results": [
            {
                "title": "Source",
                "url": "https://example.com/source",
                "date": "2026-01-01",
                "snippet": "Useful snippet",
            }
        ],
    }
