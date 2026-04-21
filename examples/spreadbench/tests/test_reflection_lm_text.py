"""Regression tests for GEPA reflection-LM response normalization."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from lib.optimize import _coerce_reflection_lm_text  # noqa: E402


def test_reflection_lm_text_accepts_classic_dspy_list_response():
    assert _coerce_reflection_lm_text([" proposed instructions "]) == (
        " proposed instructions "
    )


def test_reflection_lm_text_extracts_dict_list_item():
    response = [{"text": "new skill instructions"}]

    assert _coerce_reflection_lm_text(response) == "new skill instructions"


def test_reflection_lm_text_extracts_openai_responses_payload():
    response = {
        "id": "resp_123",
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "<proposal>new skill instructions</proposal>",
                    }
                ],
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    assert _coerce_reflection_lm_text(response) == (
        "<proposal>new skill instructions</proposal>"
    )


def test_reflection_lm_text_extracts_chat_completion_payload():
    response = {
        "choices": [
            {"message": {"content": "chat-style instructions"}},
        ]
    }

    assert _coerce_reflection_lm_text(response) == "chat-style instructions"


def test_reflection_lm_text_raises_for_unknown_dict_shape():
    with pytest.raises(TypeError, match="non-text response"):
        _coerce_reflection_lm_text({"usage": {"input_tokens": 10}})
