"""Tests for the nested triple-quote SyntaxError handling.

When an RLM emits Python code that wraps a multi-line string in ``\"\"\"``
and that content itself contains ``\"\"\"``, Python's tokenizer fails. We
do NOT auto-rewrite the code — the agent gets an enhanced error message
with an actionable hint and can self-correct on its next REPL iteration.

These tests anchor the Python-level bug (so we notice if Python ever
fixes it) and verify the detection heuristic used by the enhanced error
path in ``PredictRLM._aexecute_iteration``.
"""

from __future__ import annotations

import ast

import pytest


class TestPythonParserBugAnchor:
    """Anchor: confirm Python itself rejects nested triple-quotes.

    If these ever stop failing, Python changed its parser and the
    enhanced error hint in PredictRLM is no longer needed.
    """

    def test_nested_triple_quote_literal_rejected_by_parser(self):
        broken = 'x = """pre\n"""inside"""\npost"""\nprint(x)'
        with pytest.raises(SyntaxError):
            ast.parse(broken)

    def test_nested_triple_quote_with_code_block_content_rejected(self):
        broken = (
            'new_instructions = """# Skill\n'
            "\n"
            "```python\n"
            "def compute():\n"
            '    """docstring lives here"""\n'
            "    return 42\n"
            "```\n"
            '"""\n'
            "SUBMIT(new_instructions=new_instructions)\n"
        )
        with pytest.raises(SyntaxError):
            ast.parse(broken)


class TestTripleQuoteDetectionHeuristic:
    """The detection heuristic: code.count('\"\"\"') >= 3 plus error keywords."""

    def test_heuristic_fires_on_even_count(self):
        code = 'x = """pre\n"""inside"""\npost"""\nprint(x)'
        assert code.count('"""') >= 3
        with pytest.raises(SyntaxError):
            ast.parse(code)
        err = ""
        try:
            ast.parse(code)
        except SyntaxError as e:
            err = str(e).lower()
        assert "invalid syntax" in err or "unterminated" in err

    def test_heuristic_fires_on_odd_count(self):
        code = (
            'new_instructions = """# Skill\n'
            "\n"
            "```python\n"
            "def compute():\n"
            '    """docstring lives here"""\n'
            "    return 42\n"
            "```\n"
            '"""\n'
        )
        assert code.count('"""') >= 3
        with pytest.raises(SyntaxError):
            ast.parse(code)

    def test_heuristic_does_not_fire_on_valid_triple_quoted_string(self):
        code = 'x = """hello\nworld"""\nprint(x)'
        assert code.count('"""') == 2  # exactly 2 = one pair, no nesting
        ast.parse(code)  # should not raise

    def test_heuristic_does_not_fire_on_non_triple_quote_syntax_error(self):
        code = "x = \n"
        assert code.count('"""') == 0
        with pytest.raises(SyntaxError):
            ast.parse(code)
