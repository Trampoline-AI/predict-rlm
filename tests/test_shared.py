"""Tests for _shared.py: format_tool_docs_full and build_rlm_signatures."""

from unittest.mock import MagicMock, patch

import dspy

from predict_rlm._shared import build_rlm_signatures, format_tool_docs_full


class TestFormatToolDocsFull:
    def test_empty_tools_returns_empty_string(self):
        assert format_tool_docs_full({}) == ""

    def test_single_tool_with_signature_and_docstring(self):
        def fetch_page(url: str, timeout: int) -> str:
            """Fetch a web page.

            Args:
                url: The URL to fetch.
                timeout: Request timeout in seconds.

            Returns:
                The page HTML content.
            """
            return ""

        result = format_tool_docs_full({"fetch_page": fetch_page})
        assert "### `await fetch_page(url: str, timeout: int) -> str`" in result
        assert "Fetch a web page." in result
        assert "Args:" in result
        assert "Returns:" in result

    def test_tool_without_docstring(self):
        def no_docs(x: str) -> str:
            return x

        no_docs.__doc__ = None
        result = format_tool_docs_full({"no_docs": no_docs})
        assert "No description" in result

    def test_tool_with_uninspectable_signature(self):
        mock_fn = MagicMock(spec=lambda: None)
        # Override signature inspection to raise
        mock_fn.__doc__ = "A mock tool."
        with patch("predict_rlm._shared.inspect.signature", side_effect=ValueError):
            result = format_tool_docs_full({"mock_fn": mock_fn})
        assert "mock_fn(...)" in result
        assert "A mock tool." in result

    def test_multiple_tools_all_listed(self):
        def tool_a() -> str:
            """Tool A."""
            return ""

        def tool_b() -> str:
            """Tool B."""
            return ""

        result = format_tool_docs_full({"tool_a": tool_a, "tool_b": tool_b})
        assert "tool_a" in result
        assert "tool_b" in result
        assert "## Additional Tools" in result

    def test_header_mentions_async_and_gather(self):
        def dummy() -> str:
            """Dummy."""
            return ""

        result = format_tool_docs_full({"dummy": dummy})
        assert "await" in result
        assert "asyncio.gather()" in result


class TestBuildRlmSignatures:
    ACTION_TEMPLATE = (
        "You have inputs: {inputs}.\n"
        "Output fields:\n{output_fields}\n"
        "Submit: {final_output_names}"
    )

    def test_returns_action_and_extract_signatures(self):
        sig = dspy.Signature("question -> answer")
        action, extract = build_rlm_signatures(
            sig, self.ACTION_TEMPLATE, {}, format_tool_docs_full
        )
        assert "variables_info" in action.input_fields
        assert "repl_history" in action.input_fields
        assert "iteration" in action.input_fields
        assert "reasoning" in action.output_fields
        assert "code" in action.output_fields

        assert "variables_info" in extract.input_fields
        assert "repl_history" in extract.input_fields
        assert "answer" in extract.output_fields

    def test_tool_docs_in_action_instructions(self):
        def my_tool(x: str) -> str:
            """Does something useful."""
            return x

        sig = dspy.Signature("question -> answer")
        action, _ = build_rlm_signatures(
            sig, self.ACTION_TEMPLATE, {"my_tool": my_tool}, format_tool_docs_full
        )
        assert "my_tool" in action.instructions

    def test_skill_instructions_appended(self):
        sig = dspy.Signature("question -> answer")
        action, _ = build_rlm_signatures(
            sig,
            self.ACTION_TEMPLATE,
            {},
            format_tool_docs_full,
            skill_instructions="Use pdfplumber for PDF extraction.",
        )
        assert "## Skills" in action.instructions
        assert "Use pdfplumber for PDF extraction." in action.instructions

    def test_file_instructions_appended(self):
        sig = dspy.Signature("question -> answer")
        action, _ = build_rlm_signatures(
            sig,
            self.ACTION_TEMPLATE,
            {},
            format_tool_docs_full,
            file_instructions="Input files are mounted at /sandbox/input.",
        )
        assert "Input files are mounted at /sandbox/input." in action.instructions

    def test_original_signature_instructions_preserved(self):
        sig = dspy.Signature("question -> answer", "Be concise and accurate.")
        action, extract = build_rlm_signatures(
            sig, self.ACTION_TEMPLATE, {}, format_tool_docs_full
        )
        assert "Be concise and accurate." in action.instructions
        assert "Be concise and accurate." in extract.instructions

    def test_no_skill_or_file_instructions(self):
        sig = dspy.Signature("question -> answer")
        action, _ = build_rlm_signatures(
            sig, self.ACTION_TEMPLATE, {}, format_tool_docs_full
        )
        assert "## Skills" not in action.instructions

    def test_extract_sig_includes_output_fields(self):
        sig = dspy.Signature("question -> answer, confidence: float")
        _, extract = build_rlm_signatures(
            sig, self.ACTION_TEMPLATE, {}, format_tool_docs_full
        )
        assert "answer" in extract.output_fields
        assert "confidence" in extract.output_fields
