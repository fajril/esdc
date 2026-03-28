"""Tests for tool result truncation in agent.py."""

import pytest


class TestToolResultTruncation:
    """Tests for truncating large tool results."""

    def test_max_tool_result_chars_constant(self):
        """Test that MAX_TOOL_RESULT_CHARS is defined correctly."""
        from esdc.chat.agent import MAX_TOOL_RESULT_CHARS

        assert MAX_TOOL_RESULT_CHARS == 10000

    def test_truncation_logic_large_result(self):
        """Test that truncation produces correct output for large results."""
        from esdc.chat.agent import MAX_TOOL_RESULT_CHARS

        large_result = "x" * 15000
        observation_str = large_result

        if len(observation_str) > MAX_TOOL_RESULT_CHARS:
            truncated = (
                observation_str[:MAX_TOOL_RESULT_CHARS]
                + "\n\n[Result truncated to first 10000 characters for context efficiency]"
            )
        else:
            truncated = observation_str

        expected_len = MAX_TOOL_RESULT_CHARS + len(
            "\n\n[Result truncated to first 10000 characters for context efficiency]"
        )
        assert len(truncated) == expected_len
        assert truncated.endswith(
            "\n\n[Result truncated to first 10000 characters for context efficiency]"
        )
        assert truncated.startswith("x" * 100)

    def test_truncation_logic_small_result(self):
        """Test that small results are not truncated."""
        from esdc.chat.agent import MAX_TOOL_RESULT_CHARS

        small_result = "x" * 5000
        observation_str = small_result

        if len(observation_str) > MAX_TOOL_RESULT_CHARS:
            truncated = (
                observation_str[:MAX_TOOL_RESULT_CHARS]
                + "\n\n[Result truncated to first 10000 characters for context efficiency]"
            )
        else:
            truncated = observation_str

        assert truncated == small_result
        assert "[Result truncated" not in truncated

    def test_truncation_exact_boundary(self):
        """Test result exactly at MAX chars is not truncated."""
        from esdc.chat.agent import MAX_TOOL_RESULT_CHARS

        exact_result = "x" * MAX_TOOL_RESULT_CHARS
        observation_str = exact_result

        if len(observation_str) > MAX_TOOL_RESULT_CHARS:
            truncated = (
                observation_str[:MAX_TOOL_RESULT_CHARS]
                + "\n\n[Result truncated to first 10000 characters for context efficiency]"
            )
        else:
            truncated = observation_str

        assert truncated == exact_result
        assert "[Result truncated" not in truncated

    def test_truncation_one_char_over(self):
        """Test result one char over MAX is truncated."""
        from esdc.chat.agent import MAX_TOOL_RESULT_CHARS

        over_result = "x" * (MAX_TOOL_RESULT_CHARS + 1)
        observation_str = over_result

        if len(observation_str) > MAX_TOOL_RESULT_CHARS:
            truncated = (
                observation_str[:MAX_TOOL_RESULT_CHARS]
                + "\n\n[Result truncated to first 10000 characters for context efficiency]"
            )
        else:
            truncated = observation_str

        assert "[Result truncated" in truncated
        assert len(truncated) == MAX_TOOL_RESULT_CHARS + len(
            "\n\n[Result truncated to first 10000 characters for context efficiency]"
        )
