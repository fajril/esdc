"""Benchmark suite for message conversion functions."""

import json

import pytest

from esdc.server.agent_wrapper import convert_messages_to_langchain


class TestMessageConversionPerformance:
    """Benchmark message conversion performance."""

    @pytest.fixture
    def small_conversation(self):
        """10 message conversation."""
        return [
            {"role": "user", "content": "What are reserves?"},
            {"role": "assistant", "content": "I'll help you with that."},
            {"role": "user", "content": "Show me national reserves"},
            {"role": "assistant", "content": "Here are the national reserves..."},
            {"role": "user", "content": "What about state reserves?"},
            {"role": "assistant", "content": "State reserves data..."},
            {"role": "user", "content": "Compare them"},
            {"role": "assistant", "content": "Comparison..."},
            {"role": "user", "content": "Thank you"},
            {"role": "assistant", "content": "You're welcome!"},
        ]

    @pytest.fixture
    def medium_conversation(self):
        """50 message conversation (25 exchanges)."""
        messages = []
        for i in range(25):
            messages.append({"role": "user", "content": f"Question {i}: " + "x" * 100})
            messages.append(
                {"role": "assistant", "content": f"Answer {i}: " + "y" * 200}
            )
        return messages

    @pytest.fixture
    def large_conversation(self):
        """60 message conversation with tool calls."""
        messages = []
        for i in range(10):
            messages.append({"role": "user", "content": f"Question {i}"})
            messages.append(
                {
                    "role": "assistant",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": f"call_{i}_{j}",
                            "name": "query_data",
                            "arguments": json.dumps({"filter": "test", "id": j}),
                        }
                        for j in range(3)
                    ],
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "output": [
                        {
                            "type": "function_call_output",
                            "call_id": f"call_{i}_{j}",
                            "output": "Result data..." * 10,
                        }
                        for j in range(3)
                    ],
                }
            )
        return messages

    def test_baseline_small_conversation(self, benchmark, small_conversation):
        """Benchmark: 10 message conversion."""
        result = benchmark(convert_messages_to_langchain, small_conversation)
        assert len(result) == 10

    def test_baseline_medium_conversation(self, benchmark, medium_conversation):
        """Benchmark: 50 message conversion."""
        result = benchmark(convert_messages_to_langchain, medium_conversation)
        assert len(result) == 50

    def test_baseline_large_conversation(self, benchmark, large_conversation):
        """Benchmark: 60 message with tools."""
        result = benchmark(convert_messages_to_langchain, large_conversation)
        assert len(result) > 0

    def test_repeated_conversion(self, benchmark, small_conversation):
        """Benchmark: Same conversation converted 100 times."""

        def convert_100_times():
            for _ in range(100):
                convert_messages_to_langchain(small_conversation)

        benchmark(convert_100_times)

    def test_json_parsing_overhead(self, benchmark):
        """Benchmark: JSON parsing in tool calls."""
        messages = [
            {
                "role": "assistant",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": f"call_{i}",
                        "name": "test",
                        "arguments": json.dumps(
                            {
                                "key": "value",
                                "nested": {"data": [1, 2, 3]},
                                "id": i,
                            }
                        ),
                    }
                    for i in range(10)
                ],
            }
        ]

        result = benchmark(convert_messages_to_langchain, messages)
        assert len(result) == 10


if __name__ == "__main__":
    pytest.main([__file__, "--benchmark-only", "-v"])
