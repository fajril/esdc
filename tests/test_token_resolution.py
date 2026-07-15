"""resolve_context_tokens prefers provider-reported usage."""

from langchain_core.messages import AIMessage, HumanMessage

from esdc.chat.token_counter import resolve_context_tokens


def test_prefers_reported_usage_from_last_ai_message():
    msgs = [
        HumanMessage(content="hi"),
        AIMessage(
            content="old",
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        ),
        HumanMessage(content="again"),
        AIMessage(
            content="new",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
            },
        ),
    ]
    tokens, exact = resolve_context_tokens(msgs)
    assert tokens == 120
    assert exact is True


def test_falls_back_to_estimate_without_usage():
    msgs = [HumanMessage(content="x" * 400), AIMessage(content="y" * 400)]
    tokens, exact = resolve_context_tokens(msgs)
    assert exact is False
    assert tokens >= 200  # 800 chars // 4 heuristic


def test_zero_total_usage_is_ignored():
    msgs = [
        AIMessage(
            content="z" * 400,
            usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
    ]
    tokens, exact = resolve_context_tokens(msgs)
    assert exact is False
