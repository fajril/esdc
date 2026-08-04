"""Unit tests for esdc.llm_text.strip_thinking_tags."""

from esdc.llm_text import strip_thinking_tags


def test_strip_thinking_tags_removes_qwen3_blocks():
    text = (
        "<thinking>User asks about reserves. Output JSON like "
        '{"title": "..."}</thinking>\n\n{"title": "Cadangan nasional"}'
    )
    assert strip_thinking_tags(text) == '\n\n{"title": "Cadangan nasional"}'


def test_strip_thinking_tags_handles_think_variant_and_unbalanced():
    assert strip_thinking_tags("before <think>a</think> after") == "before  after"
    # Response cut off mid-reasoning: drop from the opening tag to the end, so
    # the unterminated reasoning prose cannot become the title.
    assert strip_thinking_tags("a <thinking>unclosed") == "a "
    assert strip_thinking_tags("<think>reasoning only, truncated") == ""
    # A closing tag with no opener is a stray token, not a block: keep content.
    assert strip_thinking_tags("kept</think> also kept") == "kept also kept"
    assert strip_thinking_tags("plain text") == "plain text"


def test_strip_thinking_tags_block_containing_braces_leaves_json_tail():
    text = (
        "<thinking>Reasoning with braces: {nested: {a: 1}} and more "
        'thoughts</thinking>\n\n{"title": "Cadangan nasional"}'
    )
    assert strip_thinking_tags(text) == '\n\n{"title": "Cadangan nasional"}'
