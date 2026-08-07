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


def test_strip_thinking_tags_mixed_close_spelling_is_balanced():
    # Models are not consistent about the close spelling; a mixed block must
    # be stripped as balanced, not treated as truncated reasoning (which
    # would drop everything to end-of-string, JSON included).
    assert (
        strip_thinking_tags('<thinking>plan briefly</think>\n{"a": 1}') == '\n{"a": 1}'
    )
    assert (
        strip_thinking_tags('<think>plan briefly</thinking>\n{"a": 1}') == '\n{"a": 1}'
    )


def test_strip_thinking_tags_tolerates_attributes_and_newlines_in_tags():
    assert strip_thinking_tags('<think lang="en">r</think>{"a": 1}') == '{"a": 1}'
    assert strip_thinking_tags("<thinking\n>r</thinking>tail") == "tail"


def test_strip_thinking_tags_leaves_tags_that_merely_start_with_think():
    # The word boundary after think/thinking is what keeps the attribute
    # tolerance above from swallowing unrelated tags. Without it, <thinker>
    # matches as an opener and the unclosed pass drops the rest of the text.
    text = '<thinker>not reasoning</thinker> {"a": 1}'
    assert strip_thinking_tags(text) == text
