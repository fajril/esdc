
from esdc.corpus.metadata import (
    METADATA_PROMPT,
    METADATA_PROMPT_IMAGE,
    llm_extract,
    normalize_metadata,
    parse_llm_json,
)


def test_parse_llm_json_strips_fences():
    raw = '```json\n{"doc_type": "surat"}\n```'
    assert parse_llm_json(raw) == {"doc_type": "surat"}


def test_parse_llm_json_invalid_returns_empty():
    assert parse_llm_json("sorry, I cannot") == {}


def test_normalize_clamps_vocabulary():
    out = normalize_metadata({"doc_type": "invoice", "doc_level": "galaxy"})
    assert out["doc_type"] == "other"
    assert out["doc_level"] == "unknown"


def test_llm_extract_uses_caller():
    result = llm_extract("# Surat\nisi", lambda prompt: '{"doc_type": "surat"}')
    assert result["doc_type"] == "surat"
    assert result["doc_level"] == "unknown"  # normalized default


def test_metadata_prompt_image_excludes_markdown_section_includes_keys():
    assert "Document markdown:" not in METADATA_PROMPT_IMAGE
    assert "doc_type" in METADATA_PROMPT_IMAGE
    assert "extras" in METADATA_PROMPT_IMAGE
    assert METADATA_PROMPT_IMAGE != METADATA_PROMPT


def test_parse_llm_json_nested_braces_in_extras():
    raw = 'here you go {"extras": {"peserta": ["A"]}} thanks'
    assert parse_llm_json(raw) == {"extras": {"peserta": ["A"]}}


def test_parse_llm_json_trailing_chatter_no_closing_brace():
    # Regex is greedy `\{.*\}`; since there is no further `}` after the
    # JSON's own closing brace, the greedy match still lands on exactly
    # the JSON object and parses successfully (documents actual behavior,
    # not changing the regex).
    raw = 'here you go {"a": 1} thanks'
    assert parse_llm_json(raw) == {"a": 1}
