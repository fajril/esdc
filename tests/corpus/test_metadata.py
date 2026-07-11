
from esdc.chat.domain_knowledge import doc_schema
from esdc.corpus import metadata
from esdc.corpus.metadata import (
    METADATA_PROMPT,
    METADATA_PROMPT_IMAGE,
    llm_extract,
    normalize_metadata,
    parse_llm_json,
)

# Pinned copy of the pre-refactor METADATA_PROMPT literal. Do not edit — this
# is the byte-identity contract the doc_schema.yaml refactor must preserve.
EXPECTED_METADATA_PROMPT = """You extract metadata from Indonesian oil & gas official documents.
Given the markdown of a document, return ONLY a JSON object with these keys
(use null when unknown, never guess):
- doc_type: "surat" (official letter) | "mom" (minutes of meeting)
  | "ba" (berita acara) | "other"
- doc_number: the document/letter number exactly as written
- doc_date: ISO date YYYY-MM-DD
- subject: perihal or meeting title
- sender: issuing organization or signatory org
- recipient: addressed organization (letters only)
- doc_level: "wk" | "field" | "project" | "unknown" — the scope this document is about
- wk_name: list of working area (wilayah kerja) names mentioned
  (e.g. ["Rokan", "Mahakam"])
- field_name: list of field (lapangan) names mentioned (e.g. ["Duri", "Minas"])
- project_name: list of project or POD names mentioned (e.g. ["POD Duri", "POD Minas"])
- extras: object with doc_type-specific fields, e.g. for mom:
  {{"peserta": [...], "keputusan": [...]}}

Document markdown:
---
{markdown}
---
JSON:"""


def test_metadata_prompt_is_byte_identical_to_legacy():
    assert metadata.METADATA_PROMPT == EXPECTED_METADATA_PROMPT


def test_vocab_tuples_derived_from_schema():
    assert metadata.DOC_TYPES == ("surat", "mom", "ba", "other")
    assert metadata.DOC_LEVELS == ("wk", "field", "project", "unknown")


def test_prompt_contains_every_llm_field():
    for name in doc_schema.llm_field_names():
        assert f"- {name}:" in metadata.METADATA_PROMPT


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
