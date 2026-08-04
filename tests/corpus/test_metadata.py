from esdc.chat.domain_knowledge import doc_schema
from esdc.corpus import metadata
from esdc.corpus.metadata import (
    METADATA_PROMPT,
    METADATA_PROMPT_IMAGE,
    llm_extract,
    normalize_metadata,
    parse_llm_json,
)

# Pinned copy of the current METADATA_PROMPT literal (post doc_topic split).
# Do not edit casually — this is the byte-identity contract the
# doc_schema.yaml -> METADATA_PROMPT render must preserve.
EXPECTED_METADATA_PROMPT = """You extract metadata from Indonesian oil & gas official documents.
Given the markdown of a document, return ONLY a JSON object with these keys
(use null when unknown, never guess):
- doc_type: one of "uu" (undang-undang) | "perpu" (peraturan pengganti UU) | "mk"
  (putusan Mahkamah Konstitusi) | "pp" (peraturan pemerintah) | "perpres"
  (peraturan presiden) | "permen" (peraturan menteri) | "kepres"
  (keputusan presiden) | "kepmen" (keputusan menteri) | "ptk" (pedoman
  tata kerja SKK Migas) | "sop" (standard operating procedure) |
  "letter" (official letter: persetujuan/edaran/umum) | "mom" (minutes
  of meeting) | "ba" (berita acara) | "note" (non-binding note) |
  "contract" (binding commercial contract, e.g. PSC/GSA) | "book"
  (bound proposal/approval book, e.g. POD/WP&B) | "others"
- doc_topic: list of business object(s) this document concerns: "pod_i" (POD I,
  first/ministerial POD) | "pod" (POD) | "pofd" (POFD) | "opl" (OPL) |
  "opll" (OPLL, optimasi pengembangan lapangan-lapangan) | "wpnb"
  (WP&B) | "afe" (AFE) | "psc" (production sharing contract) | "gsa"
  (gas sales agreement) | "monitoring_pod" (monitoring POD) | "others"
  — usually exactly one value
- doc_number: the document/letter number exactly as written
- doc_date: ISO date YYYY-MM-DD
- subject: perihal or meeting title
- sender: issuing organization or signatory org
- recipient: addressed organization (letters only)
- doc_level: "wk" | "field" | "project" | "regulation" | "unknown" — the scope this document is about
- wk_name: list of working area (wilayah kerja) names mentioned
  (e.g. ["Rokan", "Mahakam"])
- field_name: list of field (lapangan) names mentioned (e.g. ["Duri", "Minas"])
- project_name: list of project or POD names mentioned (e.g. ["POD Duri", "POD Minas"])
- pod_name: list of POD/plan-of-development name(s) this document approves or discusses, exactly as written (e.g. "POD I Lapangan Abadi", "Optimasi Pengembangan Lapangan Pedada"); null if none
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
    assert metadata.DOC_TYPES == (
        "uu",
        "perpu",
        "mk",
        "pp",
        "perpres",
        "permen",
        "kepres",
        "kepmen",
        "ptk",
        "sop",
        "letter",
        "mom",
        "ba",
        "note",
        "contract",
        "book",
        "others",
    )
    assert metadata.DOC_LEVELS == ("wk", "field", "project", "regulation", "unknown")
    assert metadata.DOC_TOPICS == (
        "pod_i",
        "pod",
        "pofd",
        "opl",
        "opll",
        "wpnb",
        "afe",
        "psc",
        "gsa",
        "monitoring_pod",
        "others",
    )


def test_doc_types_new_vocab():
    assert metadata.DOC_TYPES == (
        "uu",
        "perpu",
        "mk",
        "pp",
        "perpres",
        "permen",
        "kepres",
        "kepmen",
        "ptk",
        "sop",
        "letter",
        "mom",
        "ba",
        "note",
        "contract",
        "book",
        "others",
    )


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
    assert out["doc_type"] == "others"
    assert out["doc_level"] == "unknown"


def test_normalize_metadata_maps_legacy_doc_types():
    assert normalize_metadata({"doc_type": "surat"})["doc_type"] == "letter"
    assert normalize_metadata({"doc_type": "other"})["doc_type"] == "others"


def test_normalize_metadata_clamps_unknown_to_others():
    assert normalize_metadata({"doc_type": "invoice"})["doc_type"] == "others"


def test_normalize_metadata_is_case_insensitive():
    assert normalize_metadata({"doc_type": "UU"})["doc_type"] == "uu"
    assert normalize_metadata({"doc_type": "Psc"})["doc_type"] == "contract"
    assert normalize_metadata({"doc_type": "Surat"})["doc_type"] == "letter"
    assert normalize_metadata({"doc_level": "WK"})["doc_level"] == "wk"


def test_normalize_metadata_non_string_doc_type_clamped():
    assert normalize_metadata({"doc_type": 3})["doc_type"] == "others"
    assert normalize_metadata({"doc_level": ["wk"]})["doc_level"] == "unknown"


def test_llm_extract_uses_caller():
    result = llm_extract("# Surat\nisi", lambda prompt: '{"doc_type": "letter"}')
    assert result["doc_type"] == "letter"
    assert result["doc_level"] == "unknown"  # normalized default


def test_llm_extract_includes_filename_hint():
    from esdc.corpus import metadata

    seen = {}

    def caller(prompt: str) -> str:
        seen["prompt"] = prompt
        return "{}"

    metadata.llm_extract("body text", caller, filename="letter-2024.pdf")
    assert (
        "Filename (may hint doc_type/date/subject): letter-2024.pdf" in seen["prompt"]
    )
    assert "body text" in seen["prompt"]


def test_llm_extract_no_filename_matches_base_prompt():
    from esdc.corpus import metadata

    seen = {}

    def caller(prompt: str) -> str:
        seen["prompt"] = prompt
        return "{}"

    metadata.llm_extract("body text", caller)
    assert "Filename (may hint" not in seen["prompt"]


def test_metadata_image_prompt_with_and_without_filename():
    from esdc.corpus import metadata

    base = metadata.metadata_image_prompt()
    assert base == metadata.METADATA_PROMPT_IMAGE
    hinted = metadata.metadata_image_prompt("scan.pdf")
    assert hinted.startswith("Filename (may hint doc_type/date/subject): scan.pdf")
    assert metadata.METADATA_PROMPT_IMAGE in hinted


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


# --------------------------------------------------------------------------
# legacy doc_type -> doc_type + doc_topic seeding
# --------------------------------------------------------------------------


def test_normalize_metadata_seeds_topic_from_legacy_psc():
    out = normalize_metadata({"doc_type": "psc"})
    assert out["doc_type"] == "contract"
    assert out["doc_topic"] == ["psc"]


def test_normalize_metadata_seeds_topic_from_legacy_gsa():
    out = normalize_metadata({"doc_type": "gsa"})
    assert out["doc_type"] == "contract"
    assert out["doc_topic"] == ["gsa"]


def test_normalize_metadata_seeds_topic_from_legacy_pod():
    out = normalize_metadata({"doc_type": "pod"})
    assert out["doc_type"] == "book"
    assert out["doc_topic"] == ["pod"]


def test_normalize_metadata_legacy_seed_does_not_clobber_existing_topic():
    out = normalize_metadata({"doc_type": "pod", "doc_topic": ["pod_i"]})
    assert out["doc_type"] == "book"
    assert out["doc_topic"] == ["pod_i"]


# --------------------------------------------------------------------------
# doc_topic normalization
# --------------------------------------------------------------------------


def test_normalize_metadata_topic_scalar_wrapped_in_list():
    assert normalize_metadata({"doc_topic": "wpnb"})["doc_topic"] == ["wpnb"]


def test_normalize_metadata_topic_invalid_entry_clamped_to_others():
    assert normalize_metadata({"doc_topic": "bogus"})["doc_topic"] == ["others"]
    assert normalize_metadata({"doc_topic": ["wpnb", "bogus"]})["doc_topic"] == [
        "wpnb",
        "others",
    ]


def test_normalize_metadata_topic_deduped():
    assert normalize_metadata({"doc_topic": ["wpnb", "wpnb", "afe"]})["doc_topic"] == [
        "wpnb",
        "afe",
    ]


def test_normalize_metadata_topic_none_stays_none():
    assert normalize_metadata({"doc_topic": None})["doc_topic"] is None
    assert normalize_metadata({})["doc_topic"] is None


# --------------------------------------------------------------------------
# deterministic doc_level rules
# --------------------------------------------------------------------------


def test_normalize_metadata_doc_type_rule_sets_regulation_and_strips_entities():
    out = normalize_metadata(
        {
            "doc_type": "uu",
            "doc_level": "wk",
            "wk_name": ["Rokan"],
            "field_name": ["Duri"],
            "project_name": ["POD Duri"],
        }
    )
    assert out["doc_level"] == "regulation"
    assert out["wk_name"] is None
    assert out["field_name"] is None
    assert out["project_name"] is None


def test_normalize_metadata_topic_rule_single_implied_level_applies():
    out = normalize_metadata({"doc_topic": ["pod"], "doc_level": "field"})
    assert out["doc_level"] == "project"


def test_normalize_metadata_topic_rule_conflicting_implied_level_unchanged():
    out = normalize_metadata({"doc_topic": ["pod", "wpnb"], "doc_level": "field"})
    assert out["doc_level"] == "field"


def test_normalize_metadata_topic_rule_multiple_topics_same_implied_level_applies():
    out = normalize_metadata({"doc_topic": ["pod", "afe"], "doc_level": "field"})
    assert out["doc_level"] == "project"


def test_normalize_metadata_doc_type_rule_beats_topic_rule():
    out = normalize_metadata({"doc_type": "uu", "doc_topic": ["pod"]})
    assert out["doc_level"] == "regulation"


def test_normalize_metadata_no_rule_types_and_topics_untouched():
    out = normalize_metadata(
        {"doc_type": "letter", "doc_topic": ["gsa"], "doc_level": "wk"}
    )
    assert out["doc_type"] == "letter"
    assert out["doc_topic"] == ["gsa"]
    assert out["doc_level"] == "wk"


# --------------------------------------------------------------------------
# reasoning block handling (Qwen3, DeepSeek-R1)
# --------------------------------------------------------------------------


def test_parse_llm_json_strips_thinking_block_with_braces():
    # Reasoning models include <thinking>…</thinking> blocks in content.
    # The block contains braces that defeat the greedy \{.*\} regex if not
    # stripped first: it matches from the first { in <thinking> to the last
    # } in the JSON, consuming part of the reasoning, then fails to parse.
    raw = """<thinking>
    Let me analyze this document structure. I see { and } braces in the reasoning.
    The actual JSON should come after.
    </thinking>
    {"doc_type": "letter"}"""
    assert parse_llm_json(raw) == {"doc_type": "letter"}


def test_parse_llm_json_no_tags_unchanged():
    # Existing behavior: tag-free input is unaffected.
    raw = '{"doc_type": "book"}'
    assert parse_llm_json(raw) == {"doc_type": "book"}


def test_parse_llm_json_thinking_with_invalid_json_still_returns_empty():
    # Thinking block stripped, but resulting content is still unparseable.
    raw = """<thinking>
    Some reasoning with { and } inside.
    </thinking>
    not valid json at all"""
    assert parse_llm_json(raw) == {}
