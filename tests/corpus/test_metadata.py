import duckdb

from esdc.corpus.metadata import (
    METADATA_PROMPT,
    METADATA_PROMPT_IMAGE,
    llm_extract,
    load_canonical_names,
    normalize_metadata,
    parse_llm_json,
    resolve_entity,
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


def test_resolve_exact_case_insensitive():
    name, conf = resolve_entity("rokan", ["Rokan", "Mahakam"])
    assert name == "Rokan" and conf == 1.0


def test_resolve_substring():
    name, conf = resolve_entity("WK Rokan (PHR)", ["Rokan", "Mahakam"])
    assert name == "Rokan" and 0.8 <= conf < 1.0


def test_resolve_fuzzy():
    name, conf = resolve_entity("Mahakem", ["Rokan", "Mahakam"])
    assert name == "Mahakam" and 0.6 <= conf < 1.0


def test_resolve_no_match_returns_none():
    name, conf = resolve_entity("Blok Antah Berantah", ["Rokan", "Mahakam"])
    assert name is None and conf == 0.0


def test_load_canonical_names_distinct_non_null():
    conn = duckdb.connect()
    conn.execute(
        "CREATE TABLE project_resources (wk_name VARCHAR, field_name VARCHAR, "
        "project_name VARCHAR)"
    )
    conn.execute(
        "INSERT INTO project_resources VALUES "
        "('Rokan', 'Minas', 'POD-1'), "
        "('Rokan', 'Duri', NULL), "
        "(NULL, NULL, NULL), "
        "('Mahakam', 'Minas', 'POD-1')"
    )
    names = load_canonical_names(conn)
    assert sorted(names["wk_name"]) == ["Mahakam", "Rokan"]
    assert sorted(names["field_name"]) == ["Duri", "Minas"]
    assert names["project_name"] == ["POD-1"]


def test_load_canonical_names_missing_table_returns_empty_lists():
    conn = duckdb.connect()
    names = load_canonical_names(conn)
    assert names == {"wk_name": [], "field_name": [], "project_name": []}


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
