from __future__ import annotations

import json

import pytest

from esdc.knowledge.extractor import extract_knowledge
from esdc.knowledge.guideline import load_guideline

META = {"doc_type": "mom", "subject": "MoM POD I Duri", "doc_date": "2023-04-01"}


def _caller(payload):
    return lambda prompt: json.dumps(payload)


def test_happy_path_parses_entities_claims_unknowns():
    payload = {
        "entities": [{"type": "pod", "name": "POD I Duri"}],
        "claims": [
            {
                "type": "issue",
                "subject": "POD I Duri",
                "subject_type": "pod",
                "predicate": "delay_cause",
                "value": "rig availability",
                "evidence": "drilling delayed due to rig availability",
            }
        ],
        "unknown_types": [
            {"kind": "claim_type", "name": "hse_incident", "why": "recurring"}
        ],
    }
    result = extract_knowledge("# MoM", META, load_guideline(), _caller(payload))
    assert result.entities == [{"type": "pod", "name": "POD I Duri"}]
    assert result.claims[0]["predicate"] == "delay_cause"
    assert result.unknown_types[0]["name"] == "hse_incident"


def test_bad_items_are_dropped_not_fatal():
    payload = {
        "entities": [
            {"type": "starship", "name": "Enterprise"},  # unknown entity type
            {"type": "field"},  # missing name
            {"type": "field", "name": "Duri"},
        ],
        "claims": [
            {"type": "not_a_type", "predicate": "x", "value": 1},  # bad type
            {"type": "economics", "value": 5},  # no predicate
            {"type": "economics", "predicate": "npv_musd", "value": 100},
        ],
        "unknown_types": "garbage",
    }
    result = extract_knowledge("# Doc", META, load_guideline(), _caller(payload))
    assert result.entities == [{"type": "field", "name": "Duri"}]
    assert len(result.claims) == 1
    assert result.unknown_types == []


def test_code_fenced_json_is_parsed():
    payload = {"entities": [], "claims": [], "unknown_types": []}
    caller = lambda p: f"```json\n{json.dumps(payload)}\n```"  # noqa: E731
    result = extract_knowledge("# Doc", META, load_guideline(), caller)
    assert result.claims == []


def test_unparseable_response_raises_value_error():
    with pytest.raises(ValueError):
        extract_knowledge(
            "# Doc", META, load_guideline(), lambda p: "I cannot help with that"
        )


def test_braces_but_invalid_json_raises_value_error():
    """Regression: malformed JSON with braces present must raise, not silently parse as {}."""
    with pytest.raises(ValueError):
        extract_knowledge(
            "# Doc",
            META,
            load_guideline(),
            lambda p: "Result: {entities: [], claims: []}",  # unquoted keys
        )


def test_thinking_tags_with_json_like_content_extracts_real_json():
    """Reasoning models emit <think>…</think> with JSON-like content inside.

    Greedy {.*} must not match from a brace inside the thinking block to the
    closing brace of the real JSON. The fix: strip thinking tags before regex,
    keep original for error diagnostics.
    """
    real_payload = {
        "entities": [{"type": "field", "name": "Duri"}],
        "claims": [],
        "unknown_types": [],
    }
    thinking_content = '{"entities": ["note: something mentioned in thinking"]}'
    response = (
        f"<think>Reasoning: {thinking_content}</think>\n{json.dumps(real_payload)}"
    )

    result = extract_knowledge("# Doc", META, load_guideline(), lambda p: response)
    assert result.entities == [{"type": "field", "name": "Duri"}]
    assert result.claims == []


def test_tag_free_response_unchanged():
    """Responses without thinking tags must work exactly as before."""
    payload = {
        "entities": [{"type": "pod", "name": "POD I Duri"}],
        "claims": [],
        "unknown_types": [],
    }
    result = extract_knowledge("# Doc", META, load_guideline(), _caller(payload))
    assert result.entities == [{"type": "pod", "name": "POD I Duri"}]


def test_error_message_preserves_original_raw_with_thinking_tags():
    """Error messages must show original response, including thinking tags."""
    thinking_content = '{"invalid": json without quotes}'
    response = f"<think>Reasoning</think>\n{{{thinking_content}}}"

    with pytest.raises(ValueError) as exc_info:
        extract_knowledge("# Doc", META, load_guideline(), lambda p: response)

    error_msg = str(exc_info.value)
    # The error message starts with the first 100 chars of raw, which includes
    # the <think> tag if present (since it's at the beginning)
    assert "<think>" in error_msg
