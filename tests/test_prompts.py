"""Contracts for the compact core system prompt."""

from esdc.chat.prompts import get_system_prompt


def test_core_prompt_has_groovy_contract_without_embedded_ksmi_table() -> None:
    prompt = get_system_prompt()
    required_phrases = (
        "groovy_report",
        "groovy_isactive",
        "nullable",
        "labelled",
        "FTS",
        "semantic",
        "SQL",
        "knowledge_traversal",
        "report_year",
        "uncertainty",
        "esdc.sqlite",
        "read-only analytics",
    )
    normalized_prompt = " ".join(prompt.split())
    assert all(phrase in normalized_prompt for phrase in required_phrases)
    assert "| Level | Name | Class |" not in prompt
    assert "Project Maturity Levels" not in prompt
    assert "## Domain Definitions" not in prompt
    assert "Do not reproduce or infer" in prompt
    assert len(prompt) <= 12_000
