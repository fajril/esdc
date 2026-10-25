"""Offline regression tests for strategic summary prompt construction."""

import json

import pytest

from esdc.summarizer import build_summary_prompt


@pytest.mark.parametrize(
    ("report_year", "analysis_year", "outlook_year"),
    [(2025, 2042, 2039), (2026, 2043, 2040), (2027, 2044, 2041)],
)
@pytest.mark.parametrize("level", ["nkri", "working_area"])
def test_strategic_prompt_uses_supplied_year_metadata(
    report_year, analysis_year, outlook_year, level
):
    prompt = build_summary_prompt(
        level=level,
        year=report_year,
        entity_name="Test entity",
        source_level="strategic_analysis",
        source_items=[],
        metrics={},
        strategic_data={
            "report_year": report_year,
            "analysis_year": analysis_year,
            "outlook_year": outlook_year,
        },
    )

    data_block = prompt.split("========== DATA ANALISIS STRATEGIS ==========\n\n", 1)[1]
    data_json = data_block.split("\n\n==========", 1)[0]
    emitted = json.loads(data_json)
    assert emitted["report_year"] == report_year
    assert emitted["analysis_year"] == analysis_year
    assert emitted["outlook_year"] == outlook_year
    assert "`analysis_year` dan `outlook_year` dari metadata" in prompt
    assert "`{report_year} + 1`" not in prompt


def test_non_strategic_prompt_excludes_strategic_contract(monkeypatch):
    monkeypatch.setattr(
        "esdc.summarizer._load_skill_instructions",
        lambda: "CUSTOM STRATEGIC SKILL CONTENT",
    )
    prompt = build_summary_prompt(
        level="field",
        year=2025,
        entity_name="Test field",
        source_level="field_resources",
        source_items=[],
        metrics={},
    )

    assert "Kontrak ringkasan strategis" not in prompt
    assert "CUSTOM STRATEGIC SKILL CONTENT" not in prompt


@pytest.mark.parametrize("custom_instructions", [None, "CUSTOM SKILL CONTENT"])
def test_strategic_prompt_retains_fallback_or_custom_skill_instructions(
    monkeypatch, custom_instructions
):
    monkeypatch.setattr(
        "esdc.summarizer._load_skill_instructions", lambda: custom_instructions
    )
    prompt = build_summary_prompt(
        level="nkri",
        year=2025,
        entity_name="NKRI",
        source_level="strategic_analysis",
        source_items=[],
        metrics={},
        strategic_data={
            "report_year": 2025,
            "analysis_year": 2026,
            "outlook_year": 2027,
        },
    )

    expected = (
        "Konteks KSMI ringkas:" if custom_instructions is None else custom_instructions
    )
    assert expected in prompt
    assert "Kontrak ringkasan strategis" in prompt
