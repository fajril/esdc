"""Tests for knowledge graph bootstrapping."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from esdc.knowledge.bootstrap import (
    generate_guideline_yaml,
    init_guideline,
    sample_documents,
)
from esdc.knowledge.guideline import load_guideline

LLM_PAYLOAD = {
    "claim_types": {
        "economics": "NPV, IRR, capex. Predicate examples: npv_musd, irr_pct.",
        "Bad Name!": "should be dropped",
        "commitment": "wells, facilities, onstream dates.",
        "empty_desc": "",
    },
    "doc_type_hints": {
        "letter": "Focus on approval terms.",
        "mom": "Focus on decisions and action items.",
        "hallucinated_type": "no such doc_type in corpus",
    },
}


def _llm(prompt: str) -> str:
    return json.dumps(LLM_PAYLOAD)


def test_sample_documents_stratified_by_doc_type(sqlite_conn):
    samples = sample_documents(sqlite_conn, per_type=2)
    types = {s["doc_type"] for s in samples}
    assert types == {"letter", "mom", "permen"}
    assert all(s["excerpt"] for s in samples)


def test_generate_yaml_validates_and_keeps_fixed_entity_types(sqlite_conn):
    text = generate_guideline_yaml(sqlite_conn, _llm)
    data = yaml.safe_load(text)
    assert data["version"] == 1
    # entity types copied from packaged guideline, not from the LLM
    assert set(data["entity_types"]) == {"pod", "project", "field", "working_area"}
    assert set(data["claim_types"]) == {"economics", "commitment"}
    assert set(data["doc_type_hints"]) == {"letter", "mom"}


def test_generate_raises_when_no_valid_claim_types(sqlite_conn):
    bad = json.dumps({"claim_types": {"Bad!": "x"}, "doc_type_hints": {}})
    with pytest.raises(ValueError):
        generate_guideline_yaml(sqlite_conn, lambda p: bad)


def test_init_guideline_writes_loadable_file_and_respects_force(
    sqlite_conn, tmp_path: Path, monkeypatch
):
    from esdc.configs import Config

    monkeypatch.setattr(Config, "get_db_dir", classmethod(lambda cls: tmp_path))
    out = init_guideline(sqlite_conn=sqlite_conn, llm_caller=_llm)
    assert out == tmp_path / "guideline.yaml"
    g = load_guideline(out)
    assert "economics" in g.claim_types
    assert "pod" in g.entity_types
    with pytest.raises(FileExistsError):
        init_guideline(sqlite_conn=sqlite_conn, llm_caller=_llm)
    init_guideline(sqlite_conn=sqlite_conn, llm_caller=_llm, force=True)
