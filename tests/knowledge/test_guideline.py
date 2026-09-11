"""Tests for extraction guidelines."""

from __future__ import annotations

from pathlib import Path

from esdc.knowledge.guideline import build_extraction_prompt, load_guideline


def test_load_packaged_guideline():
    from esdc.knowledge import guideline as guideline_mod

    # explicit packaged path: a user-level ~/.esdc/guideline.yaml on the dev
    # machine must not influence this test
    g = load_guideline(guideline_mod._DEFAULT_PATH)
    assert g.version == 1
    assert "pod" in g.entity_types
    assert "economics" in g.claim_types
    assert len(g.content_hash) == 64


def test_hash_changes_when_file_changes(tmp_path: Path):
    src = Path("esdc/knowledge/guideline.yaml").read_text(encoding="utf-8")
    p = tmp_path / "guideline.yaml"
    p.write_text(src, encoding="utf-8")
    h1 = load_guideline(p).content_hash
    p.write_text(src + "\n# tweak\n", encoding="utf-8")
    h2 = load_guideline(p).content_hash
    assert h1 != h2


def test_user_level_guideline_overrides_packaged(tmp_path: Path, monkeypatch):
    from esdc.configs import Config
    from esdc.knowledge import guideline as guideline_mod

    monkeypatch.setattr(Config, "get_db_dir", classmethod(lambda cls: tmp_path))
    # no user file -> packaged default
    assert guideline_mod.default_guideline_path() == guideline_mod._DEFAULT_PATH
    # user file present -> it wins
    user_file = tmp_path / "guideline.yaml"
    user_file.write_text(
        "version: 2\nentity_types:\n  pod: x\nclaim_types:\n  economics: y\n",
        encoding="utf-8",
    )
    assert guideline_mod.default_guideline_path() == user_file
    assert load_guideline().version == 2


def test_prompt_contains_types_metadata_and_truncated_markdown():
    g = load_guideline()
    meta = {
        "doc_type": "mom",
        "subject": "MoM Monitoring POD I Duri",
        "doc_date": "2023-04-01",
    }
    prompt = build_extraction_prompt(g, meta, "A" * 50000, max_chars=1000)
    assert "economics" in prompt
    assert "MoM Monitoring POD I Duri" in prompt
    # Verify markdown body was truncated to max_chars (1000).
    # Extract body between "Document markdown:\n---\n" and "\n---"
    body_start = prompt.find("Document markdown:\n---\n") + len(
        "Document markdown:\n---\n"
    )
    body_end = prompt.rfind("\n---")
    truncated_body = prompt[body_start:body_end]
    assert truncated_body == "A" * 1000
    # doc_type hint for mom is included
    assert "action items" in prompt
    assert "JSON" in prompt
