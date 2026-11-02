"""Tests for entity resolution."""

from __future__ import annotations

import sqlite3

import pytest

from esdc.knowledge.extractor import ExtractionResult
from esdc.knowledge.resolver import resolve_extraction
from esdc.pod_registry.migrations import MigrationRefusedError
from tests.knowledge.conftest import POD_DURI, POD_DURI_REV1, PROJECT_DURI


def test_pod_mention_resolves_via_pod_matcher(sqlite_conn, duck_conn):
    ext = ExtractionResult(entities=[{"type": "pod", "name": "POD I Duri Revisi 1"}])
    res = resolve_extraction("DOC-B", ext, sqlite_conn, duck_conn)
    assert any(
        e.rel == "ABOUT_POD"
        and e.dst_id == POD_DURI_REV1
        and e.method == "llm_extracted"
        and e.confidence == 0.7
        for e in res.edges
    )


def test_superseded_master_name_is_not_a_resolution_target(sqlite_conn, duck_conn):
    """Only current identities are indexed, so a superseded name stays unresolved."""
    ext = ExtractionResult(entities=[{"type": "pod", "name": "POD I Duri"}])
    res = resolve_extraction("DOC-B", ext, sqlite_conn, duck_conn)
    assert res.edges == []
    assert res.unresolved == [{"type": "pod", "name": "POD I Duri"}]
    assert POD_DURI not in {e.dst_id for e in res.edges}


def test_field_exact_and_fuzzy_and_unresolved(sqlite_conn, duck_conn):
    ext = ExtractionResult(
        entities=[
            {"type": "field", "name": "duri"},  # exact (case-insens.)
            {"type": "field", "name": "Kampung Bar"},  # fuzzy
            {"type": "field", "name": "Atlantis"},  # unresolvable
        ]
    )
    res = resolve_extraction("DOC-B", ext, sqlite_conn, duck_conn)
    by_dst = {e.dst_id: e for e in res.edges if e.rel == "ABOUT_FIELD"}
    assert by_dst["Duri"].confidence == 0.9
    assert by_dst["Kampung Baru"].confidence == 0.75
    assert res.unresolved == [{"type": "field", "name": "Atlantis"}]


def test_project_resolves_to_project_id(sqlite_conn, duck_conn):
    ext = ExtractionResult(entities=[{"type": "project", "name": "Duri Steamflood"}])
    res = resolve_extraction("DOC-A", ext, sqlite_conn, duck_conn)
    assert any(e.rel == "ABOUT_PROJECT" and e.dst_id == PROJECT_DURI for e in res.edges)


def test_duck_mirror_is_not_a_name_source(sqlite_conn, duck_conn):
    """The disposable cache mirror can never name or resolve a canonical entity."""
    duck_conn.execute(
        "INSERT INTO project_resources VALUES "
        "('P-DUCK-ONLY', 'Duri Steamflood Mirror', 'Ghost Field', 'Ghost WK', 2024, 'x')"
    )
    ext = ExtractionResult(
        entities=[
            {"type": "project", "name": "Duri Steamflood Mirror"},
            {"type": "field", "name": "Ghost Field"},
            {"type": "working_area", "name": "Ghost WK"},
        ]
    )
    res = resolve_extraction("DOC-B", ext, sqlite_conn, duck_conn)
    assert res.edges == []
    assert [u["name"] for u in res.unresolved] == [
        "Duri Steamflood Mirror",
        "Ghost Field",
        "Ghost WK",
    ]

    # m_project stays the authority even when the mirror carries the same name.
    duck_conn.execute(
        "INSERT INTO project_resources VALUES "
        "('P-DUCK-ONLY', 'Duri Steamflood', 'Duri', 'Rokan', 2024, 'x')"
    )
    res = resolve_extraction(
        "DOC-B",
        ExtractionResult(entities=[{"type": "project", "name": "Duri Steamflood"}]),
        sqlite_conn,
        duck_conn,
    )
    assert [e.dst_id for e in res.edges if e.rel == "ABOUT_PROJECT"] == [PROJECT_DURI]


def test_claim_subject_resolution(sqlite_conn, duck_conn):
    ext = ExtractionResult(
        entities=[{"type": "pod", "name": "POD I Duri Revisi 1"}],
        claims=[
            {
                "type": "issue",
                "subject": "POD I Duri Revisi 1",
                "subject_type": "pod",
                "predicate": "delay_cause",
                "value": "rig",
                "evidence": "q",
            },
            {
                "type": "decision",
                "subject": None,
                "subject_type": None,
                "predicate": "follow_up",
                "value": "send revised POD",
                "evidence": None,
            },
        ],
    )
    res = resolve_extraction("DOC-B", ext, sqlite_conn, duck_conn)
    resolved = [c for c in res.claims if c.subject_id]
    unattached = [c for c in res.claims if c.subject_id is None]
    assert resolved[0].subject_type == "pod"
    assert resolved[0].subject_id == POD_DURI_REV1
    assert len(unattached) == 1
    assert all(c.doc_id == "DOC-B" for c in res.claims)


def test_resolver_refuses_a_source_that_is_not_the_current_schema(duck_conn):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        with pytest.raises(MigrationRefusedError):
            resolve_extraction(
                "DOC-B",
                ExtractionResult(entities=[{"type": "pod", "name": "POD I Duri"}]),
                conn,
                duck_conn,
            )
        assert not conn.in_transaction
    finally:
        conn.close()
