"""Tests for entity resolution."""

from __future__ import annotations

from esdc.knowledge.extractor import ExtractionResult
from esdc.knowledge.resolver import resolve_extraction


def test_pod_mention_resolves_via_pod_matcher(sqlite_conn, duck_conn):
    ext = ExtractionResult(entities=[{"type": "pod", "name": "POD I Duri"}])
    res = resolve_extraction("DOC-B", ext, sqlite_conn, duck_conn)
    assert any(
        e.rel == "ABOUT_POD"
        and e.dst_id == "PL-2019-0001-2-2-0"
        and e.method == "llm_extracted"
        and e.confidence == 0.7
        for e in res.edges
    )


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
    assert any(e.rel == "ABOUT_PROJECT" and e.dst_id == "PRJ-001" for e in res.edges)


def test_claim_subject_resolution(sqlite_conn, duck_conn):
    ext = ExtractionResult(
        entities=[{"type": "pod", "name": "POD I Duri"}],
        claims=[
            {
                "type": "issue",
                "subject": "POD I Duri",
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
    assert resolved[0].subject_id == "PL-2019-0001-2-2-0"
    assert len(unattached) == 1
    assert all(c.doc_id == "DOC-B" for c in res.claims)
