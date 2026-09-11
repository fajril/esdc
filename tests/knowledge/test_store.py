"""Tests for the knowledge store."""

from __future__ import annotations

from esdc.knowledge.store import Claim, Edge, KnowledgeStore


def _store(sqlite_conn):
    store = KnowledgeStore(sqlite_conn)
    store.ensure_tables()
    return store


def test_upsert_edges_dedupes_and_keeps_best_confidence(sqlite_conn):
    store = _store(sqlite_conn)
    e_low = Edge(
        "document",
        "DOC-B",
        "ABOUT_POD",
        "pod",
        "PL-2019-0001-2-2-0",
        confidence=0.8,
        method="suggested_promoted",
    )
    e_high = Edge(
        "document",
        "DOC-B",
        "ABOUT_POD",
        "pod",
        "PL-2019-0001-2-2-0",
        confidence=1.0,
        method="letter_num_exact",
    )
    assert store.upsert_edges([e_low]) == 1
    assert store.upsert_edges([e_high, e_low]) == 2  # rows touched, not created
    edges = store.edges_for("document", "DOC-B")
    assert len(edges) == 1
    assert edges[0].confidence == 1.0
    assert edges[0].method == "letter_num_exact"


def test_edges_for_matches_source_and_destination(sqlite_conn):
    store = _store(sqlite_conn)
    store.upsert_edges(
        [
            Edge(
                "pod",
                "PL-2019-0001-2-2-0",
                "HAS_PROJECT",
                "project",
                "PRJ-001",
                method="registry",
            )
        ]
    )
    assert len(store.edges_for("pod", "PL-2019-0001-2-2-0")) == 1
    assert len(store.edges_for("project", "PRJ-001")) == 1
    assert store.edges_for("field", "Duri") == []


def test_replace_claims_is_idempotent_per_doc(sqlite_conn):
    store = _store(sqlite_conn)
    claims = [
        Claim(
            "DOC-A",
            "economics",
            "npv_musd",
            100,
            subject_type="pod",
            subject_id="PL-2022-0002-2-2-1",
            evidence="NPV 100 MUSD",
        ),
        Claim("DOC-A", "decision", "approval", "approved"),
    ]
    assert store.replace_claims("DOC-A", claims) == 2
    assert store.replace_claims("DOC-A", claims[1:]) == 1
    assert len(store.claims_for_doc("DOC-A")) == 1
    got = store.claims_for("pod", "PL-2022-0002-2-2-1")
    assert got == []  # the remaining claim has no subject


def test_claim_value_roundtrips_json(sqlite_conn):
    store = _store(sqlite_conn)
    store.replace_claims(
        "DOC-A",
        [
            Claim(
                "DOC-A",
                "economics",
                "cost_breakdown",
                {"capex": 1.5, "opex": [1, 2]},
                subject_type="pod",
                subject_id="PL-2022-0002-2-2-1",
            )
        ],
    )
    claim = store.claims_for("pod", "PL-2022-0002-2-2-1")[0]
    assert claim.value == {"capex": 1.5, "opex": [1, 2]}


def test_proposal_bump_accumulates_evidence(sqlite_conn):
    store = _store(sqlite_conn)
    store.bump_proposal("claim_type", "wellhead_incident", "DOC-A")
    store.bump_proposal("claim_type", "wellhead_incident", "DOC-B")
    store.bump_proposal("claim_type", "wellhead_incident", "DOC-A")  # same doc
    props = store.pending_proposals()
    assert len(props) == 1
    assert props[0]["name"] == "wellhead_incident"
    assert props[0]["evidence_count"] == 2
    assert sorted(props[0]["sample_doc_ids"]) == ["DOC-A", "DOC-B"]


def test_learn_state_tracks_source_hash(sqlite_conn):
    store = _store(sqlite_conn)
    assert store.needs_learn("DOC-A", "h1") is True
    store.mark_learned("DOC-A", "h1")
    assert store.needs_learn("DOC-A", "h1") is False
    assert store.needs_learn("DOC-A", "h2") is True  # guideline/doc changed
    assert store.learned_count() == 1


def test_delete_doc_edges_only_removes_that_docs_edges(sqlite_conn):
    store = _store(sqlite_conn)
    store.upsert_edges(
        [
            Edge("document", "DOC-A", "ABOUT_POD", "pod", "P1", method="m"),
            Edge("document", "DOC-B", "ABOUT_POD", "pod", "P1", method="m"),
        ]
    )
    assert store.delete_doc_edges("DOC-A") == 1
    assert store.edges_for("document", "DOC-A") == []
    assert len(store.edges_for("document", "DOC-B")) == 1
