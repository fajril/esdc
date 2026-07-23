from __future__ import annotations

from pathlib import Path

import pytest

from esdc.chat.domain_knowledge.instance_graph import InstanceGraphManager
from esdc.knowledge.linker import run_deterministic_linking
from esdc.knowledge.store import KnowledgeStore


@pytest.fixture
def learned_sqlite(sqlite_conn, duck_conn, tmp_path: Path) -> Path:
    store = KnowledgeStore(sqlite_conn)
    store.ensure_tables()
    run_deterministic_linking(sqlite_conn, duck_conn, store)
    sqlite_conn.commit()
    return tmp_path / "esdc.sqlite"


@pytest.fixture(autouse=True)
def _reset_singleton():
    InstanceGraphManager.reset_for_tests()
    yield
    InstanceGraphManager.reset_for_tests()


def test_unavailable_before_learn(tmp_path: Path):
    from esdc.pod_registry.store import get_sqlite_connection

    conn = get_sqlite_connection(tmp_path / "fresh.sqlite")
    conn.close()
    mgr = InstanceGraphManager(sqlite_path=tmp_path / "fresh.sqlite")
    assert mgr.is_available() is False


def test_find_resolves_pod_by_name(learned_sqlite: Path):
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    hits = mgr.find("POD Duri")
    assert hits
    top = hits[0]
    assert top["entity_type"] == "pod"
    assert top["entity_id"] in ("PL-2019-0001-2-2-0", "PL-2022-0002-2-2-1")


def test_find_ranks_entities_over_documents(learned_sqlite: Path):
    """Type-priority ranking must hold even for a non-top same-type hit.

    Query choice ("Duri") is deliberate, not arbitrary: the fixture seeds
    TWO PODs matching it ("POD I Duri", "POD I Duri Revisi 1") plus one
    Field "Duri" (empirically confirmed via mgr.find("Duri") -- 2 pod hits,
    1 field hit, 2 document hits). That gives a *non-top* entity hit
    (the weaker POD) to check against the lower-priority Field hit.

    Under the OLD per-table max-normalized scoring, every table's top hit
    collapsed to exactly 1.0: POD's top ("POD I Duri") -> 1.0, Field's only
    hit ("Duri") -> 1.0, but the second/weaker POD hit ("POD I Duri Revisi
    1") normalized to < 1.0. Sorting by normalized score then put the
    Field hit (1.0) BETWEEN the two POD hits (1.0, then <1.0) -- i.e. NOT
    both PODs before Field. The fixed type-priority-then-raw-score ranking
    keeps all pod-priority (0) hits ahead of field-priority (2) hits
    regardless of BM25 magnitude, so both PODs must precede Field.
    """
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    hits = mgr.find("Duri", top_k=10)
    types = [h["entity_type"] for h in hits]
    assert types.count("pod") >= 2
    assert "field" in types
    assert "document" in types
    pod_positions = [i for i, t in enumerate(types) if t == "pod"]
    field_position = types.index("field")
    assert max(pod_positions) < field_position, (
        f"expected both POD hits before the Field hit, got order={types}"
    )
    assert field_position < types.index("document")


def test_find_orders_same_type_by_raw_score(learned_sqlite: Path):
    """Within one entity type, ties are broken by RAW (uncollapsed) score.

    Raw BM25 is comparable within the same FTS index, so this should not
    fall back to insertion order -- and it should not be per-table
    max-normalized either. The OLD implementation forced every table's top
    hit to exactly 1.0 before sorting; asserting the top score here is NOT
    1.0 (plus that the two scores are genuinely distinct raw magnitudes,
    not just a normalized order) fails under that old behavior and passes
    under the fixed raw-score implementation.
    """
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    hits = mgr.find("POD Duri", top_k=10)
    pod_hits = [h for h in hits if h["entity_type"] == "pod"]
    assert len(pod_hits) >= 2
    scores = [h["score"] for h in pod_hits]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] != 1.0
    assert scores[0] > scores[1]


def test_neighbors_groups_by_relation(learned_sqlite: Path):
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    n = mgr.neighbors("pod", "PL-2019-0001-2-2-0")
    assert any(item["entity_id"] == "PRJ-001" for item in n.get("HAS_PROJECT", []))
    revises = n.get("REVISES", [])
    assert any(
        item["entity_id"] == "PL-2022-0002-2-2-1" and item["direction"] == "inbound"
        for item in revises
    )
    about = n.get("ABOUT_POD", [])
    assert any(item["entity_type"] == "document" for item in about)


def test_cypher_query_passthrough(learned_sqlite: Path):
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    rows = mgr.query(
        "MATCH (d:Document)-[:ABOUT_POD]->(p:POD) "
        "RETURN d.doc_id, p.pod_id ORDER BY d.doc_id"
    )
    assert len(rows) >= 2
