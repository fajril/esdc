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
    """Type-priority policy decides ranking, not per-table score collapse.

    "Duri" matches both the Field "Duri" and Document subjects containing
    "Duri" (e.g. "MoM Monitoring POD I Duri"). Entity-resolution policy
    requires real entity nodes to always outrank Document nodes here,
    regardless of raw BM25 magnitude on either side.
    """
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    hits = mgr.find("Duri", top_k=10)
    types = [h["entity_type"] for h in hits]
    assert "field" in types
    assert "document" in types
    assert types.index("field") < types.index("document")


def test_find_orders_same_type_by_raw_score(learned_sqlite: Path):
    """Within one entity type, ties are broken by raw score.

    Raw BM25 is comparable within the same FTS index, so this should not
    fall back to insertion order.
    """
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    hits = mgr.find("POD Duri", top_k=10)
    pod_hits = [h for h in hits if h["entity_type"] == "pod"]
    assert len(pod_hits) >= 2
    scores = [h["score"] for h in pod_hits]
    assert scores == sorted(scores, reverse=True)


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
