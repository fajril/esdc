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
