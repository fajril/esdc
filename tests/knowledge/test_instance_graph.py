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


def test_find_trailing_backslash_does_not_raise(learned_sqlite: Path):
    r"""A trailing backslash in the query text must not swallow the closing.

    quote of the QUERY_FTS_INDEX string literal. The hand-rolled escaping
    previously used in find() (only `'` -> `\'`) left a lone backslash at
    the end of the Cypher string literal, which escapes the closing quote
    instead of being escaped itself -- every per-table query then raised
    and find() silently degraded to an empty result (surfacing as a
    misleading not_found in explore_entity). _cypher_escape escapes
    backslashes first, so this must return normally instead of raising.
    """
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    hits = mgr.find("Duri\\")
    assert isinstance(hits, list)


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


def test_find_resolves_project_with_real_name(
    learned_sqlite: Path, duck_conn, tmp_path: Path
):
    """Project FTS lookup must exist and surface the real duckdb name.

    Without a Project FTS index, find() never returns project hits at all
    -- this is the Issue #1 bug (entity_type='project' always not_found).

    duck_conn is closed before constructing the graph: it holds the only
    read-write connection to the tmp duckdb file for this test, and
    InstanceGraphManager opens its own read-only connection to the same
    file (deliberately, to avoid contending with a live portal on a real
    DB) -- duckdb refuses a second connection with a different
    configuration while the first is still open.
    """
    duck_conn.close()
    mgr = InstanceGraphManager(
        sqlite_path=learned_sqlite, duckdb_path=tmp_path / "esdc.duckdb"
    )
    hits = mgr.find("Duri Steamflood")
    assert hits
    projects = [h for h in hits if h["entity_type"] == "project"]
    assert projects, (
        f"expected a project hit, got types={[h['entity_type'] for h in hits]}"
    )
    top = projects[0]
    assert top["entity_id"] == "PRJ-001"
    assert top["name"] == "Duri Steamflood"


def test_project_neighbor_reports_real_name(
    learned_sqlite: Path, duck_conn, tmp_path: Path
):
    """A Project reached as a POD neighbor must show its real name.

    Today Project nodes are created with project_name = project_id (the
    graph is built only from sqlite kg_edge, which has no names) -- this
    asserts the duckdb-backed real name instead of the opaque "PRJ-001".
    """
    duck_conn.close()
    mgr = InstanceGraphManager(
        sqlite_path=learned_sqlite, duckdb_path=tmp_path / "esdc.duckdb"
    )
    n = mgr.neighbors("pod", "PL-2019-0001-2-2-0")
    has_project = n.get("HAS_PROJECT", [])
    assert any(
        item["entity_id"] == "PRJ-001" and item["name"] == "Duri Steamflood"
        for item in has_project
    ), f"expected real project name among neighbors, got {has_project}"


def test_find_entity_type_filter_avoids_cross_type_truncation(
    learned_sqlite: Path, duck_conn, tmp_path: Path
):
    """entity_type filter must apply BEFORE top_k truncation, not after.

    "Duri Steamflood" matches both POD entities ("POD I Duri", "POD I Duri
    Revisi 1" via the "Duri" token) and the Project "Duri Steamflood". POD
    outranks Project under _TYPE_PRIORITY, so with top_k=1 a plain find()
    surfaces a POD, not the project -- demonstrating that a caller asking
    specifically for entity_type="project" would otherwise lose the match
    to higher-priority cross-type hits before ever getting to filter.
    """
    duck_conn.close()
    mgr = InstanceGraphManager(
        sqlite_path=learned_sqlite, duckdb_path=tmp_path / "esdc.duckdb"
    )
    plain = mgr.find("Duri Steamflood", top_k=1)
    assert plain
    assert plain[0]["entity_type"] == "pod", (
        "test setup assumption broken: expected POD to outrank Project in "
        f"the unfiltered top_k=1 result, got {plain}"
    )

    filtered = mgr.find("Duri Steamflood", top_k=1, entity_type="project")
    assert filtered
    assert all(h["entity_type"] == "project" for h in filtered)
    assert filtered[0]["entity_id"] == "PRJ-001"
    assert filtered[0]["name"] == "Duri Steamflood"


def test_project_names_degrade_gracefully_without_duckdb(learned_sqlite: Path):
    """A missing/unreadable duckdb must not break graph availability.

    Project nodes must fall back to project_id as name -- exactly today's
    behavior -- rather than the graph failing to build.
    """
    mgr = InstanceGraphManager(
        sqlite_path=learned_sqlite,
        duckdb_path=Path("/nonexistent/path/esdc.duckdb"),
    )
    assert mgr.is_available() is True
    hits = mgr.find("PRJ-001", entity_type="project")
    assert hits
    assert hits[0]["entity_id"] == "PRJ-001"
    assert hits[0]["name"] == "PRJ-001"


def test_project_names_degrade_gracefully_on_connection_failure(
    learned_sqlite: Path, tmp_path: Path, monkeypatch
):
    """An existing-but-unreadable duckdb (e.g. a live portal lock) must not.

    break graph availability either -- same graceful-degradation contract
    as the missing-file case above, but exercising the except-branch of
    _load_project_names rather than the exists()-check short circuit.
    """
    import esdc.chat.domain_knowledge.instance_graph as instance_graph_module

    locked_duckdb = tmp_path / "esdc.duckdb"
    locked_duckdb.write_bytes(b"not a real duckdb file")

    def _raise(*args, **kwargs):
        raise OSError("simulated live-portal lock")

    monkeypatch.setattr(instance_graph_module, "get_duckdb_connection", _raise)

    mgr = InstanceGraphManager(sqlite_path=learned_sqlite, duckdb_path=locked_duckdb)
    assert mgr.is_available() is True
    hits = mgr.find("PRJ-001", entity_type="project")
    assert hits
    assert hits[0]["entity_id"] == "PRJ-001"
    assert hits[0]["name"] == "PRJ-001"
