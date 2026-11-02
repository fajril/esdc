"""Freshness on the current schema: an unregistered database is not fresh."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import fields

import pytest

from esdc.api_snapshot import ApiProjectResourceRow
from esdc.knowledge.freshness import (
    KnowledgeFreshness,
    knowledge_freshness,
    knowledge_freshness_markers,
)
from esdc.pod_registry.migrations import _validate_existing_v2_registry
from tests.knowledge.conftest import (
    POD_DURI,
    POD_DURI_REV1,
    POD_KAMPUNG,
    PROJECT_DURI,
    PROJECT_KAMPUNG,
)


@pytest.mark.parametrize(
    "read_freshness",
    [knowledge_freshness, knowledge_freshness_markers],
    ids=["generation_basis", "committed_markers"],
)
def test_missing_registry_state_is_not_fresh(
    read_freshness: Callable[[sqlite3.Connection], KnowledgeFreshness],
) -> None:
    """A database without the current registry state cannot report fresh knowledge."""
    with closing(sqlite3.connect(":memory:")) as conn:
        freshness = read_freshness(conn)
        assert freshness.fresh is False


def test_canonical_fixture_seeds_the_current_single_schema(
    sqlite_conn: sqlite3.Connection,
) -> None:
    """The shared fixture is the canonical contract, seeded through current services."""
    _validate_existing_v2_registry(sqlite_conn)

    pods = {
        str(row["pod_id"]): row
        for row in sqlite_conn.execute(
            "SELECT pod_id_seq, pod_uuid, pod_id, pod_order_no, superseded_by, is_latest "
            "FROM m_pod"
        )
    }
    assert set(pods) == {POD_DURI, POD_DURI_REV1, POD_KAMPUNG}
    assert int(pods[POD_DURI]["is_latest"]) == 0
    assert int(pods[POD_DURI]["superseded_by"]) == int(pods[POD_DURI_REV1]["pod_id_seq"])
    assert [int(pods[name]["is_latest"]) for name in (POD_DURI_REV1, POD_KAMPUNG)] == [1, 1]
    assert [
        int(pods[name]["pod_order_no"]) for name in (POD_DURI, POD_DURI_REV1, POD_KAMPUNG)
    ] == [1, 2, 3]

    ledger = sqlite_conn.execute(
        "SELECT pod_id, pod_uuid, pod_order_no FROM _pod_identity_ledger ORDER BY pod_id_seq"
    ).fetchall()
    assert len(ledger) == 3
    for row in ledger:
        pod = pods[str(row["pod_id"])]
        assert str(row["pod_uuid"]) == str(pod["pod_uuid"])
        assert int(row["pod_order_no"]) == int(pod["pod_order_no"])

    projects = sorted(
        str(row["project_id"]) for row in sqlite_conn.execute("SELECT project_id FROM m_project")
    )
    assert projects == sorted([PROJECT_DURI, PROJECT_KAMPUNG])

    relations = sorted(
        (str(row["pod_id"]), str(row["project_id"]))
        for row in sqlite_conn.execute(
            "SELECT m.pod_id, pp.project_id FROM pod_project pp "
            "JOIN m_pod m ON m.pod_id_seq = pp.pod_id_seq"
        )
    )
    assert relations == sorted(
        [
            (POD_DURI, PROJECT_DURI),
            (POD_DURI_REV1, PROJECT_DURI),
            (POD_KAMPUNG, PROJECT_KAMPUNG),
        ]
    )

    columns = tuple(
        str(row[1]) for row in sqlite_conn.execute("PRAGMA table_info(api_project_resources)")
    )
    assert columns == tuple(field.name for field in fields(ApiProjectResourceRow))
    assert sqlite_conn.execute("SELECT COUNT(*) FROM api_project_resources").fetchone()[0] == 3

    assert knowledge_freshness(sqlite_conn).fresh is True
    assert sqlite_conn.execute("PRAGMA foreign_key_check").fetchall() == []
