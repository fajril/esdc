"""End-to-end tests for project-resource API refresh and recovery."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import esdc.api_projection
from esdc.esdc import (
    rebuild_project_resources_projection,
    refresh_project_resources_from_api,
)


def _payload() -> list[dict[str, object]]:
    first = {
        "project_id": "P-1",
        "report_year": 2025,
        "project_name": "One",
        "field_geom": "POINT (1 2)",
    }
    second = {
        "field_geom": "NOT WKT",
        "project_name": "Two",
        "report_year": 2025,
        "project_id": "P-2",
    }
    return [first, second]


def _snapshot(path: Path) -> tuple[str, int, str]:
    conn = sqlite3.connect(path)
    try:
        row = conn.execute(
            "SELECT revision, row_count, publication_state FROM api_snapshot "
            "WHERE source_table = 'project_resources'"
        ).fetchone()
        assert row is not None
        return row
    finally:
        conn.close()


def test_refresh_and_rebuild_recover_without_refetch(
    isolated_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    from esdc.configs import Config

    payload = _payload()
    calls: list[str] = []

    def download(*_: object) -> bytes:
        calls.append("download")
        return json.dumps(payload).encode()

    monkeypatch.setattr("esdc.esdc.esdc_downloader", download)
    monkeypatch.setattr("esdc.esdc.load_data_to_db", lambda *_: calls.append("duck-loader"))

    def fail_checkpoint(_: object) -> None:
        raise RuntimeError("post-commit failure")

    original_checkpoint = esdc.api_projection._checkpoint
    monkeypatch.setattr("esdc.api_projection._checkpoint", fail_checkpoint)
    with pytest.raises(RuntimeError, match="analytical state requires rebuild"):
        refresh_project_resources_from_api(
            username="user", password="password", report_year=None
        )

    sqlite_path = Config.get_db_dir() / "esdc.sqlite"
    revision, row_count, state = _snapshot(sqlite_path)
    assert revision
    assert row_count == 2
    assert state == "analytics_stale"
    assert calls == ["download"]

    monkeypatch.setattr("esdc.api_projection._checkpoint", original_checkpoint)

    def fail_download(*_: object) -> bytes:
        raise AssertionError("rebuild must not fetch")

    monkeypatch.setattr("esdc.esdc.esdc_downloader", fail_download)
    publication = rebuild_project_resources_projection()

    assert publication.revision == revision
    assert publication.row_count == 2
    assert publication.invalid_geometry_count == 1
    assert _snapshot(sqlite_path)[2] == "published"
    assert calls == ["download"]
