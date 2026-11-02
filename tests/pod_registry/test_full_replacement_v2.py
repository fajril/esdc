"""One-time full POD replacement exercised through the operator CLI."""

import json
import sqlite3

import duckdb
import pytest
from typer.testing import CliRunner

from esdc.configs import Config
from esdc.esdc import app
from esdc.pod_registry.sqlite_import import (
    PodImportError,
    load_prepared_pod_batch,
    prepare_pod_import,
)
from esdc.pod_registry.store import get_sqlite_connection
from tests.knowledge import conftest as knowledge
from tests.pod_registry.test_migrations_v2 import _legacy_connection


@pytest.fixture()
def replacement_world(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "get_db_dir", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(
        Config, "get_cache_dir", classmethod(lambda cls: tmp_path / "cache")
    )
    cache = tmp_path / "esdc.duckdb"
    monkeypatch.setattr(Config, "get_db_file", classmethod(lambda cls: cache))
    incoming = tmp_path / "incoming.sqlite"
    conn = get_sqlite_connection(incoming)
    try:
        knowledge._seed_knowledge_world(conn)
        masters = [
            tuple(row)
            for row in conn.execute(
                "SELECT pod_id_seq,pod_uuid FROM m_pod ORDER BY pod_id_seq"
            )
        ]
    finally:
        conn.close()
    target = tmp_path / "esdc.sqlite"
    conn = _legacy_connection(target)
    conn.close()
    return incoming, target, cache, {900: masters[0][0], 901: masters[1][0]}, masters


def test_full_replacement_preserves_docs_and_advances_once(replacement_world):
    incoming, target, cache, mapping, masters = replacement_world
    batch = prepare_pod_import(
        incoming,
        "sqlite",
        target_path=target,
        stage_snapshots=False,
        approved_crosswalk=mapping,
    )
    result, _ = load_prepared_pod_batch(batch, sqlite_path=target, duck_path=cache)
    assert result.source_generation == 1
    conn = sqlite3.connect(target.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        assert (
            conn.execute(
                "SELECT pod_id_seq,pod_uuid FROM m_pod ORDER BY pod_id_seq"
            ).fetchall()
            == masters
        )
        assert conn.execute("SELECT * FROM documents ORDER BY doc_id").fetchall() == [
            ("DOC-001", "retained source one"),
            ("DOC-002", "retained source two"),
        ]
        assert conn.execute(
            "SELECT pod_id_seq,doc_id FROM pod_document ORDER BY doc_id"
        ).fetchall() == [(mapping[900], "DOC-001"), (mapping[901], "DOC-002")]
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert conn.execute(
            "SELECT type FROM sqlite_master WHERE name='pod_monitoring_api'"
        ).fetchone() == ("table",)
    finally:
        conn.close()


def test_replacement_refuses_missing_map_before_writer(replacement_world):
    incoming, target, _, _, _ = replacement_world
    before = target.read_bytes()
    with pytest.raises(PodImportError, match="--pod-crosswalk"):
        prepare_pod_import(
            incoming, "sqlite", target_path=target, stage_snapshots=False
        )
    assert target.read_bytes() == before


def test_replacement_refuses_concurrent_reference_edit(replacement_world):
    incoming, target, cache, mapping, _ = replacement_world
    batch = prepare_pod_import(
        incoming,
        "sqlite",
        target_path=target,
        stage_snapshots=False,
        approved_crosswalk=mapping,
    )
    conn = sqlite3.connect(target)
    try:
        conn.execute("UPDATE r_institution SET institution='Changed after prepare'")
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(PodImportError, match="changed after"):
        load_prepared_pod_batch(batch, sqlite_path=target, duck_path=cache)
    conn = sqlite3.connect(target.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        assert conn.execute("SELECT institution FROM r_institution").fetchall() == [
            ("Changed after prepare",)
        ]
        assert conn.execute("SELECT COUNT(*) FROM m_pod").fetchone() == (2,)
    finally:
        conn.close()


def test_replacement_late_refresh_failure_restores_old_schema(
    replacement_world, monkeypatch
):
    from esdc.pod_registry import read_models

    incoming, target, cache, mapping, _ = replacement_world
    batch = prepare_pod_import(
        incoming,
        "sqlite",
        target_path=target,
        stage_snapshots=False,
        approved_crosswalk=mapping,
    )

    def fail(*args, **kwargs):
        raise RuntimeError("injected physical refresh failure")

    monkeypatch.setattr(read_models, "rebuild_pod_read_models", fail)
    with pytest.raises(RuntimeError, match="injected physical"):
        load_prepared_pod_batch(batch, sqlite_path=target, duck_path=cache)
    conn = sqlite3.connect(target.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        assert conn.execute("SELECT id FROM m_pod ORDER BY id").fetchall() == [
            (900,),
            (901,),
        ]
        assert conn.execute(
            "SELECT pod_id,doc_id FROM pod_document ORDER BY doc_id"
        ).fetchall() == [(900, "DOC-001"), (901, "DOC-002")]
        assert (
            conn.execute(
                "SELECT name FROM sqlite_master WHERE name='pod_registry_state'"
            ).fetchall()
            == []
        )
    finally:
        conn.close()


def test_public_full_old_schema_replacement_keeps_docs_and_uses_explicit_map(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(Config, "get_db_dir", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(
        Config, "get_cache_dir", classmethod(lambda cls: tmp_path / "cache")
    )
    cache = tmp_path / "esdc.duckdb"
    monkeypatch.setattr(Config, "get_db_file", classmethod(lambda cls: cache))
    incoming = tmp_path / "incoming.sqlite"
    source = get_sqlite_connection(incoming)
    try:
        knowledge._seed_knowledge_world(source)
        masters = [
            tuple(row)
            for row in source.execute(
                "SELECT pod_id_seq,pod_uuid FROM m_pod ORDER BY pod_id_seq"
            )
        ]
    finally:
        source.close()
    target = tmp_path / "esdc.sqlite"
    old = _legacy_connection(target)
    try:
        documents = [
            tuple(row) for row in old.execute("SELECT * FROM documents ORDER BY doc_id")
        ]
    finally:
        old.close()
    mapping = tmp_path / "crosswalk.json"
    mapping.write_text(json.dumps({"900": masters[0][0], "901": masters[1][0]}))

    result = CliRunner().invoke(
        app,
        ["load", "--from-sqlite", str(incoming), "--pod-crosswalk", str(mapping)],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))

    reader = sqlite3.connect(target.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        assert (
            reader.execute("SELECT * FROM documents ORDER BY doc_id").fetchall()
            == documents
        )
        assert (
            reader.execute(
                "SELECT pod_id_seq,pod_uuid FROM m_pod ORDER BY pod_id_seq"
            ).fetchall()
            == masters
        )
        assert reader.execute(
            "SELECT pod_id_seq,doc_id FROM pod_document ORDER BY doc_id"
        ).fetchall() == [(masters[0][0], "DOC-001"), (masters[1][0], "DOC-002")]
        assert reader.execute(
            "SELECT source_generation FROM pod_registry_state"
        ).fetchone() == (1,)
        assert reader.execute("PRAGMA foreign_key_check").fetchall() == []
        objects = dict(reader.execute("SELECT name,type FROM sqlite_master"))
        assert objects["pod_monitoring_api"] == objects["pod_forecast_api"] == "table"
        assert "project_pod" not in objects
        assert "pod_revision" not in objects
    finally:
        reader.close()
    projection = duckdb.connect(str(cache), read_only=True)
    try:
        assert (
            projection.execute(
                "SELECT pod_id_seq,CAST(pod_uuid AS VARCHAR) FROM m_pod ORDER BY pod_id_seq"
            ).fetchall()
            == masters
        )
    finally:
        projection.close()


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ("{bad", "Invalid JSON"),
        ("[]", "JSON object"),
        ('{"900": true}', "Invalid POD crosswalk target"),
        ('{"900": 1.5}', "Invalid POD crosswalk target"),
        ('{"900": "1"}', "Invalid POD crosswalk target"),
        ('{"900": 1, "900": 2}', "Duplicate raw"),
        ('{"900": 1, "0900": 2}', "Duplicate normalized"),
        ('{"bad": 1}', "Invalid POD crosswalk key"),
    ],
)
def test_invalid_crosswalk_cli_refuses_without_writing(
    replacement_world, payload, reason
):
    incoming, target, cache, _, _ = replacement_world
    mapping_file = target.with_name("crosswalk.json")
    mapping_file.write_text(payload)
    before = target.read_bytes()
    result = CliRunner().invoke(
        app,
        [
            "load",
            "--from-sqlite",
            str(incoming),
            "--pod-crosswalk",
            str(mapping_file),
            "--dry-run",
        ],
    )
    assert result.exit_code == 1, result.output
    assert reason in result.output
    assert target.read_bytes() == before
    assert not cache.exists()


def test_valid_crosswalk_cli_dry_run_is_read_only(replacement_world):
    incoming, target, cache, mapping, _ = replacement_world
    mapping_file = target.with_name("crosswalk.json")
    mapping_file.write_text(json.dumps(mapping))
    before = target.read_bytes()
    result = CliRunner().invoke(
        app,
        [
            "load",
            "--from-sqlite",
            str(incoming),
            "--pod-crosswalk",
            str(mapping_file),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "not opened for writing" in result.output
    assert target.read_bytes() == before
    assert not cache.exists()


def test_crosswalk_unknown_target_refuses_before_writing(replacement_world):
    incoming, target, cache, mapping, _ = replacement_world
    mapping[900] = 999999
    before = target.read_bytes()
    with pytest.raises(PodImportError, match="absent from incoming"):
        prepare_pod_import(
            incoming,
            "sqlite",
            target_path=target,
            approved_crosswalk=mapping,
            stage_snapshots=False,
        )
    assert target.read_bytes() == before
    assert not cache.exists()


def test_unknown_dependency_refuses_before_writing(replacement_world):
    incoming, target, cache, mapping, _ = replacement_world
    conn = sqlite3.connect(target)
    try:
        conn.execute(
            "CREATE TABLE external_pod_fk(pod_id INTEGER REFERENCES m_pod(id))"
        )
        conn.commit()
    finally:
        conn.close()
    before = target.read_bytes()
    with pytest.raises(PodImportError, match="external_pod_fk"):
        prepare_pod_import(
            incoming,
            "sqlite",
            target_path=target,
            approved_crosswalk=mapping,
            stage_snapshots=False,
        )
    assert target.read_bytes() == before
    assert not cache.exists()


def test_old_target_without_protected_links_needs_no_map(replacement_world):
    incoming, target, cache, _, _ = replacement_world
    conn = sqlite3.connect(target)
    try:
        conn.execute("DELETE FROM pod_document")
        conn.commit()
    finally:
        conn.close()
    batch = prepare_pod_import(
        incoming,
        "sqlite",
        target_path=target,
        stage_snapshots=False,
    )
    result, _ = load_prepared_pod_batch(
        batch,
        sqlite_path=target,
        duck_path=cache,
        publish=False,
    )
    assert result.source_generation == 1
    assert not cache.exists()


def test_old_master_only_target_requires_full_load(replacement_world):
    incoming, target, cache, _, _ = replacement_world
    before = target.read_bytes()
    with pytest.raises(PodImportError, match="requires a full"):
        prepare_pod_import(
            incoming,
            "sqlite",
            target_path=target,
            profile="master-only",
            stage_snapshots=False,
        )
    assert target.read_bytes() == before
    assert not cache.exists()


def test_collapsed_document_links_are_rejected(replacement_world):
    incoming, target, cache, mapping, _ = replacement_world
    conn = sqlite3.connect(target)
    try:
        conn.execute("INSERT INTO pod_document VALUES(901, 'DOC-001')")
        conn.commit()
    finally:
        conn.close()
    mapping[901] = mapping[900]
    before = target.read_bytes()
    with pytest.raises(PodImportError, match="collapse duplicate"):
        prepare_pod_import(
            incoming,
            "sqlite",
            target_path=target,
            approved_crosswalk=mapping,
            stage_snapshots=False,
        )
    assert target.read_bytes() == before
    assert not cache.exists()
