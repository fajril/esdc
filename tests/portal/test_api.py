import duckdb
from fastapi.testclient import TestClient

import esdc.configs as configs
from esdc.corpus.store import _SQLITE_DOC_DDL
from esdc.pod_registry.store import get_sqlite_connection
from esdc.portal.app import create_portal_app


def _patch_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(configs.Config, "get_db_dir", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(
        configs.Config, "get_db_file", classmethod(lambda cls: tmp_path / "esdc.duckdb")
    )


def _seed_documents(doc_ids):
    conn = get_sqlite_connection()
    conn.execute(_SQLITE_DOC_DDL)
    for _i, doc_id in enumerate(doc_ids):
        conn.execute(
            "INSERT INTO documents (doc_id, file_name, file_path, file_hash,"
            " markdown, extraction_method, embedding_model)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (doc_id, f"{doc_id}.pdf", f"/x/{doc_id}.pdf", doc_id * 4,
             "# isi", "native", "fake-embed"),
        )
    conn.commit()
    conn.close()


def _seed_duckdb_entities(
    tmp_path, wk_name=None, field_name=None, project_name=None,
    mirror_documents=True, mirror_doc_ids=(),
):
    """Seed the DuckDB file `_patch_dirs` points Config.get_db_file() at.

    Populates the `project_resources` lookup table EntityResolver queries,
    plus (optionally) a `documents` mirror table pre-seeded before
    `_refresh_mirror_after_save`'s wholesale rebuild runs. Mirrors the setup
    in tests/portal/test_document_entities.py's
    test_default_resolver_and_mirror_share_db_path.
    """
    path = tmp_path / "esdc.duckdb"
    conn = duckdb.connect(str(path))
    conn.execute(
        "CREATE TABLE project_resources"
        " (wk_name VARCHAR, field_name VARCHAR, project_name VARCHAR)"
    )
    conn.execute(
        "INSERT INTO project_resources VALUES (?, ?, ?)",
        [wk_name, field_name, project_name],
    )
    if mirror_documents:
        conn.execute(
            "CREATE TABLE documents (doc_id VARCHAR PRIMARY KEY, wk_name JSON,"
            " field_name JSON, project_name JSON)"
        )
        for doc_id in mirror_doc_ids:
            conn.execute(
                "INSERT INTO documents VALUES (?, NULL, NULL, NULL)", [doc_id]
            )
    conn.close()
    return path


def _client(monkeypatch, tmp_path):
    _patch_dirs(monkeypatch, tmp_path)
    conn = get_sqlite_connection()
    conn.execute("INSERT INTO r_institution (code, institution) VALUES (4, 'SKK Migas')")
    conn.execute("INSERT INTO r_pod_type (code, pod_type) VALUES (1, 'POD I')")
    conn.commit()
    conn.close()
    return TestClient(create_portal_app())


def test_get_table_rows_and_refs(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    resp = client.get("/api/tables/m_pod")
    assert resp.status_code == 200
    body = resp.json()
    assert body["rows"] == []
    assert body["refs"]["institutions"] == [
        {"code": 4, "institution": "SKK Migas"}
    ]


def test_get_unknown_table_404(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    assert client.get("/api/tables/nope").status_code == 404


def test_save_insert_returns_generated_pod_id(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    resp = client.post("/api/tables/m_pod/save", json={"inserts": [{
        "id": 1, "pod_name": "POD Baru", "approval_date": "2026-07-15",
        "institution_code": 4, "pod_type_code": 1, "rev_num": 0,
    }]})
    assert resp.status_code == 200
    assert resp.json()["generated"][0]["pod_id"] == "PL-2026-0001-4-1-0"


def test_save_validation_error_422(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    resp = client.post("/api/tables/m_pod/save", json={"inserts": [{
        "id": 1, "pod_name": "Bad", "approval_date": "2026-07-15",
        "institution_code": 99, "pod_type_code": 1, "rev_num": 0,
    }]})
    assert resp.status_code == 422
    assert resp.json()["errors"]


def test_save_does_not_run_publish(monkeypatch, tmp_path):
    # Publish moved out of the save round-trip (client fires /api/publish
    # after save); a broken publish must not fail or slow the save itself.
    client = _client(monkeypatch, tmp_path)

    def boom(**kw):
        raise RuntimeError("duckdb locked")

    monkeypatch.setattr("esdc.portal.app.publish_pod_registry", boom)
    resp = client.post("/api/tables/m_pod/save", json={"inserts": [{
        "id": 1, "pod_name": "POD Baru", "approval_date": "2026-07-15",
        "institution_code": 4, "pod_type_code": 1, "rev_num": 0,
    }]})
    assert resp.status_code == 200
    assert "publish_error" not in resp.json()


def test_publish_endpoint_reports_error(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)

    def boom(**kw):
        raise RuntimeError("duckdb locked")

    monkeypatch.setattr("esdc.portal.app.publish_pod_registry", boom)
    resp = client.post("/api/publish")
    assert resp.status_code == 500
    body = resp.json()
    assert body["ok"] is False
    assert "duckdb locked" in body["error"]


def test_projects_autocomplete_empty_when_no_duckdb(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    resp = client.get("/api/projects", params={"q": "P-24"})
    assert resp.status_code == 200
    assert resp.json() == []


def _insert_pod(client):
    resp = client.post("/api/tables/m_pod/save", json={"inserts": [{
        "id": 1, "pod_name": "POD Baru", "approval_date": "2026-07-15",
        "institution_code": 4, "pod_type_code": 1, "rev_num": 0,
    }]})
    assert resp.status_code == 200


def test_pod_document_refs_include_documents(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    _seed_documents(["ccbd4f3f27635c76"])
    resp = client.get("/api/tables/pod_document")
    assert resp.status_code == 200
    body = resp.json()
    assert body["rows"] == []
    assert body["refs"]["documents"] == {
        "ccbd4f3f27635c76": "ccbd4f3f27635c76.pdf"
    }


def test_pod_document_save_rejects_unknown_doc_id(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    _insert_pod(client)
    _seed_documents(["ccbd4f3f27635c76"])
    resp = client.post("/api/tables/pod_document/save", json={"inserts": [
        {"pod_id": 1, "doc_id": "nope"},
    ]})
    assert resp.status_code == 422
    assert "nope" in resp.json()["errors"][0]["message"]


def test_pod_document_save_and_roundtrip(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    _insert_pod(client)
    _seed_documents(["ccbd4f3f27635c76"])
    resp = client.post("/api/tables/pod_document/save", json={"inserts": [
        {"pod_id": 1, "doc_id": "ccbd4f3f27635c76"},
    ]})
    assert resp.status_code == 200
    rows = client.get("/api/tables/pod_document").json()["rows"]
    assert rows == [{"pod_id": 1, "doc_id": "ccbd4f3f27635c76"}]


def test_pod_document_save_skips_doc_check_when_corpus_unavailable(
    monkeypatch, tmp_path
):
    # No SQLite documents table -> validation degrades gracefully (like
    # project ids), it must not hard-fail the save.
    client = _client(monkeypatch, tmp_path)
    _insert_pod(client)
    resp = client.post("/api/tables/pod_document/save", json={"inserts": [
        {"pod_id": 1, "doc_id": "ccbd4f3f27635c76"},
    ]})
    assert resp.status_code == 200


def test_documents_grid_rows_and_linked_count(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    _insert_pod(client)
    _seed_documents(["ccbd4f3f27635c76", "def456"])
    resp = client.post("/api/tables/pod_document/save", json={"inserts": [
        {"pod_id": 1, "doc_id": "ccbd4f3f27635c76"},
    ]})
    assert resp.status_code == 200

    rows = client.get("/api/tables/documents").json()["rows"]
    by_id = {r["doc_id"]: r for r in rows}
    assert by_id["ccbd4f3f27635c76"]["linked_pods"] == 1
    assert by_id["def456"]["linked_pods"] == 0


def test_documents_grid_includes_pod_name_and_suggestions(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    _seed_documents(["abc123"])
    conn = get_sqlite_connection()
    conn.execute(
        "UPDATE documents SET pod_name = ?, suggested_pod_ids = ?"
        " WHERE doc_id = 'abc123'",
        ('["POD I Lapangan Abadi"]', '["PL-2019-0300-4-1-0"]'),
    )
    conn.commit()
    conn.close()
    rows = client.get("/api/tables/documents").json()["rows"]
    assert rows[0]["pod_name"] == '["POD I Lapangan Abadi"]'
    assert rows[0]["suggested_pod_ids"] == '["PL-2019-0300-4-1-0"]'


def test_documents_table_empty_when_absent(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    resp = client.get("/api/tables/documents")
    assert resp.status_code == 200
    assert resp.json()["rows"] == []


def test_documents_save_updates_entity_fields(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    _seed_documents(["D1"])
    _seed_duckdb_entities(tmp_path, wk_name="Rokan", mirror_doc_ids=["D1"])

    resp = client.post("/api/tables/documents/save", json={"updates": [
        {"doc_id": "D1", "wk_name": "Rokan"},
    ]})

    assert resp.status_code == 200
    body = resp.json()
    assert body["applied"]["updates"] == 1
    assert body["warnings"] == []
    rows = client.get("/api/tables/documents").json()["rows"]
    assert rows[0]["wk_name"] == '["Rokan"]'


def test_documents_save_inserts_and_deletes_rejected_422(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    resp = client.post("/api/tables/documents/save", json={
        "inserts": [{"doc_id": "D2"}],
        "deletes": [{"doc_id": "D1"}],
    })
    assert resp.status_code == 422
    kinds = {e["kind"] for e in resp.json()["errors"]}
    assert kinds == {"insert", "delete"}


def test_documents_save_unknown_name_rejected_422(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    _seed_documents(["D1"])
    _seed_duckdb_entities(tmp_path, wk_name="Rokan", mirror_doc_ids=["D1"])

    resp = client.post("/api/tables/documents/save", json={"updates": [
        {"doc_id": "D1", "wk_name": "Nonexistent"},
    ]})

    assert resp.status_code == 422
    assert "Nonexistent" in resp.json()["errors"][0]["message"]


def test_documents_save_warnings_passthrough(monkeypatch, tmp_path):
    """A refresh that loses the DuckDB lock degrades to a warning, not a
    failed save.

    `refresh_mirror()` rebuilds `documents` wholesale (`CREATE OR REPLACE
    TABLE`), so a missing mirror table no longer reproduces a failure --
    unlike the old row-by-row UPDATE mirror, it just creates the table.
    Instead, hold a read-only DuckDB connection open on the same file:
    DuckDB refuses a second connection under a different configuration,
    which is exactly what happens if `esdc corpus commit` still holds
    the write lock when the portal save's refresh runs.
    """
    client = _client(monkeypatch, tmp_path)
    _seed_documents(["D1"])
    _seed_duckdb_entities(tmp_path, wk_name="Rokan", mirror_documents=False)

    lock_conn = duckdb.connect(str(tmp_path / "esdc.duckdb"), read_only=True)
    try:
        resp = client.post("/api/tables/documents/save", json={"updates": [
            {"doc_id": "D1", "wk_name": "Rokan"},
        ]})
    finally:
        lock_conn.close()

    assert resp.status_code == 200
    body = resp.json()
    assert body["warnings"]
    assert "esdc corpus sync" in body["warnings"][0]
