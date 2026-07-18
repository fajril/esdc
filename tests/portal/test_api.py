from fastapi.testclient import TestClient

import esdc.configs as configs
from esdc.portal.app import create_portal_app
from esdc.pod_registry.store import get_sqlite_connection


def _patch_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(configs.Config, "get_db_dir", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(
        configs.Config, "get_db_file", classmethod(lambda cls: tmp_path / "esdc.duckdb")
    )


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
    monkeypatch.setattr(
        "esdc.portal.app._known_documents",
        lambda: {"ccbd4f3f27635c76": "letter_a.pdf"},
    )
    resp = client.get("/api/tables/pod_document")
    assert resp.status_code == 200
    body = resp.json()
    assert body["rows"] == []
    assert body["refs"]["documents"] == {"ccbd4f3f27635c76": "letter_a.pdf"}


def test_pod_document_save_rejects_unknown_doc_id(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    _insert_pod(client)
    monkeypatch.setattr(
        "esdc.portal.app._known_documents",
        lambda: {"ccbd4f3f27635c76": "letter_a.pdf"},
    )
    resp = client.post("/api/tables/pod_document/save", json={"inserts": [
        {"pod_id": 1, "doc_id": "nope"},
    ]})
    assert resp.status_code == 422
    assert "nope" in resp.json()["errors"][0]["message"]


def test_pod_document_save_and_roundtrip(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    _insert_pod(client)
    monkeypatch.setattr(
        "esdc.portal.app._known_documents",
        lambda: {"ccbd4f3f27635c76": "letter_a.pdf"},
    )
    resp = client.post("/api/tables/pod_document/save", json={"inserts": [
        {"pod_id": 1, "doc_id": "ccbd4f3f27635c76"},
    ]})
    assert resp.status_code == 200
    rows = client.get("/api/tables/pod_document").json()["rows"]
    assert rows == [{"pod_id": 1, "doc_id": "ccbd4f3f27635c76"}]


def test_pod_document_save_skips_doc_check_when_corpus_unavailable(
    monkeypatch, tmp_path
):
    # No DuckDB documents table -> validation degrades gracefully (like
    # project ids), it must not hard-fail the save.
    client = _client(monkeypatch, tmp_path)
    _insert_pod(client)
    resp = client.post("/api/tables/pod_document/save", json={"inserts": [
        {"pod_id": 1, "doc_id": "ccbd4f3f27635c76"},
    ]})
    assert resp.status_code == 200
