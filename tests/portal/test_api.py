"""Current portal HTTP semantics: exact source edits and publication-only retries."""

import sqlite3
from contextlib import closing
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from esdc.configs import Config
from esdc.pod_registry.changesets import apply_changeset
from esdc.pod_registry.store import get_sqlite_connection
from esdc.portal import app as portal


@pytest.fixture()
def target(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "get_db_dir", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(
        Config, "get_db_file", classmethod(lambda cls: tmp_path / "cache.duckdb")
    )
    path = tmp_path / "esdc.sqlite"
    conn = get_sqlite_connection(path)
    conn.close()
    return TestClient(portal.create_portal_app()), path


def source_rows(path, table):
    with closing(sqlite3.connect(path)) as conn:
        return conn.execute(f'SELECT * FROM "{table}" ORDER BY 1').fetchall()


def create_pod(client, **fields):
    gen = client.get("/api/tables/m_pod").json()["source_generation"]
    return client.post(
        "/api/tables/m_pod/save",
        json={
            "source_generation": gen,
            "inserts": [
                {
                    "approval_date": "2024-01-01",
                    "institution_id": 1,
                    "pod_type_id": 1,
                    "pod_name": "POD Example",
                    "pod_letter_num": "POD/001",
                    **fields,
                }
            ],
        },
    )


def test_read_pages_and_empty_documents_never_create_source(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "get_db_dir", classmethod(lambda cls: tmp_path))
    client = TestClient(portal.create_portal_app())
    for table in ("m_pod", "documents"):
        response = client.get(f"/api/tables/{table}")
        assert response.status_code == 200
        assert response.json()["rows"] == []
    assert not (tmp_path / "esdc.sqlite").exists()
    assert client.get("/api/tables/m_pod?size=201").status_code == 422
    assert client.get("/api/tables/not_a_table").status_code == 404


def test_saved_source_survives_failed_publish_and_retry_never_reissues(
    target, monkeypatch
):
    client, path = target
    response = create_pod(client, capital_drilling_pod="12345678901234567.123")
    assert response.status_code == 200, response.text
    saved = response.json()
    assert saved["source_committed"] and saved["publication"] == "pending"
    gen = saved["source_generation"]
    masters = source_rows(path, "m_pod")
    ledger = source_rows(path, "_pod_identity_ledger")
    real = portal.publish_pod_registry
    calls = []

    def fail_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("cache locked")
        return real(*args, **kwargs)

    monkeypatch.setattr(portal, "publish_pod_registry", fail_once)
    failed = client.post("/api/publish", json={"source_generation": gen})
    assert failed.status_code == 503 and failed.json()["publication"] == "pending"
    assert failed.json()["source_committed"]
    for _ in range(2):
        published = client.post("/api/publish", json={"source_generation": gen})
        assert published.status_code == 200, published.text
        assert published.json()["published_generation"] == gen
    assert source_rows(path, "m_pod") == masters
    assert source_rows(path, "_pod_identity_ledger") == ledger
    current = client.get("/api/tables/m_pod").json()
    assert current["source_generation"] == gen
    assert current["rows"][0]["capital_drilling_pod"] == "12345678901234567.123"


def test_source_validation_and_stale_edit_are_structured(target):
    client, path = target
    assert create_pod(client).status_code == 200
    row = client.get("/api/tables/m_pod").json()["rows"][0]
    before = source_rows(path, "m_pod")
    stale = client.post(
        "/api/tables/m_pod/save",
        json={
            "source_generation": 0,
            "updates": [{"pod_id_seq": row["pod_id_seq"], "pod_name": "stale"}],
        },
    )
    assert stale.status_code == 409 and stale.json()["conflict"]
    invalid = client.post(
        "/api/tables/m_pod/save",
        json={
            "source_generation": 1,
            "updates": [
                {"pod_id_seq": row["pod_id_seq"], "capital_drilling_pod": "0.0001"}
            ],
        },
    )
    assert invalid.status_code == 422
    assert invalid.json()["errors"][0]["field"] == "capital_drilling_pod"
    assert source_rows(path, "m_pod") == before
    for table in ("r_pod_stage", "pod_forecast_api"):
        assert client.post(f"/api/tables/{table}/save", json={}).status_code == 405


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"source_generation": True},
        {"source_generation": -1},
        {"source_generation": "1"},
        [],
    ],
)
def test_publish_requires_exact_generation_without_cache_write(target, payload):
    client, path = target
    result = client.post("/api/publish", json=payload)
    assert result.status_code == 422
    assert not (path.parent / "cache.duckdb").exists()


def test_stale_publication_refused_before_cache_write(target):
    client, path = target
    assert create_pod(client).status_code == 200
    assert client.post("/api/publish", json={"source_generation": 0}).status_code == 409
    assert not (path.parent / "cache.duckdb").exists()


def test_publish_checks_projection_error_even_when_report_is_empty(target, monkeypatch):
    client, _ = target
    assert create_pod(client).status_code == 200
    monkeypatch.setattr(
        portal,
        "publish_pod_registry",
        lambda *a, **k: SimpleNamespace(projection_error="projection failed"),
    )
    result = client.post("/api/publish", json={"source_generation": 1})
    assert result.status_code == 503
    assert "projection failed" in result.json()["error"]


def test_chart_full_series_preserves_null_exact_values_and_diagnostic_scope(target):
    client, path = target
    assert create_pod(client).status_code == 200
    pod = client.get("/api/tables/m_pod").json()["rows"][0]["pod_id_seq"]
    conn = sqlite3.connect(path)
    try:
        conn.executemany(
            "INSERT INTO pod_forecast_api (pod_id_seq,reported_date,year_frcast,usf_oil_pod,usf_oil_otlk) VALUES (?, '2024-12-31', ?, ?, ?)",
            [
                (pod, y, "9007199254740993.001", None if y == 2050 else "0.000")
                for y in range(2000, 2151)
            ],
        )
        conn.execute(
            "INSERT INTO pod_read_model_diagnostic (source_generation,code,message,pod_id_seq,reported_date) VALUES (1,'GLOBAL','missing basis',NULL,NULL)"
        )
        conn.commit()
    finally:
        conn.close()
    query = {"pod_id_seq": pod, "reported_date": "2024-12-31", "component": "oil"}
    result = client.get("/api/chart", params=query)
    assert result.status_code == 200, result.text
    body = result.json()
    assert len(body["series"]) == 151 and body["unit"] == "MSTBY"
    assert body["series"][50] == {
        "year": 2050,
        "approved": "9007199254740993.001",
        "outlook": None,
    }
    diag = client.get(
        "/api/diagnostics", params={k: v for k, v in query.items() if k != "component"}
    ).json()
    assert diag["total"] == len(diag["rows"])
    assert any(row["code"] == "GLOBAL" for row in diag["rows"])
    assert diag["basis"]["source_generation"] == 1
    query["reported_date"] = "20241231"
    assert client.get("/api/chart", params=query).status_code == 422
    query["reported_date"] = "2024-12-31"
    query["component"] = "oil FROM m_pod"
    assert client.get("/api/chart", params=query).status_code == 422


def test_lookup_includes_manual_master_without_resources(target):
    client, path = target
    response = apply_changeset(
        "m_project",
        {
            "source_generation": 0,
            "inserts": [
                {
                    "project_id": "P-0000001-01",
                    "project_name": "Manual Project",
                    "is_active": False,
                }
            ],
        },
        path,
    )
    assert response.ok, response.errors
    result = client.get("/api/projects", params={"q": "Manual"})
    assert result.status_code == 200
    assert result.json() == [
        {"value": "P-0000001-01", "label": "P-0000001-01 — Manual Project (inactive)"}
    ]


def test_download_workbook_is_read_only_and_cleans_its_temporary_directory(
    target, monkeypatch
):
    from contextlib import closing
    from io import BytesIO

    from openpyxl import load_workbook

    client, path = target
    assert create_pod(client).status_code == 200
    before = source_rows(path, "m_pod")
    temporary = []
    real = portal.TemporaryDirectory

    def temporary_directory(**kwargs):
        result = real(dir=path.parent, **kwargs)
        temporary.append(result.name)
        return result

    monkeypatch.setattr(portal, "TemporaryDirectory", temporary_directory)
    response = client.get("/api/backup")
    assert response.status_code == 200, response.text
    assert "attachment" in response.headers["content-disposition"]
    with closing(load_workbook(BytesIO(response.content), read_only=True)) as workbook:
        assert workbook["m_pod"].max_row is None or workbook["m_pod"].max_row >= 2
        assert "__pod_identity_ledger" in workbook.sheetnames
    from pathlib import Path

    assert temporary and all(not Path(name).exists() for name in temporary)
    assert source_rows(path, "m_pod") == before


def test_diagnostics_pages_include_every_global_and_filtered_row(target):
    client, path = target
    with closing(sqlite3.connect(path)) as conn:
        conn.executemany(
            "INSERT INTO pod_read_model_diagnostic (source_generation,code,message,pod_id_seq,reported_date) VALUES (0,?,?,NULL,NULL)",
            [("D" + str(i), "Global " + str(i)) for i in range(405)],
        )
        conn.execute(
            "INSERT INTO pod_read_model_diagnostic (source_generation,code,message,pod_id_seq,reported_date) VALUES (0,'OTHER','not selected',2,'2025-12-31')"
        )
        conn.commit()
    pages = [
        client.get(
            "/api/diagnostics",
            params={"pod_id_seq": 1, "reported_date": "2024-12-31", "page": page},
        ).json()
        for page in (1, 2, 3)
    ]
    assert [len(page["rows"]) for page in pages] == [200, 200, 5]
    assert all(page["total"] == 405 and page["last_page"] == 3 for page in pages)
    ids = [row["diagnostic_id"] for page in pages for row in page["rows"]]
    assert len(ids) == len(set(ids)) == 405 and ids == sorted(ids)
    assert client.get("/api/diagnostics", params={"page": 0}).status_code == 422


def test_document_conflict_retains_generation_checked_under_writer_lock(
    target, monkeypatch
):
    from esdc.portal.document_entities import apply_document_entity_changeset as real
    from tests.portal.test_document_entities import FakeResolver

    client, path = target

    def interleave(changes):
        created = create_pod(client)
        assert created.status_code == 200, created.text
        return real(
            changes,
            sqlite_path=path,
            db_path=path.parent / "cache.duckdb",
            resolver=FakeResolver(),
        )

    monkeypatch.setattr(portal, "apply_document_entity_changeset", interleave)
    response = client.post(
        "/api/tables/documents/save",
        json={"source_generation": 0, "updates": [{"doc_id": "D1", "wk_name": ""}]},
    )
    assert response.status_code == 409, response.text
    assert response.json()["source_generation"] == 1
    assert response.json()["conflict"] and not response.json()["source_committed"]
