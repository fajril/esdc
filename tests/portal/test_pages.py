from fastapi.testclient import TestClient

import esdc.configs as configs
from esdc.portal.app import create_portal_app


def _client(monkeypatch, tmp_path):
    monkeypatch.setattr(configs.Config, "get_db_dir", classmethod(lambda cls: tmp_path))
    return TestClient(create_portal_app())


def test_root_redirects_to_pods(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code == 307
    assert resp.headers["location"] == "/pods"


def test_pods_page_renders_grid_config(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    resp = client.get("/pods")
    assert resp.status_code == 200
    assert "tabulator" in resp.text
    assert '"m_pod"' in resp.text          # table name passed to JS
    assert "Superseded By" in resp.text


def test_all_pages_render(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    for path in ("/pods", "/links", "/revisions", "/references"):
        assert client.get(path).status_code == 200, path
