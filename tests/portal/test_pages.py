"""Current contract pages expose all fields without opening a source writer."""

import json
import re

from fastapi.testclient import TestClient

from esdc.configs import Config
from esdc.pod_registry.contract import load_pod_contract
from esdc.portal.app import create_portal_app


def _client(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "get_db_dir", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(
        Config, "get_db_file", classmethod(lambda cls: tmp_path / "cache.duckdb")
    )
    return TestClient(create_portal_app())


def configs(text):
    return {
        name: json.loads(body)
        for name, body in re.findall(
            r'<script type="application/json" id="config-([^"]+)">(.*?)</script>',
            text,
            re.S,
        )
    }


def test_root_and_revisions_redirect_to_current_pod_editor(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    for route in ("/", "/revisions"):
        response = client.get(route, follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == "/pods"
    assert not (tmp_path / "esdc.sqlite").exists()


def test_all_pages_use_current_tables_and_readonly_results(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    routes = {
        "/pods": {"m_pod"},
        "/projects": {"m_project"},
        "/links": {"pod_project"},
        "/monitoring": {"pod_monitoring"},
        "/forecast": {"pod_forecast"},
        "/results": {"pod_monitoring_api", "pod_forecast_api"},
        "/references": {"r_institution", "r_pod_type", "r_pod_stage"},
        "/documents": {"documents", "pod_document"},
    }
    for route, expected in routes.items():
        response = client.get(route)
        assert response.status_code == 200, (route, response.text)
        actual = configs(response.text)
        assert set(actual) == expected
        assert (
            '"project_pod"' not in response.text
            and '"pod_revision"' not in response.text
        )
        if route in {"/results", "/references"}:
            assert all(config["readonly"] for config in actual.values())
    assert not (tmp_path / "esdc.sqlite").exists()


def test_every_pod_field_follows_the_packaged_dictionary_order(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    contract = load_pod_contract()
    for route, table in (("/pods", "m_pod"), ("/monitoring", "pod_monitoring")):
        response = client.get(route)
        actual = configs(response.text)[table]
        assert [column["field"] for column in actual["columns"]] == [
            column.name for column in contract.tables[table].columns
        ]
        assert actual["columnChooser"] is True
        readonly = {
            column["field"] for column in actual["columns"] if column["readonly"]
        }
        assert {"created_at", "created_by", "updated_at", "updated_by"} <= readonly
    assert not (tmp_path / "esdc.sqlite").exists()
