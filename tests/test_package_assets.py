"""Wheel packaging checks for runtime-loaded ESDC resources."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def test_wheel_contains_and_loads_contract_and_sql_resources(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    staged_source = tmp_path / "source"
    staged_source.mkdir()
    shutil.copy2(repo_root / "pyproject.toml", staged_source / "pyproject.toml")
    shutil.copy2(repo_root / "LICENSE", staged_source / "LICENSE")
    shutil.copytree(repo_root / "esdc", staged_source / "esdc")
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    build_env = os.environ.copy()
    command = ["uv", "build", "--wheel", "--out-dir", str(wheel_dir)]
    build_python = os.environ.get("ESDC_TEST_BUILD_PYTHON")
    if build_python:
        command.extend(["--offline", "--no-build-isolation", "--python", build_python])
    result = subprocess.run(
        command,
        cwd=staged_source,
        env=build_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    wheel_path = next(wheel_dir.glob("*.whl"))
    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
    assert "esdc/schemas/pod.yaml" in names
    assert "esdc/sql/pod_registry_v2.sql" in names
    assert "esdc/sql/pod_reference_seed.sql" in names
    assert any(name.startswith("esdc/sql/") and name.endswith(".sql") for name in names)

    required_paths = [
        "esdc/model_settings.py",
        "esdc/model_runtime.py",
        "esdc/model_catalog.py",
        "esdc/model_config_conversion.py",
        "esdc/model_config_apply.py",
        "esdc/model_wizard.py",
        "esdc/portal/templates/base.html",
        "esdc/portal/templates/grid.html",
        "esdc/portal/static/portal.js",
        "esdc/portal/static/editors.js",
        "esdc/portal/static/chart.js",
        "esdc/portal/static/results.js",
        "esdc/portal/static/portal.css",
        "esdc/portal/static/tabulator.min.js",
        "esdc/portal/static/tabulator.min.css",
    ]
    for path in required_paths:
        assert path in names, f"missing packaged asset: {path}"

    for removed in (
        "esdc/chat/domain_knowledge/pod_schema.yaml",
        "esdc/chat/domain_knowledge/pod_registry_schema.yaml",
    ):
        assert removed not in names, f"stale asset present: {removed}"

    installed_target = tmp_path / "installed"
    installed_target.mkdir()
    with zipfile.ZipFile(wheel_path) as wheel:
        wheel.extractall(installed_target)
    outside_checkout = tmp_path / "outside"
    outside_checkout.mkdir()
    installed_env = os.environ.copy()
    installed_env["PYTHONPATH"] = str(installed_target)
    installed_env.pop("PYTHONHOME", None)
    installed_code = (
        "import sqlite3; "
        "from importlib.resources import files; "
        "from esdc.pod_registry.contract import load_pod_contract; "
        "from esdc.pod_registry.migrations import initialize_registry_v2; "
        "from esdc.model_settings import default_model_settings; "
        "from esdc.model_config_conversion import convert_model_settings; "
        "assert default_model_settings()['rerank']['enabled'] is False; "
        "assert convert_model_settings({}).config['model_settings_version'] == 1; "
        "contract = load_pod_contract(); "
        "db = sqlite3.connect(':memory:'); "
        "db.execute('PRAGMA foreign_keys = ON'); "
        "initialize_registry_v2(db); "
        "sql = files('esdc').joinpath('sql').joinpath('create_table_project_resources.sql').read_text(); "
        "registry_sql = files('esdc').joinpath('sql').joinpath('pod_registry_v2.sql').read_text(); "
        "seed_sql = files('esdc').joinpath('sql').joinpath('pod_reference_seed.sql').read_text(); "
        "assert 'CREATE TABLE' in sql.upper(); "
        "assert 'CREATE TABLE IF NOT EXISTS POD_MONITORING_API' in registry_sql.upper(); "
        "assert 'BPPKA' in seed_sql and 'BPMA' in seed_sql; "
        "monitoring = len(db.execute('PRAGMA table_info(pod_monitoring_api)').fetchall()); "
        "forecast = len(db.execute('PRAGMA table_info(pod_forecast_api)').fetchall()); "
        "institutions = db.execute('SELECT COUNT(*) FROM r_institution').fetchone()[0]; "
        "types = db.execute('SELECT COUNT(*) FROM r_pod_type').fetchone()[0]; "
        "stages = db.execute('SELECT COUNT(*) FROM r_pod_stage').fetchone()[0]; "
        "from fastapi.testclient import TestClient; "
        "from esdc.portal.app import create_portal_app; "
        "app = create_portal_app(); "
        "client = TestClient(app); "
        "pods = client.get('/pods'); "
        "static = client.get('/static/results.js'); "
        "assert pods.status_code == 200; "
        "assert static.status_code == 200 and 'initResults' in static.text; "
        "print(contract.contract_id, contract.contract_version, contract.dictionary_version, "
        "monitoring, forecast, institutions, types, stages, len(sql))"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", installed_code],
        cwd=outside_checkout,
        env=installed_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert loaded.returncode == 0, loaded.stdout + loaded.stderr
    fields = loaded.stdout.split()
    from esdc.pod_registry.contract import load_pod_contract

    contract = load_pod_contract()
    assert fields[:3] == [
        contract.contract_id,
        str(contract.contract_version),
        contract.dictionary_version,
    ]
    assert [int(value) for value in fields[3:8]] == [176, 24, 5, 4, 4]
    assert int(fields[8]) > 0
