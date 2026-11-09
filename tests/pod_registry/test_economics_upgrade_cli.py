"""Explicit `esdc pod-upgrade-economics` command regressions."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from esdc.esdc import app
from tests.pod_registry.economics_fixture import (
    build_registry,
    columns,
    connect,
    dump,
    renames,
    sentinels,
)

runner = CliRunner()


def _invoke(*args: str):
    return runner.invoke(
        app, ["pod-upgrade-economics", *args], catch_exceptions=False
    )


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "config"
    monkeypatch.setenv("ESDC_CONFIG_DIR", str(root))
    monkeypatch.setenv("ESDC_DB_DIR", str(root))
    monkeypatch.setenv("ESDC_DB_FILE", str(root / "esdc.duckdb"))


def _payload(stdout: str) -> dict[str, Any]:
    return json.loads(stdout)


def test_upgrade_cli_dry_run_is_read_only(tmp_path: Path) -> None:
    old = tmp_path / "old.sqlite"
    build_registry(old)
    before = old.read_bytes()

    result = _invoke("--database", str(old))
    assert result.exit_code == 0, result.output
    payload = _payload(result.stdout)
    assert payload["status"] == "upgrade_required"
    assert len(payload["renames"]) == 24
    assert payload["blockers"] == []
    assert old.read_bytes() == before

    current = tmp_path / "current.sqlite"
    build_registry(current, contract_version=2, renamed=True)
    current_before = current.read_bytes()
    result = _invoke("--database", str(current))
    assert result.exit_code == 0, result.output
    payload = _payload(result.stdout)
    assert payload["status"] == "current"
    assert payload["renames"] == []
    assert current.read_bytes() == current_before


def test_upgrade_cli_apply_requires_backup_path(tmp_path: Path) -> None:
    old = tmp_path / "old.sqlite"
    build_registry(old)
    before = dump(old)

    result = _invoke("--database", str(old), "--apply")
    assert result.exit_code != 0
    assert dump(old) == before


def test_upgrade_cli_backup_preserves_pre_upgrade_database(tmp_path: Path) -> None:
    old = tmp_path / "old.sqlite"
    backup = tmp_path / "backup.sqlite"
    build_registry(old)
    before = dump(old)
    values = sentinels()

    result = _invoke(
        "--database", str(old), "--apply", "--backup", str(backup)
    )
    assert result.exit_code == 0, result.output
    assert _payload(result.stdout)["status"] == "current"

    conn = sqlite3.connect(old)
    try:
        assert tuple(columns(conn, "m_pod")) == tuple(
            "npv_ctr_pod" if name == "npv_pf_ctr_pod" else name
            for name in columns(conn, "m_pod")
        )
        assert conn.execute(
            "SELECT contract_version FROM pod_registry_state WHERE singleton_id = 1"
        ).fetchone()[0] == 2
    finally:
        conn.close()

    assert dump(backup) == before
    old_by_new = {(table, new): old for table, old, new in renames()}
    conn = connect(backup)
    try:
        assert "npv_pf_ctr_pod" in columns(conn, "m_pod")
        assert "npv_ctr_pod" not in columns(conn, "m_pod")
        assert conn.execute(
            "SELECT contract_version FROM pod_registry_state WHERE singleton_id = 1"
        ).fetchone()[0] == 1
        for (table, new), expected in values.items():
            stored = conn.execute(
                f'SELECT "{old_by_new[(table, new)]}" FROM "{table}" WHERE rowid = 1'
            ).fetchone()[0]
            assert stored == expected
    finally:
        conn.close()


def test_upgrade_cli_refuses_unsafe_paths(tmp_path: Path) -> None:
    old = tmp_path / "old.sqlite"
    build_registry(old)
    before = dump(old)

    missing = tmp_path / "missing.sqlite"
    result = _invoke("--database", str(missing))
    assert result.exit_code != 0

    result = _invoke(
        "--database", str(old), "--apply", "--backup", str(old)
    )
    assert result.exit_code != 0
    assert dump(old) == before

    link = tmp_path / "link.sqlite"
    link.symlink_to(old)
    result = _invoke(
        "--database", str(old), "--apply", "--backup", str(link)
    )
    assert result.exit_code != 0
    assert dump(old) == before

    existing = tmp_path / "existing.sqlite"
    existing.write_bytes(b"do not overwrite")
    result = _invoke(
        "--database", str(old), "--apply", "--backup", str(existing)
    )
    assert result.exit_code != 0
    assert existing.read_bytes() == b"do not overwrite"
    assert dump(old) == before


def test_upgrade_cli_blocked_layout_and_backup_failure_preserve_target(
    tmp_path: Path,
) -> None:
    blocked = tmp_path / "blocked.sqlite"
    build_registry(blocked)
    conn = connect(blocked)
    try:
        conn.execute("DROP TRIGGER trg_m_pod_identity_immutable")
        conn.commit()
    finally:
        conn.close()
    blocked_before = dump(blocked)

    backup = tmp_path / "blocked-backup.sqlite"
    result = _invoke(
        "--database", str(blocked), "--apply", "--backup", str(backup)
    )
    assert result.exit_code != 0
    assert not backup.exists()
    assert dump(blocked) == blocked_before

    old = tmp_path / "old.sqlite"
    build_registry(old)
    old_before = dump(old)
    unwritable = tmp_path / "missing-parent" / "backup.sqlite"
    result = _invoke(
        "--database", str(old), "--apply", "--backup", str(unwritable)
    )
    assert result.exit_code != 0
    assert dump(old) == old_before
    assert not unwritable.exists()


def test_upgrade_cli_current_database_is_noop(tmp_path: Path) -> None:
    current = tmp_path / "current.sqlite"
    build_registry(current, contract_version=2, renamed=True)
    backup = tmp_path / "unnecessary.sqlite"
    before = dump(current)

    result = _invoke(
        "--database", str(current), "--apply", "--backup", str(backup)
    )
    assert result.exit_code == 0, result.output
    payload = _payload(result.stdout)
    assert payload["status"] == "current"
    assert not backup.exists()
    assert dump(current) == before


def test_upgrade_cli_current_database_with_existing_backup_is_noop(
    tmp_path: Path,
) -> None:
    current = tmp_path / "current.sqlite"
    build_registry(current, contract_version=2, renamed=True)
    backup = tmp_path / "already-used.sqlite"
    backup.write_bytes(b"keep this")
    before = dump(current)

    result = _invoke(
        "--database", str(current), "--apply", "--backup", str(backup)
    )

    assert result.exit_code == 0, result.output
    assert _payload(result.stdout)["status"] == "current"
    assert backup.read_bytes() == b"keep this"
    assert dump(current) == before


def test_upgrade_cli_rejects_database_sidecar_backup(tmp_path: Path) -> None:
    old = tmp_path / "old.sqlite"
    build_registry(old)
    before = dump(old)
    sidecar = Path(f"{old}-wal")

    result = _invoke(
        "--database", str(old), "--apply", "--backup", str(sidecar)
    )

    assert result.exit_code != 0
    assert dump(old) == before
    assert not sidecar.exists()


def test_upgrade_apply_does_not_create_a_missing_database(tmp_path: Path) -> None:
    from esdc.commands import pod as pod_module

    missing = tmp_path / "gone.sqlite"
    backup = tmp_path / "backup.sqlite"

    with pytest.raises(sqlite3.OperationalError):
        pod_module._apply(missing, backup)

    assert not missing.exists()
    assert not backup.exists()
