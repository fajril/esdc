"""Regression tests for isolated fetch test configuration."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def outer_db_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a process-level database override outside the test fixture."""
    monkeypatch.setenv("ESDC_DB_DIR", str(tmp_path / "outer"))


def test_isolated_config_overrides_outer_db_dir(isolated_config: Path) -> None:
    """All database paths should resolve inside the isolated test directory."""
    from esdc.configs import Config
    from esdc.pod_registry.store import get_esdc_sqlite_path

    expected_dir = isolated_config / ".esdc"
    outer_dir = isolated_config / "outer"

    assert Config.get_db_dir() == expected_dir
    assert get_esdc_sqlite_path() == expected_dir / "esdc.sqlite"
    assert Config.get_db_file() == expected_dir / "esdc.duckdb"
    assert Config.get_db_dir() != outer_dir
