"""Generic `esdc load` must not overwrite POD-owned DuckDB projection objects."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import duckdb
import pytest
import yaml
from openpyxl import Workbook

from esdc.configs import Config
from esdc.loaders import (
    _METADATA_TABLE,
    LoadSchemaError,
    load_excel_to_duckdb,
    load_schema_from_yaml,
)
from esdc.pod_registry.contract import load_pod_contract
from esdc.pod_registry.ownership import pod_reserved_names
from esdc.pod_registry.publication import publish_v2_registry
from esdc.pod_registry.store import get_sqlite_connection
from tests.knowledge.conftest import _seed_knowledge_world

GENERIC_COLUMNS = [{"name": "pod_name", "type": "string", "description": "Name"}]
GENERIC_DESCRIPTION = "Generic workbook the caller owns"


@pytest.fixture
def pod_projection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Seed the canonical source and publish the current DuckDB projection."""
    sqlite_path = tmp_path / "esdc.sqlite"
    duck_path = tmp_path / "esdc.duckdb"
    # Isolate Config before anything resolves it, so nothing touches the active DB.
    monkeypatch.setattr(Config, "get_db_file", classmethod(lambda cls: duck_path))
    monkeypatch.setattr(Config, "get_db_dir", classmethod(lambda cls: tmp_path))
    conn = get_sqlite_connection(sqlite_path)
    try:
        _seed_knowledge_world(conn)
    finally:
        conn.close()
    publish_v2_registry(sqlite_path, duck_path)
    return sqlite_path, duck_path


def _duck_shape(db_path: Path, table: str) -> tuple[tuple[str, ...], int]:
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        columns = tuple(
            str(row[1])
            for row in conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        )
        row = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()
        assert row is not None
        rows = int(row[0])
    finally:
        conn.close()
    return columns, rows


def _duck_tables(db_path: Path) -> list[str]:
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables ORDER BY table_name"
        ).fetchall()
    finally:
        conn.close()
    return [str(row[0]) for row in rows]


def _duck_metadata(db_path: Path, table: str) -> tuple[str, ...] | None:
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        row = conn.execute(
            f"SELECT description FROM {_METADATA_TABLE} WHERE table_name = ?",
            [table],
        ).fetchone()
    finally:
        conn.close()
    return None if row is None else tuple(str(value) for value in row)


def _duck_publication(db_path: Path) -> tuple[int, str]:
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        row = conn.execute(
            "SELECT source_generation, source_sha256 FROM pod_registry_publication"
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    return int(row[0]), str(row[1])


def _sqlite_generation(sqlite_path: Path) -> int:
    uri = sqlite_path.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        row = conn.execute("SELECT source_generation FROM pod_registry_state").fetchone()
    finally:
        conn.close()
    assert row is not None
    return int(row[0])


def _generic_schema(path: Path, table_name: str) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "table_name": table_name,
                "description": GENERIC_DESCRIPTION,
                "sheet_name": "Sheet1",
                "columns": GENERIC_COLUMNS,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _generic_workbook(path: Path) -> Path:
    book = Workbook()
    sheet = book.active
    assert sheet is not None
    sheet.title = "Sheet1"
    sheet.append(["pod_name"])
    sheet.append(["Generic replacement bypass"])
    book.save(path)
    book.close()
    return path


def test_mixed_case_rejected_before_any_mutation(
    tmp_path: Path, pod_projection: tuple[Path, Path]
) -> None:
    """The parent probe shape: mixed-case `M_POD` resolves to the published `m_pod`."""
    sqlite_path, duck_path = pod_projection
    before = _duck_shape(duck_path, "m_pod")
    assert len(before[0]) == 82 and before[1] == 3
    metadata_before = _duck_metadata(duck_path, "m_pod")
    publication_before = _duck_publication(duck_path)
    generation_before = _sqlite_generation(sqlite_path)

    schema = _generic_schema(tmp_path / "schema.yaml", "M_POD")
    excel = _generic_workbook(tmp_path / "data.xlsx")

    with pytest.raises(LoadSchemaError, match="POD-owned DuckDB object 'm_pod'"):
        load_excel_to_duckdb(excel, schema)

    assert _duck_shape(duck_path, "m_pod") == before
    assert _duck_metadata(duck_path, "m_pod") == metadata_before
    assert _duck_publication(duck_path) == publication_before
    assert _sqlite_generation(sqlite_path) == generation_before
    assert _duck_metadata(duck_path, "M_POD") is None

    # The boundary precedes Excel read, database open, and directory creation:
    # the same ownership refusal is raised for a source that cannot be read.
    with pytest.raises(LoadSchemaError, match="POD-owned DuckDB object 'm_pod'"):
        load_excel_to_duckdb(tmp_path / "never-read.xlsx", schema)
    assert _duck_shape(duck_path, "m_pod") == before


@pytest.mark.parametrize(
    "table_name",
    (
        "m_pod",
        "M_POD",
        "pod_project",
        "pod_monitoring_api",
        "pod_forecast_api",
        "r_pod_stage",
        "pod_document",
        "pod_registry",
        "pod_registry_publication",
        "pod_value_case",
    ),
)
def test_representative_reserved_names_are_owner_derived(
    tmp_path: Path, pod_projection: tuple[Path, Path], table_name: str
) -> None:
    """Reserved names come from the contract and publisher, not a prefix rule."""
    _, duck_path = pod_projection
    contract = load_pod_contract()
    reserved = pod_reserved_names(contract)
    assert all(name.casefold() in reserved for name in contract.tables)
    assert reserved.get(table_name.casefold()) is not None

    before = _duck_shape(duck_path, "m_pod")
    excel = _generic_workbook(tmp_path / "data.xlsx")
    canonical = reserved[table_name.casefold()]
    metadata_before = _duck_metadata(duck_path, canonical)
    with pytest.raises(LoadSchemaError, match=f"POD-owned DuckDB object '{canonical}'"):
        load_excel_to_duckdb(excel, _generic_schema(tmp_path / "schema.yaml", table_name))

    assert _duck_shape(duck_path, "m_pod") == before
    assert _duck_metadata(duck_path, canonical) == metadata_before
    assert metadata_before != (GENERIC_DESCRIPTION,)


@pytest.mark.parametrize("table_name", ("custom_pod_notes", "pod_notes", "my_pod_projection"))
def test_pod_names_the_contract_does_not_own_remain_loadable(
    tmp_path: Path, pod_projection: tuple[Path, Path], table_name: str
) -> None:
    """A broad POD prefix ban would reject these; generic loads must not."""
    _, duck_path = pod_projection
    assert pod_reserved_names(load_pod_contract()).get(table_name.casefold()) is None

    before = _duck_shape(duck_path, "m_pod")
    excel = _generic_workbook(tmp_path / "data.xlsx")
    result = load_excel_to_duckdb(excel, _generic_schema(tmp_path / "schema.yaml", table_name))
    assert result.table_name == table_name
    assert _duck_shape(duck_path, table_name) == (("pod_name",), 1)
    assert _duck_metadata(duck_path, table_name) == (GENERIC_DESCRIPTION,)
    assert _duck_shape(duck_path, "m_pod") == before


def test_missing_target_absent_and_unrelated_load_still_works(
    tmp_path: Path, pod_projection: tuple[Path, Path]
) -> None:
    sqlite_path, duck_path = pod_projection
    before = _duck_shape(duck_path, "m_pod")
    tables_before = _duck_tables(duck_path)
    assert "pod_value_case" not in tables_before
    metadata_before = _duck_metadata(duck_path, "m_pod")
    publication_before = _duck_publication(duck_path)
    generation_before = _sqlite_generation(sqlite_path)

    excel = _generic_workbook(tmp_path / "data.xlsx")

    # Ownership is derived, not existence-based: an absent reserved target is
    # still refused and remains absent.
    with pytest.raises(LoadSchemaError, match="POD-owned DuckDB object 'pod_value_case'"):
        load_excel_to_duckdb(excel, _generic_schema(tmp_path / "reserved.yaml", "pod_value_case"))
    assert _duck_tables(duck_path) == tables_before
    assert _duck_metadata(duck_path, "pod_value_case") is None

    result = load_excel_to_duckdb(
        excel, _generic_schema(tmp_path / "custom.yaml", "custom_pod_notes")
    )
    assert result.table_name == "custom_pod_notes"
    assert _duck_shape(duck_path, "custom_pod_notes") == (("pod_name",), 1)
    assert _duck_metadata(duck_path, "custom_pod_notes") == (GENERIC_DESCRIPTION,)
    assert _duck_tables(duck_path) == sorted([*tables_before, "custom_pod_notes"])

    assert _duck_shape(duck_path, "m_pod") == before
    assert _duck_metadata(duck_path, "m_pod") == metadata_before
    assert _duck_publication(duck_path) == publication_before
    assert _sqlite_generation(sqlite_path) == generation_before


def test_read_only_schema_parser_still_accepts_pod_owned_names(
    tmp_path: Path, pod_projection: tuple[Path, Path]
) -> None:
    """The boundary is the mutating load, not the read-only parser or template."""
    _, duck_path = pod_projection
    before = _duck_shape(duck_path, "m_pod")
    parsed = load_schema_from_yaml(_generic_schema(tmp_path / "dictionary.yaml", "m_pod"))
    assert parsed.table_name == "m_pod"
    assert _duck_shape(duck_path, "m_pod") == before
