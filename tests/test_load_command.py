"""Tests for the load CLI command."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import date

import duckdb
import pandas as pd
import pytest
import yaml
from openpyxl import load_workbook

from esdc.chat.tools import knowledge_traversal
from esdc.configs import Config
from esdc.esdc import app
from esdc.loaders import (
    LoadSchemaError,
    build_loaded_schema_graph,
    load_schema_from_yaml,
)
from esdc.pod_registry.contract import load_pod_contract
from esdc.pod_registry.store import get_esdc_sqlite_path
from tests.pod_registry.test_importer import (
    INPUT_TABLES,
    _counts,
    _row,
    _write_workbook,
)


def _write_schema(path, *, columns=None, links=None):
    schema = {
        "table_name": "sample_table",
        "description": "Sample loaded table",
        "sheet_name": "Sheet1",
        "columns": columns
        or [
            {
                "name": "project_id",
                "type": "string",
                "description": "ID proyek",
                "aliases": ["project code", "kode proyek"],
            },
            {
                "name": "volume",
                "type": "double",
                "description": "Volume sumber daya",
                "unit": "MMBOE",
            },
        ],
    }
    if links is not None:
        schema["links"] = links
    path.write_text(yaml.safe_dump(schema, sort_keys=False), encoding="utf-8")
    return path


def _write_excel(path, rows):
    df = pd.DataFrame(rows)
    df.to_excel(path, index=False, sheet_name="Sheet1")
    return path


class TestLoadSchemaParser:
    def test_schema_valid(self, tmp_path):
        schema = load_schema_from_yaml(_write_schema(tmp_path / "schema.yaml"))

        assert schema.table_name == "sample_table"
        assert schema.sheet_name == "Sheet1"
        assert schema.columns[0].name == "project_id"
        assert schema.columns[1].duckdb_type == "DOUBLE"

    def test_schema_missing_required_field(self, tmp_path):
        path = tmp_path / "schema.yaml"
        path.write_text("table_name: sample_table\n", encoding="utf-8")

        with pytest.raises(LoadSchemaError, match="description"):
            load_schema_from_yaml(path)

    def test_schema_duplicate_column_name(self, tmp_path):
        path = _write_schema(
            tmp_path / "schema.yaml",
            columns=[
                {"name": "project_id", "type": "string", "description": "A"},
                {"name": "project_id", "type": "string", "description": "B"},
            ],
        )

        with pytest.raises(LoadSchemaError, match="Duplicate column name"):
            load_schema_from_yaml(path)

    def test_schema_unknown_type(self, tmp_path):
        path = _write_schema(
            tmp_path / "schema.yaml",
            columns=[
                {"name": "project_id", "type": "unknown", "description": "A"},
            ],
        )

        with pytest.raises(LoadSchemaError, match="Unsupported column type"):
            load_schema_from_yaml(path)

    def test_schema_valid_with_links_and_maps_to(self, tmp_path):
        path = _write_schema(
            tmp_path / "schema.yaml",
            columns=[
                {"name": "project_id", "type": "string", "description": "Project ID"},
                {
                    "name": "volume",
                    "type": "double",
                    "description": "Volume",
                    "maps_to": ["RESOURCES"],
                },
            ],
            links=[
                {
                    "entity_type": "Project",
                    "column": "project_id",
                }
            ],
        )

        schema = load_schema_from_yaml(path)

        assert schema.links[0].entity_type == "Project"
        assert schema.links[0].target_key == "project_id"
        assert schema.columns[1].maps_to == ("RESOURCES",)

    def test_schema_invalid_link_entity_type(self, tmp_path):
        path = _write_schema(
            tmp_path / "schema.yaml",
            links=[{"entity_type": "Basin", "column": "project_id"}],
        )

        with pytest.raises(LoadSchemaError, match="Unsupported links"):
            load_schema_from_yaml(path)

    def test_schema_link_column_must_exist(self, tmp_path):
        path = _write_schema(
            tmp_path / "schema.yaml",
            links=[{"entity_type": "Project", "column": "missing_id"}],
        )

        with pytest.raises(LoadSchemaError, match="is not in columns"):
            load_schema_from_yaml(path)

    def test_loaded_schema_graph_builds_metadata_edges(self, tmp_path):
        path = _write_schema(
            tmp_path / "schema.yaml",
            columns=[
                {"name": "project_id", "type": "string", "description": "Project ID"},
                {
                    "name": "volume",
                    "type": "double",
                    "description": "Volume",
                    "maps_to": ["RESOURCES"],
                },
            ],
            links=[{"entity_type": "Project", "column": "project_id"}],
        )
        schema = load_schema_from_yaml(path)
        schema_yaml = yaml.safe_dump(schema.raw, sort_keys=False)

        conn = build_loaded_schema_graph(
            [("sample_table", "Sample loaded table", "Sheet1", schema_yaml)]
        )
        result = conn.execute(
            "MATCH (t:LoadedTable)-[:LOADED_TABLE_HAS_COLUMN]->(c:LoadedColumn) "
            "RETURN t.table_name, c.name"
        )
        rows = []
        while result.has_next():
            rows.append(tuple(result.get_next()))

        assert rows == [("sample_table", "project_id"), ("sample_table", "volume")]


class TestLoadCommand:
    def test_load_help(self, runner):
        result = runner.invoke(app, ["load", "--help"])

        assert result.exit_code == 0
        assert "--from-excel" in result.stdout
        assert "--schema" in result.stdout
        assert "--schema-pod" in result.stdout
        assert "esdc schema --generate --from-excel data.xlsx" in result.stdout

    def test_load_requires_schema_or_schema_pod(self, runner, tmp_path):
        excel_path = _write_excel(tmp_path / "data.xlsx", [{"project_id": "A-1"}])

        result = runner.invoke(app, ["load", "--from-excel", str(excel_path)])

        assert result.exit_code == 1
        assert "Excel POD loads require --schema-pod or --pod-registry" in result.stdout

    def test_load_excel_creates_table_and_metadata(
        self, runner, isolated_config, tmp_path
    ):
        schema_path = _write_schema(tmp_path / "schema.yaml")
        excel_path = _write_excel(
            tmp_path / "data.xlsx",
            [
                {"project_id": "A-1", "volume": 12.5},
                {"project_id": "A-2", "volume": 20.0},
            ],
        )

        result = runner.invoke(
            app,
            ["load", "--from-excel", str(excel_path), "--schema", str(schema_path)],
        )

        assert result.exit_code == 0
        assert "Loaded 2 rows x 2 columns into 'sample_table'" in result.stdout

        conn = duckdb.connect(str(Config.get_db_file()))
        try:
            count_row = conn.execute("SELECT COUNT(*) FROM sample_table").fetchone()
            assert count_row is not None
            count = count_row[0]
            metadata = conn.execute(
                "SELECT description FROM _loaded_table_schemas WHERE table_name = ?",
                ["sample_table"],
            ).fetchone()
        finally:
            conn.close()
        assert count == 2
        assert metadata == ("Sample loaded table",)

    def test_load_pod_schema_builtin(self, runner, isolated_config, tmp_path):
        """--schema-pod resolves the packaged canonical contract, not a YAML template."""
        contract = load_pod_contract()
        rows = {table: [_row(contract, table)] for table in INPUT_TABLES}
        excel_path = _write_workbook(tmp_path / "pod.xlsx", contract, rows)

        result = runner.invoke(
            app, ["load", "--from-excel", str(excel_path), "--schema-pod"]
        )

        assert result.exit_code == 0
        assert "m_pod: 1 rows" in result.stdout
        assert "POD registry published to DuckDB." in result.stdout
        assert _counts(get_esdc_sqlite_path()) == dict.fromkeys(INPUT_TABLES, 1)

        conn = duckdb.connect(str(Config.get_db_file()))
        try:
            published = {
                row[0]
                for row in conn.execute(
                    "SELECT table_name FROM information_schema.tables"
                ).fetchall()
            }
            pod_row = conn.execute("SELECT pod_id, approval_date FROM m_pod").fetchone()
            api_count = conn.execute(
                "SELECT COUNT(*) FROM pod_monitoring_api"
            ).fetchone()
        finally:
            conn.close()
        assert set(INPUT_TABLES) <= published
        assert pod_row == ("PL-2024-0001-1-1-0", date(2024, 1, 1))
        assert api_count is not None and api_count[0] == 1

    def test_load_replaces_existing_table(self, runner, isolated_config, tmp_path):
        schema_path = _write_schema(tmp_path / "schema.yaml")
        first_excel = _write_excel(
            tmp_path / "data1.xlsx",
            [{"project_id": "A-1", "volume": 12.5}],
        )
        second_excel = _write_excel(
            tmp_path / "data2.xlsx",
            [
                {"project_id": "B-1", "volume": 99.0},
                {"project_id": "B-2", "volume": 100.0},
            ],
        )

        first = runner.invoke(
            app,
            ["load", "--from-excel", str(first_excel), "--schema", str(schema_path)],
        )
        second = runner.invoke(
            app,
            ["load", "--from-excel", str(second_excel), "--schema", str(schema_path)],
        )

        assert first.exit_code == 0
        assert second.exit_code == 0
        conn = duckdb.connect(str(Config.get_db_file()))
        try:
            rows = conn.execute(
                "SELECT project_id FROM sample_table ORDER BY project_id"
            ).fetchall()
        finally:
            conn.close()
        assert rows == [("B-1",), ("B-2",)]

    def test_load_excel_column_mismatch_fails(self, runner, isolated_config, tmp_path):
        schema_path = _write_schema(tmp_path / "schema.yaml")
        excel_path = _write_excel(
            tmp_path / "data.xlsx",
            [{"project_id": "A-1", "unexpected": 12.5}],
        )

        result = runner.invoke(
            app,
            ["load", "--from-excel", str(excel_path), "--schema", str(schema_path)],
        )

        assert result.exit_code == 1
        assert "Excel columns do not match schema" in result.stdout
        assert "missing columns: volume" in result.stdout
        assert "unexpected columns: unexpected" in result.stdout

    def test_load_excel_missing_sheet_fails(self, runner, isolated_config, tmp_path):
        schema_path = _write_schema(tmp_path / "schema.yaml")
        excel_path = tmp_path / "data.xlsx"
        pd.DataFrame([{"project_id": "A-1", "volume": 12.5}]).to_excel(
            excel_path, index=False, sheet_name="OtherSheet"
        )

        result = runner.invoke(
            app,
            ["load", "--from-excel", str(excel_path), "--schema", str(schema_path)],
        )

        assert result.exit_code == 1
        assert "Excel sheet 'Sheet1' not found" in result.stdout

    def test_load_excel_type_conversion_fails(self, runner, isolated_config, tmp_path):
        schema_path = _write_schema(tmp_path / "schema.yaml")
        excel_path = _write_excel(
            tmp_path / "data.xlsx",
            [{"project_id": "A-1", "volume": "not-a-number"}],
        )

        result = runner.invoke(
            app,
            ["load", "--from-excel", str(excel_path), "--schema", str(schema_path)],
        )

        assert result.exit_code == 1
        assert "Column 'volume' cannot be converted to double" in result.stdout

    def test_load_with_links_warns_for_unmatched_ids(
        self, runner, isolated_config, tmp_path
    ):
        Config.init_config()
        Config.get_db_dir().mkdir(parents=True, exist_ok=True)
        conn = duckdb.connect(str(Config.get_db_file()))
        try:
            conn.execute("CREATE TABLE project_resources (project_id TEXT)")
            conn.execute("INSERT INTO project_resources VALUES ('A-1')")
        finally:
            conn.close()
        schema_path = _write_schema(
            tmp_path / "schema.yaml",
            links=[{"entity_type": "Project", "column": "project_id"}],
        )
        excel_path = _write_excel(
            tmp_path / "data.xlsx",
            [
                {"project_id": "A-1", "volume": 12.5},
                {"project_id": "A-2", "volume": 20.0},
            ],
        )

        result = runner.invoke(
            app,
            ["load", "--from-excel", str(excel_path), "--schema", str(schema_path)],
        )

        assert result.exit_code == 0
        assert (
            "Warning: 1 value(s) in 'project_id' did not match Project.project_id"
            in result.stdout
        )

    def test_knowledge_traversal_finds_loaded_table_and_column(
        self, runner, isolated_config, tmp_path
    ):
        schema_path = _write_schema(
            tmp_path / "schema.yaml",
            columns=[
                {
                    "name": "project_id",
                    "type": "string",
                    "description": "ID proyek",
                    "aliases": ["project code", "kode proyek"],
                },
                {
                    "name": "volume",
                    "type": "double",
                    "description": "Volume sumber daya",
                    "unit": "MMBOE",
                    "maps_to": ["RESOURCES"],
                },
            ],
            links=[{"entity_type": "Project", "column": "project_id"}],
        )
        excel_path = _write_excel(
            tmp_path / "data.xlsx",
            [{"project_id": "A-1", "volume": 12.5}],
        )
        load_result = runner.invoke(
            app,
            ["load", "--from-excel", str(excel_path), "--schema", str(schema_path)],
        )
        assert load_result.exit_code == 0

        table_result = knowledge_traversal.invoke(
            {"topic": "definition", "entity": "sample_table"}
        )
        column_result = knowledge_traversal.invoke(
            {"topic": "definition", "entity": "kode proyek"}
        )

        assert "Loaded table: sample_table" in table_result
        assert "project_id: ID proyek" in table_result
        assert "project_id links to Project.project_id" in table_result
        assert "Loaded column: sample_table.project_id" in column_result
        assert "Description: ID proyek" in column_result
        volume_result = knowledge_traversal.invoke(
            {"topic": "definition", "entity": "volume"}
        )
        assert "Maps to KSMI: RESOURCES" in volume_result
        assert "A-1" not in table_result
        assert "A-1" not in column_result


class TestPodLoadValidation:
    def test_load_pod_undeclared_column_fails_before_the_target_is_written(
        self, runner, isolated_config, tmp_path
    ):
        contract = load_pod_contract()
        rows = {table: [_row(contract, table)] for table in INPUT_TABLES}
        excel_path = _write_workbook(
            tmp_path / "pod.xlsx",
            contract,
            rows,
            extra_header=("m_pod", "not_declared"),
        )

        result = runner.invoke(
            app, ["load", "--from-excel", str(excel_path), "--schema-pod"]
        )

        assert result.exit_code == 1
        assert "Error: POD import failed:" in result.stdout
        assert "not declared for m_pod" in result.stdout
        assert not get_esdc_sqlite_path().exists()

    def test_load_pod_dry_run_does_not_open_the_target_for_writing(
        self, runner, isolated_config, tmp_path
    ):
        contract = load_pod_contract()
        rows = {table: [_row(contract, table)] for table in INPUT_TABLES}
        excel_path = _write_workbook(tmp_path / "pod.xlsx", contract, rows)

        result = runner.invoke(
            app,
            ["load", "--from-excel", str(excel_path), "--schema-pod", "--dry-run"],
        )

        assert result.exit_code == 0
        assert "m_pod: 1 rows" in result.stdout
        assert "target was not opened for writing" in result.stdout
        assert not get_esdc_sqlite_path().exists()

    def test_load_pod_master_only_batch_that_would_orphan_committed_facts_is_refused(
        self, runner, isolated_config, tmp_path
    ):
        contract = load_pod_contract()
        rows = {table: [_row(contract, table)] for table in INPUT_TABLES}
        excel_path = _write_workbook(tmp_path / "pod.xlsx", contract, rows)
        runner.invoke(app, ["load", "--from-excel", str(excel_path), "--schema-pod"])

        later = {table: [_row(contract, table, 2)] for table in INPUT_TABLES}
        master_only = {
            table: later[table] for table in ("m_project", "m_pod", "pod_project")
        }
        result = runner.invoke(
            app,
            [
                "load",
                "--from-excel",
                str(_write_workbook(tmp_path / "master.xlsx", contract, master_only)),
                "--pod-registry",
            ],
        )

        assert result.exit_code == 1
        assert "would orphan preserved pod_monitoring facts" in result.stdout
        committed = _counts(get_esdc_sqlite_path())
        assert committed == dict.fromkeys(INPUT_TABLES, 1)

    def test_load_pod_projection_failure_reports_the_committed_sqlite(
        self, runner, isolated_config, tmp_path, monkeypatch
    ):
        from esdc.pod_registry.publish import publish_pod_registry

        contract = load_pod_contract()
        rows = {table: [_row(contract, table)] for table in INPUT_TABLES}
        excel_path = _write_workbook(tmp_path / "pod.xlsx", contract, rows)

        def boom(*args, **kwargs):
            raise RuntimeError("projection broke")

        monkeypatch.setattr("esdc.pod_registry.publish.publish_pod_registry", boom)
        result = runner.invoke(
            app, ["load", "--from-excel", str(excel_path), "--schema-pod"]
        )

        assert result.exit_code == 1
        assert "SQLite source committed, but DuckDB publication raised" in result.stdout
        assert "do not re-import the source" in result.stdout
        assert _counts(get_esdc_sqlite_path())["m_pod"] == 1
        with closing(sqlite3.connect(get_esdc_sqlite_path())) as source:
            before = tuple(source.execute("SELECT * FROM m_pod").fetchall())
            ledger = tuple(
                source.execute("SELECT * FROM _pod_identity_ledger").fetchall()
            )
        monkeypatch.setattr(
            "esdc.pod_registry.publish.publish_pod_registry", publish_pod_registry
        )
        recovery = runner.invoke(app, ["rebuild-api-analytics"])
        assert recovery.exit_code == 0, recovery.output
        assert '"fresh": true' in recovery.stdout
        with closing(sqlite3.connect(get_esdc_sqlite_path())) as source:
            assert tuple(source.execute("SELECT * FROM m_pod").fetchall()) == before
            assert (
                tuple(source.execute("SELECT * FROM _pod_identity_ledger").fetchall())
                == ledger
            )
            assert (
                source.execute(
                    "SELECT source_generation FROM pod_registry_state"
                ).fetchone()[0]
                == 1
            )


class TestSchemaCommand:
    def test_schema_help(self, runner):
        result = runner.invoke(app, ["schema", "--help"])

        assert result.exit_code == 0
        assert "--generate" in result.stdout
        assert "--from-excel" in result.stdout
        assert "--output" in result.stdout
        assert "--overwrite" in result.stdout
        assert "--schema-pod" in result.stdout

    def test_schema_pod_copies_template_to_current_directory(
        self, runner, tmp_path, monkeypatch
    ):
        from esdc.pod_registry.contract import load_pod_contract

        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["schema", "--schema-pod"])

        output_path = tmp_path / "pod_template.xlsx"
        assert result.exit_code == 0
        assert output_path.exists()
        assert f"Generated POD workbook template at {output_path}" in result.stdout
        workbook = load_workbook(output_path, data_only=True)
        try:
            assert workbook.sheetnames == [
                "README",
                "m_project",
                "m_pod",
                "pod_project",
                "pod_monitoring",
                "pod_forecast",
                "pod_monitoring_api",
                "pod_forecast_api",
                "__pod_metadata",
            ]
            contract = load_pod_contract()
            for table_name in (
                "m_project",
                "m_pod",
                "pod_project",
                "pod_monitoring",
                "pod_forecast",
            ):
                assert [cell.value for cell in workbook[table_name][1]] == [
                    column.name for column in contract.tables[table_name].columns
                ]
            for table_name in ("pod_monitoring_api", "pod_forecast_api"):
                sheet = workbook[table_name]
                assert sheet.protection.sheet
                assert [cell.value for cell in sheet[1]] == [
                    column.name for column in contract.tables[table_name].columns
                ]
            assert workbook["pod_forecast"].max_row == 1
            metadata = {
                str(row[0]): row[1]
                for row in workbook["__pod_metadata"].iter_rows(
                    min_row=2, values_only=True
                )
                if row[0] is not None
            }
            assert metadata == {
                "contract_id": contract.contract_id,
                "contract_version": contract.contract_version,
                "dictionary_version": contract.dictionary_version,
            }
        finally:
            workbook.close()

    def test_schema_pod_existing_output_fails_without_overwrite(self, runner, tmp_path):
        output_path = tmp_path / "pod_template.xlsx"
        output_path.write_text("keep: true\n", encoding="utf-8")

        result = runner.invoke(
            app,
            ["schema", "--schema-pod", "--output", str(output_path)],
        )

        assert result.exit_code == 1
        assert "Output schema already exists" in result.stdout
        assert output_path.read_text(encoding="utf-8") == "keep: true\n"

    def test_generate_schema_default_output_in_current_directory(
        self, runner, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        excel_path = _write_excel(
            tmp_path / "Data Source.xlsx",
            [{"project_id": "A-1", "volume": 12.5}],
        )

        result = runner.invoke(
            app,
            ["schema", "--generate", "--from-excel", str(excel_path)],
        )

        output_path = tmp_path / "Data Source.schema.yaml"
        assert result.exit_code == 0
        assert output_path.exists()
        assert f"Schema: {output_path}" in result.stdout
        schema = load_schema_from_yaml(output_path)
        assert schema.table_name == "data_source"
        assert schema.sheet_name == "Sheet1"
        assert [column.name for column in schema.columns] == ["project_id", "volume"]
        assert schema.links[0].entity_type == "Project"
        assert schema.links[0].column == "project_id"

    def test_generate_schema_explicit_output(self, runner, tmp_path):
        excel_path = _write_excel(
            tmp_path / "data.xlsx",
            [{"project_id": "A-1", "volume": 12.5}],
        )
        output_path = tmp_path / "schemas" / "data.yaml"

        result = runner.invoke(
            app,
            [
                "schema",
                "--generate",
                "--from-excel",
                str(excel_path),
                "--output",
                str(output_path),
            ],
        )

        assert result.exit_code == 0
        assert output_path.exists()
        schema = load_schema_from_yaml(output_path)
        assert schema.table_name == "data"
        assert schema.columns[1].type == "double"

    def test_generate_schema_detects_standard_links(self, runner, tmp_path):
        excel_path = _write_excel(
            tmp_path / "linked.xlsx",
            [
                {
                    "project_id": "P-1",
                    "field_id": "F-1",
                    "wk_id": "WK-1",
                    "volume": 12.5,
                }
            ],
        )
        output_path = tmp_path / "linked.schema.yaml"

        result = runner.invoke(
            app,
            [
                "schema",
                "--generate",
                "--from-excel",
                str(excel_path),
                "--output",
                str(output_path),
            ],
        )

        assert result.exit_code == 0
        data = yaml.safe_load(output_path.read_text(encoding="utf-8"))
        assert data["links"] == [
            {
                "entity_type": "Project",
                "column": "project_id",
                "target_key": "project_id",
            },
            {"entity_type": "Field", "column": "field_id", "target_key": "field_id"},
            {
                "entity_type": "WorkingArea",
                "column": "wk_id",
                "target_key": "wk_id",
            },
        ]

    def test_generate_schema_omits_links_without_standard_columns(
        self, runner, tmp_path
    ):
        excel_path = _write_excel(
            tmp_path / "plain.xlsx",
            [{"name": "Alpha", "volume": 12.5}],
        )
        output_path = tmp_path / "plain.schema.yaml"

        result = runner.invoke(
            app,
            [
                "schema",
                "--generate",
                "--from-excel",
                str(excel_path),
                "--output",
                str(output_path),
            ],
        )

        assert result.exit_code == 0
        data = yaml.safe_load(output_path.read_text(encoding="utf-8"))
        assert "links" not in data

    def test_generate_schema_existing_output_fails_without_overwrite(
        self, runner, tmp_path
    ):
        excel_path = _write_excel(
            tmp_path / "data.xlsx",
            [{"project_id": "A-1", "volume": 12.5}],
        )
        output_path = tmp_path / "schema.yaml"
        output_path.write_text("keep: true\n", encoding="utf-8")

        result = runner.invoke(
            app,
            [
                "schema",
                "--generate",
                "--from-excel",
                str(excel_path),
                "--output",
                str(output_path),
            ],
        )

        assert result.exit_code == 1
        assert "Output schema already exists" in result.stdout
        assert output_path.read_text(encoding="utf-8") == "keep: true\n"

    def test_generate_schema_existing_output_overwrites(self, runner, tmp_path):
        excel_path = _write_excel(
            tmp_path / "data.xlsx",
            [{"project_id": "A-1", "volume": 12.5}],
        )
        output_path = tmp_path / "schema.yaml"
        output_path.write_text("old: true\n", encoding="utf-8")

        result = runner.invoke(
            app,
            [
                "schema",
                "--generate",
                "--from-excel",
                str(excel_path),
                "--output",
                str(output_path),
                "--overwrite",
            ],
        )

        assert result.exit_code == 0
        schema = load_schema_from_yaml(output_path)
        assert schema.table_name == "data"
        assert schema.columns[0].description == "TODO: describe project_id"

    def test_generate_schema_infers_supported_types(self, runner, tmp_path):
        excel_path = tmp_path / "typed.xlsx"
        df = pd.DataFrame(
            {
                "name": ["Alpha", "Beta"],
                "count": [1, 2],
                "volume": [12.5, 20.25],
                "active": [True, False],
                "report_date": [
                    pd.Timestamp("2024-01-01"),
                    pd.Timestamp("2024-01-02"),
                ],
                "updated_at": [
                    pd.Timestamp("2024-01-01 12:30"),
                    pd.Timestamp("2024-01-02 13:45"),
                ],
            }
        )
        df.to_excel(excel_path, index=False, sheet_name="Sheet1")
        output_path = tmp_path / "typed.schema.yaml"

        result = runner.invoke(
            app,
            [
                "schema",
                "--generate",
                "--from-excel",
                str(excel_path),
                "--output",
                str(output_path),
            ],
        )

        assert result.exit_code == 0
        schema = load_schema_from_yaml(output_path)
        types = {column.name: column.type for column in schema.columns}
        assert types == {
            "name": "string",
            "count": "integer",
            "volume": "double",
            "active": "boolean",
            "report_date": "date",
            "updated_at": "datetime",
        }


class TestLoadPodRegistryCommand:
    def test_load_pod_registry_uses_shared_master_only_pipeline(
        self, runner, monkeypatch, tmp_path
    ):
        from esdc.pod_registry.models import ImportResult, PreparedPodBatch

        called = {}
        xlsx = tmp_path / "pod.xlsx"
        xlsx.write_bytes(b"stub")
        batch = PreparedPodBatch(
            source="xlsx",
            contract_id="esdc.pod",
            contract_version=1,
            dictionary_version="0.4.0",
            headers={},
            rows={"m_pod": ({}, {})},
            profile="master-only",
            source_path=xlsx,
        )

        def fake_prepare(path, source, *, profile, approved_crosswalk=None):
            assert approved_crosswalk is None
            called["prepare"] = (path, source, profile)
            return batch

        def fake_apply(prepared, *, sqlite_path, duck_path):
            called["apply"] = (prepared, sqlite_path, duck_path)
            return ImportResult({"m_pod": 2}, 7, True), None

        monkeypatch.setattr("esdc.esdc.prepare_pod_import", fake_prepare)
        monkeypatch.setattr("esdc.esdc.load_prepared_pod_batch", fake_apply)

        result = runner.invoke(
            app, ["load", "--from-excel", str(xlsx), "--pod-registry"]
        )

        assert result.exit_code == 0
        assert called["prepare"] == (xlsx, "xlsx", "master-only")
        assert called["apply"][0] is batch
        assert "m_pod: 2 rows" in result.stdout
        assert "generation 7" in result.stdout

    def test_load_sqlite_defaults_to_full_profile(self, runner, monkeypatch, tmp_path):
        from esdc.pod_registry.models import ImportResult, PreparedPodBatch

        source = tmp_path / "source.sqlite"
        source.write_bytes(b"sqlite source stub")
        called = {}
        tables = (
            "m_project",
            "m_pod",
            "pod_project",
            "pod_monitoring",
            "pod_forecast",
        )
        batch = PreparedPodBatch(
            source="sqlite",
            contract_id="esdc.pod",
            contract_version=1,
            dictionary_version="0.4.0",
            headers={},
            rows=dict.fromkeys(tables, ()),
            profile="full",
            source_path=source,
        )

        def fake_prepare(path, source_kind, *, profile, approved_crosswalk=None):
            assert approved_crosswalk is None
            called["prepare"] = (path, source_kind, profile)
            return batch

        def fake_apply(prepared, *, sqlite_path, duck_path):
            called["apply"] = (prepared, sqlite_path, duck_path)
            return ImportResult({}, 3, True), None

        monkeypatch.setattr("esdc.esdc.prepare_pod_import", fake_prepare)
        monkeypatch.setattr("esdc.esdc.load_prepared_pod_batch", fake_apply)
        result = runner.invoke(app, ["load", "--from-sqlite", str(source)])

        assert result.exit_code == 0
        assert called["prepare"] == (source, "sqlite", "full")
        assert called["apply"][0] is batch
        assert "SQLite POD source committed at generation 3." in result.stdout

    def test_load_rejects_both_source_options(self, runner, tmp_path):
        excel = _write_excel(tmp_path / "input.xlsx", [{"project_id": "P-1"}])
        source = tmp_path / "source.sqlite"
        source.write_bytes(b"sqlite source stub")

        result = runner.invoke(
            app,
            [
                "load",
                "--from-excel",
                str(excel),
                "--from-sqlite",
                str(source),
                "--schema-pod",
            ],
        )

        assert result.exit_code == 1
        assert "specify exactly one of --from-excel or --from-sqlite" in result.stdout

    def test_load_requires_exactly_one_mode(self, runner, tmp_path):
        xlsx = tmp_path / "pod.xlsx"
        xlsx.write_bytes(b"")

        result = runner.invoke(
            app, ["load", "--from-excel", str(xlsx), "--pod-registry", "--schema-pod"]
        )

        assert result.exit_code == 1
