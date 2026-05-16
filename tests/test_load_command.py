from __future__ import annotations

import duckdb
import pandas as pd
import pytest
import yaml
from openpyxl import load_workbook

from esdc.chat.tools import knowledge_traversal
from esdc.configs import Config
from esdc.esdc import app
from esdc.loaders import (
    POD_MONITORING_SHEET_NAME,
    POD_PLAN_SHEET_NAME,
    POD_PROJECT_SHEET_NAME,
    POD_SCHEMA_PATH,
    LoadSchemaError,
    build_loaded_schema_graph,
    load_schema_from_yaml,
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


def _write_pod_excel(path):
    schema = load_schema_from_yaml(POD_SCHEMA_PATH)
    metric_columns = [
        column
        for column in schema.columns
        if column.name
        not in {
            "report_date",
            "effective_date",
            "case_type",
            "pod_id",
            "pod_letter_num",
            "pod_name",
            "pod_scope",
            "supercedes_by",
        }
    ]

    plan_row = {
        "pod_id": "POD-001",
        "pod_letter_num": "POD-L-001",
        "pod_name": "POD Alpha",
        "pod_scope": "Development",
        "supercedes_by": "",
        "report_date": pd.Timestamp("2024-01-01"),
        "effective_date": pd.Timestamp("2024-12-31"),
    }
    monitoring_row = {
        "pod_id": "POD-001",
        "case_type": "outlook",
        "report_date": pd.Timestamp("2024-04-01"),
        "effective_date": pd.Timestamp("2024-03-31"),
    }
    for column in metric_columns:
        plan_row[column.name] = 1.0
        monitoring_row[column.name] = 2.0

    project_row = {"pod_id": "POD-001", "project_id": "PRJ-001"}

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame([plan_row]).to_excel(
            writer, index=False, sheet_name=POD_PLAN_SHEET_NAME, startrow=1
        )
        pd.DataFrame([project_row]).to_excel(
            writer, index=False, sheet_name=POD_PROJECT_SHEET_NAME, startrow=1
        )
        pd.DataFrame([monitoring_row]).to_excel(
            writer, index=False, sheet_name=POD_MONITORING_SHEET_NAME, startrow=1
        )
    workbook = load_workbook(path)
    try:
        sheet_descriptions = {
            POD_PLAN_SHEET_NAME: "Baseline POD plan data.",
            POD_PROJECT_SHEET_NAME: "Many-to-many POD to project mapping.",
            POD_MONITORING_SHEET_NAME: "POD outlook and actual monitoring data.",
        }
        for sheet_name, description in sheet_descriptions.items():
            ws = workbook[sheet_name]
            ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ws.max_column)
            ws.cell(1, 1).value = description
    finally:
        workbook.save(path)
        workbook.close()
    return path


class TestLoadSchemaParser:
    def test_schema_valid(self, tmp_path):
        schema = load_schema_from_yaml(_write_schema(tmp_path / "schema.yaml"))

        assert schema.table_name == "sample_table"
        assert schema.sheet_name == "Sheet1"
        assert schema.columns[0].name == "project_id"
        assert schema.columns[1].duckdb_type == "DOUBLE"

    def test_pod_schema_template_loads(self):
        schema = load_schema_from_yaml(POD_SCHEMA_PATH)

        assert schema.table_name == "pod_plan"
        assert schema.sheet_name == "POD"
        assert schema.links == ()
        assert any(column.name == "pod_name" for column in schema.columns)
        assert any(column.name == "supercedes_by" for column in schema.columns)
        assert any("ProducingLicense" in column.maps_to for column in schema.columns)

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
        assert "specify exactly one of --schema or --schema-pod" in result.stdout

    def test_load_excel_creates_table_and_metadata(self, runner, isolated_config, tmp_path):
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
            count = conn.execute("SELECT COUNT(*) FROM sample_table").fetchone()[0]
            metadata = conn.execute(
                "SELECT description FROM _loaded_table_schemas WHERE table_name = ?",
                ["sample_table"],
            ).fetchone()
        finally:
            conn.close()
        assert count == 2
        assert metadata == ("Sample loaded table",)

    def test_load_pod_schema_builtin(self, runner, isolated_config, tmp_path):
        excel_path = _write_pod_excel(tmp_path / "pod.xlsx")

        result = runner.invoke(
            app,
            ["load", "--from-excel", str(excel_path), "--schema-pod"],
        )

        assert result.exit_code == 0
        assert "Loaded 1 rows x 40 columns into 'pod_plan'" in result.stdout
        assert "Loaded 1 rows x 2 columns into 'pod_project'" in result.stdout
        assert "Loaded 1 rows x 37 columns into 'pod_monitoring'" in result.stdout
        conn = duckdb.connect(str(Config.get_db_file()))
        try:
            plan_row = conn.execute(
                "SELECT pod_name, pod_id FROM pod_plan"
            ).fetchone()
            project_row = conn.execute(
                "SELECT pod_id, project_id FROM pod_project"
            ).fetchone()
            monitoring_row = conn.execute(
                "SELECT pod_id, case_type, effective_date FROM pod_monitoring"
            ).fetchone()
            metadata = conn.execute(
                "SELECT description FROM _loaded_table_schemas WHERE table_name = ? ORDER BY table_name",
                ["pod_plan"],
            ).fetchone()
        finally:
            conn.close()
        assert plan_row == ("POD Alpha", "POD-001")
        assert project_row == ("POD-001", "PRJ-001")
        assert monitoring_row[0] == "POD-001"
        assert monitoring_row[1] == "outlook"
        assert "Baseline POD plan data." in metadata[0]

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


def _make_pod_rows(
    plan_pod_id: str = "POD-001",
    project_pod_id: str = "POD-001",
    monitoring_pod_id: str = "POD-001",
    project_id: str = "PRJ-001",
):
    schema = load_schema_from_yaml(POD_SCHEMA_PATH)
    metric_columns = [
        column
        for column in schema.columns
        if column.name
        not in {
            "report_date",
            "effective_date",
            "case_type",
            "pod_id",
            "pod_letter_num",
            "pod_name",
            "pod_scope",
            "supercedes_by",
        }
    ]
    plan_row = {
        "pod_id": plan_pod_id,
        "pod_letter_num": "POD-L-001",
        "pod_name": "POD Alpha",
        "pod_scope": "Development",
        "supercedes_by": "",
        "report_date": pd.Timestamp("2024-01-01"),
        "effective_date": pd.Timestamp("2024-12-31"),
    }
    monitoring_row = {
        "pod_id": monitoring_pod_id,
        "case_type": "outlook",
        "report_date": pd.Timestamp("2024-04-01"),
        "effective_date": pd.Timestamp("2024-03-31"),
    }
    for column in metric_columns:
        plan_row[column.name] = 1.0
        monitoring_row[column.name] = 2.0
    project_row = {"pod_id": project_pod_id, "project_id": project_id}
    return plan_row, project_row, monitoring_row


def _write_pod_excel_with_data(path, plan_row, project_row, monitoring_row):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame([plan_row]).to_excel(
            writer, index=False, sheet_name=POD_PLAN_SHEET_NAME, startrow=1
        )
        pd.DataFrame([project_row]).to_excel(
            writer, index=False, sheet_name=POD_PROJECT_SHEET_NAME, startrow=1
        )
        pd.DataFrame([monitoring_row]).to_excel(
            writer, index=False, sheet_name=POD_MONITORING_SHEET_NAME, startrow=1
        )
    workbook = load_workbook(path)
    try:
        descriptions = {
            POD_PLAN_SHEET_NAME: "Baseline POD plan data.",
            POD_PROJECT_SHEET_NAME: "Many-to-many POD to project mapping.",
            POD_MONITORING_SHEET_NAME: "POD outlook and actual monitoring data.",
        }
        for sheet_name, description in descriptions.items():
            ws = workbook[sheet_name]
            ws.merge_cells(
                start_row=1, start_column=1, end_row=1, end_column=ws.max_column
            )
            ws.cell(1, 1).value = description
    finally:
        workbook.save(path)
        workbook.close()
    return path


class TestPodLoadValidation:
    def test_load_pod_views_exist(self, runner, isolated_config, tmp_path):
        Config.init_config()
        Config.get_db_dir().mkdir(parents=True, exist_ok=True)
        conn = duckdb.connect(str(Config.get_db_file()))
        conn.execute(
            "CREATE TABLE project_resources "
            "(project_id TEXT, report_date TEXT, project_name TEXT, "
            "project_stage TEXT, project_class TEXT, field_name TEXT, wk_name TEXT)"
        )
        conn.execute(
            "INSERT INTO project_resources VALUES "
            "('PRJ-001', '2024-06-01', 'Test Project', "
            "'DEVELOPMENT', 'CLASS A', 'Field Alpha', 'WK Alpha')"
        )
        conn.close()

        excel_path = _write_pod_excel(tmp_path / "pod.xlsx")
        result = runner.invoke(
            app, ["load", "--from-excel", str(excel_path), "--schema-pod"]
        )
        assert result.exit_code == 0

        conn = duckdb.connect(str(Config.get_db_file()))
        try:
            views = conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_type = 'VIEW'"
            ).fetchall()
            view_names = {v[0] for v in views}
            assert "pod_economics" in view_names
            assert "pod_project_economics" in view_names

            eco_rows = conn.execute(
                "SELECT case_type, pod_id, lifting_oil FROM pod_economics ORDER BY case_type"
            ).fetchall()
            assert len(eco_rows) == 2
            assert eco_rows[0][0] == "outlook"
            assert eco_rows[0][2] == 2.0
            assert eco_rows[1][0] == "plan"
            assert eco_rows[1][2] == 1.0

            plan_letter = conn.execute(
                "SELECT pod_letter_num FROM pod_economics WHERE case_type = 'plan'"
            ).fetchone()[0]
            mon_letter = conn.execute(
                "SELECT pod_letter_num FROM pod_economics WHERE case_type = 'outlook'"
            ).fetchone()[0]
            assert plan_letter == "POD-L-001"
            assert mon_letter is None

            proj_eco = conn.execute(
                "SELECT pod_id, project_id, case_type FROM pod_project_economics ORDER BY case_type"
            ).fetchall()
            assert len(proj_eco) == 2
            assert proj_eco[0][1] == "PRJ-001"
            assert proj_eco[1][1] == "PRJ-001"
        finally:
            conn.close()

    def test_load_pod_broken_project_pod_id_fails(self, runner, isolated_config, tmp_path):
        plan_row, _, monitoring_row = _make_pod_rows(
            plan_pod_id="POD-001",
            monitoring_pod_id="POD-001",
        )
        project_row = {"pod_id": "POD-999", "project_id": "PRJ-001"}
        excel_path = _write_pod_excel_with_data(
            tmp_path / "pod.xlsx", plan_row, project_row, monitoring_row
        )
        result = runner.invoke(
            app, ["load", "--from-excel", str(excel_path), "--schema-pod"]
        )
        assert result.exit_code == 1
        assert "pod_project references pod_id values not found in pod_plan" in result.stdout

    def test_load_pod_broken_monitoring_pod_id_fails(self, runner, isolated_config, tmp_path):
        plan_row, project_row, _ = _make_pod_rows(
            plan_pod_id="POD-001",
            project_pod_id="POD-001",
        )
        schema = load_schema_from_yaml(POD_SCHEMA_PATH)
        metric_columns = [
            col
            for col in schema.columns
            if col.name
            not in {
                "report_date",
                "effective_date",
                "case_type",
                "pod_id",
                "pod_letter_num",
                "pod_name",
                "pod_scope",
                "supercedes_by",
            }
        ]
        monitoring_row = {
            "pod_id": "POD-999",
            "case_type": "outlook",
            "report_date": pd.Timestamp("2024-04-01"),
            "effective_date": pd.Timestamp("2024-03-31"),
        }
        for col in metric_columns:
            monitoring_row[col.name] = 2.0
        excel_path = _write_pod_excel_with_data(
            tmp_path / "pod.xlsx", plan_row, project_row, monitoring_row
        )
        result = runner.invoke(
            app, ["load", "--from-excel", str(excel_path), "--schema-pod"]
        )
        assert result.exit_code == 1
        assert "pod_monitoring references pod_id values not found in pod_plan" in result.stdout

    def test_load_pod_unknown_project_id_warns(self, runner, isolated_config, tmp_path):
        Config.init_config()
        Config.get_db_dir().mkdir(parents=True, exist_ok=True)
        conn = duckdb.connect(str(Config.get_db_file()))
        conn.execute(
            "CREATE TABLE project_resources "
            "(project_id TEXT, report_date TEXT, project_name TEXT, "
            "project_stage TEXT, project_class TEXT, field_name TEXT, wk_name TEXT)"
        )
        conn.execute(
            "INSERT INTO project_resources VALUES "
            "('KNOWN-001', '2024-06-01', '', '', '', '', '')"
        )
        conn.close()

        excel_path = _write_pod_excel(tmp_path / "pod.xlsx")
        result = runner.invoke(
            app, ["load", "--from-excel", str(excel_path), "--schema-pod"]
        )
        assert result.exit_code == 0
        assert "Warning" in result.stdout
        assert "project_id" in result.stdout
        assert "did not match" in result.stdout


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
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["schema", "--schema-pod"])

        output_path = tmp_path / "pod_template.xlsx"
        assert result.exit_code == 0
        assert output_path.exists()
        assert f"Generated POD workbook template at {output_path}" in result.stdout
        workbook = load_workbook(output_path)
        try:
            assert workbook.sheetnames == [
                POD_PLAN_SHEET_NAME,
                POD_PROJECT_SHEET_NAME,
                POD_MONITORING_SHEET_NAME,
            ]
            plan_headers = [cell.value for cell in workbook[POD_PLAN_SHEET_NAME][2]]
            project_headers = [cell.value for cell in workbook[POD_PROJECT_SHEET_NAME][2]]
            monitoring_headers = [
                cell.value for cell in workbook[POD_MONITORING_SHEET_NAME][2]
            ]
        finally:
            workbook.close()
        assert plan_headers[:4] == ["pod_id", "pod_letter_num", "pod_name", "pod_scope"]
        assert project_headers == ["pod_id", "project_id"]
        assert monitoring_headers[:4] == [
            "pod_id",
            "case_type",
            "report_date",
            "effective_date",
        ]

        assert "pod_plan_ids" in workbook.defined_names
        plan_sheet = workbook[POD_PLAN_SHEET_NAME]
        project_sheet = workbook[POD_PROJECT_SHEET_NAME]
        monitoring_sheet = workbook[POD_MONITORING_SHEET_NAME]
        assert len(plan_sheet.data_validations.dataValidation) == 0
        assert len(project_sheet.data_validations.dataValidation) == 1
        assert len(monitoring_sheet.data_validations.dataValidation) == 2
        proj_dv = project_sheet.data_validations.dataValidation[0]
        mon_dv_case = monitoring_sheet.data_validations.dataValidation[0]
        mon_dv_pod = monitoring_sheet.data_validations.dataValidation[1]
        assert "pod_plan_ids" in proj_dv.formula1
        assert proj_dv.sqref == "A3:A1048576"
        assert "pod_plan_ids" in mon_dv_pod.formula1
        assert mon_dv_pod.sqref == "A3:A1048576"
        assert "outlook" in mon_dv_case.formula1
        assert mon_dv_case.sqref == "B3:B1048576"

    def test_schema_pod_existing_output_fails_without_overwrite(
        self, runner, tmp_path
    ):
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
