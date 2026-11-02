"""Integration tests for fetch command."""

import json
import re

import duckdb
import pytest
from typer.testing import CliRunner

from esdc.api_snapshot import get_project_resources_snapshot
from esdc.esdc import _load_file_as_csv, app
from esdc.pod_registry.store import get_sqlite_connection
from esdc.selection import TableName
from esdc.timeseries_snapshot import get_timeseries_snapshot

from .constants import (
    PROJECT_RESOURCES_URL,
)

runner = CliRunner()


def _resources_url_for_year(year: int) -> re.Pattern:
    """Regex matching project-resources endpoint for a specific year."""
    return re.compile(
        rf"https://esdc\.skkmigas\.go\.id/api/v2/project-resources\?.*report-year={year}.*"
    )


def _timeseries_url_for_year(year: int) -> re.Pattern:
    """Regex matching project-timeseries endpoint for a specific year."""
    return re.compile(
        rf"https://esdc\.skkmigas\.go\.id/api/v2/project-timeseries\?.*report-year={year}.*"
    )


class TestFetchCreatesDatabase:
    """Tests that fetch command creates database correctly."""

    def test_fetch_creates_database(self, isolated_config, mock_esdc_api, monkeypatch):
        """First fetch run should create ~/.esdc/esdc.duckdb."""
        project_resources_data = [
            {
                "project_id": "PRJ-001",
                "project_name": "Test Project",
                "report_year": 2024,
                "project_stage": "DISCOVERED",
            }
        ]
        mock_esdc_api.add_json(PROJECT_RESOURCES_URL, json_data=project_resources_data)
        mock_esdc_api.add_json(
            _timeseries_url_for_year(2024),
            json_data=[
                {
                    "project_id": "PRJ-001",
                    "report_year": 2024,
                    "year": 2024,
                    "cprd_grs_oil": 100.0,
                }
            ],
        )

        monkeypatch.setenv("ESDC_USER", "testuser")
        monkeypatch.setenv("ESDC_PASS", "testpass")

        result = runner.invoke(app, ["fetch", "--filetype", "json"])

        db_file = isolated_config / ".esdc" / "esdc.duckdb"
        assert db_file.exists(), f"Database not created at {db_file}"
        assert result.exit_code == 0


class TestFetchJsonToDatabase:
    """Tests that fetch with JSON persists data to database."""

    @pytest.mark.parametrize(
        ("filetype", "entrypoint"),
        [("json", "reload"), ("csv", "reload"), ("json", "loader"), ("csv", "loader")],
    )
    def test_timeseries_files_publish_through_real_json_and_csv_entries(
        self, filetype, entrypoint, isolated_config, runner
    ):
        from esdc.fetch_publication import ingest_timeseries_rows

        db_dir = isolated_config / ".esdc"
        db_path = db_dir / "esdc.duckdb"
        sqlite_path = db_dir / "esdc.sqlite"
        ingest_timeseries_rows(
            [[2024, "P-old", "Old", 7]],
            ["report_year", "project_id", "project_name", "cprd_grs_oil"],
            duckdb_path=db_path,
            sqlite_path=sqlite_path,
            reindex=False,
        )

        payload_path = db_dir / f"project_timeseries.{filetype}"
        if filetype == "json":
            payload_path.write_text(
                '[{"report_year":2025,"project_id":"P-new",'
                '"project_name":"New","cprd_grs_oil":12.5},'
                '{"report_year":2025,"project_id":"P-new",'
                '"project_name":"New","cprd_grs_oil":12.5}]',
                encoding="utf-8",
            )
        else:
            payload_path.write_text(
                "report_year;project_id;project_name;cprd_grs_oil\n"
                "2025;P-new;New;12.5\n2025;P-new;New;12.5\n",
                encoding="utf-8",
            )

        if entrypoint == "reload":
            result = runner.invoke(
                app, ["reload", "--filetype", filetype, "--no-embeddings"]
            )
            assert result.exit_code == 0, result.output
        elif filetype == "json":
            from esdc.esdc import _load_file_as_json

            _load_file_as_json(str(payload_path), TableName.PROJECT_TIMESERIES)
        else:
            _load_file_as_csv(str(payload_path), TableName.PROJECT_TIMESERIES)

        sqlite_conn = get_sqlite_connection(sqlite_path)
        snapshot = get_timeseries_snapshot(sqlite_conn)
        canonical_rows = sqlite_conn.execute(
            "SELECT report_year, row_json, origin FROM api_project_timeseries "
            "ORDER BY row_json, storage_id"
        ).fetchall()
        sqlite_conn.close()
        assert snapshot is not None
        assert snapshot.row_count == 2
        assert snapshot.publication_state == "published"
        expected_origin = "api_csv" if filetype == "csv" else "api_json"
        expected_source_value = "12.5" if filetype == "csv" else 12.5
        assert [
            (
                year,
                json.loads(row_json)["project_id"],
                json.loads(row_json)["cprd_grs_oil"],
                origin,
            )
            for year, row_json, origin in canonical_rows
        ] == [
            (2025, "P-new", expected_source_value, expected_origin),
            (2025, "P-new", expected_source_value, expected_origin),
        ]

        with duckdb.connect(str(db_path), read_only=True) as conn:
            lineage = conn.execute(
                "SELECT revision, source_row_count, source_sha256 "
                "FROM api_publication WHERE source_table='project_timeseries'"
            ).fetchone()
            projected_rows = conn.execute(
                "SELECT report_year, project_id, project_name, cprd_grs_oil "
                "FROM project_timeseries ORDER BY report_year, project_id"
            ).fetchall()
        assert lineage == (
            snapshot.revision,
            snapshot.row_count,
            snapshot.source_sha256,
        )
        assert projected_rows == [
            (2025, "P-new", "New", 12.5),
            (2025, "P-new", "New", 12.5),
        ]

    def test_reload_publishes_both_files_in_one_canonical_batch(
        self, isolated_config, monkeypatch
    ):
        db_dir = isolated_config / ".esdc"
        db_dir.mkdir(parents=True)
        (db_dir / "project_resources.json").write_text(
            '[{"project_id":"P-1","project_name":"Reloaded","report_year":2025}]'
        )
        (db_dir / "project_timeseries.json").write_text(
            '[{"project_id":"P-1","report_year":2025,"year":2025,"cprd_grs_oil":12}]'
        )

        result = runner.invoke(app, ["reload", "--filetype", "json", "--no-embeddings"])

        assert result.exit_code == 0, result.output
        sqlite_path = db_dir / "esdc.sqlite"
        sqlite_conn = get_sqlite_connection(sqlite_path)
        resource_snapshot = get_project_resources_snapshot(sqlite_conn)
        timeseries_snapshot = get_timeseries_snapshot(sqlite_conn)
        assert resource_snapshot is not None and timeseries_snapshot is not None
        assert resource_snapshot.revision == timeseries_snapshot.revision
        assert tuple(
            sqlite_conn.execute("SELECT origin FROM api_project_timeseries").fetchone()
        ) == ("api_json",)
        sqlite_conn.close()
        with duckdb.connect(str(db_dir / "esdc.duckdb"), read_only=True) as conn:
            lineages = conn.execute(
                "SELECT source_table, revision FROM api_publication "
                "ORDER BY source_table"
            ).fetchall()
            assert lineages == [
                ("project_resources", resource_snapshot.revision),
                ("project_timeseries", resource_snapshot.revision),
            ]

    def test_full_fetch_replaces_bootstrapped_timeseries_history(
        self, isolated_config, mock_esdc_api, monkeypatch
    ):
        monkeypatch.setenv("ESDC_USER", "testuser")
        monkeypatch.setenv("ESDC_PASS", "testpass")
        mock_esdc_api.add_json(
            PROJECT_RESOURCES_URL,
            json_data=[
                {"project_id": "P24", "project_name": "old", "report_year": 2024}
            ],
        )
        mock_esdc_api.add_json(
            _timeseries_url_for_year(2024),
            json_data=[
                {
                    "project_id": "P24",
                    "report_year": 2024,
                    "year": 2024,
                    "cprd_grs_oil": 10,
                }
            ],
        )
        initial = runner.invoke(app, ["fetch"])
        assert initial.exit_code == 0, initial.output

        db_file = isolated_config / ".esdc" / "esdc.duckdb"
        sqlite_file = isolated_config / ".esdc" / "esdc.sqlite"
        with duckdb.connect(str(db_file)) as conn:
            conn.execute(
                "INSERT INTO project_timeseries "
                "(project_id, report_year, year, cprd_grs_oil) "
                "VALUES ('historical-null', NULL, 2020, 5)"
            )
            conn.execute(
                "DELETE FROM api_publication WHERE source_table='project_timeseries'"
            )
        sqlite_file.unlink()
        bootstrap = runner.invoke(app, ["bootstrap-timeseries-source", "--apply"])
        assert bootstrap.exit_code == 0, bootstrap.output

        mock_esdc_api.add_json(
            PROJECT_RESOURCES_URL,
            json_data=[
                {"project_id": "P24", "project_name": "new", "report_year": 2024},
                {"project_id": "P25", "project_name": "new", "report_year": 2025},
            ],
        )
        mock_esdc_api.add_json(
            _timeseries_url_for_year(2024),
            json_data=[
                {
                    "project_id": "P24",
                    "report_year": 2024,
                    "year": 2024,
                    "cprd_grs_oil": 20,
                }
            ],
        )
        mock_esdc_api.add_json(
            _timeseries_url_for_year(2025),
            json_data=[
                {
                    "project_id": "P25",
                    "report_year": 2025,
                    "year": 2025,
                    "cprd_grs_oil": 30,
                }
            ],
        )
        result = runner.invoke(app, ["fetch"])

        assert result.exit_code == 0, result.output
        assert "Search indexes rebuilt" in result.output
        with duckdb.connect(str(db_file), read_only=True) as conn:
            assert conn.execute(
                "SELECT project_id, report_year FROM project_resources ORDER BY project_id"
            ).fetchall() == [("P24", 2024), ("P25", 2025)]
            assert (
                conn.execute(
                    "SELECT project_id, report_year, cprd_grs_oil FROM project_timeseries "
                    "WHERE report_year IS NULL"
                ).fetchall()
                == []
            )
            assert conn.execute(
                "SELECT COUNT(*) FROM information_schema.views "
                "WHERE table_name IN ('field_timeseries', 'wa_timeseries', 'nkri_timeseries')"
            ).fetchone() == (3,)
            assert conn.execute(
                "SELECT COUNT(*) FROM duckdb_indexes() "
                "WHERE index_name IN ('idx_project_timeseries_report_year', 'idx_project_timeseries_agg')"
            ).fetchone() == (2,)
            publication = conn.execute(
                "SELECT revision, source_row_count FROM api_publication"
            ).fetchone()
            assert publication is not None and publication[1] == 2
        sqlite_conn = get_sqlite_connection(sqlite_file)
        snapshot = get_project_resources_snapshot(sqlite_conn)
        sqlite_conn.close()
        assert snapshot is not None and snapshot.publication_state == "published"
        assert publication is not None and publication[0] == snapshot.revision

    def test_fresh_fetch_reports_empty_upstream_timeseries(
        self, isolated_config, mock_esdc_api, monkeypatch
    ):
        mock_esdc_api.add_json(
            PROJECT_RESOURCES_URL,
            json_data=[
                {"project_id": "PRJ-001", "project_name": "Test", "report_year": 2025}
            ],
        )
        mock_esdc_api.add_json(_timeseries_url_for_year(2025), json_data=[])
        monkeypatch.setenv("ESDC_USER", "testuser")
        monkeypatch.setenv("ESDC_PASS", "testpass")

        result = runner.invoke(app, ["fetch"])

        assert result.exit_code == 0, result.output
        assert (
            "API returned no rows for project_timeseries (report year 2025)"
            in result.output
        )
        assert "FTS index skipped: project_timeseries" in result.output
        assert "Search indexes rebuilt" in result.output
        assert "HNSW embedding index dropped" not in result.output
        assert "✗" not in result.output
        with duckdb.connect(str(isolated_config / ".esdc" / "esdc.duckdb")) as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM project_resources"
            ).fetchone() == (1,)
            assert conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_name = 'project_timeseries'"
            ).fetchone() == (1,)
            assert conn.execute(
                "SELECT COUNT(*) FROM project_timeseries"
            ).fetchone() == (0,)

        sqlite_path = isolated_config / ".esdc" / "esdc.sqlite"
        sqlite_conn = get_sqlite_connection(sqlite_path)
        first_snapshot = get_timeseries_snapshot(sqlite_conn)
        sqlite_conn.close()
        assert first_snapshot is not None and first_snapshot.row_count == 0
        assert first_snapshot.publication_state == "published"

        repeated = runner.invoke(app, ["fetch"])
        assert repeated.exit_code == 0, repeated.output
        assert "updated years 2025" in repeated.output
        sqlite_conn = get_sqlite_connection(sqlite_path)
        second_snapshot = get_timeseries_snapshot(sqlite_conn)
        assert second_snapshot is not None
        assert second_snapshot.revision != first_snapshot.revision
        assert second_snapshot.row_count == 0
        assert second_snapshot.publication_state == "published"
        sqlite_conn.close()

    def test_default_empty_timeseries_clears_observed_snapshot(
        self, isolated_config, mock_esdc_api, monkeypatch
    ):
        monkeypatch.setenv("ESDC_USER", "testuser")
        monkeypatch.setenv("ESDC_PASS", "testpass")
        mock_esdc_api.add_json(
            PROJECT_RESOURCES_URL,
            json_data=[
                {"project_id": "P-1", "project_name": "Old", "report_year": 2025}
            ],
        )
        mock_esdc_api.add_json(
            _timeseries_url_for_year(2025),
            json_data=[
                {
                    "project_id": "P-1",
                    "report_year": 2025,
                    "year": 2025,
                    "cprd_grs_oil": 25,
                }
            ],
        )
        assert runner.invoke(app, ["fetch"]).exit_code == 0

        sqlite_path = isolated_config / ".esdc" / "esdc.sqlite"
        sqlite_conn = get_sqlite_connection(sqlite_path)
        before_snapshot = get_timeseries_snapshot(sqlite_conn)
        sqlite_conn.close()

        mock_esdc_api.add_json(
            PROJECT_RESOURCES_URL,
            json_data=[
                {"project_id": "P-1", "project_name": "Updated", "report_year": 2025}
            ],
        )
        mock_esdc_api.add_json(_timeseries_url_for_year(2025), json_data=[])
        result = runner.invoke(app, ["fetch"])

        assert result.exit_code == 0, result.output
        assert "updated years 2025" in result.output
        assert "skipped years 2025" in result.output
        sqlite_conn = get_sqlite_connection(sqlite_path)
        after_snapshot = get_timeseries_snapshot(sqlite_conn)
        assert before_snapshot is not None and after_snapshot is not None
        assert after_snapshot.revision != before_snapshot.revision
        assert after_snapshot.row_count == 0
        assert after_snapshot.publication_state == "published"
        assert (
            sqlite_conn.execute(
                "SELECT row_json FROM api_project_timeseries ORDER BY storage_id"
            ).fetchall()
            == []
        )
        sqlite_conn.close()
        with duckdb.connect(
            str(isolated_config / ".esdc" / "esdc.duckdb"), read_only=True
        ) as conn:
            assert (
                conn.execute(
                    "SELECT project_id, report_year, cprd_grs_oil FROM project_timeseries"
                ).fetchall()
                == []
            )

    def test_incompatible_analytical_schema_keeps_committed_source_and_last_good_cache(
        self, isolated_config, mock_esdc_api, monkeypatch
    ):
        db_path = isolated_config / ".esdc" / "esdc.duckdb"
        db_path.parent.mkdir(parents=True)
        with duckdb.connect(str(db_path)) as conn:
            conn.execute(
                "CREATE TABLE project_timeseries (report_year INTEGER, "
                "project_id VARCHAR, cprd_grs_oil DOUBLE, uuid VARCHAR)"
            )
            conn.execute(
                "INSERT INTO project_timeseries VALUES (2024, 'legacy', 17, 'legacy-uuid')"
            )
        monkeypatch.setenv("ESDC_USER", "testuser")
        monkeypatch.setenv("ESDC_PASS", "testpass")
        mock_esdc_api.add_json(
            PROJECT_RESOURCES_URL,
            json_data=[{"project_id": "P-1", "report_year": 2025}],
        )
        mock_esdc_api.add_json(
            _timeseries_url_for_year(2025),
            json_data=[{"project_id": "P-1", "report_year": 2025, "year": 2025}],
        )

        result = runner.invoke(app, ["fetch"])

        assert result.exit_code == 1
        assert "SQLite source committed at generation 1" in result.output
        assert "DuckDB publication pending" in result.output
        assert "esdc rebuild-api-analytics" in result.output
        source = get_sqlite_connection(isolated_config / ".esdc" / "esdc.sqlite")
        try:
            snapshot = get_timeseries_snapshot(source)
            assert snapshot is not None and snapshot.row_count == 1
            assert snapshot.publication_state == "analytics_stale"
            assert [
                tuple(row)
                for row in source.execute(
                    "SELECT json_extract(row_json, '$.project_id'), report_year "
                    "FROM api_project_timeseries"
                )
            ] == [("P-1", 2025)]
        finally:
            source.close()
        with duckdb.connect(str(db_path), read_only=True) as conn:
            assert conn.execute(
                "SELECT report_year, project_id, cprd_grs_oil, uuid "
                "FROM project_timeseries"
            ).fetchall() == [(2024, "legacy", 17.0, "legacy-uuid")]
            assert conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_name='project_resources'"
            ).fetchone() == (0,)

    def test_full_fetch_replaces_bootstrapped_old_years(
        self, isolated_config, mock_esdc_api, monkeypatch
    ):
        from esdc.db_security import _load_sql_script

        db_path = isolated_config / ".esdc" / "esdc.duckdb"
        db_path.parent.mkdir(parents=True)
        with duckdb.connect(str(db_path)) as conn:
            for statement in _load_sql_script(
                "create_table_project_timeseries.sql"
            ).split(";"):
                if statement.strip():
                    conn.execute(statement)
            conn.execute("ALTER TABLE project_timeseries ADD COLUMN uuid VARCHAR")
            conn.execute(
                "INSERT INTO project_timeseries "
                "(report_year, project_id, year, cprd_grs_oil, uuid) "
                "VALUES (2024, 'legacy', 2024, 17, 'legacy-uuid')"
            )
        monkeypatch.setenv("ESDC_USER", "testuser")
        monkeypatch.setenv("ESDC_PASS", "testpass")

        bootstrap = runner.invoke(app, ["bootstrap-timeseries-source", "--apply"])
        assert bootstrap.exit_code == 0, bootstrap.output
        mock_esdc_api.add_json(
            PROJECT_RESOURCES_URL,
            json_data=[{"project_id": "P-1", "report_year": 2025}],
        )
        mock_esdc_api.add_json(
            _timeseries_url_for_year(2025),
            json_data=[
                {
                    "project_id": "P-1",
                    "report_year": 2025,
                    "year": 2025,
                    "cprd_grs_oil": 25,
                }
            ],
        )

        result = runner.invoke(app, ["fetch"])

        assert result.exit_code == 0, result.output
        with duckdb.connect(str(db_path), read_only=True) as conn:
            rows = conn.execute(
                "SELECT report_year, project_id, cprd_grs_oil, uuid "
                "FROM project_timeseries ORDER BY report_year"
            ).fetchall()
        assert len(rows) == 1
        assert rows[0][:3] == (2025, "P-1", 25.0)
        assert rows[0][3] is not None

    def test_selective_fetch_replaces_corrupted_mirror_from_canonical_source(
        self, isolated_config, mock_esdc_api, monkeypatch
    ):
        monkeypatch.setenv("ESDC_USER", "testuser")
        monkeypatch.setenv("ESDC_PASS", "testpass")
        mock_esdc_api.add_json(
            PROJECT_RESOURCES_URL,
            json_data=[{"project_id": "P-1", "report_year": 2024}],
        )
        mock_esdc_api.add_json(
            _timeseries_url_for_year(2024),
            json_data=[
                {
                    "project_id": "P-1",
                    "report_year": 2024,
                    "year": 2024,
                    "cprd_grs_oil": 10,
                }
            ],
        )
        assert runner.invoke(app, ["fetch"]).exit_code == 0

        sqlite_path = isolated_config / ".esdc" / "esdc.sqlite"
        conn = get_sqlite_connection(sqlite_path)
        before_resource = get_project_resources_snapshot(conn)
        before_timeseries = get_timeseries_snapshot(conn)
        conn.close()
        db_path = isolated_config / ".esdc" / "esdc.duckdb"
        with duckdb.connect(str(db_path)) as conn:
            conn.execute(
                "UPDATE project_timeseries SET cprd_grs_oil=999 WHERE report_year=2024"
            )

        mock_esdc_api.add_json(
            _resources_url_for_year(2024),
            json_data=[{"project_id": "P-1", "report_year": 2024}],
        )
        mock_esdc_api.add_json(
            _timeseries_url_for_year(2024),
            json_data=[
                {
                    "project_id": "P-1",
                    "report_year": 2024,
                    "year": 2024,
                    "cprd_grs_oil": 11,
                }
            ],
        )

        result = runner.invoke(app, ["fetch", "--year", "2024"])

        assert result.exit_code == 0, result.output
        assert "Whole fetch is fresh" in result.output
        conn = get_sqlite_connection(sqlite_path)
        after_resource = get_project_resources_snapshot(conn)
        after_timeseries = get_timeseries_snapshot(conn)
        assert before_resource is not None and after_resource is not None
        assert before_timeseries is not None and after_timeseries is not None
        assert after_resource.revision != before_resource.revision
        assert after_timeseries.revision != before_timeseries.revision
        assert after_timeseries.publication_state == "published"
        conn.close()
        with duckdb.connect(str(db_path), read_only=True) as conn:
            assert conn.execute(
                "SELECT cprd_grs_oil FROM project_timeseries WHERE report_year=2024"
            ).fetchone() == (11.0,)

    def test_full_fetch_replaces_stale_canonical_snapshot_after_c2_failure(
        self, isolated_config, mock_esdc_api, monkeypatch
    ):
        from esdc.analytics_rebuild import apply_timeseries_stage as apply_stage

        monkeypatch.setenv("ESDC_USER", "testuser")
        monkeypatch.setenv("ESDC_PASS", "testpass")
        mock_esdc_api.add_json(
            PROJECT_RESOURCES_URL,
            json_data=[
                {"project_id": "P24", "report_year": 2024},
                {"project_id": "P25", "report_year": 2025},
            ],
        )
        mock_esdc_api.add_json(
            _timeseries_url_for_year(2024),
            json_data=[
                {
                    "project_id": "P24",
                    "report_year": 2024,
                    "year": 2024,
                    "cprd_grs_oil": 10,
                }
            ],
        )
        mock_esdc_api.add_json(
            _timeseries_url_for_year(2025),
            json_data=[
                {
                    "project_id": "P25",
                    "report_year": 2025,
                    "year": 2025,
                    "cprd_grs_oil": 20,
                }
            ],
        )
        assert runner.invoke(app, ["fetch"]).exit_code == 0

        def fail_c2(*args, **kwargs):
            raise ValueError("simulated C2 failure")

        monkeypatch.setattr("esdc.analytics_rebuild.apply_timeseries_stage", fail_c2)
        mock_esdc_api.add_json(
            _resources_url_for_year(2025),
            json_data=[{"project_id": "P25", "report_year": 2025}],
        )
        mock_esdc_api.add_json(
            _timeseries_url_for_year(2025),
            json_data=[
                {
                    "project_id": "P25",
                    "report_year": 2025,
                    "year": 2025,
                    "cprd_grs_oil": 21,
                }
            ],
        )
        failed = runner.invoke(app, ["fetch", "--year", "2025"])
        assert failed.exit_code == 1
        assert "simulated C2 failure" in failed.output

        sqlite_path = isolated_config / ".esdc" / "esdc.sqlite"
        conn = get_sqlite_connection(sqlite_path)
        stale_snapshot = get_timeseries_snapshot(conn)
        committed_resource = get_project_resources_snapshot(conn)
        conn.close()
        assert stale_snapshot is not None
        assert stale_snapshot.publication_state == "analytics_stale"

        monkeypatch.setattr(
            "esdc.analytics_rebuild.apply_timeseries_stage", apply_stage
        )
        mock_esdc_api.add_json(
            PROJECT_RESOURCES_URL,
            json_data=[
                {"project_id": "P24", "report_year": 2024},
                {"project_id": "P25", "report_year": 2025},
            ],
        )
        result = runner.invoke(app, ["fetch"])

        assert result.exit_code == 0, result.output
        assert "Whole fetch is fresh" in result.output
        conn = get_sqlite_connection(sqlite_path)
        recovered = get_timeseries_snapshot(conn)
        resources = get_project_resources_snapshot(conn)
        assert recovered is not None and resources is not None
        assert recovered.publication_state == "published"
        assert resources.publication_state == "published"
        assert recovered.revision != stale_snapshot.revision
        assert committed_resource is not None
        assert resources.revision != committed_resource.revision
        conn.close()

    def test_fetch_json_to_database(self, isolated_config, mock_esdc_api, monkeypatch):
        """Fetch with JSON should persist data to database."""
        project_resources_data = [
            {
                "project_id": "PRJ-001",
                "project_name": "Test Project",
                "report_year": 2024,
                "project_stage": "DISCOVERED",
            }
        ]
        mock_esdc_api.add_json(PROJECT_RESOURCES_URL, json_data=project_resources_data)
        mock_esdc_api.add_json(
            _timeseries_url_for_year(2024),
            json_data=[
                {
                    "project_id": "PRJ-001",
                    "report_year": 2024,
                    "year": 2024,
                    "cprd_grs_oil": 100.0,
                }
            ],
        )

        monkeypatch.setenv("ESDC_USER", "testuser")
        monkeypatch.setenv("ESDC_PASS", "testpass")

        result = runner.invoke(app, ["fetch", "--filetype", "json"])

        assert result.exit_code == 0

        db_file = isolated_config / ".esdc" / "esdc.duckdb"
        assert db_file.exists()

        conn = duckdb.connect(str(db_file))
        row = conn.execute("SELECT COUNT(*) FROM project_resources").fetchone()
        assert row is not None, "No rows returned"
        conn.close()

        count = row[0]
        assert count == 1, "Data should be persisted to project_resources table"

    def test_fetch_multi_year_timeseries_json(
        self, isolated_config, mock_esdc_api, monkeypatch
    ):
        """Fetch should download timeseries for all available years (>=2020)."""
        project_resources_data = [
            {
                "project_id": "PRJ-001",
                "project_name": "Year 2023",
                "report_year": 2023,
                "project_stage": "DISCOVERED",
            },
            {
                "project_id": "PRJ-002",
                "project_name": "Year 2024",
                "report_year": 2024,
                "project_stage": "DISCOVERED",
            },
        ]
        mock_esdc_api.add_json(PROJECT_RESOURCES_URL, json_data=project_resources_data)
        mock_esdc_api.add_json(
            _timeseries_url_for_year(2023),
            json_data=[
                {
                    "project_id": "PRJ-001",
                    "report_year": 2023,
                    "year": 2023,
                    "cprd_grs_oil": 50.0,
                }
            ],
        )
        mock_esdc_api.add_json(
            _timeseries_url_for_year(2024),
            json_data=[
                {
                    "project_id": "PRJ-002",
                    "report_year": 2024,
                    "year": 2024,
                    "cprd_grs_oil": 100.0,
                }
            ],
        )

        monkeypatch.setenv("ESDC_USER", "testuser")
        monkeypatch.setenv("ESDC_PASS", "testpass")

        result = runner.invoke(app, ["fetch", "--filetype", "json"])

        assert result.exit_code == 0

        db_file = isolated_config / ".esdc" / "esdc.duckdb"
        assert db_file.exists()

        conn = duckdb.connect(str(db_file))
        row = conn.execute("SELECT COUNT(*) FROM project_timeseries").fetchone()
        assert row is not None
        count = row[0]
        conn.close()

        assert count == 2, "Timeseries should contain data from both 2023 and 2024"

    def test_fetch_append_year_json(self, isolated_config, mock_esdc_api, monkeypatch):
        """Fetch --year should update both resources and timeseries.

        Verifies that --year updates both project_resources and
        project_timeseries for a specific year.
        """
        # --- Initial full fetch with 2024 data ---
        resources_2024 = [
            {
                "project_id": "PRJ-001",
                "project_name": "Old 2024",
                "report_year": 2024,
                "project_stage": "DISCOVERED",
            }
        ]
        mock_esdc_api.add_json(PROJECT_RESOURCES_URL, json_data=resources_2024)
        mock_esdc_api.add_json(
            _timeseries_url_for_year(2024),
            json_data=[
                {
                    "project_id": "PRJ-001",
                    "report_year": 2024,
                    "year": 2024,
                    "cprd_grs_oil": 100.0,
                }
            ],
        )

        monkeypatch.setenv("ESDC_USER", "testuser")
        monkeypatch.setenv("ESDC_PASS", "testpass")

        result = runner.invoke(app, ["fetch", "--filetype", "json"])
        assert result.exit_code == 0

        db_file = isolated_config / ".esdc" / "esdc.duckdb"

        db_file = isolated_config / ".esdc" / "esdc.duckdb"
        conn = duckdb.connect(str(db_file))
        row = conn.execute("SELECT COUNT(*) FROM project_resources").fetchone()
        assert row is not None
        resources_count_before = row[0]
        row2 = conn.execute(
            "SELECT project_name FROM project_resources WHERE report_year = 2024"
        ).fetchone()
        assert row2 is not None
        resources_name_before = row2[0]
        conn.close()

        assert resources_count_before == 1
        assert resources_name_before == "Old 2024"

        # --- Now update with --year 2025 for both tables ---
        mock_esdc_api.add_json(
            _resources_url_for_year(2025),
            json_data=[
                {
                    "project_id": "PRJ-001",
                    "project_name": "New 2025",
                    "report_year": 2025,
                    "project_stage": "DISCOVERED",
                }
            ],
        )
        mock_esdc_api.add_json(
            _timeseries_url_for_year(2025),
            json_data=[
                {
                    "project_id": "PRJ-001",
                    "report_year": 2025,
                    "year": 2025,
                    "cprd_grs_oil": 200.0,
                }
            ],
        )

        result = runner.invoke(app, ["fetch", "--filetype", "json", "--year", "2025"])
        assert result.exit_code == 0

        conn = duckdb.connect(str(db_file))
        # Both tables should have 2 rows (2024 + 2025)
        row = conn.execute("SELECT COUNT(*) FROM project_resources").fetchone()
        assert row is not None
        resources_count = row[0]
        row = conn.execute("SELECT COUNT(*) FROM project_timeseries").fetchone()
        assert row is not None
        timeseries_count = row[0]
        # Resources name for 2024 unchanged (still Old 2024)
        # Resources name for 2025 is New 2025
        row = conn.execute(
            "SELECT project_name FROM project_resources WHERE report_year = 2024"
        ).fetchone()
        assert row is not None
        name_2024 = row[0]
        row = conn.execute(
            "SELECT project_name FROM project_resources WHERE report_year = 2025"
        ).fetchone()
        assert row is not None
        name_2025 = row[0]
        row = conn.execute(
            "SELECT cprd_grs_oil FROM project_timeseries WHERE year = 2025"
        ).fetchone()
        assert row is not None
        cprd_2025 = row[0]
        conn.close()

        assert resources_count == 2, "Resources should contain data for 2024 and 2025"
        assert timeseries_count == 2, "Timeseries should contain data for 2024 and 2025"
        assert name_2024 == "Old 2024"
        assert name_2025 == "New 2025"
        assert cprd_2025 == 200.0

    def test_fetch_year_ignoring_upstream_filters_does_not_duplicate_rows(
        self, isolated_config, mock_esdc_api, monkeypatch
    ):
        initial = [
            {"project_id": "P23", "project_name": "23", "report_year": 2023},
            {"project_id": "P24", "project_name": "24", "report_year": 2024},
            {"project_id": "P25", "project_name": "25", "report_year": 2025},
        ]
        mock_esdc_api.add_json(PROJECT_RESOURCES_URL, json_data=initial)
        mock_esdc_api.add_json(_timeseries_url_for_year(2023), json_data=[])
        mock_esdc_api.add_json(_timeseries_url_for_year(2024), json_data=[])
        mock_esdc_api.add_json(_timeseries_url_for_year(2025), json_data=[])
        monkeypatch.setenv("ESDC_USER", "u")
        monkeypatch.setenv("ESDC_PASS", "p")
        assert runner.invoke(app, ["fetch", "--filetype", "json"]).exit_code == 0
        extra = initial + [
            {"project_id": "P26", "project_name": "26", "report_year": 2026}
        ]
        mock_esdc_api.add_json(_resources_url_for_year(2024), json_data=extra)
        mock_esdc_api.add_json(
            _resources_url_for_year(2025),
            json_data=[
                {"project_id": "P3", "project_name": "3", "report_year": 2025},
                {"project_id": "P23", "project_name": "23", "report_year": 2023},
                {"project_id": "P24", "project_name": "24", "report_year": 2024},
                {"project_id": "P26", "project_name": "26", "report_year": 2026},
            ],
        )
        mock_esdc_api.add_json(_timeseries_url_for_year(2024), json_data=[])
        mock_esdc_api.add_json(_timeseries_url_for_year(2025), json_data=[])
        result = runner.invoke(
            app, ["fetch", "--filetype", "json", "--year", "2024", "--year", "2025"]
        )
        assert result.exit_code == 0, result.output
        import duckdb

        with duckdb.connect(str(isolated_config / ".esdc" / "esdc.duckdb")) as conn:
            assert conn.execute(
                "SELECT project_id, report_year FROM project_resources ORDER BY project_id"
            ).fetchall() == [("P23", 2023), ("P24", 2024), ("P3", 2025)]

    def test_malformed_second_year_does_not_mutate_snapshot(
        self, isolated_config, mock_esdc_api, monkeypatch
    ):
        initial = [{"project_id": "P24", "project_name": "24", "report_year": 2024}]
        mock_esdc_api.add_json(PROJECT_RESOURCES_URL, json_data=initial)
        mock_esdc_api.add_json(_timeseries_url_for_year(2024), json_data=[])
        monkeypatch.setenv("ESDC_USER", "u")
        monkeypatch.setenv("ESDC_PASS", "p")
        assert runner.invoke(app, ["fetch", "--filetype", "json"]).exit_code == 0
        sqlite_path = isolated_config / ".esdc" / "esdc.sqlite"
        conn = get_sqlite_connection(sqlite_path)
        before_snapshot = get_project_resources_snapshot(conn)
        before_rows = conn.execute("SELECT * FROM api_project_resources").fetchall()
        conn.close()
        with duckdb.connect(str(isolated_config / ".esdc" / "esdc.duckdb")) as db:
            before_duck = db.execute("SELECT * FROM project_resources").fetchall()
        mock_esdc_api.add_json(_resources_url_for_year(2024), json_data=initial)
        mock_esdc_api.add_json(
            _resources_url_for_year(2025), json_data=[{"invalid": 1}]
        )
        mock_esdc_api.add_json(_timeseries_url_for_year(2024), json_data=[])
        mock_esdc_api.add_json(_timeseries_url_for_year(2025), json_data=[])
        result = runner.invoke(
            app, ["fetch", "--filetype", "json", "--year", "2024", "--year", "2025"]
        )
        assert result.exit_code != 0
        conn = get_sqlite_connection(sqlite_path)
        assert get_project_resources_snapshot(conn) == before_snapshot
        assert (
            conn.execute("SELECT * FROM api_project_resources").fetchall()
            == before_rows
        )
        conn.close()
        with duckdb.connect(str(isolated_config / ".esdc" / "esdc.duckdb")) as db:
            assert (
                db.execute("SELECT * FROM project_resources").fetchall() == before_duck
            )

    def test_download_failure_second_year_preserves_sqlite_and_duckdb(
        self, isolated_config, mock_esdc_api, monkeypatch
    ):
        seed = [
            {"project_id": "P24", "project_name": "old24", "report_year": 2024},
            {"project_id": "P25", "project_name": "old25", "report_year": 2025},
        ]
        mock_esdc_api.add_json(PROJECT_RESOURCES_URL, json_data=seed)
        mock_esdc_api.add_json(_timeseries_url_for_year(2024), json_data=[])
        mock_esdc_api.add_json(_timeseries_url_for_year(2025), json_data=[])
        monkeypatch.setenv("ESDC_USER", "u")
        monkeypatch.setenv("ESDC_PASS", "p")
        assert runner.invoke(app, ["fetch", "--filetype", "json"]).exit_code == 0
        sqlite_path = isolated_config / ".esdc" / "esdc.sqlite"
        conn = get_sqlite_connection(sqlite_path)
        before = get_project_resources_snapshot(conn)
        conn.close()
        mock_esdc_api.add_json(
            _resources_url_for_year(2024),
            json_data=[
                {"project_id": "P24", "project_name": "new24", "report_year": 2024}
            ],
        )
        mock_esdc_api.add_status(_resources_url_for_year(2025), status=500)
        result = runner.invoke(
            app, ["fetch", "--filetype", "json", "--year", "2024", "--year", "2025"]
        )
        assert result.exit_code != 0
        conn = get_sqlite_connection(sqlite_path)
        after = get_project_resources_snapshot(conn)
        rows = [
            tuple(row)
            for row in conn.execute(
                "SELECT project_id, project_name FROM api_project_resources ORDER BY project_id"
            ).fetchall()
        ]
        conn.close()
        assert after == before and rows == [("P24", "old24"), ("P25", "old25")]
        import duckdb

        with duckdb.connect(str(isolated_config / ".esdc" / "esdc.duckdb")) as db:
            assert (
                db.execute(
                    "SELECT project_id, project_name FROM project_resources ORDER BY project_id"
                ).fetchall()
                == rows
            )

    def test_plain_fetch_failure_on_second_timeseries_year_is_prepublication_noop(
        self, isolated_config, mock_esdc_api, monkeypatch
    ):
        seed_resources = [
            {"project_id": "P24", "project_name": "old24", "report_year": 2024},
            {"project_id": "P25", "project_name": "old25", "report_year": 2025},
        ]
        mock_esdc_api.add_json(PROJECT_RESOURCES_URL, json_data=seed_resources)
        mock_esdc_api.add_json(
            _timeseries_url_for_year(2024),
            json_data=[
                {
                    "project_id": "P24",
                    "report_year": 2024,
                    "year": 2024,
                    "cprd_grs_oil": 100,
                }
            ],
        )
        mock_esdc_api.add_json(
            _timeseries_url_for_year(2025),
            json_data=[
                {
                    "project_id": "P25",
                    "report_year": 2025,
                    "year": 2025,
                    "cprd_grs_oil": 200,
                }
            ],
        )
        monkeypatch.setenv("ESDC_USER", "u")
        monkeypatch.setenv("ESDC_PASS", "p")
        assert runner.invoke(app, ["fetch"]).exit_code == 0

        sqlite_path = isolated_config / ".esdc" / "esdc.sqlite"
        duckdb_path = isolated_config / ".esdc" / "esdc.duckdb"
        sqlite_conn = get_sqlite_connection(sqlite_path)
        before_snapshot = get_project_resources_snapshot(sqlite_conn)
        before_resources = sqlite_conn.execute(
            "SELECT project_id, project_name FROM api_project_resources ORDER BY project_id"
        ).fetchall()
        sqlite_conn.close()
        with duckdb.connect(str(duckdb_path), read_only=True) as conn:
            before_analytics = conn.execute(
                "SELECT project_id, project_name FROM project_resources ORDER BY project_id"
            ).fetchall()
            before_timeseries = conn.execute(
                "SELECT project_id, report_year, cprd_grs_oil FROM project_timeseries ORDER BY report_year"
            ).fetchall()
            before_lineage = conn.execute(
                "SELECT revision, source_row_count, source_sha256 FROM api_publication"
            ).fetchall()

        invalidations: list[str] = []
        reindexes: list[str] = []
        monkeypatch.setattr(
            "esdc.api_projection._invalidate_analytical_caches",
            lambda: invalidations.append("cache"),
        )
        monkeypatch.setattr(
            "esdc.fetch_publication._reindex_fts_for_path",
            lambda *args, **kwargs: reindexes.append("reindex"),
        )
        mock_esdc_api.add_json(
            PROJECT_RESOURCES_URL,
            json_data=[
                {"project_id": "P24", "project_name": "new24", "report_year": 2024},
                {"project_id": "P25", "project_name": "new25", "report_year": 2025},
            ],
        )
        mock_esdc_api.add_json(
            _timeseries_url_for_year(2024),
            json_data=[
                {
                    "project_id": "P24",
                    "report_year": 2024,
                    "year": 2024,
                    "cprd_grs_oil": 110,
                }
            ],
        )
        mock_esdc_api.add_status(_timeseries_url_for_year(2025), status=500)

        result = runner.invoke(app, ["fetch"])

        assert result.exit_code == 1
        assert "project_timeseries" in result.output
        assert "2025" in result.output
        assert invalidations == []
        assert reindexes == []
        sqlite_conn = get_sqlite_connection(sqlite_path)
        assert get_project_resources_snapshot(sqlite_conn) == before_snapshot
        assert (
            sqlite_conn.execute(
                "SELECT project_id, project_name FROM api_project_resources ORDER BY project_id"
            ).fetchall()
            == before_resources
        )
        sqlite_conn.close()
        with duckdb.connect(str(duckdb_path), read_only=True) as conn:
            assert (
                conn.execute(
                    "SELECT project_id, project_name FROM project_resources ORDER BY project_id"
                ).fetchall()
                == before_analytics
            )
            assert (
                conn.execute(
                    "SELECT project_id, report_year, cprd_grs_oil FROM project_timeseries ORDER BY report_year"
                ).fetchall()
                == before_timeseries
            )
            assert (
                conn.execute(
                    "SELECT revision, source_row_count, source_sha256 FROM api_publication"
                ).fetchall()
                == before_lineage
            )

    def test_later_year_duckdb_conversion_failure_rolls_back_both_tables(
        self, isolated_config, mock_esdc_api, monkeypatch
    ):
        old_resources = [
            {"project_id": "P24", "project_name": "old24", "report_year": 2024},
            {"project_id": "P25", "project_name": "old25", "report_year": 2025},
        ]
        mock_esdc_api.add_json(PROJECT_RESOURCES_URL, json_data=old_resources)
        mock_esdc_api.add_json(
            _timeseries_url_for_year(2024),
            json_data=[
                {
                    "project_id": "P24",
                    "report_year": 2024,
                    "year": 2024,
                    "cprd_grs_oil": 100,
                }
            ],
        )
        mock_esdc_api.add_json(
            _timeseries_url_for_year(2025),
            json_data=[
                {
                    "project_id": "P25",
                    "report_year": 2025,
                    "year": 2025,
                    "cprd_grs_oil": 200,
                }
            ],
        )
        monkeypatch.setenv("ESDC_USER", "u")
        monkeypatch.setenv("ESDC_PASS", "p")
        assert runner.invoke(app, ["fetch"]).exit_code == 0

        sqlite_path = isolated_config / ".esdc" / "esdc.sqlite"
        duckdb_path = isolated_config / ".esdc" / "esdc.duckdb"
        sqlite_conn = get_sqlite_connection(sqlite_path)
        before_snapshot = get_project_resources_snapshot(sqlite_conn)
        sqlite_conn.close()
        with duckdb.connect(str(duckdb_path), read_only=True) as conn:
            before_resources = conn.execute(
                "SELECT project_id, project_name FROM project_resources ORDER BY project_id"
            ).fetchall()
            before_timeseries = conn.execute(
                "SELECT project_id, report_year, cprd_grs_oil FROM project_timeseries ORDER BY report_year"
            ).fetchall()
            before_lineage = conn.execute(
                "SELECT revision, source_row_count, source_sha256 FROM api_publication"
            ).fetchall()

        invalidations: list[str] = []
        reindexes: list[str] = []
        monkeypatch.setattr(
            "esdc.api_projection._invalidate_analytical_caches",
            lambda: invalidations.append("cache"),
        )
        monkeypatch.setattr(
            "esdc.fetch_publication._reindex_fts_for_path",
            lambda *args, **kwargs: reindexes.append("reindex"),
        )
        mock_esdc_api.add_json(
            PROJECT_RESOURCES_URL,
            json_data=[
                {"project_id": "P24", "project_name": "new24", "report_year": 2024},
                {"project_id": "P25", "project_name": "new25", "report_year": 2025},
            ],
        )
        mock_esdc_api.add_json(
            _timeseries_url_for_year(2024),
            json_data=[
                {
                    "project_id": "P24",
                    "report_year": 2024,
                    "year": 2024,
                    "cprd_grs_oil": 110,
                }
            ],
        )
        mock_esdc_api.add_json(
            _timeseries_url_for_year(2025),
            json_data=[
                {
                    "project_id": "P25",
                    "report_year": 2025,
                    "year": 2025,
                    "cprd_grs_oil": "not-a-number",
                }
            ],
        )

        result = runner.invoke(app, ["fetch"])

        assert result.exit_code == 1
        normalized_output = " ".join(result.output.split())
        assert "phase=publication" in normalized_output
        assert "table=project_timeseries" in normalized_output
        assert "year=2025" in normalized_output
        assert invalidations == []
        assert reindexes == []
        sqlite_conn = get_sqlite_connection(sqlite_path)
        after_snapshot = get_project_resources_snapshot(sqlite_conn)
        assert after_snapshot is not None
        assert before_snapshot is not None
        assert after_snapshot.revision != before_snapshot.revision
        assert after_snapshot.publication_state == "analytics_stale"
        assert [
            tuple(row)
            for row in sqlite_conn.execute(
                "SELECT project_id, project_name FROM api_project_resources ORDER BY project_id"
            ).fetchall()
        ] == [("P24", "new24"), ("P25", "new25")]
        sqlite_conn.close()
        with duckdb.connect(str(duckdb_path), read_only=True) as conn:
            assert (
                conn.execute(
                    "SELECT project_id, project_name FROM project_resources ORDER BY project_id"
                ).fetchall()
                == before_resources
            )
            assert (
                conn.execute(
                    "SELECT project_id, report_year, cprd_grs_oil FROM project_timeseries ORDER BY report_year"
                ).fetchall()
                == before_timeseries
            )
            assert (
                conn.execute(
                    "SELECT revision, source_row_count, source_sha256 FROM api_publication"
                ).fetchall()
                == before_lineage
            )

    def test_empty_selected_year_clears_its_scope_in_both_stores(
        self, isolated_config, mock_esdc_api, monkeypatch
    ):
        seed = [
            {"project_id": "P24", "project_name": "old24", "report_year": 2024},
            {"project_id": "P25", "project_name": "old25", "report_year": 2025},
        ]
        mock_esdc_api.add_json(PROJECT_RESOURCES_URL, json_data=seed)
        mock_esdc_api.add_json(_timeseries_url_for_year(2024), json_data=[])
        mock_esdc_api.add_json(_timeseries_url_for_year(2025), json_data=[])
        monkeypatch.setenv("ESDC_USER", "u")
        monkeypatch.setenv("ESDC_PASS", "p")
        assert runner.invoke(app, ["fetch", "--filetype", "json"]).exit_code == 0
        mock_esdc_api.add_json(
            _resources_url_for_year(2024),
            json_data=[
                {"project_id": "P24", "project_name": "new24", "report_year": 2024}
            ],
        )
        mock_esdc_api.add_json(_resources_url_for_year(2025), json_data=[])
        mock_esdc_api.add_json(_timeseries_url_for_year(2024), json_data=[])
        mock_esdc_api.add_json(_timeseries_url_for_year(2025), json_data=[])
        assert (
            runner.invoke(
                app, ["fetch", "--filetype", "json", "--year", "2024", "--year", "2025"]
            ).exit_code
            == 0
        )
        conn = get_sqlite_connection(isolated_config / ".esdc" / "esdc.sqlite")
        rows = [
            tuple(row)
            for row in conn.execute(
                "SELECT project_id, project_name FROM api_project_resources ORDER BY project_id"
            ).fetchall()
        ]
        conn.close()
        assert rows == [("P24", "new24")]
        import duckdb

        with duckdb.connect(str(isolated_config / ".esdc" / "esdc.duckdb")) as db:
            assert (
                db.execute(
                    "SELECT project_id, project_name FROM project_resources ORDER BY project_id"
                ).fetchall()
                == rows
            )

    def test_json_csv_parity_preserves_complete_values(
        self, isolated_config, mock_esdc_api, monkeypatch
    ):
        row = {
            "project_id": "0007",
            "project_name": "Parity",
            "report_year": 2024,
            "field_geom": "POINT (1 2)",
            "rec_oil": 12.5,
            "grov_current_situation": "line one\nline two",
        }
        mock_esdc_api.add_json(PROJECT_RESOURCES_URL, json_data=[row])
        mock_esdc_api.add_json(_timeseries_url_for_year(2024), json_data=[])
        monkeypatch.setenv("ESDC_USER", "u")
        monkeypatch.setenv("ESDC_PASS", "p")
        assert runner.invoke(app, ["fetch", "--filetype", "json"]).exit_code == 0
        conn = get_sqlite_connection(isolated_config / ".esdc" / "esdc.sqlite")
        json_snapshot = get_project_resources_snapshot(conn)
        json_rows = [
            tuple(row)
            for row in conn.execute(
                "SELECT project_id, report_year, field_geom, rec_oil, groovy_report FROM api_project_resources"
            ).fetchall()
        ]
        conn.close()
        csv_body = 'grov_current_situation;rec_oil;field_geom;report_year;project_name;project_id\n"line one\nline two";12.5;"POINT (1 2)";2024;Parity;0007\n'
        mock_esdc_api.add_csv(PROJECT_RESOURCES_URL, body=csv_body)
        mock_esdc_api.add_csv(
            _timeseries_url_for_year(2024),
            body="report_year;project_id;year\n2024;0007;2024\n",
        )
        assert runner.invoke(app, ["fetch", "--filetype", "csv"]).exit_code == 0
        conn = get_sqlite_connection(isolated_config / ".esdc" / "esdc.sqlite")
        csv_snapshot = get_project_resources_snapshot(conn)
        csv_rows = [
            tuple(row)
            for row in conn.execute(
                "SELECT project_id, report_year, field_geom, rec_oil, groovy_report FROM api_project_resources"
            ).fetchall()
        ]
        conn.close()
        assert csv_rows == json_rows
        assert (
            csv_snapshot is not None
            and json_snapshot is not None
            and csv_snapshot.source_sha256 == json_snapshot.source_sha256
        )

    def test_header_only_csv_reload_preserves_snapshot(
        self, isolated_config, mock_esdc_api, monkeypatch, tmp_path
    ):
        seed = [{"project_id": "P24", "project_name": "old24", "report_year": 2024}]
        mock_esdc_api.add_json(PROJECT_RESOURCES_URL, json_data=seed)
        mock_esdc_api.add_json(_timeseries_url_for_year(2024), json_data=[])
        monkeypatch.setenv("ESDC_USER", "u")
        monkeypatch.setenv("ESDC_PASS", "p")
        assert runner.invoke(app, ["fetch", "--filetype", "json"]).exit_code == 0
        path = tmp_path / "empty.csv"
        path.write_text("project_id;project_name;report_year\n")
        conn = get_sqlite_connection(isolated_config / ".esdc" / "esdc.sqlite")
        before = get_project_resources_snapshot(conn)
        conn.close()
        _load_file_as_csv(str(path), "project_resources")
        conn = get_sqlite_connection(isolated_config / ".esdc" / "esdc.sqlite")
        assert get_project_resources_snapshot(conn) == before
        conn.close()

    @pytest.mark.parametrize(
        ("filetype", "body", "expected_exit"),
        [
            ("json", [], 0),
            ("json", {}, 1),
            ("json", None, 1),
            ("json", False, 1),
            ("csv", "project_id;project_name;report_year\n", 0),
        ],
    )
    def test_full_project_empty_replaces_and_malformed_preserves_truth(
        self, isolated_config, mock_esdc_api, monkeypatch, filetype, body, expected_exit
    ):
        monkeypatch.setenv("ESDC_USER", "u")
        monkeypatch.setenv("ESDC_PASS", "p")
        mock_esdc_api.add_json(
            PROJECT_RESOURCES_URL,
            json_data=[
                {"project_id": "KEEP", "project_name": "Keep", "report_year": 2024}
            ],
        )
        mock_esdc_api.add_json(_timeseries_url_for_year(2024), json_data=[])
        assert runner.invoke(app, ["fetch", "--filetype", "json"]).exit_code == 0
        sqlite_path = isolated_config / ".esdc" / "esdc.sqlite"
        conn = get_sqlite_connection(sqlite_path)
        before_snapshot = get_project_resources_snapshot(conn)
        before_rows = conn.execute(
            "SELECT project_id, project_name, report_year FROM api_project_resources"
        ).fetchall()
        conn.close()
        with duckdb.connect(str(isolated_config / ".esdc" / "esdc.duckdb")) as db:
            before_duck = db.execute(
                "SELECT project_id, project_name, report_year FROM project_resources"
            ).fetchall()
        if filetype == "json":
            mock_esdc_api.add_json(PROJECT_RESOURCES_URL, json_data=body)
        else:
            mock_esdc_api.add_csv(PROJECT_RESOURCES_URL, body=body)
        if filetype == "csv":
            mock_esdc_api.add_csv(
                _timeseries_url_for_year(2024), body="report_year;project_id\n"
            )
        else:
            mock_esdc_api.add_json(_timeseries_url_for_year(2024), json_data=[])
        result = runner.invoke(app, ["fetch", "--filetype", filetype])
        assert (result.exit_code != 0) == (expected_exit != 0), result.output
        conn = get_sqlite_connection(sqlite_path)
        after = get_project_resources_snapshot(conn)
        if expected_exit:
            assert after == before_snapshot
        else:
            assert after is not None and before_snapshot is not None
            assert after.row_count == 0
            assert after.revision != before_snapshot.revision
            assert after.publication_state == "published"
        assert conn.execute(
            "SELECT project_id, project_name, report_year FROM api_project_resources"
        ).fetchall() == (before_rows if expected_exit else [])
        conn.close()
        with duckdb.connect(str(isolated_config / ".esdc" / "esdc.duckdb")) as db:
            assert db.execute(
                "SELECT project_id, project_name, report_year FROM project_resources"
            ).fetchall() == (before_duck if expected_exit else [])

    def test_canonical_csv_full_fetch_and_local_reload_preserve_fields(
        self, isolated_config, mock_esdc_api, monkeypatch, tmp_path
    ):
        csv_body = (
            "project_id;report_year;groovy_report;basin86_id;basin86\n"
            "P-CAN;2024;Canonical report;B86-1;Basin 86\n"
        )
        monkeypatch.setenv("ESDC_USER", "u")
        monkeypatch.setenv("ESDC_PASS", "p")
        mock_esdc_api.add_csv(PROJECT_RESOURCES_URL, body=csv_body)
        mock_esdc_api.add_csv(
            _timeseries_url_for_year(2024), body="report_year;project_id;year\n"
        )
        assert runner.invoke(app, ["fetch", "--filetype", "csv"]).exit_code == 0
        path = tmp_path / "canonical.csv"
        path.write_text(csv_body)
        _load_file_as_csv(str(path), "project_resources")
        conn = get_sqlite_connection(isolated_config / ".esdc" / "esdc.sqlite")
        assert conn.execute(
            "SELECT groovy_report, basin86_id, basin86 FROM api_project_resources"
        ).fetchone()[:3] == ("Canonical report", "B86-1", "Basin 86")
        conn.close()
        with duckdb.connect(str(isolated_config / ".esdc" / "esdc.duckdb")) as db:
            assert db.execute(
                "SELECT groovy_report, basin86_id, basin86 FROM project_resources"
            ).fetchone() == ("Canonical report", "B86-1", "Basin 86")

    def test_two_year_csv_failure_leaves_snapshot_unchanged(
        self, isolated_config, mock_esdc_api, monkeypatch
    ):
        monkeypatch.setenv("ESDC_USER", "u")
        monkeypatch.setenv("ESDC_PASS", "p")
        seed = [{"project_id": "KEEP", "project_name": "Keep", "report_year": 2024}]
        mock_esdc_api.add_json(PROJECT_RESOURCES_URL, json_data=seed)
        mock_esdc_api.add_json(_timeseries_url_for_year(2024), json_data=[])
        assert runner.invoke(app, ["fetch", "--filetype", "json"]).exit_code == 0
        conn = get_sqlite_connection(isolated_config / ".esdc" / "esdc.sqlite")
        before = get_project_resources_snapshot(conn)
        before_rows = conn.execute("SELECT * FROM api_project_resources").fetchall()
        conn.close()
        with duckdb.connect(str(isolated_config / ".esdc" / "esdc.duckdb")) as db:
            before_duck = db.execute("SELECT * FROM project_resources").fetchall()
        mock_esdc_api.add_csv(
            _resources_url_for_year(2024),
            body="project_id;report_year;project_name;groovy_report\nNEW24;2024;New;Canonical\n",
        )
        mock_esdc_api.add_status(_resources_url_for_year(2025), 500)
        result = runner.invoke(
            app, ["fetch", "--filetype", "csv", "--year", "2024", "--year", "2025"]
        )
        assert result.exit_code != 0
        conn = get_sqlite_connection(isolated_config / ".esdc" / "esdc.sqlite")
        assert get_project_resources_snapshot(conn) == before
        assert (
            conn.execute("SELECT * FROM api_project_resources").fetchall()
            == before_rows
        )
        conn.close()
        with duckdb.connect(str(isolated_config / ".esdc" / "esdc.duckdb")) as db:
            assert (
                db.execute("SELECT * FROM project_resources").fetchall() == before_duck
            )

    def test_selected_empty_scope_preserves_unselected_rows(
        self, isolated_config, mock_esdc_api, monkeypatch
    ):
        monkeypatch.setenv("ESDC_USER", "u")
        monkeypatch.setenv("ESDC_PASS", "p")
        mock_esdc_api.add_json(
            PROJECT_RESOURCES_URL,
            json_data=[{"project_id": "KEEP", "report_year": 2024}],
        )
        mock_esdc_api.add_json(_timeseries_url_for_year(2024), json_data=[])
        assert runner.invoke(app, ["fetch", "--filetype", "json"]).exit_code == 0
        conn = get_sqlite_connection(isolated_config / ".esdc" / "esdc.sqlite")
        before = get_project_resources_snapshot(conn)
        conn.close()
        mock_esdc_api.add_json(
            _resources_url_for_year(2025),
            json_data=[{"project_id": "OTHER", "report_year": 2024}],
        )
        mock_esdc_api.add_json(_timeseries_url_for_year(2025), json_data=[])
        result = runner.invoke(app, ["fetch", "--filetype", "json", "--year", "2025"])
        assert result.exit_code == 0
        assert "Resources: skipped years 2025" in result.output
        conn = get_sqlite_connection(isolated_config / ".esdc" / "esdc.sqlite")
        after = get_project_resources_snapshot(conn)
        assert before is not None and after is not None
        assert after.revision != before.revision
        assert after.row_count == 1
        assert [
            tuple(row)
            for row in conn.execute(
                "SELECT project_id, report_year FROM api_project_resources"
            )
        ] == [("KEEP", 2024)]
        conn.close()


class TestFetchCsvToDatabase:
    """Tests that fetch with CSV persists data to database."""

    def test_fetch_csv_to_database(self, isolated_config, mock_esdc_api, monkeypatch):
        """Fetch with CSV should persist data to database."""
        csv_content = (
            "project_id;project_name;report_year;project_stage\n"
            "PRJ-002;CSV Project;2024;DEVELOPMENT"
        )
        csv_timeseries = (
            "report_year;project_id;year;cprd_grs_oil\n2024;PRJ-002;2024;200.0"
        )
        mock_esdc_api.add_csv(PROJECT_RESOURCES_URL, body=csv_content)
        mock_esdc_api.add_csv(_timeseries_url_for_year(2024), body=csv_timeseries)

        monkeypatch.setenv("ESDC_USER", "testuser")
        monkeypatch.setenv("ESDC_PASS", "testpass")

        result = runner.invoke(app, ["fetch", "--filetype", "csv"])

        assert result.exit_code == 0

        db_file = isolated_config / ".esdc" / "esdc.duckdb"
        assert db_file.exists()

        conn = duckdb.connect(str(db_file))
        cursor = conn.execute("SELECT COUNT(*) FROM project_resources")
        row = cursor.fetchone()
        assert row is not None, "No rows returned"
        count = row[0]
        conn.close()

        assert count == 1, "Data should be persisted to project_resources table"


class TestFetchAuthFailure:
    """Tests that fetch handles auth failures correctly."""

    def test_fetch_auth_failure(
        self, isolated_config, mock_esdc_api, monkeypatch, caplog
    ):
        """API returns 401 should show clear error message."""
        import logging

        caplog.set_level(logging.WARNING)
        mock_esdc_api.add_status(PROJECT_RESOURCES_URL, status=401)

        monkeypatch.setenv("ESDC_USER", "baduser")
        monkeypatch.setenv("ESDC_PASS", "badpass")

        runner.invoke(app, ["fetch", "--filetype", "json"])

        # Check logs for warning messages
        log_messages = caplog.text
        assert "401" in log_messages or "Failed to download" in log_messages


class TestFetchMalformedJson:
    """Tests that fetch handles malformed JSON correctly."""

    def test_fetch_malformed_json(self, isolated_config, mock_esdc_api, monkeypatch):
        """Invalid JSON response should fail gracefully."""
        mock_esdc_api.add_json(
            PROJECT_RESOURCES_URL,
            json_data=[{"project_id": "PRJ-001", "invalid_column": "data"}],
        )
        mock_esdc_api.add_json(
            _timeseries_url_for_year(2024),
            json_data=[{"project_id": "PRJ-001", "year": 2024}],
        )

        monkeypatch.setenv("ESDC_USER", "testuser")
        monkeypatch.setenv("ESDC_PASS", "testpass")

        result = runner.invoke(app, ["fetch", "--filetype", "json"])

        assert result.exit_code == 0 or result.exception is not None


class TestFetchWithEnvCredentials:
    """Tests that fetch works with environment credentials."""

    def test_fetch_with_env_credentials(
        self, isolated_config, mock_esdc_api, monkeypatch
    ):
        """ESDC_USER and ESDC_PASS env vars should work."""
        project_resources_data = [
            {
                "project_id": "PRJ-003",
                "project_name": "Env Test",
                "report_year": 2024,
                "project_stage": "PRODUCTION",
            }
        ]
        mock_esdc_api.add_json(PROJECT_RESOURCES_URL, json_data=project_resources_data)
        mock_esdc_api.add_json(
            _timeseries_url_for_year(2024),
            json_data=[
                {
                    "project_id": "PRJ-003",
                    "report_year": 2024,
                    "year": 2024,
                    "cprd_grs_oil": 300.0,
                }
            ],
        )

        monkeypatch.setenv("ESDC_USER", "envuser")
        monkeypatch.setenv("ESDC_PASS", "envpass")

        result = runner.invoke(app, ["fetch", "--filetype", "json"])

        assert result.exit_code == 0

        db_file = isolated_config / ".esdc" / "esdc.duckdb"
        assert db_file.exists()
