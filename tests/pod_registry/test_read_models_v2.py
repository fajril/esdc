"""POD v2 physical read-model rebuild regressions."""

from __future__ import annotations

import importlib
import sqlite3
from datetime import date
from pathlib import Path

import pytest

from esdc.api_snapshot import ApiProjectResourceRow, ApiSnapshot
from esdc.timeseries_snapshot import (
    CanonicalTimeseriesRow,
    ensure_timeseries_tables,
    normalize_timeseries_row,
    replace_timeseries_in_transaction,
)


def _connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    migrations = importlib.import_module("esdc.pod_registry.migrations")
    migrations.initialize_registry_v2(conn)
    return conn


def _insert_pod(
    conn: sqlite3.Connection,
    *,
    approval_date: date = date(2024, 1, 1),
    daily_slot: int = 1,
    pod_name: str = "POD one",
    reported_date: str = "2024-06-30",
) -> int:
    identity = importlib.import_module("esdc.pod_registry.identity")
    issued = identity.issue_identity(
        conn,
        key=identity.ApprovalKey(
            approval_date=approval_date,
            daily_slot=daily_slot,
            institution_id=1,
            pod_type_id=1,
            rev_num=0,
        ),
        actor=f"FA:test-read-model-{daily_slot}",
    )
    conn.execute(
        "INSERT INTO m_pod (pod_id_seq, pod_uuid, approval_date, institution_id, "
        "pod_type_id, rev_num, pod_order_no, pod_id, pod_name, pod_letter_num, "
        "is_latest, capital_drilling_pod, irr_ctr_pod) "
        "VALUES (?, ?, ?, 1, 1, 0, ?, ?, ?, 'Letter one', 1, '1.235', '0.123456')",
        (
            issued.pod_id_seq,
            str(issued.pod_uuid),
            issued.approval_date.isoformat(),
            issued.pod_order_no,
            issued.pod_id,
            pod_name,
        ),
    )
    conn.execute(
        "INSERT INTO pod_monitoring (monpod_uuid, pod_id_seq, pod_id, reported_date, "
        "is_completed, pod_stage_id, capital_drilling_real, irr_ctr_real) "
        "VALUES (?, ?, ?, ?, 0, 1, '2.345', '0.987654')",
        (f"monitor-{issued.pod_id_seq}", issued.pod_id_seq, issued.pod_id, reported_date),
    )
    return issued.pod_id_seq


def _install_snapshot_tables(conn: sqlite3.Connection) -> None:
    columns = tuple(ApiProjectResourceRow.__dataclass_fields__)
    conn.execute(
        "CREATE TABLE api_project_resources ("
        + ", ".join(f'"{name}" TEXT' for name in columns)
        + ")"
    )
    conn.execute(
        "CREATE TABLE api_snapshot ("
        "source_table TEXT PRIMARY KEY, revision TEXT NOT NULL, row_count INTEGER NOT NULL, "
        "source_sha256 TEXT NOT NULL, fetched_at TEXT NOT NULL, publication_state TEXT NOT NULL, "
        "invalid_geometry_count INTEGER NOT NULL DEFAULT 0)"
    )
    ensure_timeseries_tables(conn)


def _map_project(conn: sqlite3.Connection, pod_id_seq: int, project_id: str) -> None:
    conn.execute(
        "INSERT INTO m_project (project_id, project_name, is_active) VALUES (?, ?, 1)",
        (project_id, f"Project {project_id}"),
    )
    conn.execute(
        "INSERT INTO pod_project (pod_project_uuid, pod_id_seq, project_id, is_main_project, "
        "is_auto_inferred) VALUES (?, ?, ?, 1, 0)",
        (f"map-{project_id}", pod_id_seq, project_id),
    )


def _resource(
    project_id: str,
    report_year: int,
    *,
    case: str = "2. Middle Value",
    level: str = "E4. Plan approved",
    oil_reserve: str = "0",
    oil_sales: str | None = None,
    con_sales: str | None = None,
    ga_sales: str | None = None,
    gn_sales: str | None = None,
    oc_sales: str | None = None,
    an_sales: str | None = None,
    project_name_value: str | None = None,
    field_name_value: str | None = None,
    wk_name: str | None = None,
) -> ApiProjectResourceRow:
    return ApiProjectResourceRow(
        project_id=project_id,
        report_year=report_year,
        project_name=project_name_value or f"Name {project_id}",
        field_name=field_name_value or f"Field {project_id}",
        wk_name=wk_name,
        project_level=level,
        uncert_level=case,
        res_oil=oil_reserve,
        res_con="1.111",
        res_ga="2.222",
        res_gn="3.333",
        res_oc="4.444",
        res_an="5.555",
        rec_oil="6.666",
        rec_con="7.777",
        rec_ga="8.888",
        rec_gn="9.999",
        rec_oc="10.111",
        rec_an="11.222",
        prj_ioip="12.333",
        prj_igip="13.444",
        cprd_sls_oil=oil_sales,
        cprd_sls_con=con_sales,
        cprd_sls_ga=ga_sales,
        cprd_sls_gn=gn_sales,
        cprd_sls_oc=oc_sales,
        cprd_sls_an=an_sales,
    )


def _timeseries(
    project_id: str,
    report_year: int,
    year: int,
    *,
    slf_oil: str | None = None,
    slf_con: str | None = None,
    slf_ga: str | None = None,
    slf_gn: str | None = None,
) -> CanonicalTimeseriesRow:
    header = (
        "project_id",
        "report_year",
        "year",
        "slf_oil",
        "slf_con",
        "slf_ga",
        "slf_gn",
    )
    return normalize_timeseries_row(
        header,
        (project_id, report_year, year, slf_oil, slf_con, slf_ga, slf_gn),
        origin="api_json",
    )


def _seed_forecast_context(
    conn: sqlite3.Connection, *, oil: str = "1.123", con: str = "2.345"
) -> int:
    pod_id_seq = _insert_pod(conn)
    _map_project(conn, pod_id_seq, "P-1")
    conn.execute(
        "INSERT INTO pod_forecast (frcast_uuid, pod_id_seq, year_frcast, usf_oil_pod, "
        "usf_con_pod, usf_ga_pod, usf_gn_pod) "
        "VALUES ('forecast-input', ?, 2025, ?, ?, '3.456', '4.567')",
        (pod_id_seq, oil, con),
    )
    _install_snapshot_tables(conn)
    conn.commit()

    from esdc.api_snapshot import replace_project_resources_snapshot

    replace_project_resources_snapshot(
        conn,
        [_resource("P-1", 2024)],
        ApiSnapshot("project_resources", "source-resource", 1, "", "now", "committed"),
    )
    return pod_id_seq


def test_rebuild_uses_caller_transaction_and_keeps_exact_inputs() -> None:
    conn = _connection()
    try:
        pod_id_seq = _insert_pod(conn)
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")

        read_models = importlib.import_module("esdc.pod_registry.read_models")
        report = read_models.rebuild_pod_read_models(conn)

        assert conn.in_transaction
        assert report.monitoring_rows == 1
        assert report.source_generation == 0
        assert conn.execute(
            "SELECT pod_id_seq, reported_date, monpod_id, capital_drilling_pod, "
            "irr_ctr_pod, capital_drilling_real, irr_ctr_real "
            "FROM pod_monitoring_api"
        ).fetchone() == (
            pod_id_seq,
            "2024-06-30",
            1,
            "1.24",
            "0.123456",
            "2.35",
            "0.987654",
        )
        assert conn.execute(
            "SELECT capital_drilling_pod, irr_ctr_pod FROM m_pod WHERE pod_id_seq=?",
            (pod_id_seq,),
        ).fetchone() == ("1.235", "0.123456")
        assert conn.execute(
            "SELECT type FROM sqlite_master WHERE name='pod_monitoring_api'"
        ).fetchone() == ("table",)
        conn.rollback()
    finally:
        conn.close()


def test_rebuild_rejects_unsupported_dictionary_metadata() -> None:
    conn = _connection()
    try:
        _insert_pod(conn)
        conn.commit()
        conn.execute(
            "UPDATE pod_registry_state SET dictionary_version='0.3.9' WHERE singleton_id=1"
        )
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")

        read_models = importlib.import_module("esdc.pod_registry.read_models")
        with pytest.raises(RuntimeError, match="supported v2 registry"):
            read_models.rebuild_pod_read_models(conn)

        assert conn.execute(
            "SELECT COUNT(*) FROM pod_monitoring_api"
        ).fetchone() == (0,)
        conn.rollback()
    finally:
        conn.close()


def test_refresh_rejects_missing_v2_singleton_before_output_writes() -> None:
    conn = _connection()
    try:
        _insert_pod(conn)
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")
        read_models = importlib.import_module("esdc.pod_registry.read_models")
        read_models.rebuild_pod_read_models(conn)
        conn.commit()
        previous = conn.execute(
            "SELECT reported_date, capital_drilling_real FROM pod_monitoring_api"
        ).fetchall()
        conn.execute("DELETE FROM pod_registry_state WHERE singleton_id=1")
        conn.commit()

        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(RuntimeError, match="supported v2 registry"):
            read_models.refresh_pod_after_source_change(conn, actor="FA:test")

        assert conn.execute(
            "SELECT reported_date, capital_drilling_real FROM pod_monitoring_api"
        ).fetchall() == previous
        assert conn.in_transaction
        conn.rollback()
    finally:
        conn.close()


def test_file_rebuild_requires_shared_writer_policy_before_writing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "raw-reader.sqlite"
    initialized = sqlite3.connect(path)
    try:
        migrations = importlib.import_module("esdc.pod_registry.migrations")
        migrations.initialize_registry_v2(initialized)
        initialized.commit()
    finally:
        initialized.close()

    conn = sqlite3.connect(path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        read_models = importlib.import_module("esdc.pod_registry.read_models")
        with pytest.raises(RuntimeError, match="Configure file-backed registry"):
            read_models.rebuild_pod_read_models(conn)

        assert conn.execute(
            "SELECT COUNT(*) FROM pod_monitoring_api"
        ).fetchone() == (0,)
        assert conn.in_transaction
        conn.rollback()
    finally:
        conn.close()


def test_approved_forecast_without_snapshot_context_is_preserved_and_omitted() -> None:
    conn = _connection()
    try:
        pod_id_seq = _insert_pod(conn)
        conn.execute(
            "INSERT INTO pod_forecast (frcast_uuid, pod_id_seq, year_frcast, usf_oil_pod, "
            "usf_con_pod, usf_ga_pod, usf_gn_pod) "
            "VALUES ('forecast-contextless', ?, 2025, '1.123', '2.345', '3.456', '4.567')",
            (pod_id_seq,),
        )
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")

        read_models = importlib.import_module("esdc.pod_registry.read_models")
        report = read_models.rebuild_pod_read_models(conn)

        assert report.forecast_rows == 0
        assert any(
            diagnostic.code == "forecast_reporting_context_missing"
            and diagnostic.pod_id_seq == pod_id_seq
            and diagnostic.year_frcast == 2025
            for diagnostic in report.diagnostics
        )
        assert conn.execute(
            "SELECT usf_oil_pod, usf_con_pod FROM pod_forecast WHERE pod_id_seq=?",
            (pod_id_seq,),
        ).fetchone() == ("1.123", "2.345")
        assert conn.execute(
            "SELECT COUNT(*) FROM pod_forecast_api"
        ).fetchone() == (0,)
        conn.rollback()
    finally:
        conn.close()


def test_approved_forecast_is_reused_across_annual_snapshot_contexts() -> None:
    conn = _connection()
    try:
        pod_id_seq = _insert_pod(conn)
        _map_project(conn, pod_id_seq, "P-1")
        conn.execute(
            "INSERT INTO pod_forecast (frcast_uuid, pod_id_seq, year_frcast, usf_oil_pod, "
            "usf_con_pod, usf_ga_pod, usf_gn_pod) "
            "VALUES ('forecast-reused', ?, 2025, '1.123', '2.345', '3.456', '4.567')",
            (pod_id_seq,),
        )
        _install_snapshot_tables(conn)
        conn.commit()

        from esdc.api_snapshot import replace_project_resources_snapshot

        replace_project_resources_snapshot(
            conn,
            [_resource("P-1", 2024), _resource("P-1", 2025)],
            ApiSnapshot("project_resources", "source-resource", 2, "", "now", "committed"),
        )

        assert conn.execute(
            "SELECT reported_date, year_frcast, frcast_id, usf_oil_pod "
            "FROM pod_forecast_api WHERE pod_id_seq=? AND year_frcast=2025 "
            "ORDER BY reported_date",
            (pod_id_seq,),
        ).fetchall() == [
            ("2024-12-31", 2025, 1, "1.123"),
            ("2025-12-31", 2025, 1, "1.123"),
        ]
    finally:
        conn.close()


def test_snapshot_refresh_aggregates_middle_case_and_resolves_annual_outlook() -> None:
    conn = _connection()
    try:
        pod_id_seq = _insert_pod(conn)
        _map_project(conn, pod_id_seq, "P-1")
        _map_project(conn, pod_id_seq, "P-2")
        conn.execute(
            "UPDATE pod_registry_state SET knowledge_generation=42 WHERE singleton_id=1"
        )
        conn.execute(
            "INSERT INTO pod_forecast (frcast_uuid, pod_id_seq, year_frcast, usf_oil_pod, "
            "usf_con_pod, usf_ga_pod, usf_gn_pod) "
            "VALUES ('forecast-one', ?, 2025, '1.123', '2.345', '3.456', '4.567')",
            (pod_id_seq,),
        )
        _install_snapshot_tables(conn)
        conn.commit()

        resources = [
            _resource("P-1", 2023, oil_sales="10", con_sales="2", ga_sales="100", gn_sales="50"),
            _resource("P-2", 2023, oil_sales="20", con_sales="4", ga_sales="200", gn_sales="75"),
            _resource("P-1", 2024, oil_reserve="1.235", oil_sales="16.125", con_sales="5", ga_sales="140", gn_sales="60", wk_name="WK B", level="E7. Existing production", project_name_value="Zulu", field_name_value="Shared field"),
            _resource("P-2", 2024, oil_reserve="2.555", oil_sales="30.125", con_sales="8", ga_sales="260", gn_sales="100", wk_name="WK A", level="E0. Production", project_name_value="Alpha", field_name_value="Shared field"),
            _resource("P-1", 2024, case="1P", oil_reserve="999", wk_name="Wrong case"),
            _resource("P-2", 2024, case="3P", oil_reserve="888", wk_name="Wrong case"),
        ]
        from esdc.api_snapshot import replace_project_resources_snapshot

        replace_project_resources_snapshot(
            conn,
            resources,
            ApiSnapshot("project_resources", "source-resource", len(resources), "", "now", "committed"),
        )

        duplicate_future = _timeseries("P-1", 2024, 2025, slf_oil="40.0005", slf_con="20", slf_ga="300", slf_gn="100")
        conn.execute("BEGIN IMMEDIATE")
        replace_timeseries_in_transaction(
            conn,
            [
                _timeseries("P-1", 2024, 2024),
                _timeseries("P-2", 2024, 2024),
                duplicate_future,
                duplicate_future,
                _timeseries("P-2", 2024, 2025, slf_oil="60", slf_con="30", slf_ga="400", slf_gn="200"),
            ],
            years=None,
            revision="source-timeseries",
            fetched_at="now",
        )
        conn.commit()

        monitor = conn.execute(
            "SELECT res_oil_2p, cprd_sls_oil, wk_name_list, project_level_highest, "
            "field_name_list, project_name_list "
            "FROM pod_monitoring_api WHERE reported_date='2024-06-30'"
        ).fetchone()
        assert monitor == (
            3.79,
            46.25,
            "WK A;WK B",
            "E0. Production",
            "Shared field",
            "Alpha;Zulu",
        )

        forecasts = conn.execute(
            "SELECT year_frcast, frcast_id, usf_oil_pod, usf_oc_pod, usf_oil_otlk, "
            "usf_oc_otlk FROM pod_forecast_api WHERE reported_date='2024-12-31' "
            "ORDER BY year_frcast"
        ).fetchall()
        assert forecasts == [
            (2024, None, None, None, "16.250", "23.250"),
            (2025, 1, "1.123", "3.468", "100.001", "150.001"),
        ]
        assert conn.execute(
            "SELECT usf_oil_pod, usf_con_pod FROM pod_forecast WHERE pod_id_seq=?",
            (pod_id_seq,),
        ).fetchone() == ("1.123", "2.345")
        assert conn.execute(
            "SELECT source_generation, knowledge_generation FROM pod_registry_state"
        ).fetchone() == (2, None)
        assert not importlib.import_module(
            "esdc.knowledge.freshness"
        ).knowledge_freshness(conn).fresh
    finally:
        conn.close()


def test_monitoring_uses_current_cumulative_sales_without_prior_year() -> None:
    conn = _connection()
    try:
        pod_id_seq = _insert_pod(conn)
        _map_project(conn, pod_id_seq, "P-1")
        _map_project(conn, pod_id_seq, "P-2")
        _install_snapshot_tables(conn)
        conn.commit()

        from esdc.api_snapshot import replace_project_resources_snapshot

        replace_project_resources_snapshot(
            conn,
            [
                _resource(
                    "P-1", 2024, oil_sales="100", con_sales="25", ga_sales="5",
                    gn_sales="4", oc_sales="700", an_sales="20"
                ),
                _resource(
                    "P-2", 2024, oil_sales="30", con_sales="10", ga_sales="7",
                    gn_sales="3", oc_sales="100", an_sales="40"
                ),
            ],
            ApiSnapshot("project_resources", "source-resource", 2, "", "now", "committed"),
        )
        conn.execute("BEGIN IMMEDIATE")
        replace_timeseries_in_transaction(
            conn,
            [_timeseries("P-1", 2024, 2024), _timeseries("P-2", 2024, 2024)],
            years=None,
            revision="source-timeseries",
            fetched_at="now",
        )
        conn.commit()

        read_models = importlib.import_module("esdc.pod_registry.read_models")
        conn.execute("BEGIN IMMEDIATE")
        report = read_models.rebuild_pod_read_models(conn)

        monitor = conn.execute(
            "SELECT cprd_sls_oil, cprd_sls_con, cprd_sls_ga, cprd_sls_gn, "
            "cprd_sls_oc, cprd_sls_an FROM pod_monitoring_api "
            "WHERE reported_date='2024-06-30'"
        ).fetchone()
        forecast = conn.execute(
            "SELECT usf_oil_otlk FROM pod_forecast_api WHERE reported_date='2024-12-31' "
            "AND year_frcast=2024"
        ).fetchone()
        assert monitor == (130.0, 35.0, 12.0, 7.0, 800.0, 60.0)
        assert forecast == (None,)
        assert any(
            diagnostic.code == "resource_group_incomplete"
            and diagnostic.reported_date == "2024-12-31"
            and diagnostic.year_frcast == 2024
            and diagnostic.component == "cprd_sls_oil"
            for diagnostic in report.diagnostics
        )
        conn.rollback()
    finally:
        conn.close()


def test_multiple_monitoring_submissions_have_no_synthetic_dates_and_keep_history() -> None:
    conn = _connection()
    try:
        predecessor = _insert_pod(conn, pod_name="Earlier POD")
        successor = _insert_pod(
            conn,
            approval_date=date(2025, 1, 1),
            pod_name="Replacement POD",
            reported_date="2025-06-30",
        )
        conn.execute(
            "UPDATE m_pod SET superseded_by=?, is_latest=0 WHERE pod_id_seq=?",
            (successor, predecessor),
        )
        conn.execute(
            "INSERT INTO pod_monitoring "
            "(monpod_uuid, pod_id_seq, pod_id, reported_date, is_completed, pod_stage_id) "
            "SELECT 'earlier-second-report', pod_id_seq, pod_id, '2024-12-15', 0, 1 "
            "FROM m_pod WHERE pod_id_seq=?",
            (predecessor,),
        )
        conn.execute(
            "INSERT INTO pod_monitoring "
            "(monpod_uuid, pod_id_seq, pod_id, reported_date, is_completed, pod_stage_id) "
            "SELECT 'earlier-after-replacement', pod_id_seq, pod_id, '2025-06-30', 0, 1 "
            "FROM m_pod WHERE pod_id_seq=?",
            (predecessor,),
        )
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")

        read_models = importlib.import_module("esdc.pod_registry.read_models")
        read_models.rebuild_pod_read_models(conn)

        assert conn.execute(
            "SELECT pod_id_seq, reported_date FROM pod_monitoring_api "
            "ORDER BY reported_date, pod_id_seq"
        ).fetchall() == [
            (predecessor, "2024-06-30"),
            (predecessor, "2024-12-15"),
            (successor, "2025-06-30"),
        ]
        conn.rollback()
    finally:
        conn.close()


def test_scoped_rebuild_preserves_unselected_pod_outputs() -> None:
    conn = _connection()
    try:
        first = _insert_pod(conn, daily_slot=1)
        second = _insert_pod(conn, daily_slot=2)
        conn.commit()
        read_models = importlib.import_module("esdc.pod_registry.read_models")
        conn.execute("BEGIN IMMEDIATE")
        read_models.rebuild_pod_read_models(conn)
        conn.commit()

        before = conn.execute(
            "SELECT pod_id_seq, irr_ctr_pod FROM pod_monitoring_api "
            "ORDER BY pod_id_seq"
        ).fetchall()
        conn.execute(
            "UPDATE m_pod SET irr_ctr_pod='0.222222' WHERE pod_id_seq=?",
            (first,),
        )
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")
        read_models.rebuild_pod_read_models(conn, pod_ids=(first,))

        assert before == [(first, "0.123456"), (second, "0.123456")]
        assert conn.execute(
            "SELECT pod_id_seq, irr_ctr_pod FROM pod_monitoring_api "
            "ORDER BY pod_id_seq"
        ).fetchall() == [(first, "0.222222"), (second, "0.123456")]
        conn.rollback()
    finally:
        conn.close()


def test_predecessor_names_are_ordered_by_pod_identity() -> None:
    conn = _connection()
    try:
        first = _insert_pod(conn, daily_slot=1, pod_name="Zulu predecessor")
        second = _insert_pod(conn, daily_slot=2, pod_name="Alpha predecessor")
        successor = _insert_pod(
            conn,
            approval_date=date(2025, 1, 1),
            pod_name="Successor",
            reported_date="2025-06-30",
        )
        conn.execute(
            "UPDATE m_pod SET superseded_by=?, is_latest=0 WHERE pod_id_seq IN (?, ?)",
            (successor, first, second),
        )
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")

        read_models = importlib.import_module("esdc.pod_registry.read_models")
        read_models.rebuild_pod_read_models(conn)

        assert conn.execute(
            "SELECT preceded_by_pod_name FROM pod_monitoring_api WHERE pod_id_seq=?",
            (successor,),
        ).fetchone() == ("Zulu predecessor;Alpha predecessor",)
        conn.rollback()
    finally:
        conn.close()


def test_combined_two_p_reserves_use_zero_but_risked_null_is_preserved() -> None:
    conn = _connection()
    try:
        pod_id_seq = _insert_pod(conn)
        conn.execute(
            "UPDATE m_pod SET res_oil_2p_pod='1.5', res_oil_3p_pod='1.5', "
            "res_oil_2p_risked_pod='0.5', res_con_2p_risked_pod=NULL "
            "WHERE pod_id_seq=?",
            (pod_id_seq,),
        )
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")

        read_models = importlib.import_module("esdc.pod_registry.read_models")
        read_models.rebuild_pod_read_models(conn)

        assert conn.execute(
            "SELECT res_oc_2p_pod, res_oc_2p_risked_pod "
            "FROM pod_monitoring_api WHERE pod_id_seq=?",
            (pod_id_seq,),
        ).fetchone() == (1.5, None)
        conn.rollback()
    finally:
        conn.close()


def test_conflicting_repeated_future_timeseries_value_is_null_with_diagnostic() -> None:
    conn = _connection()
    try:
        pod_id_seq = _seed_forecast_context(conn)
        conn.execute("BEGIN IMMEDIATE")
        replace_timeseries_in_transaction(
            conn,
            [
                _timeseries("P-1", 2024, 2026, slf_oil="10", slf_con="2"),
                _timeseries("P-1", 2024, 2026, slf_oil="12", slf_con="2"),
            ],
            years=None,
            revision="source-timeseries-conflict",
            fetched_at="later",
        )
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")

        read_models = importlib.import_module("esdc.pod_registry.read_models")
        report = read_models.rebuild_pod_read_models(conn)

        assert conn.execute(
            "SELECT usf_oil_otlk FROM pod_forecast_api WHERE pod_id_seq=? "
            "AND reported_date='2024-12-31' AND year_frcast=2026",
            (pod_id_seq,),
        ).fetchone() == (None,)
        assert any(
            diagnostic.code == "timeseries_value_conflicting"
            and diagnostic.pod_id_seq == pod_id_seq
            and diagnostic.year_frcast == 2026
            and diagnostic.component == "slf_oil"
            for diagnostic in report.diagnostics
        )
        conn.rollback()
    finally:
        conn.close()


def test_invalid_approved_forecast_precision_is_rejected_before_output_replacement() -> None:
    conn = _connection()
    try:
        pod_id_seq = _seed_forecast_context(conn)
        previous = conn.execute(
            "SELECT year_frcast, usf_oil_pod FROM pod_forecast_api WHERE pod_id_seq=? "
            "ORDER BY year_frcast",
            (pod_id_seq,),
        ).fetchall()
        conn.execute(
            "UPDATE pod_forecast SET usf_oil_pod='1.1234' WHERE pod_id_seq=?",
            (pod_id_seq,),
        )
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")

        read_models = importlib.import_module("esdc.pod_registry.read_models")
        with pytest.raises(ValueError, match="scale 3"):
            read_models.rebuild_pod_read_models(conn)

        assert conn.execute(
            "SELECT year_frcast, usf_oil_pod FROM pod_forecast_api WHERE pod_id_seq=? "
            "ORDER BY year_frcast",
            (pod_id_seq,),
        ).fetchall() == previous
        conn.rollback()
    finally:
        conn.close()


def test_derived_forecast_precision_overflow_is_rejected_before_output_replacement() -> None:
    conn = _connection()
    try:
        pod_id_seq = _seed_forecast_context(conn)
        previous = conn.execute(
            "SELECT year_frcast, usf_oc_pod FROM pod_forecast_api WHERE pod_id_seq=? "
            "ORDER BY year_frcast",
            (pod_id_seq,),
        ).fetchall()
        conn.execute(
            "UPDATE pod_forecast SET usf_oil_pod='99999999999999999.999', "
            "usf_con_pod='1.000' WHERE pod_id_seq=?",
            (pod_id_seq,),
        )
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")

        read_models = importlib.import_module("esdc.pod_registry.read_models")
        with pytest.raises(ValueError, match="precision 20, scale 3"):
            read_models.rebuild_pod_read_models(conn)

        assert conn.execute(
            "SELECT year_frcast, usf_oc_pod FROM pod_forecast_api WHERE pod_id_seq=? "
            "ORDER BY year_frcast",
            (pod_id_seq,),
        ).fetchall() == previous
        conn.rollback()
    finally:
        conn.close()


def test_rounded_monitoring_precision_overflow_is_rejected_before_replacement() -> None:
    conn = _connection()
    try:
        pod_id_seq = _insert_pod(conn)
        conn.commit()
        read_models = importlib.import_module("esdc.pod_registry.read_models")
        conn.execute("BEGIN IMMEDIATE")
        read_models.rebuild_pod_read_models(conn)
        conn.commit()
        previous = conn.execute(
            "SELECT capital_drilling_pod FROM pod_monitoring_api WHERE pod_id_seq=?",
            (pod_id_seq,),
        ).fetchall()
        conn.execute(
            "UPDATE m_pod SET capital_drilling_pod='99999999999999999.999' "
            "WHERE pod_id_seq=?",
            (pod_id_seq,),
        )
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")

        with pytest.raises(ValueError, match="precision 20, scale 3"):
            read_models.rebuild_pod_read_models(conn)

        assert conn.execute(
            "SELECT capital_drilling_pod FROM pod_monitoring_api WHERE pod_id_seq=?",
            (pod_id_seq,),
        ).fetchall() == previous
        conn.rollback()
    finally:
        conn.close()


@pytest.mark.parametrize("invalid_value", ["not-a-number", "Inf"])
def test_invalid_monitoring_decimal_is_rejected_before_replacement(
    invalid_value: str,
) -> None:
    conn = _connection()
    try:
        pod_id_seq = _insert_pod(conn)
        conn.commit()
        read_models = importlib.import_module("esdc.pod_registry.read_models")
        conn.execute("BEGIN IMMEDIATE")
        read_models.rebuild_pod_read_models(conn)
        conn.commit()
        previous = conn.execute(
            "SELECT capital_drilling_pod FROM pod_monitoring_api WHERE pod_id_seq=?",
            (pod_id_seq,),
        ).fetchall()
        conn.execute(
            "UPDATE m_pod SET capital_drilling_pod=? WHERE pod_id_seq=?",
            (invalid_value, pod_id_seq),
        )
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")

        with pytest.raises(ValueError, match="finite numeric value"):
            read_models.rebuild_pod_read_models(conn)

        assert conn.execute(
            "SELECT capital_drilling_pod FROM pod_monitoring_api WHERE pod_id_seq=?",
            (pod_id_seq,),
        ).fetchall() == previous
        conn.rollback()
    finally:
        conn.close()


def test_monitoring_double_overflow_is_rejected_before_replacement() -> None:
    conn = _connection()
    try:
        pod_id_seq = _seed_forecast_context(conn)
        previous = conn.execute(
            "SELECT cprd_sls_oil FROM pod_monitoring_api WHERE pod_id_seq=?",
            (pod_id_seq,),
        ).fetchall()
        conn.execute(
            "UPDATE api_project_resources SET cprd_sls_oil='1e309' "
            "WHERE project_id='P-1' AND report_year=2024"
        )
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")
        read_models = importlib.import_module("esdc.pod_registry.read_models")

        with pytest.raises(ValueError, match="finite output precision"):
            read_models.rebuild_pod_read_models(conn)

        assert conn.execute(
            "SELECT cprd_sls_oil FROM pod_monitoring_api WHERE pod_id_seq=?",
            (pod_id_seq,),
        ).fetchall() == previous
        conn.rollback()
    finally:
        conn.close()


def test_large_finite_monitoring_double_remains_supported() -> None:
    conn = _connection()
    try:
        pod_id_seq = _insert_pod(conn)
        conn.execute(
            "UPDATE m_pod SET res_oil_2p_pod='1e30', res_oil_3p_pod='1e30' "
            "WHERE pod_id_seq=?",
            (pod_id_seq,),
        )
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")

        read_models = importlib.import_module("esdc.pod_registry.read_models")
        read_models.rebuild_pod_read_models(conn)

        assert conn.execute(
            "SELECT res_oil_2p_pod FROM pod_monitoring_api WHERE pod_id_seq=?",
            (pod_id_seq,),
        ).fetchone() == (1e30,)
        conn.rollback()
    finally:
        conn.close()


def test_snapshot_source_change_rolls_back_when_read_model_preflight_fails() -> None:
    conn = _connection()
    try:
        pod_id_seq = _seed_forecast_context(conn)
        before_snapshot = conn.execute(
            "SELECT revision FROM api_snapshot WHERE source_table='project_resources'"
        ).fetchone()
        before_outputs = conn.execute(
            "SELECT year_frcast, usf_oil_pod FROM pod_forecast_api WHERE pod_id_seq=? "
            "ORDER BY year_frcast",
            (pod_id_seq,),
        ).fetchall()
        before_generation = conn.execute(
            "SELECT source_generation FROM pod_registry_state"
        ).fetchone()
        conn.execute(
            "UPDATE pod_forecast SET usf_oil_pod='1.1234' WHERE pod_id_seq=?",
            (pod_id_seq,),
        )
        conn.commit()

        from esdc.api_snapshot import replace_project_resources_snapshot

        with pytest.raises(ValueError, match="scale 3"):
            replace_project_resources_snapshot(
                conn,
                [_resource("P-1", 2025)],
                ApiSnapshot(
                    "project_resources", "next-resource", 1, "", "later", "committed"
                ),
            )

        assert not conn.in_transaction
        assert conn.execute(
            "SELECT DISTINCT report_year FROM api_project_resources"
        ).fetchall() == [("2024",)]
        assert conn.execute(
            "SELECT revision FROM api_snapshot WHERE source_table='project_resources'"
        ).fetchone() == before_snapshot
        assert conn.execute(
            "SELECT source_generation FROM pod_registry_state"
        ).fetchone() == before_generation
        assert conn.execute(
            "SELECT year_frcast, usf_oil_pod FROM pod_forecast_api WHERE pod_id_seq=? "
            "ORDER BY year_frcast",
            (pod_id_seq,),
        ).fetchall() == before_outputs
    finally:
        conn.close()



def test_rebuild_persists_snapshot_bound_diagnostics_by_pod_scope() -> None:
    conn = _connection()
    try:
        first = _insert_pod(conn, daily_slot=1)
        second = _insert_pod(conn, daily_slot=2)
        conn.commit()
        read_models = importlib.import_module("esdc.pod_registry.read_models")
        conn.execute("BEGIN IMMEDIATE")
        report = read_models.rebuild_pod_read_models(conn)
        assert any(
            diagnostic.code == "project_mapping_missing"
            and diagnostic.pod_id_seq == first
            for diagnostic in report.diagnostics
        )
        assert any(
            diagnostic.code == "project_mapping_missing"
            and diagnostic.pod_id_seq == second
            for diagnostic in report.diagnostics
        )
        conn.commit()

        first_basis = conn.execute(
            "SELECT source_generation, resource_revision, resource_sha256, "
            "timeseries_revision, timeseries_sha256, code, pod_id_seq "
            "FROM pod_read_model_diagnostic WHERE code='project_mapping_missing' "
            "AND pod_id_seq=?",
            (first,),
        ).fetchone()
        assert first_basis == (
            0,
            None,
            None,
            None,
            None,
            "project_mapping_missing",
            first,
        )

        conn.execute(
            "INSERT INTO m_project (project_id, project_name, is_active) "
            "VALUES ('P-2', 'Project 2', 1)"
        )
        conn.execute(
            "INSERT INTO pod_project (pod_project_uuid, pod_id_seq, project_id, "
            "is_main_project, is_auto_inferred) VALUES ('map-2', ?, 'P-2', 1, 0)",
            (second,),
        )
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")
        read_models.rebuild_pod_read_models(conn, pod_ids=(second,))
        rows = conn.execute(
            "SELECT DISTINCT pod_id_seq FROM pod_read_model_diagnostic "
            "WHERE code='project_mapping_missing' ORDER BY pod_id_seq"
        ).fetchall()
        assert rows == [(first,)]
        assert conn.execute(
            "SELECT source_generation FROM pod_read_model_snapshot WHERE singleton_id=1"
        ).fetchone() == (0,)
        conn.rollback()
    finally:
        conn.close()

def test_scoped_source_change_carries_forward_valid_unselected_diagnostics(tmp_path):
    source = tmp_path / "source.sqlite"
    conn = sqlite3.connect(source)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    importlib.import_module("esdc.pod_registry.migrations").initialize_registry_v2(conn)
    first = _insert_pod(conn, daily_slot=1)
    second = _insert_pod(conn, daily_slot=2)
    conn.commit()

    read_models = importlib.import_module("esdc.pod_registry.read_models")
    conn.execute("BEGIN IMMEDIATE")
    read_models.rebuild_pod_read_models(conn)
    conn.commit()
    before = importlib.import_module(
        "esdc.pod_registry.publication"
    ).read_pod_read_model_diagnostics(source)
    assert any(
        row["pod_id_seq"] == second and row["code"] == "project_mapping_missing"
        for row in before
    )

    conn.execute("BEGIN IMMEDIATE")
    conn.execute(
        "UPDATE m_pod SET pod_name='Updated first POD' WHERE pod_id_seq=?",
        (first,),
    )
    read_models.refresh_pod_after_source_change(
        conn,
        actor="FA:test-scoped-diagnostic-generation",
        pod_ids=(first,),
    )
    conn.commit()

    after = importlib.import_module(
        "esdc.pod_registry.publication"
    ).read_pod_read_model_diagnostics(source)
    assert after
    assert all(row["source_generation"] == 1 for row in after)
    assert any(
        row["pod_id_seq"] == second and row["code"] == "project_mapping_missing"
        for row in after
    )
    conn.close()


def test_source_change_invalidation_and_rebuild_failure_roll_back_together(
    tmp_path, monkeypatch
):
    from esdc.knowledge.freshness import (
        compute_knowledge_basis,
        ensure_knowledge_state_table,
    )
    from esdc.knowledge.store import Claim, Edge, KnowledgeStore

    path = tmp_path / "rollback.sqlite"
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    importlib.import_module("esdc.pod_registry.migrations").initialize_registry_v2(
        conn
    )
    pod_id_seq = _insert_pod(conn)
    conn.commit()

    read_models = importlib.import_module("esdc.pod_registry.read_models")
    conn.execute("BEGIN IMMEDIATE")
    read_models.rebuild_pod_read_models(conn)
    conn.commit()

    store = KnowledgeStore(conn, allow_stale=True)
    store.ensure_tables()
    store.upsert_edges(
        [Edge("document", "DOC-1", "ABOUT_POD", "pod", "POD-1", method="llm_extracted")]
    )
    store.replace_claims(
        "DOC-1",
        [Claim("DOC-1", "issue", "delay_cause", "equipment", "pod", "POD-1")],
    )
    store.bump_proposal("claim_type", "issue_type", "DOC-1")
    store.mark_learned("DOC-1", "old-hash")
    ensure_knowledge_state_table(conn)
    basis = compute_knowledge_basis(conn)
    conn.execute(
        "INSERT INTO pod_knowledge_state "
        "(singleton_id,basis_sha256,source_generation) VALUES (1,?,0) "
        "ON CONFLICT(singleton_id) DO UPDATE SET basis_sha256=excluded.basis_sha256",
        (basis,),
    )
    conn.execute(
        "UPDATE pod_registry_state SET knowledge_generation=0 WHERE singleton_id=1"
    )
    conn.commit()

    old_name = conn.execute(
        "SELECT pod_name FROM m_pod WHERE pod_id_seq=?", (pod_id_seq,)
    ).fetchone()[0]
    old_state = conn.execute(
        "SELECT source_generation, knowledge_generation FROM pod_registry_state"
    ).fetchone()
    old_outputs = conn.execute(
        "SELECT * FROM pod_monitoring_api ORDER BY pod_id_seq, reported_date"
    ).fetchall()
    old_diagnostics = conn.execute(
        "SELECT source_generation, code, message, pod_id_seq, reported_date "
        "FROM pod_read_model_diagnostic ORDER BY diagnostic_id"
    ).fetchall()

    original_persist = read_models._persist_rebuild_diagnostics

    def persist_then_fail(*args, **kwargs):
        original_persist(*args, **kwargs)
        raise RuntimeError("forced failure after read-model diagnostics write")

    monkeypatch.setattr(read_models, "_persist_rebuild_diagnostics", persist_then_fail)
    conn.execute("BEGIN IMMEDIATE")
    conn.execute(
        "UPDATE m_pod SET pod_name='Uncommitted changed name' WHERE pod_id_seq=?",
        (pod_id_seq,),
    )
    with pytest.raises(RuntimeError, match="forced failure"):
        read_models.refresh_pod_after_source_change(
            conn, actor="FA:test-rebuild-rollback"
        )
    assert conn.in_transaction
    assert conn.execute("SELECT COUNT(*) FROM kg_edge").fetchone()[0] == 0
    conn.rollback()

    assert conn.execute(
        "SELECT pod_name FROM m_pod WHERE pod_id_seq=?", (pod_id_seq,)
    ).fetchone()[0] == old_name
    assert conn.execute(
        "SELECT source_generation, knowledge_generation FROM pod_registry_state"
    ).fetchone() == old_state
    assert conn.execute("SELECT * FROM pod_monitoring_api ORDER BY pod_id_seq, reported_date").fetchall() == old_outputs
    assert conn.execute(
        "SELECT source_generation, code, message, pod_id_seq, reported_date "
        "FROM pod_read_model_diagnostic ORDER BY diagnostic_id"
    ).fetchall() == old_diagnostics
    assert conn.execute("SELECT COUNT(*) FROM kg_edge").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM kg_claim").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM kg_proposal").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM learn_state").fetchone()[0] == 1
    conn.close()



def test_numeric_only_source_change_carries_current_knowledge_generation():
    from esdc.knowledge.freshness import (
        compute_knowledge_basis,
        ensure_knowledge_state_table,
        knowledge_freshness,
    )
    from esdc.knowledge.store import Edge, KnowledgeStore
    from esdc.pod_registry.read_models import refresh_pod_after_source_change

    conn = _connection()
    try:
        _insert_pod(conn)
        _install_snapshot_tables(conn)
        conn.commit()
        pod_id = conn.execute("SELECT pod_id FROM m_pod").fetchone()[0]
        store = KnowledgeStore(conn, allow_stale=True)
        store.ensure_tables()
        store.upsert_edges(
            [
                Edge(
                    "document",
                    "DOC-1",
                    "ABOUT_POD",
                    "pod",
                    pod_id,
                    method="llm_extracted",
                )
            ]
        )
        ensure_knowledge_state_table(conn)
        basis = compute_knowledge_basis(conn)
        conn.execute(
            "INSERT INTO pod_knowledge_state "
            "(singleton_id, basis_sha256, source_generation) VALUES (1, ?, 0)",
            (basis,),
        )
        conn.execute(
            "UPDATE pod_registry_state SET knowledge_generation=0 WHERE singleton_id=1"
        )
        conn.commit()

        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "UPDATE pod_monitoring SET capital_drilling_real='9.876' "
            "WHERE pod_id_seq=(SELECT pod_id_seq FROM m_pod LIMIT 1)"
        )
        refresh_pod_after_source_change(conn, actor="FA:test-numeric-only-refresh")
        conn.commit()

        assert conn.execute(
            "SELECT source_generation, knowledge_generation FROM pod_registry_state"
        ).fetchone() == (1, 1)
        assert conn.execute("SELECT COUNT(*) FROM kg_edge").fetchone()[0] == 1
        assert knowledge_freshness(conn).fresh
    finally:
        conn.close()
