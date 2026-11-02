"""Task 4 input normalization, staging, and workbook template regressions."""

import hashlib
import sqlite3
from dataclasses import replace
from decimal import Decimal
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from uuid import uuid4

import pytest
from openpyxl import Workbook, load_workbook


def _create_sqlite_source(
    path: Path,
    contract,
    *,
    extra_project_column=False,
    omit_forecast=False,
) -> None:
    conn = sqlite3.connect(path)
    try:
        for table_name in (
            "m_project",
            "m_pod",
            "pod_project",
            "pod_monitoring",
            "pod_forecast",
        ):
            if table_name == "pod_forecast" and omit_forecast:
                continue
            columns = [column.name for column in contract.tables[table_name].columns]
            if table_name == "m_project" and extra_project_column:
                columns.append("unsupported")
            ddl_columns = ", ".join(f'"{column}"' for column in columns)
            conn.execute(f'CREATE TABLE "{table_name}" ({ddl_columns})')
        conn.execute(
            "CREATE TABLE _pod_identity_ledger (approval_year, approval_date, "
            "daily_slot, pod_order_no, pod_id_seq, pod_id, issued_at, issued_by)"
        )
        conn.execute("PRAGMA user_version=1")
        if omit_forecast:
            conn.execute("CREATE VIEW pod_forecast_api AS SELECT 1 AS marker")
        resources = [
            name
            for name in import_module(
                "esdc.api_snapshot"
            ).ApiProjectResourceRow.__dataclass_fields__
            if name != "groovy_report"
        ]
        resources.extend(["id", "uuid", "project_uuid", "project_id_canonical"])
        conn.execute(
            "CREATE TABLE project_resources ("
            + ", ".join(f'"{name}"' for name in resources)
            + ")"
        )
        timeseries = list(
            import_module("esdc.timeseries_snapshot").TIMESERIES_SOURCE_COLUMNS
        )
        timeseries.extend(["id", "uuid", "project_id_canonical"])
        conn.execute(
            "CREATE TABLE project_timeseries ("
            + ", ".join(f'"{name}"' for name in timeseries)
            + ")"
        )
        conn.commit()
    finally:
        conn.close()


def _required_project_row(contract):
    row = {}
    for column in contract.tables["m_project"].columns:
        if column.name == "project_id":
            row[column.name] = "p001459f01"
        elif column.nullable:
            row[column.name] = None
        elif column.logical_type == "text":
            row[column.name] = "Example"
        elif column.logical_type == "integer":
            row[column.name] = 1
        elif column.logical_type == "decimal":
            row[column.name] = "1.25"
        elif column.logical_type == "boolean":
            row[column.name] = 1
        elif column.logical_type == "date":
            row[column.name] = "2024-01-01"
        elif column.logical_type == "datetime":
            row[column.name] = "2024-01-01T00:00:00"
        elif column.logical_type == "uuid":
            row[column.name] = str(uuid4())
        else:
            raise AssertionError(column.name)
    return row


def test_shared_pod_import_adapter_is_available() -> None:
    spec = find_spec("esdc.pod_registry.sqlite_import")

    assert spec is not None, "Task 4 must add the shared SQLite/XLSX import adapter"


def test_project_id_normalizer_preserves_alphanumeric_canonical_identity() -> None:
    spec = find_spec("esdc.pod_registry.normalization")
    assert spec is not None, "Task 4 must add shared project ID normalization"
    normalize_project_id = import_module(
        "esdc.pod_registry.normalization"
    ).normalize_project_id

    assert normalize_project_id("p001459f01") == "P-001459F-01"
    assert normalize_project_id("P-001459F-01") == "P-001459F-01"


def test_project_id_normalizer_rejects_unrecognized_forms() -> None:
    spec = find_spec("esdc.pod_registry.normalization")
    assert spec is not None, "Task 4 must add shared project ID normalization"
    normalize_project_id = import_module(
        "esdc.pod_registry.normalization"
    ).normalize_project_id

    with pytest.raises(ValueError, match="canonical project ID"):
        normalize_project_id("P-001459-F01")


def test_sqlite_prepare_accepts_blank_declared_types_and_preserves_project_id(
    tmp_path, monkeypatch
) -> None:
    contract = import_module("esdc.pod_registry.contract").load_pod_contract()
    adapter = import_module("esdc.pod_registry.sqlite_import")
    source = tmp_path / "source.sqlite"
    target = tmp_path / "target.sqlite"
    _create_sqlite_source(source, contract)
    conn = sqlite3.connect(source)
    try:
        headers = [column.name for column in contract.tables["m_project"].columns]
        row = _required_project_row(contract)
        conn.execute(
            "INSERT INTO m_project VALUES (" + ",".join("?" for _ in headers) + ")",
            [row[name] for name in headers],
        )
        conn.commit()
    finally:
        conn.close()

    batch = adapter.prepare_pod_import(
        source,
        "sqlite",
        contract,
        target_path=target,
        stage_snapshots=False,
    )
    assert batch.rows["m_project"][0]["project_id"] == "P-001459F-01"
    assert batch.profile == "full"
    assert not target.exists()



def test_sqlite_master_only_preserves_target_project_snapshots(tmp_path) -> None:
    contract = import_module("esdc.pod_registry.contract").load_pod_contract()
    adapter = import_module("esdc.pod_registry.sqlite_import")
    migrations = import_module("esdc.pod_registry.migrations")
    source = tmp_path / "master-source.sqlite"
    target = tmp_path / "target.sqlite"
    _create_sqlite_source(source, contract)

    project = _required_project_row(contract)
    project["project_id"] = "P-001459F-01"
    columns = [column.name for column in contract.tables["m_project"].columns]
    with sqlite3.connect(source) as source_conn:
        source_conn.execute(
            "INSERT INTO m_project VALUES (" + ",".join("?" for _ in columns) + ")",
            [migrations._sqlite_value(project[name]) for name in columns],
        )

    store = import_module("esdc.pod_registry.store")
    timeseries = import_module("esdc.timeseries_snapshot")
    snapshot = import_module("esdc.api_snapshot")
    with sqlite3.connect(target) as target_conn:
        target_conn.execute("PRAGMA foreign_keys=ON")
        migrations.initialize_registry_v2(target_conn)
        target_conn.execute(
            "INSERT INTO m_project VALUES ("
            + ",".join("?" for _ in columns)
            + ")",
            [migrations._sqlite_value(project[name]) for name in columns],
        )
        store.ensure_project_resources_table(target_conn)
        adapter._ensure_snapshot_metadata_table(target_conn)
        timeseries.ensure_timeseries_tables(target_conn)

        resource_columns = list(snapshot.ApiProjectResourceRow.__dataclass_fields__)
        resource_values: list[str | None] = [None] * len(resource_columns)
        resource_values[resource_columns.index("project_id")] = "P-001459F-01"
        resource_values[resource_columns.index("project_name")] = "Target snapshot"
        target_conn.execute(
            "INSERT INTO api_project_resources ("
            + ",".join(f'"{name}"' for name in resource_columns)
            + ") VALUES ("
            + ",".join("?" for _ in resource_columns)
            + ")",
            resource_values,
        )
        target_conn.execute(
            "INSERT INTO api_snapshot VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "project_resources",
                "resource-revision",
                1,
                "resource-hash",
                "2024-01-01T00:00:00Z",
                "published",
                0,
            ),
        )
        target_conn.execute(
            "INSERT INTO api_project_timeseries "
            "(report_year, row_json, row_sha256, origin, schema_version) "
            "VALUES (2024, ?, ?, 'api_json', 1)",
            ('{"project_id":"P-001459F-01","marker":"target"}', "timeseries-hash"),
        )
        timeseries.record_timeseries_provenance(
            target_conn,
            revision="timeseries-revision",
            origin="api_csv",
            details={"marker": "target"},
        )

    batch = adapter.prepare_pod_import(
        source,
        "sqlite",
        contract,
        profile="master-only",
        target_path=target,
    )
    try:
        assert batch.staged_snapshots is None
        with sqlite3.connect(target) as target_conn:
            target_conn.execute("PRAGMA foreign_keys=ON")
            target_conn.execute("BEGIN")
            adapter.apply_pod_batch_in_transaction(
                target_conn,
                batch,
                "FA:test-master-only",
                contract=contract,
                refresh=False,
            )

        with sqlite3.connect(target) as check:
            assert check.execute(
                "SELECT project_name FROM api_project_resources"
            ).fetchall() == [("Target snapshot",)]
            assert check.execute(
                "SELECT source_table, revision, source_sha256 "
                "FROM api_snapshot ORDER BY source_table"
            ).fetchall() == [
                ("project_resources", "resource-revision", "resource-hash")
            ]
            assert check.execute(
                "SELECT row_json, row_sha256, origin "
                "FROM api_project_timeseries"
            ).fetchall() == [
                ('{"project_id":"P-001459F-01","marker":"target"}', "timeseries-hash", "api_json")
            ]
            assert check.execute(
                "SELECT source_table, revision, origin, details_json "
                "FROM api_snapshot_provenance"
            ).fetchall() == [
                (
                    "project_timeseries",
                    "timeseries-revision",
                    "api_csv",
                    '{"marker":"target"}',
                )
            ]

        unexpected_stage_path = tmp_path / "unexpected-resource-stage.jsonl"
        unexpected_stage_path.write_text("", encoding="utf-8")
        unexpected_stage = adapter.ProjectSnapshotStage(
            resources_path=unexpected_stage_path,
            resource_row_count=0,
            resources_sha256=hashlib.sha256(b"[]").hexdigest(),
        )
        batch_with_snapshot = replace(batch, staged_snapshots=unexpected_stage)
        with pytest.raises(
            adapter.PodImportError,
            match="master-only profile cannot apply project snapshots",
        ), sqlite3.connect(target) as target_conn:
            target_conn.execute("BEGIN")
            adapter.apply_pod_batch_in_transaction(
                target_conn,
                batch_with_snapshot,
                "FA:test-master-only",
                contract=contract,
                refresh=False,
            )
        with sqlite3.connect(target) as check:
            assert check.execute(
                "SELECT project_name FROM api_project_resources"
            ).fetchall() == [("Target snapshot",)]
    finally:
        batch.cleanup_staging()

def test_sqlite_prepare_rejects_unknown_columns_before_writer(tmp_path) -> None:
    contract = import_module("esdc.pod_registry.contract").load_pod_contract()
    adapter = import_module("esdc.pod_registry.sqlite_import")
    source = tmp_path / "source.sqlite"
    target = tmp_path / "target.sqlite"
    _create_sqlite_source(source, contract, extra_project_column=True)

    with pytest.raises(adapter.PodImportError, match="unsupported"):
        adapter.prepare_pod_import(
            source,
            "sqlite",
            contract,
            target_path=target,
            stage_snapshots=False,
        )
    assert not target.exists()


def test_sqlite_snapshot_stage_is_owned_and_cleaned(tmp_path) -> None:
    contract = import_module("esdc.pod_registry.contract").load_pod_contract()
    adapter = import_module("esdc.pod_registry.sqlite_import")
    source = tmp_path / "mockup.sqlite"
    target = tmp_path / "target.sqlite"
    _create_sqlite_source(source, contract, omit_forecast=True)
    before = sqlite3.connect(source).execute("PRAGMA journal_mode").fetchone()[0]

    batch = adapter.prepare_pod_import(source, "sqlite", contract, target_path=target)
    stage = batch.staged_snapshots
    assert stage is not None
    assert len(batch.rows["pod_forecast"]) == 0
    assert stage.resource_row_count == 0
    assert stage.timeseries_row_count == 0
    assert stage.resources_path is not None and stage.resources_path.exists()
    stage_directory = stage.resources_path.parent
    batch.cleanup_staging()
    assert not stage_directory.exists()
    check = sqlite3.connect(source)
    try:
        assert check.execute("PRAGMA journal_mode").fetchone()[0] == before
    finally:
        check.close()
    assert not target.exists()


def test_source_target_same_file_is_rejected_before_reading(tmp_path) -> None:
    adapter = import_module("esdc.pod_registry.sqlite_import")
    source = tmp_path / "source.sqlite"
    source.write_bytes(b"not a database")

    with pytest.raises(adapter.PodImportError, match="must not be"):
        adapter.prepare_pod_import(
            source,
            "sqlite",
            target_path=source,
            stage_snapshots=False,
        )


def test_decimal_and_double_normalization_is_exact_and_finite() -> None:
    normalization = import_module("esdc.pod_registry.normalization")
    contract = import_module("esdc.pod_registry.contract").load_pod_contract()
    fixed = contract.tables["m_pod"].columns[
        next(
            index
            for index, column in enumerate(contract.tables["m_pod"].columns)
            if column.name == "capital_drilling_pod"
        )
    ]
    double = next(
        column
        for column in contract.tables["m_pod"].columns
        if column.logical_type == "decimal" and "DOUBLE" in column.data_type.upper()
    )

    assert normalization.normalize_pod_value(
        "12.345", fixed, source="sqlite"
    ) == Decimal("12.345")
    with pytest.raises(normalization.PodNormalizationError, match="finite SQL DOUBLE"):
        normalization.normalize_pod_value("1e10000", double, source="sqlite")


def test_full_xlsx_profile_rejects_missing_forecast_sheet(tmp_path) -> None:
    contract = import_module("esdc.pod_registry.contract").load_pod_contract()
    adapter = import_module("esdc.pod_registry.sqlite_import")
    source = tmp_path / "incomplete.xlsx"
    workbook = Workbook()
    active_sheet = workbook.active
    if active_sheet is not None:
        workbook.remove(active_sheet)
    for table_name in ("m_project", "m_pod", "pod_project", "pod_monitoring"):
        workbook.create_sheet(table_name).append(
            [column.name for column in contract.tables[table_name].columns]
        )
    workbook.save(source)

    with pytest.raises(adapter.PodImportError, match="missing required POD sheet"):
        adapter.prepare_pod_import(
            source, "xlsx", contract, target_path=tmp_path / "target.sqlite"
        )


def test_import_template_contains_contract_inputs_outputs_and_metadata(
    tmp_path,
) -> None:
    contract = import_module("esdc.pod_registry.contract").load_pod_contract()
    adapter = import_module("esdc.pod_registry.sqlite_import")
    destination = tmp_path / "template.xlsx"
    adapter.generate_pod_import_template(destination, contract=contract)
    workbook = load_workbook(destination, data_only=True)
    try:
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
            for row in workbook["__pod_metadata"].iter_rows(min_row=2, values_only=True)
            if row[0] is not None
        }
        assert metadata == {
            "contract_id": contract.contract_id,
            "contract_version": contract.contract_version,
            "dictionary_version": contract.dictionary_version,
        }
    finally:
        workbook.close()


IDENTITY_UUID = "4aac3a88-10f4-4c74-87d5-12246ce43ca2"
IDENTITY_TABLES = ("m_project", "m_pod", "pod_project", "pod_monitoring", "pod_forecast")


def _identity_pod_row(contract, pod_id_seq, **overrides):
    row = {}
    for column in contract.tables["m_pod"].columns:
        if column.nullable:
            row[column.name] = None
        elif column.logical_type == "text":
            row[column.name] = "Identity test"
        elif column.logical_type == "integer":
            row[column.name] = 1
        elif column.logical_type == "decimal":
            row[column.name] = "1.25"
        elif column.logical_type == "boolean":
            row[column.name] = 1
        elif column.logical_type in ("date", "datetime"):
            row[column.name] = "1997-01-01"
        elif column.logical_type == "uuid":
            row[column.name] = IDENTITY_UUID
        else:
            raise AssertionError(column.name)
    row.update(
        {
            "pod_id_seq": pod_id_seq,
            "pod_uuid": IDENTITY_UUID,
            "approval_date": "1997-01-01",
            "pod_order_no": 1,
            "pod_id": "PL-1997-0001-4-1-0",
            "institution_id": 4,
            "pod_type_id": 1,
            "rev_num": 0,
            "pod_name": "Identity test pod",
            "is_latest": 1,
        }
    )
    row.update(overrides)
    return row


def _identity_batch(contract, models, rows_by_table, *, identity_ledger=None):
    headers = {
        table_name: tuple(
            column.name for column in contract.tables[table_name].columns
        )
        for table_name in IDENTITY_TABLES
    }
    rows = dict.fromkeys(IDENTITY_TABLES, ())
    for table_name, table_rows in rows_by_table.items():
        rows[table_name] = tuple(table_rows)
    return models.PreparedPodBatch(
        source="sqlite",
        contract_id=contract.contract_id,
        contract_version=contract.contract_version,
        dictionary_version=contract.dictionary_version,
        headers=headers,
        rows=rows,
        identity_ledger=identity_ledger,
        profile="full",
    )


def _ledger_row(pod_id_seq, actor, issued_at):
    return {
        "approval_year": 1997,
        "approval_date": "1997-01-01",
        "daily_slot": 1,
        "pod_order_no": 1,
        "pod_id_seq": pod_id_seq,
        "pod_id": "PL-1997-0001-4-1-0",
        "pod_uuid": IDENTITY_UUID,
        "institution_id": 4,
        "pod_type_id": 1,
        "rev_num": 0,
        "issued_by": actor,
        "issued_at": issued_at,
    }


def _identity_project_row(contract, project_name):
    row = _required_project_row(contract)
    row["project_id"] = "P-001459F-01"
    row["project_name"] = project_name
    return row


def _open_writer(tmp_path, name):
    migrations = import_module("esdc.pod_registry.migrations")
    runtime = import_module("esdc.sqlite_runtime")
    conn = sqlite3.connect(tmp_path / name)
    runtime.require_wal_runtime()
    runtime.configure_writer(conn)
    migrations.initialize_registry_v2(conn)
    return conn


def test_common_apply_keeps_a_known_ledger_reservation_audit(tmp_path) -> None:
    """Manual and API sources alternate without rewriting a committed identity."""
    contract = import_module("esdc.pod_registry.contract").load_pod_contract()
    adapter = import_module("esdc.pod_registry.sqlite_import")
    models = import_module("esdc.pod_registry.models")

    conn = _open_writer(tmp_path, "ledger-roundtrip.sqlite")

    def committed_ledger() -> tuple:
        return conn.execute(
            "SELECT approval_year, approval_date, daily_slot, pod_order_no, pod_id_seq, "
            "pod_id, pod_uuid, institution_id, pod_type_id, rev_num, actor, issued_at "
            "FROM _pod_identity_ledger WHERE pod_id_seq = 1"
        ).fetchone()

    def current_project_name() -> str:
        return conn.execute("SELECT project_name FROM m_project").fetchone()[0]

    def current_pod_name() -> str:
        return conn.execute("SELECT pod_name FROM m_pod WHERE pod_id_seq = 1").fetchone()[0]

    try:
        first_audit = ("FA:source-initial", "2024-01-01T00:00:00Z")
        first = adapter.apply_pod_batch(
            conn,
            _identity_batch(
                contract,
                models,
                {
                    "m_pod": [_identity_pod_row(contract, 1, pod_name="Identity pod manual one")],
                    "m_project": [_identity_project_row(contract, "Manual project one")],
                },
                identity_ledger=(_ledger_row(1, *first_audit),),
            ),
            "FA:manual",
        )
        assert first.source_generation == 1
        committed = committed_ledger()
        assert committed[10:] == first_audit
        assert current_project_name() == "Manual project one"

        # The API projection carries no ledger payload, so its reservation names the
        # identity that is already committed; ordinary input values still change.
        second = adapter.apply_pod_batch(
            conn,
            _identity_batch(
                contract,
                models,
                {
                    "m_pod": [_identity_pod_row(contract, 1, pod_name="Identity pod api one")],
                    "m_project": [_identity_project_row(contract, "API project one")],
                },
            ),
            "FA:fetch",
        )
        assert second.source_generation == 2
        assert committed_ledger() == committed
        assert current_project_name() == "API project one"
        assert current_pod_name() == "Identity pod api one"

        # A later manual source with a different valid audit keeps the committed audit.
        later_audit = ("FA:source-later", "2024-02-02T00:00:00Z")
        third = adapter.apply_pod_batch(
            conn,
            _identity_batch(
                contract,
                models,
                {
                    "m_pod": [_identity_pod_row(contract, 1, pod_name="Identity pod manual two")],
                    "m_project": [_identity_project_row(contract, "Manual project two")],
                },
                identity_ledger=(_ledger_row(1, *later_audit),),
            ),
            "FA:manual",
        )
        assert third.source_generation == 3
        assert committed_ledger() == committed
        assert committed_ledger()[10:] != later_audit
        assert current_project_name() == "Manual project two"

        fourth = adapter.apply_pod_batch(
            conn,
            _identity_batch(
                contract,
                models,
                {
                    "m_pod": [_identity_pod_row(contract, 1, pod_name="Identity pod api two")],
                    "m_project": [_identity_project_row(contract, "API project two")],
                },
            ),
            "FA:fetch",
        )
        assert fourth.source_generation == 4
        assert committed_ledger() == committed
        assert current_project_name() == "API project two"
        assert current_pod_name() == "Identity pod api two"
    finally:
        conn.close()


def test_new_ledger_reservation_keeps_the_provided_source_audit_exactly(
    tmp_path,
) -> None:
    contract = import_module("esdc.pod_registry.contract").load_pod_contract()
    adapter = import_module("esdc.pod_registry.sqlite_import")
    models = import_module("esdc.pod_registry.models")

    conn = _open_writer(tmp_path, "provided-audit.sqlite")
    try:
        result = adapter.apply_pod_batch(
            conn,
            _identity_batch(
                contract,
                models,
                {"m_pod": [_identity_pod_row(contract, 1)]},
                identity_ledger=(
                    _ledger_row(1, "FA:source-initial", "2024-01-01T00:00:00Z"),
                ),
            ),
            "FA:manual",
        )

        assert result.source_generation == 1
        assert conn.execute(
            "SELECT actor, issued_at FROM _pod_identity_ledger WHERE pod_id_seq=1"
        ).fetchone() == ("FA:source-initial", "2024-01-01T00:00:00Z")
    finally:
        conn.close()


def test_reservation_without_provided_audit_uses_the_local_generated_audit(
    tmp_path,
) -> None:
    contract = import_module("esdc.pod_registry.contract").load_pod_contract()
    adapter = import_module("esdc.pod_registry.sqlite_import")
    migrations = import_module("esdc.pod_registry.migrations")
    models = import_module("esdc.pod_registry.models")

    conn = _open_writer(tmp_path, "generated-audit.sqlite")
    try:
        adapter.apply_pod_batch(
            conn,
            _identity_batch(contract, models, {"m_pod": [_identity_pod_row(contract, 1)]}),
            "FA:fetch",
        )

        actor, issued_at = conn.execute(
            "SELECT actor, issued_at FROM _pod_identity_ledger WHERE pod_id_seq=1"
        ).fetchone()
        assert actor == migrations.MIGRATION_ACTOR
        assert issued_at
    finally:
        conn.close()


def test_api_first_reservation_keeps_its_generated_audit_for_a_later_manual_source(
    tmp_path,
) -> None:
    """A locally issued reservation is never re-attributed by a later source."""
    contract = import_module("esdc.pod_registry.contract").load_pod_contract()
    adapter = import_module("esdc.pod_registry.sqlite_import")
    migrations = import_module("esdc.pod_registry.migrations")
    models = import_module("esdc.pod_registry.models")

    conn = _open_writer(tmp_path, "api-first-audit.sqlite")
    try:
        adapter.apply_pod_batch(
            conn,
            _identity_batch(contract, models, {"m_pod": [_identity_pod_row(contract, 1)]}),
            "FA:fetch",
        )
        generated = conn.execute(
            "SELECT actor, issued_at FROM _pod_identity_ledger WHERE pod_id_seq=1"
        ).fetchone()
        assert generated[0] == migrations.MIGRATION_ACTOR
        assert generated[1]

        adapter.apply_pod_batch(
            conn,
            _identity_batch(
                contract,
                models,
                {"m_pod": [_identity_pod_row(contract, 1)]},
                identity_ledger=(_ledger_row(1, "FA:source-later", "2024-03-03T00:00:00Z"),),
            ),
            "FA:manual",
        )

        assert conn.execute(
            "SELECT actor, issued_at FROM _pod_identity_ledger WHERE pod_id_seq=1"
        ).fetchone() == generated
    finally:
        conn.close()


def test_business_identity_conflicts_still_reject_atomically(tmp_path) -> None:
    """A known business identity that differs from the committed one is a conflict."""
    contract = import_module("esdc.pod_registry.contract").load_pod_contract()
    adapter = import_module("esdc.pod_registry.sqlite_import")
    api_context = import_module("esdc.pod_registry.api_context")
    models = import_module("esdc.pod_registry.models")

    conn = _open_writer(tmp_path, "identity-conflict.sqlite")

    def committed_ledger() -> tuple:
        return conn.execute(
            "SELECT approval_year, approval_date, daily_slot, pod_order_no, pod_id_seq, "
            "pod_id, pod_uuid, institution_id, pod_type_id, rev_num, actor, issued_at "
            "FROM _pod_identity_ledger WHERE pod_id_seq = 1"
        ).fetchone()

    def committed_master() -> tuple:
        return conn.execute(
            "SELECT pod_id_seq, pod_uuid, approval_date, pod_order_no, pod_id, "
            "institution_id, pod_type_id, rev_num FROM m_pod WHERE pod_id_seq = 1"
        ).fetchone()

    try:
        adapter.apply_pod_batch(
            conn,
            _identity_batch(
                contract,
                models,
                {
                    "m_pod": [_identity_pod_row(contract, 1)],
                    "m_project": [_identity_project_row(contract, "Manual project one")],
                },
                identity_ledger=(_ledger_row(1, "FA:source-initial", "2024-01-01T00:00:00Z"),),
            ),
            "FA:manual",
        )
        ledger_before = committed_ledger()
        master_before = committed_master()
        assert ledger_before[10:] == ("FA:source-initial", "2024-01-01T00:00:00Z")

        # Every variant is internally coherent, so the only problem is that it names a
        # different business identity than the permanent ledger already holds.
        variants = {
            "pod_uuid": (1, {"pod_uuid": "6a7b8c9d-0e0e-40e0-80e0-000000000000"}, "pod_uuid"),
            "pod_id_seq": (
                2,
                {},
                "identity ledger reservation already exists",
            ),
            "daily_slot": (
                1,
                {"pod_order_no": 2, "pod_id": "PL-1997-0002-4-1-0"},
                "daily_slot",
            ),
            "approval_date": (
                1,
                {
                    "approval_date": "1997-01-02",
                    "pod_order_no": 21,
                    "pod_id": "PL-1997-0021-4-1-0",
                },
                "approval_date",
            ),
            "pod_id": (1, {"institution_id": 2, "pod_id": "PL-1997-0001-2-1-0"}, "pod_id"),
        }

        for field, (pod_id_seq, overrides, message) in variants.items():
            row = _identity_pod_row(contract, pod_id_seq, **overrides)
            if field == "pod_id_seq":
                # The identity is already issued for sequence 1, so the strict insert
                # still refuses the second reservation through the uniqueness guards.
                expected: tuple[type[BaseException], ...] = (sqlite3.IntegrityError,)
            else:
                expected = (api_context.PodImportError,)
            with pytest.raises(expected, match=message):
                adapter.apply_pod_batch(
                    conn,
                    _identity_batch(contract, models, {"m_pod": [row]}),
                    "FA:fetch",
                )

            # Nothing from the rejected apply survives: the ledger and the issued
            # master row keep the committed values.
            assert committed_ledger() == ledger_before
            assert committed_master() == master_before
    finally:
        conn.close()


def test_direct_migration_insert_stays_strict_for_a_known_reservation(tmp_path) -> None:
    """Only the import apply reconciles; the migration helper still refuses actor drift."""
    contract = import_module("esdc.pod_registry.contract").load_pod_contract()
    adapter = import_module("esdc.pod_registry.sqlite_import")
    migrations = import_module("esdc.pod_registry.migrations")
    models = import_module("esdc.pod_registry.models")

    conn = _open_writer(tmp_path, "strict-helper.sqlite")
    try:
        adapter.apply_pod_batch(
            conn,
            _identity_batch(
                contract,
                models,
                {"m_pod": [_identity_pod_row(contract, 1)]},
                identity_ledger=(_ledger_row(1, "FA:source-initial", "2024-01-01T00:00:00Z"),),
            ),
            "FA:manual",
        )
        committed = conn.execute(
            "SELECT actor, issued_at FROM _pod_identity_ledger WHERE pod_id_seq=1"
        ).fetchone()
        assert committed == ("FA:source-initial", "2024-01-01T00:00:00Z")

        reservation = migrations._IdentityLedgerReservation(
            approval_year=1997,
            approval_date="1997-01-01",
            daily_slot=1,
            pod_order_no=1,
            pod_id_seq=1,
            pod_id="PL-1997-0001-4-1-0",
            pod_uuid=IDENTITY_UUID,
            institution_id=4,
            pod_type_id=1,
            rev_num=0,
            actor="FA:other-action",
            issued_at="2024-06-06T00:00:00Z",
        )
        with pytest.raises(migrations.MigrationRefusedError, match="conflicts"):
            migrations._insert_identity_ledger(conn, (reservation,))

        assert conn.execute(
            "SELECT actor, issued_at FROM _pod_identity_ledger WHERE pod_id_seq=1"
        ).fetchone() == committed
    finally:
        conn.close()
