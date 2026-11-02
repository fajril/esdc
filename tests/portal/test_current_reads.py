"""Current portal reads preserve schema, snapshot and exact numeric semantics."""

import json
import sqlite3

import pytest

from esdc.portal.data import (
    build_query,
    finite_decimal,
    register_decimal,
    serialize_row,
    strict_integer,
    validate_page_size,
)


def test_exact_query():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(
            "CREATE TABLE sample(id INTEGER PRIMARY KEY, amount TEXT, label TEXT, enabled INTEGER)"
        )
        rows = [
            (1, "9007199254740993.01", "a_%", 0),
            (2, "9007199254740993.00", "axb", 1),
            (3, "0", "zero", 0),
            (4, None, "none", None),
        ]
        conn.executemany("INSERT INTO sample VALUES (?, ?, ?, ?)", rows)

        metadata = {
            "columns": [
                {"name": "id", "kind": "integer", "storage_type": "BIGINT"},
                {"name": "amount", "kind": "decimal", "storage_type": "NUMERIC(25,2)"},
                {"name": "label", "kind": "text", "storage_type": "TEXT"},
                {"name": "enabled", "kind": "boolean", "storage_type": "BOOLEAN"},
            ],
            "key_fields": ["id"],
        }

        register_decimal(conn)

        def helper(ids_filters=None, sort=None):
            sql, _, params = build_query("sample", metadata, ids_filters, sort, "asc")
            return [row["id"] for row in conn.execute(sql, params).fetchall()]

        assert helper(sort="amount") == [4, 3, 2, 1]
        assert helper(
            [{"field": "amount", "op": "gt", "value": "9007199254740993.00"}]
        ) == [1]
        assert helper([{"field": "amount", "op": "eq", "value": "0"}]) == [3]
        assert helper([{"field": "amount", "op": "is_null", "value": True}]) == [4]
        assert helper([{"field": "enabled", "op": "eq", "value": False}]) == [1, 3]
        assert helper([{"field": "label", "op": "contains", "value": "_%"}]) == [1]

        sql, _, params = build_query(
            "sample", metadata, [{"field": "id", "op": "eq", "value": 1}], None, "asc"
        )
        row = conn.execute(sql, params).fetchone()
        serialized = serialize_row(row, metadata)
        assert serialized["amount"] == "9007199254740993.01"
        assert serialized["enabled"] is False
        json_str = json.dumps(serialized)
        parsed = json.loads(json_str)
        assert parsed["id"] == 1
        large = 9007199254740993
        row = conn.execute(
            "SELECT ? AS id, '0' AS amount, 'big' AS label, 1 AS enabled", (large,)
        ).fetchone()
        serialized = serialize_row(row, metadata)
        assert serialized["id"] == str(large)
        assert json.loads(serialized["_row_key"]) == [large]
    finally:
        conn.close()


def test_validation():
    for bad_page in [(True, 10), (0, 10), (1, 201)]:
        with pytest.raises(ValueError):
            validate_page_size(*bad_page)

    with pytest.raises(ValueError):
        strict_integer("1\n")

    for val in ["NaN", "Inf", 1.5, True]:
        with pytest.raises(ValueError):
            finite_decimal(val)


def test_injection():
    metadata = {
        "columns": [
            {"name": "id", "kind": "integer", "storage_type": "BIGINT"},
        ],
        "key_fields": ["id"],
    }

    with pytest.raises(ValueError):
        build_query(
            "sample",
            metadata,
            [{"field": "unknown_field", "op": "eq", "value": 1}],
            None,
            "asc",
        )

    with pytest.raises(ValueError):
        build_query("sample", metadata, None, "unknown_sort", "asc")

    with pytest.raises(ValueError):
        build_query(
            "sample",
            metadata,
            [{"field": "id", "op": "unknown_op", "value": 1}],
            None,
            "asc",
        )

    with pytest.raises(ValueError):
        build_query("sample", metadata, None, "id", "invalid_direction")


def test_current_metadata_and_missing_path(tmp_path):
    from esdc.portal.data import read_table_page
    from esdc.portal.tables import TABLE_CONFIGS

    assert len(TABLE_CONFIGS["m_pod"]["columns"]) == 82
    assert len(TABLE_CONFIGS["pod_monitoring"]["columns"]) == 92
    assert len(TABLE_CONFIGS["pod_monitoring_api"]["columns"]) == 176
    assert len(TABLE_CONFIGS["pod_forecast_api"]["columns"]) == 24
    assert TABLE_CONFIGS["pod_document"]["pk"] == ["pod_id_seq", "doc_id"]
    for table in (
        "r_institution",
        "r_pod_type",
        "r_pod_stage",
        "pod_monitoring_api",
        "pod_forecast_api",
    ):
        assert TABLE_CONFIGS[table]["readonly"]
        assert all(column["readonly"] for column in TABLE_CONFIGS[table]["columns"])
    path = tmp_path / "absent" / "registry.sqlite"
    result = read_table_page("m_pod", sqlite_path=path)
    assert result["initialized"] is False
    assert result["total"] == 0
    assert not path.parent.exists()


@pytest.fixture
def current_registry(tmp_path):
    from esdc.pod_registry.migrations import initialize_registry_v2
    from esdc.sqlite_runtime import configure_writer

    path = tmp_path / "current.sqlite"
    conn = sqlite3.connect(path)
    configure_writer(conn)
    conn.execute("PRAGMA foreign_keys=ON")
    initialize_registry_v2(conn)
    conn.executemany(
        "INSERT INTO m_project(project_id,project_name,is_active) VALUES (?,?,?)",
        [
            ("inactive", "Unmapped inactive", 0),
            ("percent_%", "Literal", 1),
            ("active", "Active", 1),
        ],
    )
    from datetime import date

    from esdc.pod_registry.identity import ApprovalKey, issue_identity

    for i, amount in enumerate(
        ("9007199254740993.01", "9007199254740993.00", "0", None), 1
    ):
        issued = issue_identity(
            conn,
            key=ApprovalKey(date(2026, 1, 1), i, 1, 1, 0),
            actor="FA:test-current-read",
        )
        conn.execute(
            "INSERT INTO m_pod(pod_id_seq,pod_uuid,approval_date,institution_id,"
            "pod_type_id,rev_num,pod_order_no,pod_id,pod_name,pod_letter_num,"
            "is_latest,capital_drilling_pod) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                issued.pod_id_seq,
                str(issued.pod_uuid),
                issued.approval_date.isoformat(),
                1,
                1,
                0,
                issued.pod_order_no,
                issued.pod_id,
                f"Pod {i}",
                f"Letter {i}",
                1,
                amount,
            ),
        )
    conn.executemany(
        "INSERT INTO pod_forecast_api(pod_id_seq,reported_date,year_frcast) VALUES (?,?,?)",
        [(2, "2026-01-01", 2030), (1, "2026-01-01", 2031), (1, "2026-01-01", 2030)],
    )
    conn.commit()
    conn.close()
    return path


def test_current_snapshot_lookup_and_grain(current_registry, monkeypatch):
    import esdc.portal.data as data

    statements = []
    original = data.read_only_connection

    def guarded(path):
        conn = original(path)

        def authorize(action, *args):
            assert action not in {
                sqlite3.SQLITE_INSERT,
                sqlite3.SQLITE_UPDATE,
                sqlite3.SQLITE_DELETE,
                sqlite3.SQLITE_CREATE_TABLE,
                sqlite3.SQLITE_DROP_TABLE,
            }
            return sqlite3.SQLITE_OK

        conn.set_authorizer(authorize)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(data, "read_only_connection", guarded)
    first = data.read_table_page(
        "pod_forecast_api", sqlite_path=current_registry, size=2
    )
    second = data.read_table_page(
        "pod_forecast_api", sqlite_path=current_registry, size=2, page=2
    )
    assert first["total"] == 3 and first["last_page"] == 2
    assert [json.loads(row["_row_key"]) for row in first["rows"] + second["rows"]] == [
        [1, "2026-01-01", 2030],
        [1, "2026-01-01", 2031],
        [2, "2026-01-01", 2030],
    ]
    assert all(row["frcast_id"] is None for row in first["rows"])
    assert (
        data.lookup_projects(sqlite_path=current_registry, q="inactive")[0][
            "project_id"
        ]
        == "inactive"
    )
    assert [
        row["project_id"]
        for row in data.lookup_projects(sqlite_path=current_registry, q="_%")
    ] == ["percent_%"]
    assert set(data.read_reference_options(sqlite_path=current_registry)) == {
        "pods",
        "r_institution",
        "r_pod_type",
        "r_pod_stage",
    }
    assert any("LIMIT 2 OFFSET 2" in statement for statement in statements)


def test_incompatible_existing_refuses(tmp_path):
    from esdc.portal.data import read_table_page

    path = tmp_path / "partial.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE m_pod(id INTEGER)")
    conn.close()
    with pytest.raises(ValueError, match="load a validated current dataset"):
        read_table_page("m_pod", sqlite_path=path)


def test_current_exact_numeric(current_registry):
    from esdc.portal.data import read_table_page

    page = read_table_page(
        "m_pod", sqlite_path=current_registry, sort="capital_drilling_pod"
    )
    assert [row["pod_id_seq"] for row in page["rows"]] == [4, 3, 2, 1]
    assert page["rows"][-1]["capital_drilling_pod"] == "9007199254740993.01"
    page = read_table_page(
        "m_pod",
        sqlite_path=current_registry,
        filters=[
            {
                "field": "capital_drilling_pod",
                "op": "gt",
                "value": "9007199254740993.00",
            }
        ],
    )
    assert [row["pod_id_seq"] for row in page["rows"]] == [1]


def test_ownership_and_full_metadata():
    from esdc.portal.tables import TABLE_CONFIGS

    required = {
        "field",
        "title",
        "logical_type",
        "data_type",
        "nullable",
        "unit",
        "group",
        "description",
        "readonly",
        "editable",
        "editableOnNew",
        "required",
        "editor",
        "ref",
    }
    for config in TABLE_CONFIGS.values():
        for column in config["columns"]:
            assert required <= column.keys()
    pod = {column["field"]: column for column in TABLE_CONFIGS["m_pod"]["columns"]}
    for field in ("approval_date", "institution_id", "pod_type_id", "rev_num"):
        assert pod[field]["editableOnNew"] and not pod[field]["editable"]
    for field in (
        "pod_uuid",
        "is_latest",
        "pod_id_seq",
        "pod_id",
        "pod_order_no",
        "created_at",
    ):
        assert pod[field]["readonly"]
    assert pod["superseded_by"]["editable"]
    assert pod["capital_drilling_pod"]["editor"] == "text"
    assert pod["is_latest"]["logical_type"] == "boolean"


def test_project_lookup_fresh_after_generation(current_registry):
    from esdc.portal.data import lookup_projects, read_table_page

    first = read_table_page("m_project", sqlite_path=current_registry)
    conn = sqlite3.connect(current_registry)
    conn.execute(
        "UPDATE m_project SET project_name='Renamed project' WHERE project_id='inactive'"
    )
    conn.execute(
        "UPDATE pod_registry_state SET source_generation=source_generation+1 WHERE singleton_id=1"
    )
    conn.commit()
    conn.close()
    assert (
        lookup_projects(sqlite_path=current_registry, q="Renamed")[0]["project_id"]
        == "inactive"
    )
    assert (
        read_table_page("m_project", sqlite_path=current_registry)["source_generation"]
        == first["source_generation"] + 1
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"page": True},
        {"page": 0},
        {"size": 201},
        {"sort": "id;DROP TABLE m_pod"},
        {"direction": "DROP"},
        {"sort": {}},
        {"filters": [{"field": "pod_id_seq", "op": "eq OR1=1", "value": 1}]},
    ],
)
def test_server_rejects_invalid_arguments_before_missing_target(tmp_path, kwargs):
    from esdc.portal.data import read_table_page

    with pytest.raises(ValueError):
        read_table_page("m_pod", sqlite_path=tmp_path / "missing.sqlite", **kwargs)
    assert not (tmp_path / "missing.sqlite").exists()


def test_reference_shape_is_not_assumed_empty(current_registry):
    from esdc.portal.data import read_reference_options

    conn = sqlite3.connect(current_registry)
    conn.execute("ALTER TABLE r_pod_stage RENAME COLUMN stage_definition TO other")
    conn.close()
    with pytest.raises(ValueError):
        read_reference_options(sqlite_path=current_registry)


def test_reference_options_preserve_large_issued_identity(current_registry):
    from datetime import date

    from esdc.pod_registry.identity import ApprovalKey, issue_identity
    from esdc.portal.data import read_reference_options
    from esdc.sqlite_runtime import configure_writer

    conn = sqlite3.connect(current_registry)
    configure_writer(conn)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("UPDATE sqlite_sequence SET seq=? WHERE name='m_pod'", (2**53,))
    issued = issue_identity(
        conn,
        key=ApprovalKey(date(2026, 1, 1), 5, 1, 1, 0),
        actor="FA:test-large-read-option",
    )
    assert issued.pod_id_seq == 2**53 + 1
    conn.execute(
        "INSERT INTO m_pod(pod_id_seq,pod_uuid,approval_date,institution_id,"
        "pod_type_id,rev_num,pod_order_no,pod_id,pod_name,pod_letter_num,"
        "is_latest) VALUES (?,?,?,1,1,0,?,?,?,'Large letter',1)",
        (
            issued.pod_id_seq,
            str(issued.pod_uuid),
            issued.approval_date.isoformat(),
            issued.pod_order_no,
            issued.pod_id,
            "Large identity",
        ),
    )
    conn.commit()
    conn.close()
    options = read_reference_options(sqlite_path=current_registry)
    selected = next(
        row for row in options["pods"] if row["pod_name"] == "Large identity"
    )
    assert selected["pod_id_seq"] == str(2**53 + 1)
    assert selected["pod_id"] == issued.pod_id


def test_datetime_sort_and_filter_utc_instants(current_registry):
    from esdc.portal.data import read_table_page

    stamps = {
        "inactive": "2026-01-01T00:00:00+07:00",
        "percent_%": "2025-12-31T18:00:00Z",
        "active": "2025-12-31 17:00:00",
    }
    conn = sqlite3.connect(current_registry)
    conn.executemany(
        "UPDATE m_project SET created_at=? WHERE project_id=?",
        [(stamp, key) for key, stamp in stamps.items()],
    )
    conn.commit()
    conn.close()
    result = read_table_page(
        "m_project", sqlite_path=current_registry, sort="created_at"
    )
    assert [row["project_id"] for row in result["rows"]] == [
        "active",
        "inactive",
        "percent_%",
    ]
    assert {row["project_id"]: row["created_at"] for row in result["rows"]} == stamps
    result = read_table_page(
        "m_project", sqlite_path=current_registry, sort="created_at", direction="desc"
    )
    assert [row["project_id"] for row in result["rows"]] == [
        "percent_%",
        "active",
        "inactive",
    ]
    for op, value, expected in [
        ("eq", "2026-01-01T00:00:00+07:00", ["active", "inactive"]),
        ("gt", "2025-12-31T17:00:00Z", ["percent_%"]),
        ("lte", "2025-12-31 17:00:00", ["active", "inactive"]),
    ]:
        result = read_table_page(
            "m_project",
            sqlite_path=current_registry,
            filters=[{"field": "created_at", "op": op, "value": value}],
        )
        assert [row["project_id"] for row in result["rows"]] == expected
        assert result["total"] == len(expected)


@pytest.mark.parametrize(
    "value",
    [
        "not-a-timestamp",
        "2026-02-30T00:00:00Z",
        "2026-01-01T00:00:00+24:00",
        "2026-01-01T00:00:00+01:60",
        "2026-01-01T00:00:00Z\n",
        True,
        "NaN",
    ],
)
def test_datetime_invalid_filter_rejected(current_registry, value):
    from esdc.portal.data import read_table_page

    with pytest.raises(ValueError):
        read_table_page(
            "m_project",
            sqlite_path=current_registry,
            filters=[{"field": "created_at", "op": "eq", "value": value}],
        )
