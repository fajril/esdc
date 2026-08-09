"""Tests for SQLite POD value-case truth (plan, actual, outlook)."""

# tests/pod_registry/test_value_cases.py
import sqlite3
from datetime import date, datetime

import pandas as pd
import pytest
import yaml

import esdc.configs as configs
from esdc.loaders import POD_SCHEMA_PATH, load_schema_from_yaml
from esdc.pod_registry.store import get_sqlite_connection
from esdc.pod_registry.value_cases import replace_pod_value_cases


@pytest.fixture
def schema():
    return load_schema_from_yaml(POD_SCHEMA_PATH)


def _load_pod_schema_yaml() -> dict:
    with open(POD_SCHEMA_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _seed_refs(conn):
    conn.execute("INSERT INTO r_institution (code, institution) VALUES (3, 'BP Migas')")
    conn.execute(
        "INSERT INTO r_pod_type (code, pod_type) VALUES (2, 'POD/Waterflood/EOR')"
    )


def _seed_m_pod(conn, pod_id, *, approval_date="2024-05-15", seq=1):
    conn.execute(
        "INSERT INTO m_pod (id, pod_id, pod_name, approval_date,"
        " institution_code, pod_type_code, rev_num, approval_seq)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (seq, pod_id, "POD A", approval_date, 3, 2, 0, seq),
    )


def _make_db(db_path, pod_ids):
    conn = get_sqlite_connection(db_path)
    _seed_refs(conn)
    for index, pod_id in enumerate(pod_ids, start=1):
        _seed_m_pod(conn, pod_id, seq=index)
    conn.commit()
    conn.close()


def _fetch(db_path, sql, params=()):
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def _count(db_path, table="pod_value_case"):
    conn = sqlite3.connect(db_path)
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if not exists:
            return 0
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


def _plan_df(rows):
    return pd.DataFrame(rows)


def _monitoring_df(rows):
    return pd.DataFrame(rows)


# --- contract ---


def test_schema_contract_declares_sqlite_truth():
    raw = _load_pod_schema_yaml()
    assert raw["schema_version"] == "2"
    assert raw["source_of_truth"] == "sqlite"
    assert raw["truth_table"] == "pod_value_case"


def test_schema_keeps_only_authoritative_value_case_payload():
    raw = _load_pod_schema_yaml()
    names = {column["name"] for column in raw["columns"]}
    assert {"pod_id", "case_type", "report_date", "as_of_date", "pod_scope"} <= names
    assert not {"pod_name", "pod_letter_num", "supercedes_by", "effective_date"} & names


def test_schema_report_date_has_no_approval_alias():
    raw = _load_pod_schema_yaml()
    report_date = next(
        column for column in raw["columns"] if column["name"] == "report_date"
    )
    assert "tanggal persetujuan POD" not in report_date["aliases"]


def test_schema_parses_as_load_schema():
    schema = load_schema_from_yaml(POD_SCHEMA_PATH)
    assert schema.table_name == "pod_value_case"
    names = {column.name for column in schema.columns}
    assert {"pod_id", "case_type", "report_date", "as_of_date", "pod_scope"} <= names


# --- temporal semantics ---


def test_plan_rows_become_case_type_plan(tmp_path, schema):
    db_path = tmp_path / "esdc.sqlite"
    _make_db(db_path, ["PL-2024-0001-3-2-0"])
    plan = _plan_df(
        [
            {
                "pod_id": "PL-2024-0001-3-2-0",
                "report_date": date(2024, 6, 15),
                "as_of_date": date(2024, 12, 31),
            }
        ]
    )
    count = replace_pod_value_cases(db_path, plan, pd.DataFrame(), schema)
    assert count == 1
    row = _fetch(
        db_path,
        "SELECT pod_id, case_type, report_date, as_of_date FROM pod_value_case",
    )[0]
    assert tuple(row) == ("PL-2024-0001-3-2-0", "plan", "2024-06-15", "2024-12-31")


def test_report_and_as_of_dates_are_distinct_semantics(tmp_path, schema):
    db_path = tmp_path / "esdc.sqlite"
    _make_db(db_path, ["PL-2024-0001-3-2-0"])
    plan = _plan_df(
        [
            {
                "pod_id": "PL-2024-0001-3-2-0",
                "report_date": date(2024, 6, 15),
                "as_of_date": date(2024, 12, 31),
            },
            {
                "pod_id": "PL-2024-0001-3-2-0",
                "report_date": date(2024, 8, 20),
                "as_of_date": date(2024, 6, 30),
            },
        ]
    )
    replace_pod_value_cases(db_path, plan, pd.DataFrame(), schema)
    rows = _fetch(
        db_path,
        "SELECT report_date, as_of_date FROM pod_value_case ORDER BY as_of_date",
    )
    assert [(row["report_date"], row["as_of_date"]) for row in rows] == [
        ("2024-08-20", "2024-06-30"),
        ("2024-06-15", "2024-12-31"),
    ]


def test_approval_date_comes_from_m_pod_not_value_case(tmp_path, schema):
    db_path = tmp_path / "esdc.sqlite"
    _make_db(db_path, ["PL-2024-0001-3-2-0"])
    plan = _plan_df(
        [
            {
                "pod_id": "PL-2024-0001-3-2-0",
                "report_date": date(2024, 6, 15),
                "as_of_date": date(2024, 12, 31),
            }
        ]
    )
    replace_pod_value_cases(db_path, plan, pd.DataFrame(), schema)
    columns = {
        row["name"] for row in _fetch(db_path, "PRAGMA table_info(pod_value_case)")
    }
    assert "approval_date" not in columns
    pod = _fetch(
        db_path,
        "SELECT approval_date FROM m_pod WHERE pod_id = 'PL-2024-0001-3-2-0'",
    )
    assert pod[0]["approval_date"] == "2024-05-15"


def test_monitoring_accepts_actual_and_outlook(tmp_path, schema):
    db_path = tmp_path / "esdc.sqlite"
    _make_db(db_path, ["PL-2024-0001-3-2-0"])
    monitoring = _monitoring_df(
        [
            {
                "pod_id": "PL-2024-0001-3-2-0",
                "case_type": "actual",
                "report_date": date(2025, 1, 20),
                "as_of_date": date(2024, 12, 31),
            },
            {
                "pod_id": "PL-2024-0001-3-2-0",
                "case_type": "outlook",
                "report_date": date(2025, 1, 20),
                "as_of_date": date(2025, 6, 30),
            },
        ]
    )
    count = replace_pod_value_cases(db_path, pd.DataFrame(), monitoring, schema)
    assert count == 2
    types = {
        row["case_type"]
        for row in _fetch(db_path, "SELECT case_type FROM pod_value_case")
    }
    assert types == {"actual", "outlook"}


@pytest.mark.parametrize("bad", ["plan", "budget", "Actual Plan"])
def test_monitoring_rejects_invalid_case_type(tmp_path, schema, bad):
    db_path = tmp_path / "esdc.sqlite"
    _make_db(db_path, ["PL-2024-0001-3-2-0"])
    monitoring = _monitoring_df(
        [
            {
                "pod_id": "PL-2024-0001-3-2-0",
                "case_type": bad,
                "report_date": date(2025, 1, 20),
                "as_of_date": date(2024, 12, 31),
            }
        ]
    )
    with pytest.raises(ValueError, match="case_type"):
        replace_pod_value_cases(db_path, pd.DataFrame(), monitoring, schema)
    assert _count(db_path) == 0


def test_unknown_pod_id_fails_before_any_replacement(tmp_path, schema):
    db_path = tmp_path / "esdc.sqlite"
    _make_db(db_path, ["PL-2024-0001-3-2-0"])
    plan = _plan_df(
        [
            {
                "pod_id": "PL-2024-0001-3-2-0",
                "report_date": date(2024, 6, 15),
                "as_of_date": date(2024, 12, 31),
            },
            {
                "pod_id": "PL-9999-0001-3-2-0",
                "report_date": date(2024, 6, 15),
                "as_of_date": date(2025, 6, 30),
            },
        ]
    )
    with pytest.raises(ValueError, match="pod_id"):
        replace_pod_value_cases(db_path, plan, pd.DataFrame(), schema)
    assert _count(db_path) == 0


def test_unknown_pod_id_preserves_existing_truth(tmp_path, schema):
    db_path = tmp_path / "esdc.sqlite"
    _make_db(db_path, ["PL-2024-0001-3-2-0"])
    good = _plan_df(
        [
            {
                "pod_id": "PL-2024-0001-3-2-0",
                "report_date": date(2024, 6, 15),
                "as_of_date": date(2024, 12, 31),
            }
        ]
    )
    replace_pod_value_cases(db_path, good, pd.DataFrame(), schema)
    assert _count(db_path) == 1
    bad = _plan_df(
        [
            {
                "pod_id": "PL-2024-0001-3-2-0",
                "report_date": date(2024, 6, 15),
                "as_of_date": date(2024, 12, 31),
            },
            {
                "pod_id": "PL-4040-0001-3-2-0",
                "report_date": date(2024, 6, 15),
                "as_of_date": date(2025, 6, 30),
            },
        ]
    )
    with pytest.raises(ValueError, match="pod_id"):
        replace_pod_value_cases(db_path, bad, pd.DataFrame(), schema)
    assert _count(db_path) == 1


def test_duplicate_plan_key_rejected_before_replacement(tmp_path, schema):
    db_path = tmp_path / "esdc.sqlite"
    _make_db(db_path, ["PL-2024-0001-3-2-0"])
    plan = _plan_df(
        [
            {
                "pod_id": "PL-2024-0001-3-2-0",
                "report_date": date(2024, 6, 15),
                "as_of_date": date(2024, 12, 31),
            },
            {
                "pod_id": "PL-2024-0001-3-2-0",
                "report_date": date(2024, 9, 1),
                "as_of_date": date(2024, 12, 31),
            },
        ]
    )
    with pytest.raises(ValueError, match="duplicate"):
        replace_pod_value_cases(db_path, plan, pd.DataFrame(), schema)
    assert _count(db_path) == 0


def test_duplicate_monitoring_key_rejected_before_replacement(tmp_path, schema):
    db_path = tmp_path / "esdc.sqlite"
    _make_db(db_path, ["PL-2024-0001-3-2-0"])
    monitoring = _monitoring_df(
        [
            {
                "pod_id": "PL-2024-0001-3-2-0",
                "case_type": "actual",
                "report_date": date(2025, 1, 20),
                "as_of_date": date(2024, 12, 31),
            },
            {
                "pod_id": "PL-2024-0001-3-2-0",
                "case_type": "actual",
                "report_date": date(2025, 2, 20),
                "as_of_date": date(2024, 12, 31),
            },
        ]
    )
    with pytest.raises(ValueError, match="duplicate"):
        replace_pod_value_cases(db_path, pd.DataFrame(), monitoring, schema)
    assert _count(db_path) == 0


def test_replacement_swaps_existing_truth_rows(tmp_path, schema):
    db_path = tmp_path / "esdc.sqlite"
    _make_db(db_path, ["PL-2024-0001-3-2-0"])
    first = _plan_df(
        [
            {
                "pod_id": "PL-2024-0001-3-2-0",
                "report_date": date(2024, 6, 15),
                "as_of_date": date(2024, 12, 31),
            },
            {
                "pod_id": "PL-2024-0001-3-2-0",
                "report_date": date(2024, 6, 15),
                "as_of_date": date(2025, 6, 30),
            },
        ]
    )
    replace_pod_value_cases(db_path, first, pd.DataFrame(), schema)
    assert _count(db_path) == 2
    second = _plan_df(
        [
            {
                "pod_id": "PL-2024-0001-3-2-0",
                "report_date": date(2024, 6, 15),
                "as_of_date": date(2024, 12, 31),
            }
        ]
    )
    count = replace_pod_value_cases(db_path, second, pd.DataFrame(), schema)
    assert count == 1
    rows = _fetch(db_path, "SELECT pod_id, case_type, as_of_date FROM pod_value_case")
    assert [(row["pod_id"], row["case_type"], row["as_of_date"]) for row in rows] == [
        ("PL-2024-0001-3-2-0", "plan", "2024-12-31")
    ]


def test_return_value_counts_inserted_rows(tmp_path, schema):
    db_path = tmp_path / "esdc.sqlite"
    _make_db(db_path, ["PL-2024-0001-3-2-0"])
    plan = _plan_df(
        [
            {
                "pod_id": "PL-2024-0001-3-2-0",
                "report_date": date(2024, 6, 15),
                "as_of_date": date(2024, 12, 31),
            },
            {
                "pod_id": "PL-2024-0001-3-2-0",
                "report_date": date(2024, 6, 15),
                "as_of_date": date(2025, 6, 30),
            },
        ]
    )
    monitoring = _monitoring_df(
        [
            {
                "pod_id": "PL-2024-0001-3-2-0",
                "case_type": "actual",
                "report_date": date(2025, 1, 20),
                "as_of_date": date(2024, 12, 31),
            }
        ]
    )
    count = replace_pod_value_cases(db_path, plan, monitoring, schema)
    assert count == 3
    assert _count(db_path) == 3


def test_numeric_metrics_stored_as_real(tmp_path, schema):
    db_path = tmp_path / "esdc.sqlite"
    _make_db(db_path, ["PL-2024-0001-3-2-0"])
    plan = _plan_df(
        [
            {
                "pod_id": "PL-2024-0001-3-2-0",
                "report_date": date(2024, 6, 15),
                "as_of_date": date(2024, 12, 31),
                "lifting_oil": 1234.5,
                "wap_gas": 7.25,
                "ctr_npv": None,
            }
        ]
    )
    replace_pod_value_cases(db_path, plan, pd.DataFrame(), schema)
    columns = {
        row["name"]: row["type"]
        for row in _fetch(db_path, "PRAGMA table_info(pod_value_case)")
    }
    assert columns["lifting_oil"] == "REAL"
    assert columns["wap_gas"] == "REAL"
    row = _fetch(db_path, "SELECT lifting_oil, wap_gas, ctr_npv FROM pod_value_case")[0]
    assert row["lifting_oil"] == 1234.5
    assert row["wap_gas"] == 7.25
    assert row["ctr_npv"] is None


def test_value_case_table_grain_unique_and_case_type_check(tmp_path, schema):
    db_path = tmp_path / "esdc.sqlite"
    _make_db(db_path, ["PL-2024-0001-3-2-0"])
    plan = _plan_df(
        [
            {
                "pod_id": "PL-2024-0001-3-2-0",
                "report_date": date(2024, 6, 15),
                "as_of_date": date(2024, 12, 31),
            }
        ]
    )
    replace_pod_value_cases(db_path, plan, pd.DataFrame(), schema)
    conn = sqlite3.connect(db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO pod_value_case (pod_id, case_type, report_date, as_of_date)"
                " VALUES ('PL-2024-0001-3-2-0', 'plan', '2024-07-01', '2024-12-31')"
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO pod_value_case (pod_id, case_type, report_date, as_of_date)"
                " VALUES ('PL-2024-0001-3-2-0', 'guess', '2024-07-01', '2025-06-30')"
            )
    finally:
        conn.close()


def test_invalid_as_of_date_is_rejected(tmp_path, schema):
    db_path = tmp_path / "esdc.sqlite"
    _make_db(db_path, ["PL-2024-0001-3-2-0"])
    plan = _plan_df(
        [
            {
                "pod_id": "PL-2024-0001-3-2-0",
                "report_date": date(2024, 6, 15),
                "as_of_date": "not-a-date",
            }
        ]
    )
    with pytest.raises(ValueError, match="as_of_date"):
        replace_pod_value_cases(db_path, plan, pd.DataFrame(), schema)
    assert _count(db_path) == 0


def test_missing_as_of_date_column_rejected(tmp_path, schema):
    db_path = tmp_path / "esdc.sqlite"
    _make_db(db_path, ["PL-2024-0001-3-2-0"])
    plan = pd.DataFrame(
        [{"pod_id": "PL-2024-0001-3-2-0", "report_date": date(2024, 6, 15)}]
    )
    with pytest.raises(ValueError, match="as_of_date"):
        replace_pod_value_cases(db_path, plan, pd.DataFrame(), schema)
    assert _count(db_path) == 0


def test_monitoring_without_pod_scope_stores_null(tmp_path, schema):
    db_path = tmp_path / "esdc.sqlite"
    _make_db(db_path, ["PL-2024-0001-3-2-0"])
    monitoring = _monitoring_df(
        [
            {
                "pod_id": "PL-2024-0001-3-2-0",
                "case_type": "actual",
                "report_date": date(2025, 1, 20),
                "as_of_date": date(2024, 12, 31),
            }
        ]
    )
    replace_pod_value_cases(db_path, pd.DataFrame(), monitoring, schema)
    row = _fetch(db_path, "SELECT pod_scope FROM pod_value_case")[0]
    assert row["pod_scope"] is None


def test_none_path_uses_default_db_dir(monkeypatch, tmp_path, schema):
    monkeypatch.setattr(configs.Config, "get_db_dir", classmethod(lambda cls: tmp_path))
    db_path = tmp_path / "esdc.sqlite"
    _make_db(db_path, ["PL-2024-0001-3-2-0"])
    plan = _plan_df(
        [
            {
                "pod_id": "PL-2024-0001-3-2-0",
                "report_date": date(2024, 6, 15),
                "as_of_date": date(2024, 12, 31),
            }
        ]
    )
    count = replace_pod_value_cases(None, plan, pd.DataFrame(), schema)
    assert count == 1
    assert _count(db_path) == 1


def test_replacement_does_not_touch_project_pod(tmp_path, schema):
    db_path = tmp_path / "esdc.sqlite"
    _make_db(db_path, ["PL-2024-0001-3-2-0"])
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO project_pod (pod_id, project_id) VALUES (1, 'PROJECT-CANONICAL')"
    )
    conn.commit()
    conn.close()
    plan = _plan_df(
        [
            {
                "pod_id": "PL-2024-0001-3-2-0",
                "report_date": date(2024, 6, 15),
                "as_of_date": date(2024, 12, 31),
            }
        ]
    )
    replace_pod_value_cases(db_path, plan, pd.DataFrame(), schema)
    rows = _fetch(db_path, "SELECT pod_id, project_id FROM project_pod")
    assert [(row["pod_id"], row["project_id"]) for row in rows] == [
        (1, "PROJECT-CANONICAL")
    ]


def test_pd_timestamp_dates_persist_as_date_only(tmp_path, schema):
    db_path = tmp_path / "esdc.sqlite"
    _make_db(db_path, ["PL-2024-0001-3-2-0"])
    plan = _plan_df(
        [
            {
                "pod_id": "PL-2024-0001-3-2-0",
                "report_date": pd.Timestamp("2024-06-15 10:30:00"),
                "as_of_date": pd.Timestamp("2024-12-31 23:59:59"),
            }
        ]
    )
    replace_pod_value_cases(db_path, plan, pd.DataFrame(), schema)
    row = _fetch(db_path, "SELECT report_date, as_of_date FROM pod_value_case")[0]
    assert row["report_date"] == "2024-06-15"
    assert row["as_of_date"] == "2024-12-31"


def test_python_datetime_dates_persist_as_date_only(tmp_path, schema):
    db_path = tmp_path / "esdc.sqlite"
    _make_db(db_path, ["PL-2024-0001-3-2-0"])
    plan = _plan_df(
        [
            {
                "pod_id": "PL-2024-0001-3-2-0",
                "report_date": datetime(2024, 6, 15, 9, 30),
                "as_of_date": datetime(2024, 12, 31),
            }
        ]
    )
    replace_pod_value_cases(db_path, plan, pd.DataFrame(), schema)
    row = _fetch(db_path, "SELECT report_date, as_of_date FROM pod_value_case")[0]
    assert row["report_date"] == "2024-06-15"
    assert row["as_of_date"] == "2024-12-31"


def test_foreign_key_rejects_unknown_pod_id(tmp_path, schema):
    db_path = tmp_path / "esdc.sqlite"
    _make_db(db_path, ["PL-2024-0001-3-2-0"])
    plan = _plan_df(
        [
            {
                "pod_id": "PL-2024-0001-3-2-0",
                "report_date": date(2024, 6, 15),
                "as_of_date": date(2024, 12, 31),
            }
        ]
    )
    replace_pod_value_cases(db_path, plan, pd.DataFrame(), schema)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO pod_value_case (pod_id, case_type, report_date,"
                " as_of_date) VALUES ('PL-9999-0001-3-2-0', 'actual',"
                " '2025-01-20', '2025-06-30')"
            )
            conn.commit()
    finally:
        conn.close()


def test_delete_and_reinsert_m_pod_preserves_linked_value_cases(tmp_path, schema):
    db_path = tmp_path / "esdc.sqlite"
    _make_db(db_path, ["PL-2024-0001-3-2-0"])
    plan = _plan_df(
        [
            {
                "pod_id": "PL-2024-0001-3-2-0",
                "report_date": date(2024, 6, 15),
                "as_of_date": date(2024, 12, 31),
            }
        ]
    )
    replace_pod_value_cases(db_path, plan, pd.DataFrame(), schema)
    assert _count(db_path) == 1
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        with conn:
            conn.execute("DELETE FROM m_pod WHERE pod_id = 'PL-2024-0001-3-2-0'")
            conn.execute(
                "INSERT INTO m_pod (id, pod_id, pod_name, approval_date,"
                " institution_code, pod_type_code, rev_num, approval_seq)"
                " VALUES (1, 'PL-2024-0001-3-2-0', 'POD A', '2024-05-15',"
                " 3, 2, 0, 1)"
            )
        assert _count(db_path) == 1
    finally:
        conn.close()
