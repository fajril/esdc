"""Current portal transaction, precision, and permanent identity regressions."""

import sqlite3
from pathlib import Path
from uuid import UUID

import pytest

from esdc.pod_registry.changesets import apply_changeset
from esdc.pod_registry.store import get_sqlite_connection


@pytest.fixture()
def target(tmp_path):
    path = tmp_path / "source.sqlite"
    conn = get_sqlite_connection(path)
    conn.close()
    return path


def rows(path: Path, table: str) -> list[dict]:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute(f'SELECT * FROM "{table}"')]
    finally:
        conn.close()


def gen(path):
    return rows(path, "pod_registry_state")[0]["source_generation"]


def save(path, table, **changes):
    return apply_changeset(table, {"source_generation": gen(path), **changes}, path)


def approval(**changes):
    return {
        "approval_date": "2024-01-01",
        "institution_id": 1,
        "pod_type_id": 1,
        "pod_name": "POD Example",
        "pod_letter_num": "POD/001",
        **changes,
    }


def test_exact_decimal_edit_preserves_identity_and_audit(target):
    result = save(target, "m_pod", inserts=[approval()])
    assert result.ok, result.errors
    assert result.source_committed and result.source_generation == 1
    before = rows(target, "m_pod")[0]
    assert before["pod_id"] == "PL-2024-0001-1-1-0"
    assert UUID(before["pod_uuid"]).version == 4
    ledger = rows(target, "_pod_identity_ledger")
    value = "12345678901234567.123"
    result = save(
        target,
        "m_pod",
        updates=[{"pod_id_seq": before["pod_id_seq"], "capital_drilling_pod": value}],
    )
    assert result.ok, result.errors
    after = rows(target, "m_pod")[0]
    assert after["capital_drilling_pod"] == value
    assert (after["pod_id"], after["pod_uuid"]) == (
        before["pod_id"],
        before["pod_uuid"],
    )
    assert rows(target, "_pod_identity_ledger") == ledger
    assert gen(target) == 2


@pytest.mark.parametrize(
    "patch",
    [
        {"approval_date": "2025-01-01"},
        {"pod_uuid": "forged"},
        {"created_by": "forged"},
        {"rev_num": 2},
        {"is_latest": False},
    ],
)
def test_identity_and_generated_fields_are_not_editable(target, patch):
    assert save(target, "m_pod", inserts=[approval()]).ok
    before = rows(target, "m_pod")
    result = save(
        target, "m_pod", updates=[{"pod_id_seq": before[0]["pod_id_seq"], **patch}]
    )
    assert not result.ok and rows(target, "m_pod") == before and gen(target) == 1


def test_validation_rolls_back_identity_reservation(target):
    bad = approval()
    bad.pop("pod_letter_num")
    assert not save(target, "m_pod", inserts=[bad]).ok
    assert rows(target, "_pod_identity_ledger") == [] and gen(target) == 0
    assert save(target, "m_pod", inserts=[approval()]).ok
    assert rows(target, "_pod_identity_ledger")[0]["daily_slot"] == 1


def test_deleted_master_keeps_permanent_slot(target):
    assert save(target, "m_pod", inserts=[approval()]).ok
    old = rows(target, "m_pod")[0]
    assert save(target, "m_pod", deletes=[{"pod_id_seq": old["pod_id_seq"]}]).ok
    assert save(target, "m_pod", inserts=[approval(pod_name="Historical addition")]).ok
    new = rows(target, "m_pod")[0]
    assert (
        new["pod_id"] == "PL-2024-0002-1-1-0" and new["pod_id_seq"] != old["pod_id_seq"]
    )
    assert len(rows(target, "_pod_identity_ledger")) == 2


def test_supersession_is_atomic_and_explicit_name_revision_takes_precedence(target):
    assert save(target, "m_pod", inserts=[approval()]).ok
    old = rows(target, "m_pod")[0]
    result = save(
        target,
        "m_pod",
        inserts=[
            approval(
                approval_date="2024-01-02",
                pod_name="POD Revisi II",
                predecessors=[old["pod_id_seq"]],
            )
        ],
    )
    assert result.ok, result.errors
    old, new = sorted(rows(target, "m_pod"), key=lambda row: row["pod_id_seq"])
    assert old["superseded_by"] == new["pod_id_seq"] and old["is_latest"] == 0
    assert new["pod_id"] == "PL-2024-0021-1-1-2" and new["is_latest"] == 1


def test_twenty_first_slot_and_stale_edit_leave_no_partial_writes(target):
    result = save(
        target, "m_pod", inserts=[approval(pod_name=f"POD {i}") for i in range(20)]
    )
    assert result.ok, result.errors
    before = rows(target, "_pod_identity_ledger")
    assert not save(target, "m_pod", inserts=[approval(pod_name="Overflow")]).ok
    assert rows(target, "_pod_identity_ledger") == before and gen(target) == 1
    result = apply_changeset(
        "m_pod",
        {
            "source_generation": 0,
            "updates": [{"pod_id_seq": 1, "pod_name": "Stale draft"}],
        },
        target,
    )
    assert not result.ok and result.conflict and gen(target) == 1


def test_inactive_project_mapping_and_final_reference_validation(target):
    assert save(target, "m_pod", inserts=[approval()]).ok
    assert save(
        target,
        "m_project",
        inserts=[
            {
                "project_id": "P-001459F-01",
                "project_name": "Inactive master",
                "is_active": False,
            }
        ],
    ).ok
    assert save(
        target,
        "pod_project",
        inserts=[
            {"pod_id_seq": 1, "project_id": "P-001459F-01", "is_main_project": True}
        ],
    ).ok
    link = rows(target, "pod_project")[0]
    assert link["is_auto_inferred"] == 0
    assert not save(target, "m_project", deletes=[{"project_id": "P-001459F-01"}]).ok
    assert not save(
        target,
        "pod_project",
        updates=[{"id": link["id"], "project_id": "P-0000000-00"}],
    ).ok
    assert rows(target, "pod_project") == [link] and gen(target) == 3


def test_monitoring_and_forecast_use_current_schema(target):
    assert save(target, "m_pod", inserts=[approval()]).ok
    result = save(
        target,
        "pod_monitoring",
        inserts=[
            {
                "pod_id_seq": 1,
                "reported_date": "2024-06-30",
                "pod_stage_id": 1,
                "capital_drilling_real": "98765432109876543.210",
                "well_prd_real": 0,
            }
        ],
    )
    assert result.ok, result.errors
    result = save(
        target,
        "pod_forecast",
        inserts=[{"pod_id_seq": 1, "year_frcast": 2030, "usf_oil_pod": "1.234"}],
    )
    assert result.ok, result.errors
    mon = rows(target, "pod_monitoring")[0]
    assert (
        mon["capital_drilling_real"] == "98765432109876543.210"
        and mon["well_prd_real"] == 0
    )
    assert mon["pod_id"] == "PL-2024-0001-1-1-0" and mon["created_by"] == "FA:portal"
    assert gen(target) == 3 and rows(target, "pod_forecast")[0]["year_frcast"] == 2030


@pytest.mark.parametrize(
    "table",
    [
        "r_institution",
        "r_pod_type",
        "r_pod_stage",
        "pod_monitoring_api",
        "pod_forecast_api",
    ],
)
def test_references_and_results_cannot_be_written(table, tmp_path):
    target = tmp_path / "absent" / "source.sqlite"
    with pytest.raises(ValueError):
        apply_changeset(table, {"source_generation": 0, "inserts": [{}]}, target)
    assert not target.parent.exists()


def test_document_links_preserve_raw_documents_and_protect_master(target):
    assert save(target, "m_pod", inserts=[approval()]).ok
    conn = get_sqlite_connection(target)
    try:
        conn.execute("CREATE TABLE documents(doc_id TEXT PRIMARY KEY, file_name TEXT)")
        conn.execute("INSERT INTO documents VALUES ('doc-1', 'Source document')")
        conn.commit()
    finally:
        conn.close()
    documents = rows(target, "documents")
    result = save(
        target, "pod_document", inserts=[{"pod_id_seq": "1", "doc_id": "doc-1"}]
    )
    assert result.ok, result.errors
    assert gen(target) == 2
    assert not save(target, "m_pod", deletes=[{"pod_id_seq": 1}]).ok
    assert rows(target, "documents") == documents
    assert len(rows(target, "pod_document")) == 1 and gen(target) == 2
    assert save(
        target, "pod_document", deletes=[{"pod_id_seq": "1", "doc_id": "doc-1"}]
    ).ok
    assert rows(target, "pod_document") == [] and rows(target, "documents") == documents
    assert gen(target) == 3


def test_failure_after_shared_refresh_rolls_back_ledger_source_and_generation(
    target, monkeypatch
):
    import esdc.pod_registry.changesets as changesets

    real_apply = changesets.apply_pod_batch_in_transaction

    def fail_after_apply(*args, **kwargs):
        real_apply(*args, **kwargs)
        raise RuntimeError("injected after final source refresh")

    monkeypatch.setattr(changesets, "apply_pod_batch_in_transaction", fail_after_apply)
    before = rows(target, "pod_registry_state")
    with pytest.raises(RuntimeError, match="injected after final"):
        save(target, "m_pod", inserts=[approval()])
    assert rows(target, "m_pod") == [] and rows(target, "_pod_identity_ledger") == []
    assert rows(target, "pod_registry_state") == before


def test_writer_lock_rechecks_generation_before_identity_issuance(target, monkeypatch):
    import esdc.pod_registry.changesets as changesets

    assert save(target, "m_pod", inserts=[approval()]).ok
    original = changesets.read_pod_api_context
    raced = False

    def other_writer(path):
        nonlocal raced
        captured = original(path)
        if not raced:
            raced = True
            result = save(
                path, "m_pod", updates=[{"pod_id_seq": 1, "pod_name": "Other writer"}]
            )
            assert result.ok, result.errors
        return captured

    monkeypatch.setattr(changesets, "read_pod_api_context", other_writer)
    result = save(target, "m_pod", inserts=[approval(approval_date="2024-02-01")])
    assert not result.ok and result.conflict
    assert rows(target, "m_pod")[0]["pod_name"] == "Other writer"
    assert len(rows(target, "m_pod")) == len(rows(target, "_pod_identity_ledger")) == 1
    assert gen(target) == 2


def test_mapping_edit_rebuilds_physical_results_from_local_resources(target):
    from esdc.api_snapshot import ApiSnapshot, replace_project_resources_snapshot
    from tests.pod_registry.test_read_models_v2 import _resource

    assert save(target, "m_pod", inserts=[approval()]).ok
    assert save(
        target,
        "m_project",
        inserts=[
            {"project_id": "P-001459F-01", "project_name": "One", "is_active": True},
            {"project_id": "P-001459F-02", "project_name": "Two", "is_active": False},
        ],
    ).ok
    assert save(
        target,
        "pod_project",
        inserts=[
            {"pod_id_seq": 1, "project_id": "P-001459F-01", "is_main_project": True}
        ],
    ).ok
    assert save(
        target,
        "pod_monitoring",
        inserts=[{"pod_id_seq": 1, "reported_date": "2024-06-30", "pod_stage_id": 1}],
    ).ok
    conn = get_sqlite_connection(target)
    try:
        replace_project_resources_snapshot(
            conn,
            [
                _resource("P-001459F-01", 2024, oil_reserve="10.25"),
                _resource("P-001459F-02", 2024, oil_reserve="20.75"),
            ],
            ApiSnapshot(
                "project_resources",
                "local-resource-r1",
                2,
                "",
                "2026-11-02T00:00:00+00:00",
                "committed",
            ),
        )
    finally:
        conn.close()
    assert rows(target, "pod_monitoring_api")[0]["res_oil_2p"] == 10.25
    before = gen(target)
    link = rows(target, "pod_project")[0]
    result = save(
        target,
        "pod_project",
        updates=[{"id": link["id"], "project_id": "P-001459F-02"}],
    )
    assert result.ok, result.errors
    assert result.source_generation == before + 1
    assert rows(target, "pod_monitoring_api")[0]["res_oil_2p"] == 20.75


def test_reassigning_monitoring_updates_its_derived_identity(target):
    assert save(
        target,
        "m_pod",
        inserts=[
            approval(),
            approval(approval_date="2024-01-02", pod_name="Second scope"),
        ],
    ).ok
    assert save(
        target,
        "pod_monitoring",
        inserts=[{"pod_id_seq": 1, "reported_date": "2024-06-30", "pod_stage_id": 1}],
    ).ok
    record = rows(target, "pod_monitoring")[0]
    result = save(
        target,
        "pod_monitoring",
        updates=[{"monpod_id": record["monpod_id"], "pod_id_seq": 2}],
    )
    assert result.ok, result.errors
    after = rows(target, "pod_monitoring")[0]
    assert after["pod_id_seq"] == 2 and after["pod_id"] == "PL-2024-0021-1-1-0"
    assert after["monpod_uuid"] == record["monpod_uuid"]


def test_contract_errors_preserve_operation_row_and_field(target):
    assert save(target, "m_pod", inserts=[approval()]).ok
    result = save(
        target,
        "m_pod",
        updates=[{"pod_id_seq": 1, "capital_drilling_pod": "12345678901234567.1234"}],
    )
    assert not result.ok
    assert all(
        error.kind == "update"
        and error.index == 0
        and error.field == "capital_drilling_pod"
        for error in result.errors
    )
    assert gen(target) == 1
