"""Tests for POD registry changesets."""

import esdc.configs as configs
from esdc.corpus.store import _SQLITE_DOC_DDL
from esdc.pod_registry.changesets import apply_changeset
from esdc.pod_registry.store import get_sqlite_connection


def _patch_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(configs.Config, "get_db_dir", classmethod(lambda cls: tmp_path))


def _seed_documents(doc_ids):
    conn = get_sqlite_connection()
    conn.execute(_SQLITE_DOC_DDL)
    for _i, doc_id in enumerate(doc_ids):
        conn.execute(
            "INSERT INTO documents (doc_id, file_name, file_path, file_hash,"
            " markdown, extraction_method, embedding_model)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                doc_id,
                f"{doc_id}.pdf",
                f"/x/{doc_id}.pdf",
                doc_id * 4,
                "# isi",
                "native",
                "fake-embed",
            ),
        )
    conn.commit()
    conn.close()


def _seed_refs(monkeypatch, tmp_path):
    _patch_dirs(monkeypatch, tmp_path)
    conn = get_sqlite_connection()
    conn.execute(
        "INSERT INTO r_institution (code, institution) VALUES (4, 'SKK Migas')"
    )
    conn.execute("INSERT INTO r_pod_type (code, pod_type) VALUES (1, 'POD I')")
    conn.commit()
    conn.close()


def test_insert_m_pod_generates_pod_id(monkeypatch, tmp_path):
    _seed_refs(monkeypatch, tmp_path)
    result = apply_changeset(
        "m_pod",
        {
            "inserts": [
                {
                    "id": 900,
                    "pod_name": "POD Baru",
                    "pod_letter_num": "X/2026",
                    "approval_date": "2026-07-15",
                    "institution_code": 4,
                    "pod_type_code": 1,
                    "rev_num": 0,
                }
            ]
        },
    )
    assert result.ok
    assert result.applied["inserts"] == 1
    assert result.generated[0]["pod_id"] == "PL-2026-0001-4-1-0"


def test_update_m_pod_rejects_structural_change(monkeypatch, tmp_path):
    _seed_refs(monkeypatch, tmp_path)
    apply_changeset(
        "m_pod",
        {
            "inserts": [
                {
                    "id": 900,
                    "pod_name": "POD Baru",
                    "approval_date": "2026-07-15",
                    "institution_code": 4,
                    "pod_type_code": 1,
                    "rev_num": 0,
                }
            ]
        },
    )
    result = apply_changeset(
        "m_pod", {"updates": [{"id": 900, "approval_date": "2001-01-01"}]}
    )
    assert not result.ok
    assert "locked" in result.errors[0].message


def test_update_m_pod_name_ok(monkeypatch, tmp_path):
    _seed_refs(monkeypatch, tmp_path)
    apply_changeset(
        "m_pod",
        {
            "inserts": [
                {
                    "id": 900,
                    "pod_name": "POD Baru",
                    "approval_date": "2026-07-15",
                    "institution_code": 4,
                    "pod_type_code": 1,
                    "rev_num": 0,
                }
            ]
        },
    )
    result = apply_changeset(
        "m_pod", {"updates": [{"id": 900, "pod_name": "POD Renamed"}]}
    )
    assert result.ok
    conn = get_sqlite_connection()
    try:
        assert (
            conn.execute("SELECT pod_name FROM m_pod WHERE id=900").fetchone()[0]
            == "POD Renamed"
        )
    finally:
        conn.close()


def test_delete_m_pod_blocked_when_referenced(monkeypatch, tmp_path):
    _seed_refs(monkeypatch, tmp_path)
    apply_changeset(
        "m_pod",
        {
            "inserts": [
                {
                    "id": 900,
                    "pod_name": "POD Baru",
                    "approval_date": "2026-07-15",
                    "institution_code": 4,
                    "pod_type_code": 1,
                    "rev_num": 0,
                }
            ]
        },
    )
    apply_changeset("project_pod", {"inserts": [{"pod_id": 900, "project_id": "P-1"}]})
    result = apply_changeset("m_pod", {"deletes": [{"id": 900}]})
    assert not result.ok


def test_project_pod_validates_against_known_ids(monkeypatch, tmp_path):
    _seed_refs(monkeypatch, tmp_path)
    apply_changeset(
        "m_pod",
        {
            "inserts": [
                {
                    "id": 900,
                    "pod_name": "POD Baru",
                    "approval_date": "2026-07-15",
                    "institution_code": 4,
                    "pod_type_code": 1,
                    "rev_num": 0,
                }
            ]
        },
    )
    result = apply_changeset(
        "project_pod",
        {"inserts": [{"pod_id": 900, "project_id": "P-TYPO"}]},
        known_project_ids={"P-1", "P-2"},
    )
    assert not result.ok
    assert "P-TYPO" in result.errors[0].message


def _insert_pod_900(monkeypatch, tmp_path):
    _seed_refs(monkeypatch, tmp_path)
    apply_changeset(
        "m_pod",
        {
            "inserts": [
                {
                    "id": 900,
                    "pod_name": "POD Baru",
                    "approval_date": "2026-07-15",
                    "institution_code": 4,
                    "pod_type_code": 1,
                    "rev_num": 0,
                }
            ]
        },
    )


def test_pod_document_insert_and_delete(monkeypatch, tmp_path):
    _insert_pod_900(monkeypatch, tmp_path)
    result = apply_changeset(
        "pod_document", {"inserts": [{"pod_id": 900, "doc_id": "abc123"}]}
    )
    assert result.ok
    result = apply_changeset(
        "pod_document", {"deletes": [{"pod_id": 900, "doc_id": "abc123"}]}
    )
    assert result.ok
    conn = get_sqlite_connection()
    try:
        assert conn.execute("SELECT COUNT(*) FROM pod_document").fetchone()[0] == 0
    finally:
        conn.close()


def test_pod_document_validates_against_documents_table(monkeypatch, tmp_path):
    _insert_pod_900(monkeypatch, tmp_path)
    _seed_documents(["abc123"])
    result = apply_changeset(
        "pod_document", {"inserts": [{"pod_id": 900, "doc_id": "nope"}]}
    )
    assert not result.ok
    assert "nope" in result.errors[0].message
    result = apply_changeset(
        "pod_document", {"inserts": [{"pod_id": 900, "doc_id": "abc123"}]}
    )
    assert result.ok


def test_pod_document_check_skipped_when_no_documents_table(monkeypatch, tmp_path):
    _insert_pod_900(monkeypatch, tmp_path)
    result = apply_changeset(
        "pod_document", {"inserts": [{"pod_id": 900, "doc_id": "anything"}]}
    )
    assert result.ok


def test_pod_document_unknown_pod_rejected(monkeypatch, tmp_path):
    _insert_pod_900(monkeypatch, tmp_path)
    result = apply_changeset(
        "pod_document", {"inserts": [{"pod_id": 999, "doc_id": "abc123"}]}
    )
    assert not result.ok
    assert "999" in result.errors[0].message


def test_pod_document_rejects_updates_and_duplicates(monkeypatch, tmp_path):
    _insert_pod_900(monkeypatch, tmp_path)
    result = apply_changeset(
        "pod_document", {"updates": [{"pod_id": 900, "doc_id": "abc123"}]}
    )
    assert not result.ok
    apply_changeset("pod_document", {"inserts": [{"pod_id": 900, "doc_id": "abc123"}]})
    result = apply_changeset(
        "pod_document", {"inserts": [{"pod_id": 900, "doc_id": "abc123"}]}
    )
    assert not result.ok
    assert "already exists" in result.errors[0].message


def test_delete_m_pod_blocked_by_pod_document(monkeypatch, tmp_path):
    _insert_pod_900(monkeypatch, tmp_path)
    apply_changeset("pod_document", {"inserts": [{"pod_id": 900, "doc_id": "abc123"}]})
    result = apply_changeset("m_pod", {"deletes": [{"id": 900}]})
    assert not result.ok
    assert "pod_document" in result.errors[0].message


def _seed_pod_pair(monkeypatch, tmp_path):
    _seed_refs(monkeypatch, tmp_path)
    result = apply_changeset(
        "m_pod",
        {
            "inserts": [
                {
                    "id": 900,
                    "pod_name": "POD Succ",
                    "approval_date": "2026-07-15",
                    "institution_code": 4,
                    "pod_type_code": 1,
                    "rev_num": 0,
                },
                {
                    "id": 901,
                    "pod_name": "POD Pred",
                    "approval_date": "2025-07-15",
                    "institution_code": 4,
                    "pod_type_code": 1,
                    "rev_num": 0,
                },
            ]
        },
    )
    return result.generated[0]["pod_id"], result.generated[1]["pod_id"]


def test_pod_revision_insert_full_temporal_fields(monkeypatch, tmp_path):
    succ, pred = _seed_pod_pair(monkeypatch, tmp_path)
    result = apply_changeset(
        "pod_revision",
        {
            "inserts": [
                {
                    "successor_id": succ,
                    "predecessor_id": pred,
                    "revision_effect": "full_replacement",
                    "effective_date": "2026-08-01",
                    "amended_scope": "all clauses",
                    "previous_remains_valid": False,
                }
            ]
        },
    )
    assert result.ok
    conn = get_sqlite_connection()
    try:
        row = conn.execute(
            "SELECT revision_effect, effective_date, amended_scope,"
            " previous_remains_valid FROM pod_revision"
        ).fetchone()
        assert tuple(row) == ("full_replacement", "2026-08-01", "all clauses", 0)
    finally:
        conn.close()


def test_pod_revision_insert_partial_temporal_fields(monkeypatch, tmp_path):
    succ, pred = _seed_pod_pair(monkeypatch, tmp_path)
    result = apply_changeset(
        "pod_revision",
        {
            "inserts": [
                {
                    "successor_id": succ,
                    "predecessor_id": pred,
                    "revision_effect": "partial_amendment",
                    "effective_date": "2026-09-01",
                    "amended_scope": "clause 4",
                    "previous_remains_valid": True,
                }
            ]
        },
    )
    assert result.ok
    conn = get_sqlite_connection()
    try:
        row = conn.execute(
            "SELECT revision_effect, effective_date, amended_scope,"
            " previous_remains_valid FROM pod_revision"
        ).fetchone()
        assert tuple(row) == ("partial_amendment", "2026-09-01", "clause 4", 1)
    finally:
        conn.close()


def test_pod_revision_insert_legacy_defaults_to_unknown(monkeypatch, tmp_path):
    succ, pred = _seed_pod_pair(monkeypatch, tmp_path)
    result = apply_changeset(
        "pod_revision",
        {"inserts": [{"successor_id": succ, "predecessor_id": pred}]},
    )
    assert result.ok
    conn = get_sqlite_connection()
    try:
        row = conn.execute(
            "SELECT revision_effect, effective_date, amended_scope,"
            " previous_remains_valid FROM pod_revision"
        ).fetchone()
        assert tuple(row) == ("unknown", None, None, None)
    finally:
        conn.close()


def test_pod_revision_insert_rejects_inconsistent_effect(monkeypatch, tmp_path):
    succ, pred = _seed_pod_pair(monkeypatch, tmp_path)
    result = apply_changeset(
        "pod_revision",
        {
            "inserts": [
                {
                    "successor_id": succ,
                    "predecessor_id": pred,
                    "revision_effect": "full_replacement",
                    "previous_remains_valid": True,
                }
            ]
        },
    )
    assert not result.ok
    assert "previous_remains_valid" in result.errors[0].message


def test_pod_revision_insert_rejects_unknown_effect(monkeypatch, tmp_path):
    succ, pred = _seed_pod_pair(monkeypatch, tmp_path)
    result = apply_changeset(
        "pod_revision",
        {
            "inserts": [
                {
                    "successor_id": succ,
                    "predecessor_id": pred,
                    "revision_effect": "supersedes_entirely",
                }
            ]
        },
    )
    assert not result.ok
    assert "revision_effect" in result.errors[0].message


def test_pod_revision_change_is_delete_plus_insert(monkeypatch, tmp_path):
    succ, pred = _seed_pod_pair(monkeypatch, tmp_path)
    apply_changeset(
        "pod_revision",
        {
            "inserts": [
                {
                    "successor_id": succ,
                    "predecessor_id": pred,
                    "revision_effect": "partial_amendment",
                    "previous_remains_valid": True,
                }
            ]
        },
    )
    result = apply_changeset(
        "pod_revision",
        {
            "deletes": [{"successor_id": succ, "predecessor_id": pred}],
            "inserts": [
                {
                    "successor_id": succ,
                    "predecessor_id": pred,
                    "revision_effect": "full_replacement",
                    "effective_date": "2026-08-01",
                    "amended_scope": "all clauses",
                    "previous_remains_valid": False,
                }
            ],
        },
    )
    assert result.ok
    conn = get_sqlite_connection()
    try:
        row = conn.execute(
            "SELECT revision_effect, effective_date, amended_scope,"
            " previous_remains_valid FROM pod_revision"
        ).fetchone()
        assert tuple(row) == ("full_replacement", "2026-08-01", "all clauses", 0)
    finally:
        conn.close()


def test_pod_revision_insert_rejects_invalid_previous_remains_valid(
    monkeypatch, tmp_path
):
    """An invalid non-null previous_remains_valid must fail validation.

    Previously _bool_or_none coerced it to None, so revision_effect='unknown'
    passed validation and storage silently wrote NULL.
    """
    succ, pred = _seed_pod_pair(monkeypatch, tmp_path)
    for invalid in ("garbage", 2):
        result = apply_changeset(
            "pod_revision",
            {
                "inserts": [
                    {
                        "successor_id": succ,
                        "predecessor_id": pred,
                        "revision_effect": "unknown",
                        "previous_remains_valid": invalid,
                    }
                ]
            },
        )
        assert not result.ok
        assert "previous_remains_valid" in result.errors[0].message


def test_pod_revision_rejects_self_link(monkeypatch, tmp_path):
    _seed_refs(monkeypatch, tmp_path)
    r = apply_changeset(
        "m_pod",
        {
            "inserts": [
                {
                    "id": 900,
                    "pod_name": "POD Baru",
                    "approval_date": "2026-07-15",
                    "institution_code": 4,
                    "pod_type_code": 1,
                    "rev_num": 0,
                }
            ]
        },
    )
    pod_id = r.generated[0]["pod_id"]
    result = apply_changeset(
        "pod_revision",
        {
            "inserts": [
                {
                    "successor_id": pod_id,
                    "predecessor_id": pod_id,
                }
            ]
        },
    )
    assert not result.ok


def test_constraint_violation_returns_row_error_not_exception(monkeypatch, tmp_path):
    _seed_refs(monkeypatch, tmp_path)
    apply_changeset(
        "m_pod",
        {
            "inserts": [
                {
                    "id": 900,
                    "pod_name": "POD Baru",
                    "approval_date": "2026-07-15",
                    "institution_code": 4,
                    "pod_type_code": 1,
                    "rev_num": 0,
                }
            ]
        },
    )
    result = apply_changeset("m_pod", {"updates": [{"id": 900, "pod_name": None}]})
    assert not result.ok
    assert result.errors[0].kind == "apply"
    conn = get_sqlite_connection()
    try:
        assert (
            conn.execute("SELECT pod_name FROM m_pod WHERE id=900").fetchone()[0]
            == "POD Baru"
        )
    finally:
        conn.close()


def test_atomic_no_partial_writes(monkeypatch, tmp_path):
    _seed_refs(monkeypatch, tmp_path)
    result = apply_changeset(
        "m_pod",
        {
            "inserts": [
                {
                    "id": 900,
                    "pod_name": "OK",
                    "approval_date": "2026-07-15",
                    "institution_code": 4,
                    "pod_type_code": 1,
                    "rev_num": 0,
                },
                {
                    "id": 901,
                    "pod_name": "Bad",
                    "approval_date": "2026-07-15",
                    "institution_code": 99,
                    "pod_type_code": 1,
                    "rev_num": 0,
                },
            ]
        },
    )
    assert not result.ok
    conn = get_sqlite_connection()
    try:
        assert conn.execute("SELECT COUNT(*) FROM m_pod").fetchone()[0] == 0
    finally:
        conn.close()
