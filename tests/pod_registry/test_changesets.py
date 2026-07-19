import esdc.configs as configs
from esdc.corpus.store import _SQLITE_DOC_DDL
from esdc.pod_registry.changesets import apply_changeset
from esdc.pod_registry.store import get_sqlite_connection


def _patch_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(configs.Config, "get_db_dir", classmethod(lambda cls: tmp_path))


def _seed_documents(doc_ids):
    conn = get_sqlite_connection()
    conn.execute(_SQLITE_DOC_DDL)
    for i, doc_id in enumerate(doc_ids):
        conn.execute(
            "INSERT INTO documents (doc_id, file_name, file_path, file_hash,"
            " markdown, extraction_method, embedding_model)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (doc_id, f"{doc_id}.pdf", f"/x/{doc_id}.pdf", doc_id * 4,
             "# isi", "native", "fake-embed"),
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
    apply_changeset(
        "pod_document", {"inserts": [{"pod_id": 900, "doc_id": "abc123"}]}
    )
    result = apply_changeset(
        "pod_document", {"inserts": [{"pod_id": 900, "doc_id": "abc123"}]}
    )
    assert not result.ok
    assert "already exists" in result.errors[0].message


def test_delete_m_pod_blocked_by_pod_document(monkeypatch, tmp_path):
    _insert_pod_900(monkeypatch, tmp_path)
    apply_changeset(
        "pod_document", {"inserts": [{"pod_id": 900, "doc_id": "abc123"}]}
    )
    result = apply_changeset("m_pod", {"deletes": [{"id": 900}]})
    assert not result.ok
    assert "pod_document" in result.errors[0].message


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
