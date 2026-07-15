import esdc.configs as configs
from esdc.pod_registry.changesets import apply_changeset
from esdc.pod_registry.store import get_sqlite_connection


def _patch_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(configs.Config, "get_db_dir", classmethod(lambda cls: tmp_path))


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
