# tests/pod_registry/test_publish.py
import esdc.configs as configs
from esdc.dbmanager import get_duckdb_connection
from esdc.loaders import _METADATA_TABLE
from esdc.pod_registry.publish import publish_pod_registry
from esdc.pod_registry.store import get_sqlite_connection


def _patch_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(configs.Config, "get_db_dir", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(
        configs.Config, "get_db_file", classmethod(lambda cls: tmp_path / "esdc.duckdb")
    )


def _seed(monkeypatch, tmp_path):
    _patch_dirs(monkeypatch, tmp_path)
    conn = get_sqlite_connection()
    conn.execute("INSERT INTO r_institution (code, institution) VALUES (3, 'BP Migas')")
    conn.execute("INSERT INTO r_pod_type (code, pod_type) VALUES (2, 'POD/Waterflood/EOR')")
    conn.executemany(
        "INSERT INTO m_pod (id, pod_id, pod_name, pod_letter_num, approval_date,"
        " institution_code, pod_type_code, rev_num, approval_seq)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        [
            (645, "PL-2003-0005-3-2-0", "POD Mengoepeh", "294/BP", "2003-11-21", 3, 2, 0, 5),
            (700, "PL-2005-0051-3-2-1", "POD Mengoepeh Rev", "51/BP", "2005-06-01", 3, 2, 1, 51),
        ],
    )
    conn.execute("INSERT INTO project_pod (pod_id, project_id) VALUES (645, 'P-2403431-01')")
    conn.execute("INSERT INTO pod_document (pod_id, doc_id) VALUES (645, 'ccbd4f3f27635c76')")
    conn.execute(
        "INSERT INTO pod_revision (successor_id, predecessor_id)"
        " VALUES ('PL-2005-0051-3-2-1', 'PL-2003-0005-3-2-0')"
    )
    conn.commit()
    conn.close()


def test_publish_creates_denormalized_tables(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    results = publish_pod_registry()
    assert {r.table_name for r in results} == {
        "pod_registry", "pod_project", "pod_document",
    }

    conn = get_duckdb_connection(tmp_path / "esdc.duckdb", read_only=True)
    try:
        rows = conn.execute(
            "SELECT pod_id, institution, pod_type, preceded_by, superseded_by"
            " FROM pod_registry ORDER BY approval_seq"
        ).fetchall()
        assert rows[0][1] == "BP Migas"
        assert rows[0][4] == "PL-2005-0051-3-2-1"   # original superseded by rev
        assert rows[1][3] == "PL-2003-0005-3-2-0"   # rev preceded by original
        assert rows[0][3] is None and rows[1][4] is None

        link = conn.execute("SELECT pod_id, project_id FROM pod_project").fetchall()
        assert link == [("PL-2003-0005-3-2-0", "P-2403431-01")]

        # canonical PL-... pod_id in the doc snapshot, not the surrogate int
        docs = conn.execute("SELECT pod_id, doc_id FROM pod_document").fetchall()
        assert docs == [("PL-2003-0005-3-2-0", "ccbd4f3f27635c76")]

        meta = conn.execute(
            f"SELECT table_name FROM {_METADATA_TABLE} ORDER BY table_name"
        ).fetchall()
        assert ("pod_registry",) in meta and ("pod_project",) in meta
        assert ("pod_document",) in meta
    finally:
        conn.close()


def test_publish_is_idempotent(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    publish_pod_registry()
    publish_pod_registry()  # replaces, no duplicates
    conn = get_duckdb_connection(tmp_path / "esdc.duckdb", read_only=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM pod_registry").fetchone()[0] == 2
    finally:
        conn.close()
