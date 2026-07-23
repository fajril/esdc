"""LadybugDB instance graph over learned corpus knowledge.

Disposable in-memory index rebuilt from esdc.sqlite (kg_edge + registry +
documents) on first access. Source of truth stays relational; this exists
for Cypher traversal and BM25 entity lookup in chat tools.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

import real_ladybug as lb

from esdc.configs import Config
from esdc.dbmanager import get_duckdb_connection
from esdc.pod_registry.store import get_esdc_sqlite_path

logger = logging.getLogger(__name__)

_NODE_DDL = [
    "CREATE NODE TABLE Document (doc_id STRING PRIMARY KEY, "
    "file_name STRING, doc_type STRING, doc_date STRING, subject STRING)",
    "CREATE NODE TABLE POD (pod_id STRING PRIMARY KEY, pod_name STRING, "
    "letter_num STRING, approval_date STRING, pod_type STRING, rev_num INT64)",
    "CREATE NODE TABLE Project (project_id STRING PRIMARY KEY, project_name STRING)",
    "CREATE NODE TABLE Field (name STRING PRIMARY KEY)",
    "CREATE NODE TABLE WorkingArea (name STRING PRIMARY KEY)",
]
_REL_DDL = [
    "CREATE REL TABLE ABOUT_POD (FROM Document TO POD, "
    "confidence DOUBLE, method STRING)",
    "CREATE REL TABLE ABOUT_FIELD (FROM Document TO Field, "
    "confidence DOUBLE, method STRING)",
    "CREATE REL TABLE ABOUT_WK (FROM Document TO WorkingArea, "
    "confidence DOUBLE, method STRING)",
    "CREATE REL TABLE ABOUT_PROJECT (FROM Document TO Project, "
    "confidence DOUBLE, method STRING)",
    "CREATE REL TABLE HAS_PROJECT (FROM POD TO Project)",
    "CREATE REL TABLE REVISES (FROM POD TO POD)",
    "CREATE REL TABLE IN_FIELD (FROM Project TO Field)",
    "CREATE REL TABLE IN_WK (FROM Field TO WorkingArea)",
]

# rel -> (src label, src key, dst label, dst key, has confidence/method)
_REL_MAP: dict[str, tuple[str, str, str, str, bool]] = {
    "ABOUT_POD": ("Document", "doc_id", "POD", "pod_id", True),
    "ABOUT_FIELD": ("Document", "doc_id", "Field", "name", True),
    "ABOUT_WK": ("Document", "doc_id", "WorkingArea", "name", True),
    "ABOUT_PROJECT": ("Document", "doc_id", "Project", "project_id", True),
    "HAS_PROJECT": ("POD", "pod_id", "Project", "project_id", False),
    "REVISES": ("POD", "pod_id", "POD", "pod_id", False),
    "IN_FIELD": ("Project", "project_id", "Field", "name", False),
    "IN_WK": ("Field", "name", "WorkingArea", "name", False),
}

_LABEL_TO_TYPE = {
    "Document": "document",
    "POD": "pod",
    "Project": "project",
    "Field": "field",
    "WorkingArea": "working_area",
}
_TYPE_TO_LABEL = {v: k for k, v in _LABEL_TO_TYPE.items()}

_NODE_KEY_COL = {
    "Document": "doc_id",
    "POD": "pod_id",
    "Project": "project_id",
    "Field": "name",
    "WorkingArea": "name",
}
_NODE_NAME_COL = {
    "Document": "subject",
    "POD": "pod_name",
    "Project": "project_name",
    "Field": "name",
    "WorkingArea": "name",
}

# (label, fts index name, indexed properties)
_FTS_INDEXES = [
    ("POD", "pod_fts", ["pod_name"]),
    ("Project", "project_fts", ["project_name"]),
    ("Field", "field_fts", ["name"]),
    ("WorkingArea", "wk_fts", ["name"]),
    ("Document", "document_fts", ["subject"]),
]

# Entity-resolution ranking policy for find(): LadybugDB computes BM25 stats
# (idf, avg field length) independently per FTS index, so raw scores from
# separate single-column indexes are NOT comparable across tables -- only
# within the same table/index. `find()` is used to resolve a search string to
# an entity node for graph traversal, so real entity nodes (pod/project/
# field/working_area) should always be preferred over Document nodes. Lower
# number = higher priority; this is the PRIMARY sort key. Raw (uncollapsed)
# score is only used as a tie-breaker within the same type/table, where it
# is directly comparable.
_TYPE_PRIORITY: dict[str, int] = {
    "pod": 0,
    "project": 1,
    "field": 2,
    "working_area": 3,
    "document": 4,
}


def _cypher_escape(s: str) -> str:
    s = s.replace("\\", "\\\\")
    s = s.replace("'", "\\'")
    s = s.replace('"', '\\"')
    return s


class InstanceGraphManager:
    """Rebuild and query a disposable LadybugDB instance graph.

    Built lazily from esdc.sqlite's ``kg_edge`` table plus registry and
    document metadata. When constructed with an explicit ``sqlite_path``
    (tests), each instance is independent. With no argument (production),
    callers share a module-level singleton via ``get_instance_graph()``.
    """

    _instance: InstanceGraphManager | None = None
    _class_lock: threading.Lock = threading.Lock()

    def __new__(
        cls,
        sqlite_path: Path | None = None,
        duckdb_path: Path | None = None,
    ) -> InstanceGraphManager:
        if sqlite_path is not None:
            return super().__new__(cls)
        with cls._class_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(
        self,
        sqlite_path: Path | None = None,
        duckdb_path: Path | None = None,
    ) -> None:
        # Narrow benign race: two no-arg constructions could both pass this
        # check before either sets `_ctor_done` (this runs outside
        # `_class_lock`, unlike the singleton assignment in __new__). At
        # worst that re-initializes _build_lock/_built/etc a second time on
        # the same shared instance; _build() itself is idempotent and
        # guarded by _build_lock, so a stray re-init is safe, not corrupt.
        if sqlite_path is None and getattr(self, "_ctor_done", False):
            return
        self._sqlite_path = sqlite_path or get_esdc_sqlite_path()
        self._duckdb_path = (
            duckdb_path if duckdb_path is not None else Config.get_db_file()
        )
        self._build_lock = threading.Lock()
        self._built = False
        self._available = False
        self._db: lb.Database | None = None
        self._conn: lb.Connection | None = None
        self._ctor_done = True

    @classmethod
    def reset_for_tests(cls) -> None:
        with cls._class_lock:
            if cls._instance is not None:
                cls._instance.close()
            cls._instance = None

    def close(self) -> None:
        if self._conn is not None:
            with contextlib.suppress(Exception):
                self._conn.close()
            self._conn = None
        if self._db is not None:
            with contextlib.suppress(Exception):
                self._db.close()
            self._db = None
        self._built = False
        self._available = False

    # -- build --------------------------------------------------------

    def _ensure_built(self) -> None:
        with self._build_lock:
            if self._built:
                return
            self._build()
            self._built = True

    def _build(self) -> None:
        if not self._sqlite_path.exists():
            self._available = False
            return
        conn = sqlite3.connect(self._sqlite_path)
        conn.row_factory = sqlite3.Row
        try:
            has_table = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='kg_edge'"
            ).fetchone()
            if not has_table:
                self._available = False
                return
            edge_count = conn.execute("SELECT COUNT(*) FROM kg_edge").fetchone()[0]
            if not edge_count:
                self._available = False
                return

            project_names = self._load_project_names()

            self._db = lb.Database(":memory:")
            self._conn = lb.Connection(self._db)
            self._load_extensions()
            self._create_schema()
            self._load_nodes(conn, project_names)
            self._load_edges(conn)
            self._create_fts_indexes()
            self._available = True
            logger.info("[InstanceGraph] graph_built | edges=%d", edge_count)
        except Exception as e:
            logger.error("[InstanceGraph] build_failed | error=%s", e)
            self._available = False
        finally:
            conn.close()

    def _load_project_names(self) -> dict[str, str]:
        """Look up project_id -> project_name from duckdb project_resources.

        The instance graph is built from sqlite kg_edge, which only has
        project_ids; real names live in duckdb. This is a best-effort
        enrichment: a missing/locked duckdb file must not break graph
        availability, so any failure here degrades to an empty map (Project
        nodes then fall back to project_id as their name, same as before
        this method existed).
        """
        if self._duckdb_path is None or not Path(self._duckdb_path).exists():
            return {}
        conn = None
        try:
            conn = get_duckdb_connection(self._duckdb_path, read_only=True)
            rows = conn.execute(
                """
                SELECT project_id, project_name FROM (
                    SELECT project_id, project_name,
                           ROW_NUMBER() OVER (
                               PARTITION BY project_id
                               ORDER BY report_year DESC, project_name
                           ) AS rn
                    FROM project_resources
                ) WHERE rn = 1
                """
            ).fetchall()
            return {str(pid): name for pid, name in rows if name}
        except Exception as e:
            logger.warning(
                "[InstanceGraph] project_names_unavailable | path=%s error=%s",
                self._duckdb_path,
                e,
            )
            return {}
        finally:
            if conn is not None:
                with contextlib.suppress(Exception):
                    conn.close()

    def _load_extensions(self) -> None:
        assert self._conn is not None
        self._conn.execute("INSTALL FTS")
        self._conn.execute("LOAD EXTENSION FTS")

    def _create_schema(self) -> None:
        assert self._conn is not None
        for stmt in (*_NODE_DDL, *_REL_DDL):
            try:
                self._conn.execute(stmt)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning("[InstanceGraph] schema_error | %s", e)

    def _load_nodes(
        self, conn: sqlite3.Connection, project_names: dict[str, str]
    ) -> None:
        assert self._conn is not None

        for row in conn.execute(
            "SELECT m.pod_id, m.pod_name, m.pod_letter_num, m.approval_date, "
            "t.pod_type, m.rev_num FROM m_pod m "
            "JOIN r_pod_type t ON t.code = m.pod_type_code"
        ).fetchall():
            try:
                pod_id = _cypher_escape(row["pod_id"])
                pod_name = _cypher_escape(row["pod_name"] or "")
                letter_num = _cypher_escape(row["pod_letter_num"] or "")
                approval_date = _cypher_escape(row["approval_date"] or "")
                pod_type = _cypher_escape(row["pod_type"] or "")
                rev_num = int(row["rev_num"] or 0)
                self._conn.execute(
                    f"CREATE (:POD {{pod_id: '{pod_id}', pod_name: '{pod_name}', "
                    f"letter_num: '{letter_num}', "
                    f"approval_date: '{approval_date}', "
                    f"pod_type: '{pod_type}', rev_num: {rev_num}}})"
                )
            except Exception as e:
                logger.warning("[InstanceGraph] pod_node_error | %s", e)

        for row in conn.execute(
            "SELECT doc_id, file_name, doc_type, doc_date, subject FROM documents"
        ).fetchall():
            try:
                doc_id = _cypher_escape(row["doc_id"])
                file_name = _cypher_escape(row["file_name"] or "")
                doc_type = _cypher_escape(row["doc_type"] or "")
                doc_date = _cypher_escape(row["doc_date"] or "")
                subject = _cypher_escape(row["subject"] or "")
                self._conn.execute(
                    f"CREATE (:Document {{doc_id: '{doc_id}', "
                    f"file_name: '{file_name}', doc_type: '{doc_type}', "
                    f"doc_date: '{doc_date}', subject: '{subject}'}})"
                )
            except Exception as e:
                logger.warning("[InstanceGraph] document_node_error | %s", e)

        for row in conn.execute(
            "SELECT DISTINCT dst_type, dst_id FROM kg_edge "
            "WHERE dst_type IN ('project','field','working_area') "
            "UNION "
            "SELECT DISTINCT src_type, src_id FROM kg_edge "
            "WHERE src_type IN ('project','field')"
        ).fetchall():
            entity_type, entity_id = row[0], row[1]
            try:
                esc_id = _cypher_escape(entity_id)
                if entity_type == "project":
                    esc_name = _cypher_escape(project_names.get(entity_id, entity_id))
                    self._conn.execute(
                        f"CREATE (:Project {{project_id: '{esc_id}', "
                        f"project_name: '{esc_name}'}})"
                    )
                elif entity_type == "field":
                    self._conn.execute(f"CREATE (:Field {{name: '{esc_id}'}})")
                elif entity_type == "working_area":
                    self._conn.execute(f"CREATE (:WorkingArea {{name: '{esc_id}'}})")
            except Exception as e:
                logger.warning(
                    "[InstanceGraph] entity_node_error | type=%s id=%s err=%s",
                    entity_type,
                    entity_id,
                    e,
                )

    def _load_edges(self, conn: sqlite3.Connection) -> None:
        assert self._conn is not None
        for row in conn.execute(
            "SELECT src_type, src_id, rel, dst_type, dst_id, confidence, method "
            "FROM kg_edge"
        ).fetchall():
            rel = row["rel"]
            mapping = _REL_MAP.get(rel)
            if mapping is None:
                logger.debug("[InstanceGraph] skip_unknown_rel | rel=%s", rel)
                continue
            src_label, src_key, dst_label, dst_key, has_props = mapping
            cypher = (
                f"MATCH (a:{src_label} "
                f"{{{src_key}: '{_cypher_escape(row['src_id'])}'}}), "
                f"(b:{dst_label} "
                f"{{{dst_key}: '{_cypher_escape(row['dst_id'])}'}}) "
            )
            if has_props:
                cypher += (
                    f"CREATE (a)-[:{rel} {{confidence: "
                    f"{float(row['confidence'])}, "
                    f"method: '{_cypher_escape(row['method'] or '')}'}}]->(b)"
                )
            else:
                cypher += f"CREATE (a)-[:{rel}]->(b)"
            try:
                self._conn.execute(cypher)
            except Exception as e:
                logger.warning(
                    "[InstanceGraph] edge_error | rel=%s src=%s dst=%s err=%s",
                    rel,
                    row["src_id"],
                    row["dst_id"],
                    e,
                )

    def _create_fts_indexes(self) -> None:
        assert self._conn is not None
        for table, idx_name, props in _FTS_INDEXES:
            try:
                props_literal = "[" + ", ".join(f"'{p}'" for p in props) + "]"
                self._conn.execute(
                    f"CALL CREATE_FTS_INDEX('{table}', '{idx_name}', "
                    f"{props_literal}, stemmer := 'indonesian')"
                )
                logger.debug("[InstanceGraph] fts_created | %s.%s", table, idx_name)
            except Exception as e:
                logger.warning(
                    "[InstanceGraph] fts_error | table=%s idx=%s err=%s",
                    table,
                    idx_name,
                    e,
                )

    def _execute_cypher(self, cypher: str) -> list[list[Any]]:
        """Execute a Cypher query and return result rows.

        Wraps LadybugDB's result iteration API which differs from
        the type stubs (uses has_next/get_next at runtime).
        """
        assert self._conn is not None
        result = self._conn.execute(cypher)
        rows: list[list[Any]] = []
        while result.has_next():  # type: ignore[union-attr]
            rows.append(result.get_next())  # type: ignore[union-attr]
        return rows

    # -- public API -----------------------------------------------------

    def is_available(self) -> bool:
        self._ensure_built()
        return self._available

    def find(
        self,
        text: str,
        top_k: int = 5,
        entity_type: str | None = None,
    ) -> list[dict[str, Any]]:
        self._ensure_built()
        if not self._available:
            return []
        escaped = _cypher_escape(text)
        all_results: list[dict[str, Any]] = []
        indexes = _FTS_INDEXES
        if entity_type is not None:
            # Restrict to the requested type's index BEFORE the top_k
            # truncation below, so a valid match of that type is never
            # lost to higher-priority cross-type hits filling up top_k
            # first (see explore_entity's entity_type filter).
            target_label = _TYPE_TO_LABEL.get(entity_type)
            indexes = [idx for idx in _FTS_INDEXES if idx[0] == target_label]
        for table, idx_name, _props in indexes:
            key_col = _NODE_KEY_COL[table]
            name_col = _NODE_NAME_COL[table]
            cypher = (
                f"CALL QUERY_FTS_INDEX('{table}', '{idx_name}', '{escaped}', "
                f"top := {top_k}) RETURN node.{key_col}, node.{name_col}, score"
            )
            try:
                rows = self._execute_cypher(cypher)
            except Exception as e:
                logger.warning(
                    "[InstanceGraph] find_error | table=%s error=%s", table, e
                )
                continue
            if not rows:
                continue
            for row in rows:
                all_results.append(
                    {
                        "entity_type": _LABEL_TO_TYPE[table],
                        "entity_id": row[0],
                        "name": row[1],
                        "score": row[2],
                    }
                )
        # See _TYPE_PRIORITY docstring: type priority is the primary sort
        # key (deterministic, intentional entity-resolution policy); raw
        # score only breaks ties within the same type, where it is
        # actually comparable.
        all_results.sort(
            key=lambda r: (_TYPE_PRIORITY.get(r["entity_type"], 99), -r["score"])
        )
        return all_results[:top_k]

    def neighbors(
        self, entity_type: str, entity_id: str
    ) -> dict[str, list[dict[str, Any]]]:
        self._ensure_built()
        result: dict[str, list[dict[str, Any]]] = {}
        if not self._available:
            return result
        label = _TYPE_TO_LABEL.get(entity_type)
        if label is None:
            return result
        key_col = _NODE_KEY_COL[label]
        esc_id = _cypher_escape(entity_id)

        for rel, (
            src_label,
            src_key,
            dst_label,
            dst_key,
            has_props,
        ) in _REL_MAP.items():
            items: list[dict[str, Any]] = []
            directions: list[tuple[str, str, str, str]] = []
            # Check both directions independently: self-referential rel
            # types (e.g. REVISES, POD->POD) have src_label == dst_label,
            # so an if/elif would miss one direction.
            if src_label == label:
                directions.append((dst_label, dst_key, "outbound", ""))
            if dst_label == label:
                directions.append((src_label, src_key, "inbound", ""))

            for other_label, other_key, direction, _ in directions:
                if direction == "outbound":
                    cols = [f"b.{other_key}", f"b.{_NODE_NAME_COL[other_label]}"]
                    if has_props:
                        cols += ["r.confidence", "r.method"]
                    cypher = (
                        f"MATCH (a:{label} {{{key_col}: '{esc_id}'}})"
                        f"-[r:{rel}]->(b:{other_label}) "
                        f"RETURN {', '.join(cols)}"
                    )
                else:
                    cols = [f"a.{other_key}", f"a.{_NODE_NAME_COL[other_label]}"]
                    if has_props:
                        cols += ["r.confidence", "r.method"]
                    cypher = (
                        f"MATCH (a:{other_label})-[r:{rel}]->"
                        f"(b:{label} {{{key_col}: '{esc_id}'}}) "
                        f"RETURN {', '.join(cols)}"
                    )

                try:
                    rows = self._execute_cypher(cypher)
                except Exception as e:
                    logger.warning(
                        "[InstanceGraph] neighbors_error | rel=%s error=%s", rel, e
                    )
                    continue
                for row in rows:
                    item: dict[str, Any] = {
                        "entity_type": _LABEL_TO_TYPE[other_label],
                        "entity_id": row[0],
                        "name": row[1],
                        "confidence": row[2] if has_props else None,
                        "method": row[3] if has_props else None,
                        "direction": direction,
                    }
                    items.append(item)
            if items:
                result[rel] = items
        return result

    def query(self, cypher: str) -> list[dict[str, Any]]:
        self._ensure_built()
        if not self._available:
            return []
        assert self._conn is not None
        try:
            raw = self._conn.execute(cypher)
            columns: list[str] = raw.get_column_names()  # type: ignore[union-attr]
            rows: list[list[Any]] = []
            while raw.has_next():  # type: ignore[union-attr]
                rows.append(raw.get_next())  # type: ignore[union-attr]
        except Exception as e:
            logger.warning(
                "[InstanceGraph] query_error | cypher=%s error=%s", cypher[:80], e
            )
            return []
        return [dict(zip(columns, row, strict=True)) for row in rows]


def get_instance_graph() -> InstanceGraphManager:
    """Module-level singleton for production (no-arg) use."""
    return InstanceGraphManager()
