"""LadybugDB instance graph over learned corpus knowledge.

Disposable in-memory index rebuilt from esdc.sqlite (kg_edge + registry +
documents) on first access. Source of truth stays relational; this exists
for Cypher traversal and BM25 entity lookup in chat tools.
"""

from __future__ import annotations

import contextlib
import logging
import re
import sqlite3
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import real_ladybug as lb
import yaml

from esdc.configs import Config
from esdc.dbmanager import get_duckdb_connection
from esdc.pod_registry.store import get_esdc_sqlite_path

logger = logging.getLogger(__name__)

_INSTANCE_GRAPH_SCHEMA_PATH = Path(__file__).parent / "instance_graph_schema.yaml"
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SCALAR_TYPES = frozenset({"STRING", "INT64", "DOUBLE"})
_REQUIRED_TEMPORAL_PROPS = ("valid_from", "valid_to", "properties_json")
_ABOUT_EVIDENCE_PROPS = ("confidence", "method")


class _SchemaDriftError(ValueError):
    """Raised when kg_edge no longer matches the declared graph contract."""


@dataclass(frozen=True)
class _NodeDef:
    label: str
    corpus_type: str
    key: str
    name: str
    properties: Mapping[str, str]
    fts_index: str | None
    fts_properties: tuple[str, ...]


@dataclass(frozen=True)
class _RelDef:
    rel: str
    from_label: str
    to_label: str
    properties: Mapping[str, str]


@dataclass(frozen=True)
class _InstanceGraphSchema:
    nodes: Mapping[str, _NodeDef]
    relationships: Mapping[str, _RelDef]


def _load_instance_graph_schema(
    path: Path = _INSTANCE_GRAPH_SCHEMA_PATH,
) -> _InstanceGraphSchema:
    """Load and strictly validate the executable LadybugDB topology contract."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    version = data.get("schema_version")
    if version != 1:
        raise ValueError(f"{path}: schema_version must be 1, got {version!r}")
    nodes_raw = data.get("nodes")
    rels_raw = data.get("relationships")
    if not isinstance(nodes_raw, dict):
        raise ValueError(f"{path}: nodes must be a mapping")
    if not isinstance(rels_raw, dict):
        raise ValueError(f"{path}: relationships must be a mapping")

    def validate_identifier(context: str, value: Any) -> str:
        if not isinstance(value, str) or not _IDENTIFIER_RE.match(value):
            raise ValueError(f"{context}: invalid identifier {value!r}")
        return value

    def validate_properties(context: str, props_raw: Any) -> dict[str, str]:
        if not isinstance(props_raw, dict):
            raise ValueError(f"{context}: must be a mapping")
        properties: dict[str, str] = {}
        for prop_name, prop_type in props_raw.items():
            prop_path = f"{context}.{prop_name}"
            prop_name = validate_identifier(prop_path, prop_name)
            if prop_type not in _SCALAR_TYPES:
                raise ValueError(f"{prop_path}: unsupported type {prop_type!r}")
            properties[prop_name] = prop_type
        return properties

    node_defs: dict[str, _NodeDef] = {}
    seen_corpus_types: dict[str, str] = {}
    for label, raw in nodes_raw.items():
        node_path = f"{path}: nodes.{label}"
        label = validate_identifier(node_path, label)
        if not isinstance(raw, dict):
            raise ValueError(f"{node_path} must be a mapping")
        corpus_type = validate_identifier(
            f"{node_path}.corpus_type", raw.get("corpus_type")
        )
        prior_label = seen_corpus_types.get(corpus_type)
        if prior_label is not None:
            raise ValueError(
                f"{node_path}.corpus_type duplicates {corpus_type!r} "
                f"from nodes.{prior_label}"
            )
        seen_corpus_types[corpus_type] = label
        properties = validate_properties(
            f"{node_path}.properties", raw.get("properties")
        )
        key = validate_identifier(f"{node_path}.key", raw.get("key"))
        if key not in properties:
            raise ValueError(f"{node_path}.key {key!r} must name a declared property")
        name = validate_identifier(f"{node_path}.name", raw.get("name"))
        if name not in properties:
            raise ValueError(f"{node_path}.name {name!r} must name a declared property")

        fts_index: str | None = None
        fts_properties: tuple[str, ...] = ()
        fts_raw = raw.get("fts")
        if fts_raw is not None:
            if not isinstance(fts_raw, dict):
                raise ValueError(f"{node_path}.fts must be a mapping")
            fts_index = validate_identifier(
                f"{node_path}.fts.index", fts_raw.get("index")
            )
            fts_props_raw = fts_raw.get("properties")
            if not isinstance(fts_props_raw, list) or not fts_props_raw:
                raise ValueError(f"{node_path}.fts.properties must be a non-empty list")
            for fts_prop in fts_props_raw:
                fts_prop_path = f"{node_path}.fts.properties.{fts_prop}"
                fts_prop = validate_identifier(fts_prop_path, fts_prop)
                if fts_prop not in properties:
                    raise ValueError(f"{fts_prop_path} must name a declared property")
                fts_properties += (fts_prop,)
        node_defs[label] = _NodeDef(
            label=label,
            corpus_type=corpus_type,
            key=key,
            name=name,
            properties=MappingProxyType(properties),
            fts_index=fts_index,
            fts_properties=fts_properties,
        )

    rel_defs: dict[str, _RelDef] = {}
    for rel, raw in rels_raw.items():
        rel_path = f"{path}: relationships.{rel}"
        rel = validate_identifier(rel_path, rel)
        if not isinstance(raw, dict):
            raise ValueError(f"{rel_path} must be a mapping")
        from_label = validate_identifier(f"{rel_path}.from", raw.get("from"))
        to_label = validate_identifier(f"{rel_path}.to", raw.get("to"))
        if from_label not in node_defs:
            raise ValueError(f"{rel_path}.from references unknown node {from_label!r}")
        if to_label not in node_defs:
            raise ValueError(f"{rel_path}.to references unknown node {to_label!r}")
        properties = validate_properties(
            f"{rel_path}.properties", raw.get("properties")
        )
        for temporal in _REQUIRED_TEMPORAL_PROPS:
            if temporal not in properties:
                raise ValueError(f"{rel_path}.properties.{temporal} is required")
        if rel.startswith("ABOUT_"):
            for evidence in _ABOUT_EVIDENCE_PROPS:
                if evidence not in properties:
                    raise ValueError(
                        f"{rel_path}.properties.{evidence} is required "
                        "for ABOUT_ relationships"
                    )
        rel_defs[rel] = _RelDef(
            rel=rel,
            from_label=from_label,
            to_label=to_label,
            properties=MappingProxyType(properties),
        )

    return _InstanceGraphSchema(
        nodes=MappingProxyType(node_defs),
        relationships=MappingProxyType(rel_defs),
    )


def _node_ddl(node: _NodeDef) -> str:
    """Generate one LadybugDB node table from a schema node definition."""
    ordered_props = [node.key]
    ordered_props += [p for p in node.properties if p != node.key]
    columns = ", ".join(
        f"{p} {node.properties[p]}" + (" PRIMARY KEY" if p == node.key else "")
        for p in ordered_props
    )
    return f"CREATE NODE TABLE {node.label} ({columns})"


def _rel_ddl(rel: _RelDef) -> str:
    """Generate one LadybugDB rel table from a schema relationship."""
    columns = ", ".join(f"{p} {t}" for p, t in rel.properties.items())
    return (
        f"CREATE REL TABLE {rel.rel} (FROM {rel.from_label} "
        f"TO {rel.to_label}, {columns})"
    )


_SCHEMA = _load_instance_graph_schema()
_NODE_DDL = [_node_ddl(node) for node in _SCHEMA.nodes.values()]
_REL_DDL = [_rel_ddl(rel) for rel in _SCHEMA.relationships.values()]

# rel -> (src label, src key, dst label, dst key, has confidence/method)
_REL_MAP: dict[str, tuple[str, str, str, str, bool]] = {
    rel_name: (
        rel_def.from_label,
        _SCHEMA.nodes[rel_def.from_label].key,
        rel_def.to_label,
        _SCHEMA.nodes[rel_def.to_label].key,
        set(_ABOUT_EVIDENCE_PROPS) <= set(rel_def.properties),
    )
    for rel_name, rel_def in _SCHEMA.relationships.items()
}

_LABEL_TO_TYPE = {label: node.corpus_type for label, node in _SCHEMA.nodes.items()}
_TYPE_TO_LABEL = {v: k for k, v in _LABEL_TO_TYPE.items()}

_NODE_KEY_COL = {label: node.key for label, node in _SCHEMA.nodes.items()}
_NODE_NAME_COL = {label: node.name for label, node in _SCHEMA.nodes.items()}

# (label, fts index name, indexed properties)
_FTS_INDEXES = [
    (label, node.fts_index, list(node.fts_properties))
    for label, node in _SCHEMA.nodes.items()
    if node.fts_index is not None
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


# -- public query read-only guard -------------------------------------
#
# The instance graph is a disposable read-only index rebuilt from sqlite;
# public query() must never mutate it. Native read-only connections are
# unavailable for :memory: databases, so queries are validated lexically
# before execution. find()/neighbors() build their own read-only queries
# (e.g. CALL QUERY_FTS_INDEX) and are NOT routed through this guard.

# Known Cypher clauses; multi-word clauses must precede their single-word
# parts so the alternation matches them whole (OPTIONAL MATCH vs MATCH).
_KNOWN_CLAUSES = (
    "OPTIONAL MATCH",
    "DETACH DELETE",
    "ORDER BY",
    "UNION ALL",
    "LOAD CSV",
    "MATCH",
    "WHERE",
    "WITH",
    "UNWIND",
    "RETURN",
    "SKIP",
    "LIMIT",
    "CREATE",
    "MERGE",
    "SET",
    "DELETE",
    "REMOVE",
    "DROP",
    "CALL",
    "YIELD",
    "COPY",
    "ALTER",
    "INSTALL",
    "UPDATE",
    "USE",
    "EXPLAIN",
    "PROFILE",
    "FOREACH",
    "UNION",
    "ATTACH",
    "BEGIN",
    "COMMIT",
    "ROLLBACK",
)
# The only clauses a public read query may use.
_ALLOWED_CLAUSES = frozenset(
    {
        "MATCH",
        "OPTIONAL MATCH",
        "WHERE",
        "WITH",
        "UNWIND",
        "RETURN",
        "ORDER BY",
        "SKIP",
        "LIMIT",
    }
)
_READ_START_RE = re.compile(r"(?:OPTIONAL\s+MATCH|MATCH|UNWIND)\b")
# mask string/backtick literals and // and /* */ comments so keywords and
# semicolons inside them never count toward clause/statement validation
_READ_ONLY_MASK_RE = re.compile(
    r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|`[^`]*`|//[^\n]*|/\*.*?\*/",
    re.DOTALL,
)
_CLAUSE_RE = re.compile(
    "|".join(
        r"\b" + r"\s+".join(re.escape(w) for w in c.split()) + r"\b"
        for c in _KNOWN_CLAUSES
    )
)


def _validate_read_only(cypher: str) -> None:
    """Reject anything but a single read-only Cypher statement.

    A public query must begin with MATCH / OPTIONAL MATCH / UNWIND, contain
    RETURN, use only allowlisted clauses, and hold at most one trailing
    semicolon. Literals/comments are masked before scanning, so clause
    keywords and semicolons inside them never count.
    """
    query = cypher.strip()
    if not query:
        raise ValueError("read-only: empty query")
    masked = _READ_ONLY_MASK_RE.sub(" ", query).upper().strip()
    if masked.endswith(";"):
        masked = masked[:-1].rstrip()
    if ";" in masked:
        raise ValueError("read-only: multiple statements are not allowed")
    if not _READ_START_RE.match(masked):
        raise ValueError(
            "read-only: query must begin with MATCH, OPTIONAL MATCH, or UNWIND"
        )
    if not re.search(r"\bRETURN\b", masked):
        raise ValueError("read-only: query must contain RETURN")
    for clause in _CLAUSE_RE.findall(masked):
        normalized = " ".join(clause.split())
        if normalized not in _ALLOWED_CLAUSES:
            raise ValueError(f"read-only: {normalized} is not allowed")


def _rel_props_literal(row: sqlite3.Row) -> str:
    """Format valid_from/valid_to/properties_json as a Cypher props body."""
    parts: list[str] = []
    for key in ("valid_from", "valid_to", "properties_json"):
        value = row[key]
        if value is None:
            parts.append(f"{key}: null")
        else:
            parts.append(f"{key}: '{_cypher_escape(str(value))}'")
    return ", ".join(parts)


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
            self.close()
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
        pod_label = _TYPE_TO_LABEL["pod"]
        pod_key = _NODE_KEY_COL[pod_label]

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
                    f"CREATE (:{pod_label} {{{pod_key}: '{pod_id}', "
                    f"pod_name: '{pod_name}', "
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
                doc_label = _TYPE_TO_LABEL["document"]
                doc_key = _NODE_KEY_COL[doc_label]
                doc_id = _cypher_escape(row["doc_id"])
                file_name = _cypher_escape(row["file_name"] or "")
                doc_type = _cypher_escape(row["doc_type"] or "")
                doc_date = _cypher_escape(row["doc_date"] or "")
                subject = _cypher_escape(row["subject"] or "")
                self._conn.execute(
                    f"CREATE (:{doc_label} {{{doc_key}: '{doc_id}', "
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
            if not isinstance(entity_id, str):
                continue
            try:
                esc_id = _cypher_escape(entity_id)
                label = _TYPE_TO_LABEL.get(entity_type)
                if label is None:
                    continue
                key_col = _NODE_KEY_COL[label]
                if entity_type == "project":
                    esc_name = _cypher_escape(project_names.get(entity_id, entity_id))
                    self._conn.execute(
                        f"CREATE (:{label} {{{key_col}: '{esc_id}', "
                        f"{_NODE_NAME_COL[label]}: '{esc_name}'}})"
                    )
                elif entity_type == "field" or entity_type == "working_area":
                    self._conn.execute(f"CREATE (:{label} {{{key_col}: '{esc_id}'}})")
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
            "SELECT src_type, src_id, rel, dst_type, dst_id, confidence, "
            "method, valid_from, valid_to, properties_json FROM kg_edge"
        ).fetchall():
            rel = row["rel"]
            rel_def = _SCHEMA.relationships.get(rel)
            if rel_def is None:
                raise _SchemaDriftError(
                    f"kg_edge schema_drift | rel={rel!r} "
                    f"src={row['src_type']}:{row['src_id']} "
                    f"dst={row['dst_type']}:{row['dst_id']} "
                    "unknown relationship"
                )
            expected_src = _SCHEMA.nodes[rel_def.from_label].corpus_type
            expected_dst = _SCHEMA.nodes[rel_def.to_label].corpus_type
            if row["src_type"] != expected_src or row["dst_type"] != expected_dst:
                raise _SchemaDriftError(
                    f"kg_edge schema_drift | rel={rel} "
                    f"src={row['src_type']}:{row['src_id']} "
                    f"dst={row['dst_type']}:{row['dst_id']} "
                    f"expected {expected_src}->{expected_dst}"
                )
            src_label, src_key = rel_def.from_label, _NODE_KEY_COL[rel_def.from_label]
            dst_label, dst_key = rel_def.to_label, _NODE_KEY_COL[rel_def.to_label]
            for label, key, node_id, side in (
                (src_label, src_key, row["src_id"], "src"),
                (dst_label, dst_key, row["dst_id"], "dst"),
            ):
                exists = self._execute_cypher(
                    f"MATCH (n:{label} {{{key}: '{_cypher_escape(node_id)}'}}) "
                    f"RETURN n.{key} LIMIT 1"
                )
                if not exists:
                    raise _SchemaDriftError(
                        f"kg_edge schema_drift | rel={rel} "
                        f"{side}={row[f'{side}_type']}:{node_id} "
                        f"missing projected node {label} ({key})"
                    )
            cypher = (
                f"MATCH (a:{src_label} "
                f"{{{src_key}: '{_cypher_escape(row['src_id'])}'}}), "
                f"(b:{dst_label} "
                f"{{{dst_key}: '{_cypher_escape(row['dst_id'])}'}}) "
            )
            props = _rel_props_literal(row)
            if set(_ABOUT_EVIDENCE_PROPS) <= set(rel_def.properties):
                props = (
                    f"confidence: {float(row['confidence'])}, "
                    f"method: '{_cypher_escape(row['method'] or '')}', "
                    f"{props}"
                )
            cypher += f"CREATE (a)-[:{rel} {{{props}}}]->(b)"
            self._conn.execute(cypher)

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
        _validate_read_only(cypher)
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
