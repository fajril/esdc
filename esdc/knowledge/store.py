"""SQLite-backed store for learned knowledge: edges, claims, proposals, state.

Relational tables in esdc.sqlite are the source of truth; the LadybugDB
instance graph used by chat is rebuilt from them at startup and is fully
disposable.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

_DDL = """
CREATE TABLE IF NOT EXISTS kg_edge (
    src_type   TEXT NOT NULL,
    src_id     TEXT NOT NULL,
    rel        TEXT NOT NULL,
    dst_type   TEXT NOT NULL,
    dst_id     TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    method     TEXT NOT NULL,
    evidence   TEXT,
    learned_at TEXT DEFAULT (datetime('now')),
    UNIQUE (src_type, src_id, rel, dst_type, dst_id)
);
CREATE INDEX IF NOT EXISTS idx_kg_edge_src ON kg_edge (src_type, src_id);
CREATE INDEX IF NOT EXISTS idx_kg_edge_dst ON kg_edge (dst_type, dst_id);

CREATE TABLE IF NOT EXISTS kg_claim (
    claim_id     INTEGER PRIMARY KEY,
    doc_id       TEXT NOT NULL,
    claim_type   TEXT NOT NULL,
    subject_type TEXT,
    subject_id   TEXT,
    predicate    TEXT NOT NULL,
    value_json   TEXT NOT NULL,
    evidence     TEXT,
    confidence   REAL NOT NULL DEFAULT 0.7,
    learned_at   TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_kg_claim_doc ON kg_claim (doc_id);
CREATE INDEX IF NOT EXISTS idx_kg_claim_subject
    ON kg_claim (subject_type, subject_id);

CREATE TABLE IF NOT EXISTS kg_proposal (
    id             INTEGER PRIMARY KEY,
    proposal_type  TEXT NOT NULL,
    name           TEXT NOT NULL,
    evidence_count INTEGER NOT NULL DEFAULT 0,
    sample_doc_ids TEXT NOT NULL DEFAULT '[]',
    status         TEXT NOT NULL DEFAULT 'pending',
    UNIQUE (proposal_type, name)
);

CREATE TABLE IF NOT EXISTS learn_state (
    doc_id      TEXT PRIMARY KEY,
    source_hash TEXT NOT NULL,
    learned_at  TEXT DEFAULT (datetime('now'))
);
"""

ENTITY_TYPES = ("document", "pod", "project", "field", "working_area")
RELATIONS = (
    "ABOUT_POD",
    "ABOUT_FIELD",
    "ABOUT_WK",
    "HAS_PROJECT",
    "REVISES",
    "IN_FIELD",
    "IN_WK",
)


@dataclass(frozen=True)
class Edge:
    src_type: str
    src_id: str
    rel: str
    dst_type: str
    dst_id: str
    confidence: float = 1.0
    method: str = ""
    evidence: str | None = None


@dataclass(frozen=True)
class Claim:
    doc_id: str
    claim_type: str
    predicate: str
    value: Any
    subject_type: str | None = None
    subject_id: str | None = None
    evidence: str | None = None
    confidence: float = 0.7


class KnowledgeStore:
    """CRUD over kg_edge / kg_claim / kg_proposal / learn_state."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def ensure_tables(self) -> None:
        self._conn.executescript(_DDL)
        self._conn.commit()

    # -- edges ------------------------------------------------------------

    def upsert_edges(self, edges: list[Edge]) -> int:
        cur = self._conn.executemany(
            """
            INSERT INTO kg_edge
                (src_type, src_id, rel, dst_type, dst_id,
                 confidence, method, evidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (src_type, src_id, rel, dst_type, dst_id) DO UPDATE SET
                confidence = CASE
                    WHEN excluded.confidence > kg_edge.confidence
                    THEN excluded.confidence ELSE kg_edge.confidence END,
                method = CASE
                    WHEN excluded.confidence > kg_edge.confidence
                    THEN excluded.method ELSE kg_edge.method END,
                evidence = CASE
                    WHEN excluded.confidence > kg_edge.confidence
                    THEN excluded.evidence ELSE kg_edge.evidence END
            """,
            [
                (
                    e.src_type,
                    e.src_id,
                    e.rel,
                    e.dst_type,
                    e.dst_id,
                    e.confidence,
                    e.method,
                    e.evidence,
                )
                for e in edges
            ],
        )
        self._conn.commit()
        return cur.rowcount if cur.rowcount and cur.rowcount > 0 else len(edges)

    def delete_doc_edges(self, doc_id: str) -> int:
        cur = self._conn.execute(
            "DELETE FROM kg_edge WHERE src_type = 'document' AND src_id = ?",
            (doc_id,),
        )
        self._conn.commit()
        return cur.rowcount

    def edges_for(self, entity_type: str, entity_id: str) -> list[Edge]:
        rows = self._conn.execute(
            """
            SELECT src_type, src_id, rel, dst_type, dst_id,
                   confidence, method, evidence
            FROM kg_edge
            WHERE (src_type = ? AND src_id = ?) OR (dst_type = ? AND dst_id = ?)
            ORDER BY rel, confidence DESC
            """,
            (entity_type, entity_id, entity_type, entity_id),
        ).fetchall()
        return [Edge(*row) for row in rows]

    # -- claims -----------------------------------------------------------

    def replace_claims(self, doc_id: str, claims: list[Claim]) -> int:
        self._conn.execute("DELETE FROM kg_claim WHERE doc_id = ?", (doc_id,))
        self._conn.executemany(
            """
            INSERT INTO kg_claim
                (doc_id, claim_type, subject_type, subject_id,
                 predicate, value_json, evidence, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    c.doc_id,
                    c.claim_type,
                    c.subject_type,
                    c.subject_id,
                    c.predicate,
                    json.dumps(c.value, ensure_ascii=False),
                    c.evidence,
                    c.confidence,
                )
                for c in claims
            ],
        )
        self._conn.commit()
        return len(claims)

    def _claim_rows(self, where: str, params: tuple[Any, ...]) -> list[Claim]:
        rows = self._conn.execute(
            f"""
            SELECT doc_id, claim_type, predicate, value_json,
                   subject_type, subject_id, evidence, confidence
            FROM kg_claim WHERE {where}
            """,
            params,
        ).fetchall()
        return [
            Claim(r[0], r[1], r[2], json.loads(r[3]), r[4], r[5], r[6], r[7])
            for r in rows
        ]

    def claims_for(self, subject_type: str, subject_id: str) -> list[Claim]:
        return self._claim_rows(
            "subject_type = ? AND subject_id = ?", (subject_type, subject_id)
        )

    def claims_for_doc(self, doc_id: str) -> list[Claim]:
        return self._claim_rows("doc_id = ?", (doc_id,))

    # -- proposals --------------------------------------------------------

    def bump_proposal(self, proposal_type: str, name: str, doc_id: str) -> None:
        row = self._conn.execute(
            "SELECT sample_doc_ids FROM kg_proposal "
            "WHERE proposal_type = ? AND name = ?",
            (proposal_type, name),
        ).fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO kg_proposal "
                "(proposal_type, name, evidence_count, sample_doc_ids) "
                "VALUES (?, ?, 1, ?)",
                (proposal_type, name, json.dumps([doc_id])),
            )
        else:
            docs = set(json.loads(row[0]))
            if doc_id in docs:
                self._conn.commit()
                return
            docs.add(doc_id)
            self._conn.execute(
                "UPDATE kg_proposal SET evidence_count = ?, sample_doc_ids = ? "
                "WHERE proposal_type = ? AND name = ?",
                (len(docs), json.dumps(sorted(docs)), proposal_type, name),
            )
        self._conn.commit()

    def pending_proposals(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT proposal_type, name, evidence_count, sample_doc_ids "
            "FROM kg_proposal WHERE status = 'pending' "
            "ORDER BY evidence_count DESC"
        ).fetchall()
        return [
            {
                "proposal_type": r[0],
                "name": r[1],
                "evidence_count": r[2],
                "sample_doc_ids": json.loads(r[3]),
            }
            for r in rows
        ]

    # -- learn state ------------------------------------------------------

    def needs_learn(self, doc_id: str, source_hash: str) -> bool:
        row = self._conn.execute(
            "SELECT source_hash FROM learn_state WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return row is None or row[0] != source_hash

    def mark_learned(self, doc_id: str, source_hash: str) -> None:
        self._conn.execute(
            "INSERT INTO learn_state (doc_id, source_hash) VALUES (?, ?) "
            "ON CONFLICT (doc_id) DO UPDATE SET "
            "source_hash = excluded.source_hash, "
            "learned_at = datetime('now')",
            (doc_id, source_hash),
        )
        self._conn.commit()

    def learned_count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM learn_state").fetchone()[0])
