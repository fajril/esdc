"""LLM-synthesized eval query generation, file I/O, and reconciliation.

Sampling math and fingerprint live in esdc.corpus.sampling. See
docs/superpowers/specs/2026-07-25-dynamic-corpus-queries-design.md.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class QueryMeta:
    fingerprint: str
    margin: float
    n: int
    ks: list[int]
    embedding_model: str
    generated_at: str


def write_query_file(path: Path, rows: list[dict], meta: QueryMeta) -> None:
    """Write the _meta header line followed by one query row per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"_meta": asdict(meta)}, ensure_ascii=False)]
    lines += [json.dumps(r, ensure_ascii=False) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_query_file(path: Path) -> tuple[list[dict], QueryMeta | None]:
    """Return (rows, meta). meta is None for legacy files without a header."""
    meta: QueryMeta | None = None
    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict) and "_meta" in obj:
            meta = QueryMeta(**obj["_meta"])
            continue
        rows.append(obj)
    return rows, meta
