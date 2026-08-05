"""LLM-synthesized eval query generation, file I/O, and reconciliation.

Sampling math and fingerprint live in esdc.corpus.sampling. See
docs/superpowers/specs/2026-07-25-dynamic-corpus-queries-design.md.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from esdc.corpus.sampling import (
    allocate,
    compute_sample_size,
    corpus_fingerprint,
    sample_docs,
)
from esdc.llm_text import strip_thinking_tags


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


_PROMPT = """You are building a retrieval benchmark for an Indonesian oil & gas \
document corpus. Given a document's subject and an excerpt, write ONE realistic \
search query a user would type to find THIS document. Match the language of the \
source (Indonesian or English). Return only the query text, no quotes, no prefix.

Subject: {subject}

Excerpt:
{chunk}

Query:"""


def synthesize_query(
    call: Callable[[str], str], subject: str, chunk_text: str
) -> str:
    """Ask the LLM for one realistic query grounded in this document."""
    prompt = _PROMPT.format(subject=subject or "(none)", chunk=chunk_text[:1500])
    return strip_thinking_tags(call(prompt)).strip()


def _docs_by_type(docs: list[dict]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for d in docs:
        key = d.get("doc_type") or "unknown"
        out.setdefault(key, []).append(d["doc_id"])
    return out


def _embedding_model(store) -> str:
    try:
        return store.counts().get("embedding_model", "") or ""
    except Exception:
        return ""


def _synthesize_rows(
    store,
    call: Callable[[str], str],
    doc_ids: list[str],
    progress_cb: Callable[[int, int], None] | None,
) -> list[dict]:
    rows: list[dict] = []
    total = len(doc_ids)
    for i, doc_id in enumerate(doc_ids, 1):
        content = store.sample_content(doc_id)
        if content is None:
            continue
        query = synthesize_query(call, content["subject"], content["chunk_text"])
        rows.append(
            {
                "query": query,
                "expected": [doc_id],
                "file_hash": content.get("file_hash", ""),
            }
        )
        if progress_cb:
            progress_cb(i, total)
    return rows


def _make_meta(store, rows, margin, ks) -> QueryMeta:
    return QueryMeta(
        fingerprint=corpus_fingerprint(store.fingerprint_rows()),
        margin=margin,
        n=len(rows),
        ks=list(ks),
        embedding_model=_embedding_model(store),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def generate(
    store,
    call: Callable[[str], str],
    margin: float = 0.05,
    n: int | None = None,
    seed: int = 42,
    ks: tuple[int, ...] = (1, 5, 10),
    progress_cb: Callable[[int, int], None] | None = None,
) -> tuple[list[dict], QueryMeta]:
    """Stratified sample of the corpus → one synthesized query per doc."""
    docs = store.list_documents()
    by_type = _docs_by_type(docs)
    n_docs = len(docs)
    target = n if n is not None else compute_sample_size(n_docs, margin=margin)
    target = max(0, min(target, n_docs))
    counts = {k: len(v) for k, v in by_type.items()}
    alloc = allocate(counts, target)
    doc_ids = sample_docs(by_type, alloc, seed=seed)
    rows = _synthesize_rows(store, call, doc_ids, progress_cb)
    return rows, _make_meta(store, rows, margin, ks)


def reconcile(
    store,
    call: Callable[[str], str],
    existing_rows: list[dict],
    existing_meta: QueryMeta | None,
    seed: int = 42,
    progress_cb: Callable[[int, int], None] | None = None,
) -> tuple[list[dict], QueryMeta]:
    """Drop rows for vanished docs, regenerate changed docs, add new docs."""
    docs = store.list_documents()
    live_hash = dict(store.fingerprint_rows())
    by_type = _docs_by_type(docs)

    kept: list[dict] = []
    changed_ids: list[str] = []
    for r in existing_rows:
        if not r.get("expected"):
            continue
        doc_id = r["expected"][0]
        if doc_id not in live_hash:
            continue  # removed doc — dropped
        row_hash = r.get("file_hash")
        if not row_hash:
            kept.append(r)  # legacy/no hash — keep, can't tell if changed
        elif row_hash == live_hash[doc_id]:
            kept.append(r)  # unchanged
        else:
            changed_ids.append(doc_id)  # content changed — regenerate

    kept_ids = {r["expected"][0] for r in kept}
    covered_ids = kept_ids | set(changed_ids)

    margin = existing_meta.margin if existing_meta else 0.05
    ks = tuple(existing_meta.ks) if existing_meta else (1, 5, 10)

    # Target total: preserve the existing query-set size (--refresh does not
    # resize; only --init/generate sizes the set via margin or explicit n).
    target = (
        existing_meta.n if existing_meta is not None
        else compute_sample_size(len(docs), margin=margin)
    )
    target = min(target, len(docs))
    counts = {k: len(v) for k, v in by_type.items()}
    alloc = allocate(counts, target)

    # Candidate new docs = live docs not already covered (kept or changed).
    unused_by_type = {
        t: [i for i in ids if i not in covered_ids] for t, ids in by_type.items()
    }
    kept_per_type: dict[str, int] = {}
    for did in covered_ids:
        t = next((k for k, v in by_type.items() if did in v), "unknown")
        kept_per_type[t] = kept_per_type.get(t, 0) + 1

    need = {
        t: max(0, alloc.get(t, 0) - kept_per_type.get(t, 0)) for t in by_type
    }
    budget = max(0, target - len(kept) - len(changed_ids))
    add_ids = sample_docs(unused_by_type, need, seed=seed)[:budget]
    synthesized = _synthesize_rows(
        store, call, changed_ids + add_ids, progress_cb
    )

    rows = kept + synthesized
    return rows, _make_meta(store, rows, margin, ks)
