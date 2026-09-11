"""LLM-synthesized eval query generation, file I/O, and reconciliation.

Sampling math and fingerprint live in esdc.corpus.sampling. See
docs/superpowers/specs/2026-07-25-dynamic-corpus-queries-design.md.
"""

from __future__ import annotations

import json
import random
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


# The document's subject is deliberately NOT in this prompt.
# build_context_prefix stamps the subject onto every chunk's embed_text,
# which is what both the vector index and the FTS index are built on — a
# query synthesized from the subject is a paraphrase of indexed metadata,
# and the benchmark ends up scoring itself.
_LOOKUP_PROMPT = """You are building a retrieval benchmark for an Indonesian \
oil & gas document corpus. Given an EXCERPT from one document, write ONE \
realistic search query a user would type to find THIS document. Base the query \
only on what the excerpt says — do not invent a title. Match the language of \
the source (Indonesian or English). Return only the query text, no quotes, no \
prefix.

Excerpt:
{chunk}

Query:"""


def synthesize_query(call: Callable[[str], str], chunk_text: str) -> str:
    """Ask the LLM for one realistic query grounded in this excerpt."""
    prompt = _LOOKUP_PROMPT.format(chunk=chunk_text[:1500])
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
        # Seeded on doc_id: the first chunk of a letter is the letterhead,
        # so a benchmark built from it asks about the header block.
        content = store.sample_content(doc_id, chunk_seed=doc_id)
        if content is None:
            continue
        query = synthesize_query(call, content["chunk_text"])
        rows.append(
            {
                "query": query,
                "expected": [doc_id],
                "class": "lookup",
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


_CROSS_REF_PROMPT = """You are building a retrieval benchmark for an Indonesian \
oil & gas correspondence corpus. One letter follows up on an earlier letter. \
Write ONE realistic search query a user would type when they want BOTH letters \
— the original and the follow-up. Do not mention any letter number. Write in \
Indonesian. Return only the query text.

Follow-up letter excerpt:
{chunk}

Query:"""


def generate_cross_reference(
    store, call: Callable[[str], str], limit: int = 30, seed: int = 42
) -> list[dict]:
    """Query rows for document pairs linked by a letter-number citation."""
    from esdc.corpus.citations import extract_letter_numbers, normalize_letter_number

    bodies = store.document_bodies()
    by_number: dict[str, str] = {}
    for doc_id, doc_number, _md in bodies:
        norm = normalize_letter_number(doc_number)
        if norm:
            by_number.setdefault(norm, doc_id)

    pairs: list[tuple[str, str, str]] = []  # (citing, cited, excerpt)
    for doc_id, _num, markdown in bodies:
        for cited_num in extract_letter_numbers(markdown):
            cited_id = by_number.get(cited_num)
            if cited_id and cited_id != doc_id:
                pairs.append((doc_id, cited_id, markdown[:1500]))
                break  # one query per citing document

    random.Random(seed).shuffle(pairs)
    rows: list[dict] = []
    for citing, cited, excerpt in pairs[:limit]:
        query = strip_thinking_tags(
            call(_CROSS_REF_PROMPT.format(chunk=excerpt))
        ).strip()
        if query:
            rows.append(
                {
                    "query": query,
                    "expected": [citing, cited],
                    "class": "cross_reference",
                }
            )
    return rows


_THEMATIC_PROMPT = """You are building a retrieval benchmark for an Indonesian \
oil & gas correspondence corpus. Below are the subjects of several letters that \
share a working area and a year. Write ONE realistic thematic question a user \
would type when they want to understand what these letters are collectively \
about. The question must NOT name any single letter. Write in Indonesian. \
Return only the query text.

Subjects:
{subjects}

Query:"""


def generate_thematic(
    store,
    call: Callable[[str], str],
    min_group: int = 5,
    limit: int = 20,
    seed: int = 42,
) -> list[dict]:
    """Query rows for metadata-defined themes (working area + year)."""
    groups: dict[tuple[str, int], list[dict]] = {}
    for d in store.list_documents():
        wks = d.get("wk_name") or []
        date = d.get("doc_date")
        if not wks or not date:
            continue
        year = int(str(date)[:4])
        groups.setdefault((str(wks[0]), year), []).append(d)

    eligible = [(k, v) for k, v in groups.items() if len(v) >= min_group]
    eligible.sort(key=lambda kv: (kv[0][0], kv[0][1]))
    random.Random(seed).shuffle(eligible)

    rows: list[dict] = []
    for (wk, year), docs in eligible[:limit]:
        subjects = "\n".join(f"- {d.get('subject') or ''}" for d in docs[:15])
        query = strip_thinking_tags(
            call(_THEMATIC_PROMPT.format(subjects=subjects))
        ).strip()
        if query:
            rows.append(
                {
                    "query": query,
                    "expected": [d["doc_id"] for d in docs],
                    "class": "thematic",
                    "seed_filter": {"wk_name": wk, "year": year},
                }
            )
    return rows


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
        existing_meta.n
        if existing_meta is not None
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

    need = {t: max(0, alloc.get(t, 0) - kept_per_type.get(t, 0)) for t in by_type}
    budget = max(0, target - len(kept) - len(changed_ids))
    add_ids = sample_docs(unused_by_type, need, seed=seed)[:budget]
    synthesized = _synthesize_rows(store, call, changed_ids + add_ids, progress_cb)

    rows = kept + synthesized
    return rows, _make_meta(store, rows, margin, ks)
