"""Retrieval eval: doc-level Pass@k and latency over a JSONL query set.

Line format: {"query": "...", "expected": ["<doc_id or file_name>", ...]}
A query passes at k when any of the top-k results matches an expected
doc_id or file_name. Cookbook-style Pass@k, at document granularity —
chunk-level golden labels are impractical to author by hand.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from esdc.corpus.pipeline import _progress_with_status


@dataclass
class ClassReport:
    n_queries: int = 0
    pass_at: dict[int, float] = field(default_factory=dict)
    recall_at: dict[int, float] = field(default_factory=dict)
    abstention: float | None = None


@dataclass
class EvalReport:
    n_queries: int = 0
    pass_at: dict[int, float] = field(default_factory=dict)
    mean_latency_ms: float = 0.0
    failures: list[str] = field(default_factory=list)
    by_class: dict[str, ClassReport] = field(default_factory=dict)


# Classes whose `expected` names several documents. Pass@k treats
# retrieving one of four the same as retrieving all four, so these are
# scored by Recall@k as well.
RECALL_CLASSES = frozenset({"cross_reference", "thematic"})


def row_class(row: dict[str, Any]) -> str:
    """Query class for a row.

    Rows predating classes score in their own bucket rather than being
    folded into `lookup`: they were synthesized from the document's own
    subject line, which the context prefix stamps onto every chunk, so
    they are systematically easier and must never be averaged with
    leak-free queries.
    """
    return str(row.get("class") or "lookup_legacy")


def run_eval(
    queries_path: Path,
    ks: tuple[int, ...] = (1, 5, 10),
    rerank: bool | None = None,
    store: Any | None = None,
) -> EvalReport:
    """Run every query against the corpus store and score Pass@k."""
    report = EvalReport()
    owns_store = store is None
    if store is None:
        from esdc.corpus.store import CorpusStore

        store = CorpusStore()

    lines = [
        line.strip()
        for line in Path(queries_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    n_queries = sum(1 for line in lines if _is_query_line(line))

    hits = dict.fromkeys(ks, 0)
    per_class_hits: dict[str, dict[int, int]] = {}
    per_class_n: dict[str, int] = {}
    recall_sums: dict[str, dict[int, float]] = {}
    latencies: list[float] = []
    try:
        with _progress_with_status("eval", n_queries, "queries") as p:
            for lineno, line in enumerate(lines, 1):
                try:
                    row = json.loads(line)
                    if isinstance(row, dict) and "_meta" in row:
                        continue
                    query = row["query"]
                    expected = set(row["expected"])
                    cls = row_class(row)
                except (ValueError, KeyError, TypeError) as e:
                    report.failures.append(f"line {lineno}: {e}")
                    continue

                p.file(query[:40])
                start = time.perf_counter()
                result = store.search(query, limit=max(ks), rerank=rerank)
                latencies.append((time.perf_counter() - start) * 1000)
                p.advance()

                if result.get("status") not in ("success", "no_results"):
                    report.failures.append(
                        f"line {lineno}: search {result.get('status')}: "
                        f"{result.get('message', '')}"
                    )
                    continue

                report.n_queries += 1
                results = result.get("results", [])
                per_class_n[cls] = per_class_n.get(cls, 0) + 1
                cls_hits = per_class_hits.setdefault(cls, dict.fromkeys(ks, 0))
                for k in ks:
                    top = results[:k]
                    if any(
                        r.get("doc_id") in expected or r.get("file_name") in expected
                        for r in top
                    ):
                        hits[k] += 1
                        cls_hits[k] += 1

                if cls in RECALL_CLASSES:
                    sums = recall_sums.setdefault(
                        cls, dict.fromkeys(ks, 0.0)
                    )
                    for k in ks:
                        top = results[:k]
                        matched = {
                            e
                            for e in expected
                            for r in top
                            if r.get("doc_id") == e or r.get("file_name") == e
                        }
                        if expected:
                            sums[k] += len(matched) / len(expected)
    finally:
        if owns_store:
            store.close()

    if report.n_queries:
        report.pass_at = {k: hits[k] / report.n_queries for k in ks}
        report.mean_latency_ms = sum(latencies) / len(latencies)
    for cls, n in per_class_n.items():
        sums = recall_sums.get(cls)
        report.by_class[cls] = ClassReport(
            n_queries=n,
            pass_at={k: per_class_hits[cls][k] / n for k in ks},
            recall_at={k: sums[k] / n for k in ks} if sums is not None else {},
        )
    return report


def _is_query_line(line: str) -> bool:
    """Check if a line is a query row (not _meta header)."""
    try:
        obj = json.loads(line)
        return isinstance(obj, dict) and "query" in obj and "_meta" not in obj
    except (json.JSONDecodeError, TypeError):
        return False
