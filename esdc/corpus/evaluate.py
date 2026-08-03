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
class EvalReport:
    n_queries: int = 0
    pass_at: dict[int, float] = field(default_factory=dict)
    mean_latency_ms: float = 0.0
    failures: list[str] = field(default_factory=list)


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
                for k in ks:
                    top = results[:k]
                    if any(
                        r.get("doc_id") in expected or r.get("file_name") in expected
                        for r in top
                    ):
                        hits[k] += 1
    finally:
        if owns_store:
            store.close()

    if report.n_queries:
        report.pass_at = {k: hits[k] / report.n_queries for k in ks}
        report.mean_latency_ms = sum(latencies) / len(latencies)
    return report


def _is_query_line(line: str) -> bool:
    """Check if a line is a query row (not _meta header)."""
    try:
        obj = json.loads(line)
        return isinstance(obj, dict) and "query" in obj and "_meta" not in obj
    except (json.JSONDecodeError, TypeError):
        return False
