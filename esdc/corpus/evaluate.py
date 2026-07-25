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

    hits = dict.fromkeys(ks, 0)
    latencies: list[float] = []
    try:
        for lineno, line in enumerate(
            Path(queries_path).read_text(encoding="utf-8").splitlines(), 1
        ):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if isinstance(row, dict) and "_meta" in row:
                    continue
                query = row["query"]
                expected = set(row["expected"])
            except (ValueError, KeyError, TypeError) as e:
                report.failures.append(f"line {lineno}: {e}")
                continue

            start = time.perf_counter()
            result = store.search(query, limit=max(ks), rerank=rerank)
            latencies.append((time.perf_counter() - start) * 1000)

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
