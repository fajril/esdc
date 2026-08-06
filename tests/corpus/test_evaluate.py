import json

from esdc.corpus.evaluate import row_class, run_eval


class FakeStore:
    """Returns doc-b first, doc-a second, for every query."""

    def __init__(self):
        self.rerank_args = []

    def search(self, query, limit=10, filters=None, rerank=None):
        self.rerank_args.append(rerank)
        return {
            "status": "success",
            "results": [
                {"doc_id": "doc-b", "file_name": "b.pdf"},
                {"doc_id": "doc-a", "file_name": "a.pdf"},
            ],
            "count": 2,
        }

    def close(self):
        pass


def _write_queries(tmp_path, rows):
    p = tmp_path / "queries.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def test_pass_at_k(tmp_path):
    path = _write_queries(
        tmp_path,
        [
            {"query": "q1", "expected": ["doc-b"]},   # hit at 1
            {"query": "q2", "expected": ["doc-a"]},   # hit at 2, not 1
            {"query": "q3", "expected": ["missing"]}, # never hit
        ],
    )
    report = run_eval(path, ks=(1, 2), store=FakeStore())
    assert report.n_queries == 3
    assert report.pass_at[1] == 1 / 3
    assert report.pass_at[2] == 2 / 3


def test_expected_matches_file_name_too(tmp_path):
    path = _write_queries(tmp_path, [{"query": "q", "expected": ["a.pdf"]}])
    report = run_eval(path, ks=(2,), store=FakeStore())
    assert report.pass_at[2] == 1.0


def test_rerank_flag_forwarded(tmp_path):
    path = _write_queries(tmp_path, [{"query": "q", "expected": ["doc-a"]}])
    store = FakeStore()
    run_eval(path, ks=(1,), rerank=True, store=store)
    assert store.rerank_args == [True]


def test_bad_line_recorded_as_failure(tmp_path):
    p = tmp_path / "queries.jsonl"
    p.write_text('{"query": "ok", "expected": ["doc-b"]}\nnot json\n')
    report = run_eval(p, ks=(1,), store=FakeStore())
    assert report.n_queries == 1
    assert len(report.failures) == 1


def test_row_without_class_is_lookup_legacy():
    assert row_class({"query": "q", "expected": ["d"]}) == "lookup_legacy"


def test_row_class_is_read_from_the_row():
    assert row_class({"query": "q", "expected": ["d"], "class": "lookup"}) == "lookup"


def test_per_class_pass_at_k(tmp_path):
    path = _write_queries(
        tmp_path,
        [
            {"query": "q1", "expected": ["doc-b"], "class": "lookup"},
            {"query": "q2", "expected": ["doc-a"], "class": "lookup"},
            {"query": "q3", "expected": ["missing"]},  # legacy bucket
        ],
    )
    report = run_eval(path, ks=(1, 2), store=FakeStore())
    assert report.by_class["lookup"].n_queries == 2
    assert report.by_class["lookup"].pass_at[1] == 0.5
    assert report.by_class["lookup"].pass_at[2] == 1.0
    assert report.by_class["lookup_legacy"].n_queries == 1
    assert report.by_class["lookup_legacy"].pass_at[1] == 0.0


def test_blended_report_still_present(tmp_path):
    path = _write_queries(
        tmp_path, [{"query": "q1", "expected": ["doc-b"], "class": "lookup"}]
    )
    report = run_eval(path, ks=(1,), store=FakeStore())
    assert report.n_queries == 1
    assert report.pass_at[1] == 1.0


def test_recall_at_k_for_multi_doc_class(tmp_path):
    # FakeStore returns doc-b then doc-a for every query.
    path = _write_queries(
        tmp_path,
        [
            {
                "query": "q1",
                "expected": ["doc-a", "doc-b", "doc-c", "doc-d"],
                "class": "cross_reference",
            }
        ],
    )
    report = run_eval(path, ks=(1, 2), store=FakeStore())
    cr = report.by_class["cross_reference"]
    assert cr.recall_at[1] == 0.25   # doc-b of 4
    assert cr.recall_at[2] == 0.5    # doc-b + doc-a of 4
    assert cr.pass_at[1] == 1.0      # Pass@k still reported


def test_recall_not_computed_for_lookup(tmp_path):
    path = _write_queries(
        tmp_path, [{"query": "q", "expected": ["doc-b"], "class": "lookup"}]
    )
    report = run_eval(path, ks=(1,), store=FakeStore())
    assert report.by_class["lookup"].recall_at == {}


def test_recall_ignores_empty_expected(tmp_path):
    path = _write_queries(
        tmp_path, [{"query": "q", "expected": [], "class": "thematic"}]
    )
    report = run_eval(path, ks=(1,), store=FakeStore())
    assert report.by_class["thematic"].recall_at[1] == 0.0


class ScoredStore:
    """Returns one result whose rerank_score is fixed at construction."""

    def __init__(self, score):
        self._score = score

    def search(self, query, limit=10, filters=None, rerank=None):
        hit = {"doc_id": "doc-b", "file_name": "b.pdf"}
        if self._score is not None:
            hit["rerank_score"] = self._score
        return {"status": "success", "results": [hit], "count": 1}

    def close(self):
        pass


def test_negative_is_counted_but_never_scored(tmp_path):
    """A rerank floor cannot score this class — see reranker.py's docstring.

    Measured 2026-08-06: negatives scored max 0.9999 against a true-positive
    minimum of 0.9968, so no threshold separates them. The class is kept
    (the queries are still worth carrying) but abstention stays None until a
    mechanism exists that can actually detect absence.
    """
    path = _write_queries(
        tmp_path, [{"query": "q", "expected": [], "class": "negative"}]
    )
    for score in (0.1, 0.95, None):
        report = run_eval(path, ks=(1,), rerank=True, store=ScoredStore(score))
        assert report.by_class["negative"].n_queries == 1
        assert report.by_class["negative"].abstention is None


def test_negative_excluded_from_pass_at_k(tmp_path):
    path = _write_queries(
        tmp_path,
        [
            {"query": "q1", "expected": ["doc-b"], "class": "lookup"},
            {"query": "q2", "expected": [], "class": "negative"},
        ],
    )
    report = run_eval(path, ks=(1,), store=ScoredStore(0.1))
    assert report.by_class["lookup"].pass_at[1] == 1.0
    assert report.by_class["negative"].pass_at == {}


def test_meta_header_line_is_ignored(tmp_path):
    p = tmp_path / "queries.jsonl"
    p.write_text(
        '{"_meta": {"fingerprint": "fp"}}\n'
        '{"query": "q1", "expected": ["doc-b"]}\n',
        encoding="utf-8",
    )
    report = run_eval(p, ks=(1,), store=FakeStore())
    assert report.n_queries == 1
    assert report.failures == []
