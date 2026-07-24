import json

from esdc.corpus.evaluate import EvalReport, run_eval


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
