from esdc.corpus.query_gen import QueryMeta, read_query_file, write_query_file


def test_write_then_read_roundtrip(tmp_path):
    p = tmp_path / "corpus_queries.jsonl"
    rows = [{"query": "apa isi surat?", "expected": ["doc-1"]}]
    meta = QueryMeta(
        fingerprint="abc", margin=0.05, n=1, ks=[1, 5, 10],
        embedding_model="qwen3", generated_at="2026-07-25T00:00:00",
    )
    write_query_file(p, rows, meta)

    got_rows, got_meta = read_query_file(p)
    assert got_rows == rows
    assert got_meta.fingerprint == "abc"
    assert got_meta.ks == [1, 5, 10]


def test_meta_is_first_line(tmp_path):
    p = tmp_path / "corpus_queries.jsonl"
    write_query_file(
        p, [{"query": "q", "expected": ["d"]}],
        QueryMeta("fp", 0.05, 1, [1], "m", "t"),
    )
    first = p.read_text(encoding="utf-8").splitlines()[0]
    assert first.startswith('{"_meta"')


def test_read_legacy_file_without_meta(tmp_path):
    p = tmp_path / "corpus_queries.jsonl"
    p.write_text('{"query": "q", "expected": ["d"]}\n', encoding="utf-8")
    rows, meta = read_query_file(p)
    assert meta is None
    assert rows == [{"query": "q", "expected": ["d"]}]
