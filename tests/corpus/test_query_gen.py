from esdc.corpus.query_gen import (
    QueryMeta,
    generate,
    read_query_file,
    reconcile,
    synthesize_query,
    write_query_file,
)


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


class FakeStore:
    def __init__(self, docs):
        # docs: list of {doc_id, doc_type, subject, file_hash, chunk_text}
        self._docs = {d["doc_id"]: d for d in docs}

    def list_documents(self):
        return [
            {"doc_id": d["doc_id"], "doc_type": d["doc_type"], "subject": d["subject"]}
            for d in self._docs.values()
        ]

    def fingerprint_rows(self):
        return [(d["doc_id"], d["file_hash"]) for d in self._docs.values()]

    def sample_content(self, doc_id, chunk_seed=None):
        return self._docs.get(doc_id)


def _call(prompt):  # deterministic stub LLM
    return "generated query"


def _docs(n, doc_type="letter"):
    return [
        {
            "doc_id": f"{doc_type}-{i}", "doc_type": doc_type,
            "subject": f"subject {i}", "file_hash": f"h{i}",
            "chunk_text": f"body {i}",
        }
        for i in range(n)
    ]


def test_synthesize_query_uses_caller():
    seen = {}
    def cap(prompt):
        seen["prompt"] = prompt
        return "  a question?  "
    out = synthesize_query(cap, "isi surat tentang cadangan")
    assert out == "a question?"                          # stripped
    assert "isi surat tentang cadangan" in seen["prompt"]  # chunk grounded


def test_lookup_prompt_never_contains_the_subject():
    """The subject is stamped onto every chunk's embed_text by the context.

    prefix, so a query synthesized from it is a paraphrase of indexed
    metadata and the benchmark scores itself.
    """
    seen = {}

    def cap(prompt):
        seen["prompt"] = prompt
        return "berapa cadangan terbukti lapangan itu?"

    out = synthesize_query(cap, "isi surat tentang cadangan")
    assert out == "berapa cadangan terbukti lapangan itu?"
    assert "Subject:" not in seen["prompt"]


def test_generated_rows_carry_lookup_class():
    store = FakeStore(_docs(1))
    rows, _meta = generate(store, _call, n=1)
    assert rows[0]["class"] == "lookup"


def test_synthesize_query_strips_reasoning_block():
    """A reasoning-model response must not become the benchmark query text.

    Pre-fix, `synthesize_query` only did `call(prompt).strip()`: the
    `<think>...</think>` block and its prose would survive into the query
    file verbatim (a leading/trailing plain `.strip()` cannot remove an
    embedded tagged block), silently corrupting every later Pass@k number.
    """

    def cap(prompt):
        think = "<think>The user wants a search query about reserves.</think>\n"
        return think + "apa isi surat cadangan?"

    out = synthesize_query(cap, "isi")
    assert out == "apa isi surat cadangan?"
    assert "<think>" not in out
    assert "search query" not in out


def test_generate_one_row_per_sampled_doc_with_meta():
    store = FakeStore(_docs(50))
    rows, meta = generate(store, _call, margin=0.10, seed=42)
    assert 0 < len(rows) <= 50
    assert all(r["expected"][0].startswith("letter-") for r in rows)
    assert all(len(r["expected"]) == 1 for r in rows)
    assert meta.n == len(rows)
    assert meta.fingerprint  # set from fingerprint_rows()


def test_generate_explicit_n_overrides_auto():
    store = FakeStore(_docs(50))
    rows, _ = generate(store, _call, n=7, seed=42)
    assert len(rows) == 7


def test_generate_progress_cb_called_per_query():
    store = FakeStore(_docs(20))
    calls = []
    generate(store, _call, n=5, seed=1, progress_cb=lambda d, t: calls.append((d, t)))
    assert calls[-1] == (5, 5)
    assert all(t == 5 for _, t in calls)


def test_reconcile_drops_removed_and_adds_new():
    store = FakeStore(_docs(10))
    rows, meta = generate(store, _call, n=6, seed=1)

    # Remove one sampled doc, add three new docs.
    kept_id = rows[0]["expected"][0]
    dropped_id = rows[1]["expected"][0]
    del store._docs[dropped_id]
    for d in _docs(3, doc_type="report"):
        store._docs[d["doc_id"]] = d

    new_rows, new_meta = reconcile(store, _call, rows, meta, seed=1)
    ids = {r["expected"][0] for r in new_rows}
    assert dropped_id not in ids
    assert kept_id in ids
    assert new_meta.fingerprint == __import__(
        "esdc.corpus.sampling", fromlist=["corpus_fingerprint"]
    ).corpus_fingerprint(store.fingerprint_rows())


def test_reconcile_noop_when_corpus_unchanged():
    store = FakeStore(_docs(10))
    rows, meta = generate(store, _call, n=6, seed=1)
    new_rows, new_meta = reconcile(store, _call, rows, meta, seed=1)
    assert {r["expected"][0] for r in new_rows} == {r["expected"][0] for r in rows}
    assert new_meta.fingerprint == meta.fingerprint


def test_reconcile_drops_row_with_empty_expected():
    """A hand-authored/legacy row with expected=[] must be dropped, not crash."""
    store = FakeStore(_docs(10))
    rows, meta = generate(store, _call, n=3, seed=1)
    malformed = {"query": "x", "expected": []}
    new_rows, _ = reconcile(store, _call, rows + [malformed], meta, seed=1)
    assert malformed not in new_rows
    assert all(r["expected"] for r in new_rows)


def test_reconcile_does_not_grow_past_original_size():
    store = FakeStore(_docs(10, doc_type="letter"))
    rows, meta = generate(store, _call, n=6, seed=1)   # 6 letter rows, target 6
    # Corpus gains a new type; all 6 kept rows are still live letters.
    for d in _docs(5, doc_type="report"):
        store._docs[d["doc_id"]] = d
    new_rows, _ = reconcile(store, _call, rows, meta, seed=1)
    assert len(new_rows) <= meta.n          # never exceeds original size
    assert len(new_rows) == 6               # noop-ish: kept already fills target


def test_reconcile_regenerates_doc_with_changed_file_hash():
    store = FakeStore(_docs(10))
    rows, meta = generate(store, _call, n=6, seed=1)

    changed_id = rows[0]["expected"][0]
    other_ids = [r["expected"][0] for r in rows[1:]]
    store._docs[changed_id]["file_hash"] = "NEW-HASH"
    store._docs[changed_id]["chunk_text"] = "brand new body"

    new_rows, _ = reconcile(store, _call, rows, meta, seed=1)
    new_ids = {r["expected"][0] for r in new_rows}

    # Doc is still present, still exactly one row for it.
    assert changed_id in new_ids
    assert sum(1 for r in new_rows if r["expected"][0] == changed_id) == 1

    # Its row was regenerated: file_hash reflects the NEW content.
    changed_row = next(r for r in new_rows if r["expected"][0] == changed_id)
    assert changed_row["file_hash"] == "NEW-HASH"

    # Unchanged docs' rows are untouched (identical dicts as before).
    old_by_id = {r["expected"][0]: r for r in rows if r["expected"][0] in other_ids}
    new_by_id = {r["expected"][0]: r for r in new_rows if r["expected"][0] in other_ids}
    for doc_id, old_row in old_by_id.items():
        assert new_by_id[doc_id] == old_row


def test_reconcile_keeps_legacy_row_without_file_hash():
    """A pre-existing row with no `file_hash` key is kept, not regenerated."""
    store = FakeStore(_docs(10))
    rows, meta = generate(store, _call, n=6, seed=1)

    legacy_id = rows[0]["expected"][0]
    legacy_row = dict(rows[0])
    del legacy_row["file_hash"]
    rows[0] = legacy_row

    # Even though live content differs, a legacy row (no file_hash) can't be
    # detected as changed, so it must be kept as-is.
    store._docs[legacy_id]["file_hash"] = "SOME-OTHER-HASH"

    new_rows, _ = reconcile(store, _call, rows, meta, seed=1)
    assert legacy_row in new_rows
    assert sum(1 for r in new_rows if r["expected"][0] == legacy_id) == 1
