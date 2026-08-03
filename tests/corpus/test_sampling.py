import pytest

from esdc.corpus.sampling import (
    allocate,
    compute_sample_size,
    corpus_fingerprint,
    sample_docs,
)


@pytest.mark.parametrize(
    "n_docs, expected",
    [(100, 80), (200, 132), (384, 193), (500, 218), (1000, 278), (5000, 357)],
)
def test_compute_sample_size_fpc(n_docs, expected):
    assert compute_sample_size(n_docs, margin=0.05) == expected


def test_compute_sample_size_clamps_to_population():
    assert compute_sample_size(10, margin=0.05) == 10  # n never exceeds N


def test_compute_sample_size_wider_margin_is_smaller():
    assert compute_sample_size(10_000, margin=0.10) < compute_sample_size(
        10_000, margin=0.05
    )


def test_allocate_sums_to_n_with_floor_and_largest_remainder():
    counts = {"letter": 70, "report": 25, "regulation": 5}
    alloc = allocate(counts, n=40)
    assert sum(alloc.values()) == 40
    assert all(v >= 1 for v in alloc.values())  # floor 1 per non-empty type
    assert alloc["letter"] > alloc["report"] > alloc["regulation"]


def test_allocate_never_exceeds_stratum_size():
    counts = {"letter": 3, "report": 100}
    alloc = allocate(counts, n=90)
    assert alloc["letter"] <= 3


def test_sample_docs_is_deterministic_under_seed():
    docs = {"letter": [f"L{i}" for i in range(20)], "report": [f"R{i}" for i in range(10)]}
    alloc = {"letter": 5, "report": 3}
    a = sample_docs(docs, alloc, seed=42)
    b = sample_docs(docs, alloc, seed=42)
    assert a == b
    assert len(a) == 8


def test_sample_docs_respects_allocation_counts():
    docs = {"letter": [f"L{i}" for i in range(20)], "report": [f"R{i}" for i in range(10)]}
    picked = sample_docs(docs, {"letter": 5, "report": 3}, seed=1)
    assert sum(1 for d in picked if d.startswith("L")) == 5
    assert sum(1 for d in picked if d.startswith("R")) == 3


def test_sample_docs_different_seed_differs():
    docs = {"letter": [f"L{i}" for i in range(50)]}
    assert sample_docs(docs, {"letter": 10}, seed=1) != sample_docs(
        docs, {"letter": 10}, seed=2
    )


def test_fingerprint_stable_under_reorder():
    a = corpus_fingerprint([("d1", "h1"), ("d2", "h2")])
    b = corpus_fingerprint([("d2", "h2"), ("d1", "h1")])
    assert a == b


def test_fingerprint_changes_on_hash_change():
    base = corpus_fingerprint([("d1", "h1")])
    assert base != corpus_fingerprint([("d1", "h2")])
    assert base != corpus_fingerprint([("d1", "h1"), ("d2", "h2")])
