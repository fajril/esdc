import pytest

from esdc.corpus.sampling import allocate, compute_sample_size


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
