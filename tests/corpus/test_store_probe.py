# tests/corpus/test_store_probe.py
"""corpus_meta parity probe: seeded on write paths, enforced on mismatch."""
from __future__ import annotations

import json

import pytest

from esdc.corpus.store import CorpusStore


class _FixedEmbedder:
    model = "qwen3-embedding-0.6b-q8_0"

    def __init__(self, vec):
        self._vec = list(vec)

    def generate_embedding(self, text):
        return list(self._vec)

    def generate_embeddings_batch(self, texts):
        return [list(self._vec) for _ in texts]


def _store(tmp_path, vec, name="c.duckdb"):
    return CorpusStore(db_path=tmp_path / name, embedder=_FixedEmbedder(vec))


def test_probe_column_created(tmp_path):
    s = _store(tmp_path, [1.0, 0.0, 0.0])
    s.ensure_tables(validate_model=True)
    cols = [
        r[0]
        for r in s._get_connection()
        .execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'corpus_meta'"
        )
        .fetchall()
    ]
    s.close()
    assert "probe_vec" in cols


def test_probe_seeded_on_first_write_path(tmp_path):
    s = _store(tmp_path, [1.0, 0.0, 0.0])
    s.ensure_tables(validate_model=True)
    stored = s._get_connection().execute("SELECT probe_vec FROM corpus_meta").fetchone()[0]
    s.close()
    assert json.loads(stored) == [1.0, 0.0, 0.0]


def test_matching_backend_reopens_cleanly(tmp_path):
    s = _store(tmp_path, [1.0, 0.0, 0.0])
    s.ensure_tables(validate_model=True)
    s.close()
    s2 = _store(tmp_path, [0.999, 0.001, 0.0])
    s2.ensure_tables(validate_model=True)  # must not raise
    s2.close()


def test_drifted_backend_is_rejected(tmp_path):
    s = _store(tmp_path, [1.0, 0.0, 0.0])
    s.ensure_tables(validate_model=True)
    s.close()
    s2 = _store(tmp_path, [0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="parity probe failed"):
        s2.ensure_tables(validate_model=True)
    s2.close()


def test_read_path_skips_probe(tmp_path):
    s = _store(tmp_path, [1.0, 0.0, 0.0])
    s.ensure_tables(validate_model=True)
    s.close()
    s2 = _store(tmp_path, [0.0, 1.0, 0.0])
    s2.ensure_tables()  # validate_model=False -> no probe, no raise
    s2.close()


def test_set_meta_rebaselines_probe(tmp_path):
    s = _store(tmp_path, [1.0, 0.0, 0.0])
    s.ensure_tables(validate_model=True)
    s.close()
    s2 = _store(tmp_path, [0.0, 1.0, 0.0])
    s2.set_meta("qwen3-embedding-0.6b-q8_0", 3)
    stored = s2._get_connection().execute("SELECT probe_vec FROM corpus_meta").fetchone()[0]
    s2.close()
    assert json.loads(stored) == [0.0, 1.0, 0.0]
