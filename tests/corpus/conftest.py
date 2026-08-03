import pytest

import esdc.configs as configs


@pytest.fixture(autouse=True)
def _isolated_db_dirs(tmp_path, monkeypatch):
    """Keep CorpusStore's SQLite documents table off the user's real ~/.esdc.

    Corpus tests pass explicit DuckDB paths already; the documents table
    moved to the operational SQLite db whose default location comes from
    Config.get_db_dir().
    """
    monkeypatch.setattr(
        configs.Config, "get_db_dir", classmethod(lambda cls: tmp_path)
    )
    # Corpus search() now reads Config.get_corpus_config() (rerank gating).
    # Without this, a dev machine's real ~/.esdc/config.yaml (e.g.
    # corpus.rerank: true) would leak into unit tests, triggering a real
    # llama.cpp reranker GGUF download and reordering results. Patching
    # _load_config keeps get_corpus_config() on pure CORPUS_DEFAULTS.
    monkeypatch.setattr(
        configs.Config, "_load_config", classmethod(lambda cls: None)
    )
