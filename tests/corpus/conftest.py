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
