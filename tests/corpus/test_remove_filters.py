"""Tests for CorpusStore.find_doc_ids and `esdc corpus remove` filters."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from esdc.corpus.store import CorpusStore


class FakeEmbedder:
    model = "fake-embed-v1"

    def generate_embedding(self, text):
        return [1.0, 0.0, 0.0]

    def generate_embeddings_batch(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


def _doc(doc_id: str, name: str, doc_type: str, date: str) -> dict:
    return {
        "doc_id": doc_id,
        "file_name": name,
        "file_path": f"/tmp/{name}",
        "file_hash": doc_id * 4,
        "doc_type": doc_type,
        "doc_date": date,
        "markdown": "# isi",
        "extraction_method": "native",
    }


@pytest.fixture(autouse=True)
def _isolated_registry(tmp_path, monkeypatch):
    """Keep _warn_pod_links' registry lookup off the user's real ~/.esdc."""
    import esdc.configs as configs

    monkeypatch.setattr(
        configs.Config, "get_db_dir", classmethod(lambda cls: tmp_path)
    )


@pytest.fixture
def store(tmp_path: Path) -> CorpusStore:
    s = CorpusStore(db_path=tmp_path / "corpus.duckdb", embedder=FakeEmbedder())
    s.ensure_tables()
    s.insert_document(_doc("aaaa", "letter_a.pdf", "letter", "2019-01-01"), [])
    s.insert_document(_doc("bbbb", "letter_b.pdf", "letter", "2020-01-01"), [])
    s.insert_document(_doc("cccc", "regulation.pdf", "permen", "2019-06-01"), [])
    # find_doc_ids/get_document are serving reads off the DuckDB mirror
    # (Task 8); populate it so `corpus remove` can find/check these docs.
    s.refresh_mirror()
    yield s
    s.close()


def test_find_doc_ids_by_doc_type(store):
    rows = store.find_doc_ids({"doc_type": "letter"})
    assert [r[0] for r in rows] == ["aaaa", "bbbb"]
    assert rows[0][1] == "letter_a.pdf"


def test_find_doc_ids_combines_filters(store):
    rows = store.find_doc_ids({"doc_type": "letter", "year": 2019})
    assert [r[0] for r in rows] == ["aaaa"]


def test_find_doc_ids_empty_filters_returns_all(store):
    assert len(store.find_doc_ids({})) == 3


def _invoke_remove(monkeypatch, store, args):
    import esdc.esdc as cli

    monkeypatch.setattr(cli, "_open_corpus_store", lambda: store)
    return CliRunner().invoke(cli.corpus_app, ["remove", *args])


def test_remove_no_args_errors(monkeypatch, store):
    result = _invoke_remove(monkeypatch, store, [])
    assert result.exit_code != 0
    assert "corpus clear" in result.output


def test_remove_filter_requires_yes(monkeypatch, store):
    result = _invoke_remove(monkeypatch, store, ["--doc-type", "letter"])
    assert result.exit_code != 0
    assert "--yes" in result.output
    assert store.counts()["documents"] == 3  # nothing deleted


def test_remove_filter_dry_run_lists_without_deleting(monkeypatch, store):
    result = _invoke_remove(
        monkeypatch, store, ["--doc-type", "letter", "--dry-run"]
    )
    assert result.exit_code == 0
    assert "letter_a.pdf" in result.output
    assert store.counts()["documents"] == 3


def test_remove_filter_with_yes_deletes(monkeypatch, store):
    result = _invoke_remove(
        monkeypatch, store, ["--doc-type", "letter", "--yes"]
    )
    assert result.exit_code == 0
    assert store.counts()["documents"] == 1
    assert store.get_document("cccc") is not None


def test_remove_explicit_ids_still_work_without_yes(monkeypatch, store):
    result = _invoke_remove(monkeypatch, store, ["aaaa"])
    assert result.exit_code == 0
    assert store.counts()["documents"] == 2
