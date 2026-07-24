"""Tests for the iris chat tools `search_documents` and `read_document`.

Monkeypatch choice: the tools construct ``CorpusStore()`` via
``_get_corpus_embedder()``, which resolves ``db_path`` via
``Config.get_db_file()`` and the embedder via
``esdc.corpus.embedder.InternalEmbedder`` (lazily imported inside
``_get_corpus_embedder``). We patch both module attributes so the real
constructor path is exercised against a tmp DuckDB with a FakeEmbedder.
The tool result cache is redirected to a tmp diskcache for isolation.
"""

import json
from pathlib import Path

import diskcache
import pytest

from esdc.corpus.chunker import Chunk
from esdc.corpus.store import CorpusStore


def _fake_vector(text: str) -> list[float]:
    """Deterministic text-dependent 3-dim vector, normalized.

    Same bucketing trick as tests/corpus/test_store.py so cosine
    ranking is text-dependent and testable.
    """
    v = [1.0, 1.0, 1.0]
    for ch in text.lower():
        if "a" <= ch <= "i":
            v[0] += 1.0
        elif "j" <= ch <= "r":
            v[1] += 1.0
        elif "s" <= ch <= "z":
            v[2] += 1.0
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


class FakeEmbedder:
    model = "fake-embed"

    def generate_embedding(self, text: str) -> list[float]:
        return _fake_vector(text)

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        return [_fake_vector(t) for t in texts]


DOC = {
    "doc_id": "abc123", "file_name": "s.pdf", "file_path": "/x/s.pdf",
    "file_hash": "ab" * 32, "doc_type": "surat",
    "doc_number": "SRT-1", "doc_date": "2026-01-05", "subject": "Persetujuan",
    "sender": "SKK", "recipient": "KKKS", "doc_level": "field",
    "wk_name": "Rokan", "field_name": "Duri", "project_name": None,
    "raw_entities": "{}", "metadata": "{}", "markdown": "# Surat\nisi",
    "extraction_method": "native", "page_count": 1,
}

LONG_DOC = {
    **DOC,
    "doc_id": "long01",
    "file_name": "long.pdf",
    "file_path": "/x/long.pdf",
    "file_hash": "cd" * 32,
    "markdown": "# Panjang\n" + ("isi dokumen panjang sekali " * 20),
}


@pytest.fixture
def tool_env(tmp_path: Path, monkeypatch):
    """Redirect CorpusStore defaults + tool cache to tmp resources.

    Uses importlib.import_module for lazy-loading; the old star-import
    shadowing of `esdc.chat` was fixed in __init__.py so direct imports
    now work too, but importlib is fine here and keeps the fixture
    self-contained.
    """
    import importlib

    from esdc.configs import Config

    tools_mod = importlib.import_module("esdc.chat.tools")
    embedder_mod = importlib.import_module("esdc.corpus.embedder")

    db_path = tmp_path / "corpus.duckdb"
    monkeypatch.setattr(Config, "get_db_file", classmethod(lambda cls: db_path))
    monkeypatch.setattr(embedder_mod, "InternalEmbedder", FakeEmbedder)
    monkeypatch.setattr(tools_mod, "_corpus_embedder", None)

    cache = diskcache.Cache(str(tmp_path / "tool_cache"))
    monkeypatch.setattr(tools_mod, "_get_tool_cache", lambda: cache)
    yield db_path
    cache.close()


@pytest.fixture
def populated(tool_env: Path) -> Path:
    """Build a tmp DuckDB with one short and one long document ingested."""
    store = CorpusStore(db_path=tool_env, embedder=FakeEmbedder())
    store.ensure_tables()
    store.insert_document(DOC, [Chunk(0, "Surat", "persetujuan POD lapangan Duri")])
    store.insert_document(LONG_DOC, [Chunk(0, None, "notulen rapat panjang")])
    store.rebuild_indexes()
    store.close()
    return tool_env


def test_search_documents_returns_inserted_doc(populated):
    from esdc.chat.tools import search_documents

    result = json.loads(search_documents.invoke({"query": "persetujuan POD Duri"}))
    assert result["status"] == "success"
    assert result["count"] >= 1
    top = result["results"][0]
    assert top["doc_id"] == "abc123"
    assert top["file_name"] == "s.pdf"
    assert "score" in top


def test_search_documents_filter_excludes(populated):
    from esdc.chat.tools import search_documents

    result = json.loads(
        search_documents.invoke({"query": "persetujuan POD", "doc_type": "mom"})
    )
    assert result["status"] == "no_results"
    assert result["results"] == []


def test_search_documents_doc_topic_filter(tool_env):
    from esdc.chat.tools import search_documents

    store = CorpusStore(db_path=tool_env, embedder=FakeEmbedder())
    store.ensure_tables()
    doc = dict(DOC)
    doc["doc_topic"] = ["wpnb"]
    store.insert_document(doc, [Chunk(0, None, "rencana kerja dan anggaran")])
    store.rebuild_indexes()
    store.close()

    result = json.loads(
        search_documents.invoke({"query": "rencana kerja", "doc_topic": "wpnb"})
    )
    assert result["status"] == "success"
    assert result["results"][0]["doc_topic"] == ["wpnb"]

    excluded = json.loads(
        search_documents.invoke({"query": "rencana kerja", "doc_topic": "afe"})
    )
    assert excluded["status"] == "no_results"


def test_search_documents_empty_db_not_available(tool_env):
    from esdc.chat.tools import search_documents

    result = json.loads(search_documents.invoke({"query": "persetujuan POD"}))
    assert result["status"] == "not_available"
    assert "esdc corpus extract" in result["message"]
    assert "esdc corpus commit" in result["message"]


def test_read_document_returns_markdown_and_metadata(populated):
    from esdc.chat.tools import read_document

    result = json.loads(read_document.invoke({"doc_id": "abc123"}))
    assert result["status"] == "success"
    doc = result["document"]
    assert doc["markdown"] == "# Surat\nisi"
    assert doc["truncated"] is False
    assert doc["file_name"] == "s.pdf"
    assert doc["subject"] == "Persetujuan"
    assert doc["doc_type"] == "surat"
    assert doc["wk_name"] == "Rokan"
    # Noise fields are stripped from the agent-facing payload.
    assert "embedding_model" not in doc
    assert "raw_entities" not in doc
    assert "metadata" not in doc


def test_read_document_truncates_markdown(populated):
    from esdc.chat.tools import read_document

    result = json.loads(read_document.invoke({"doc_id": "long01", "max_chars": 50}))
    assert result["status"] == "success"
    doc = result["document"]
    assert doc["truncated"] is True
    assert len(doc["markdown"]) <= 50


def test_read_document_unknown_id_not_found(populated):
    from esdc.chat.tools import read_document

    result = json.loads(read_document.invoke({"doc_id": "nope"}))
    assert result["status"] == "not_found"
    assert result["doc_id"] == "nope"


class ExplodingStore:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("store exploded")


def test_tools_never_raise_on_store_explosion(tool_env, monkeypatch):
    import importlib

    from esdc.chat.tools import read_document, search_documents

    store_mod = importlib.import_module("esdc.corpus.store")

    monkeypatch.setattr(store_mod, "CorpusStore", ExplodingStore)

    result = json.loads(search_documents.invoke({"query": "anything at all"}))
    assert result["status"] == "error"

    result = json.loads(read_document.invoke({"doc_id": "abc123"}))
    assert result["status"] == "error"


def test_tools_registered_in_agent():
    """Grep-level check: agent.py imports and binds both tools."""
    import importlib
    import inspect

    agent_mod = importlib.import_module("esdc.chat.agent")

    source = inspect.getsource(agent_mod)
    # once in the import block, once in the default tools list
    assert source.count("search_documents") >= 2
    assert source.count("read_document") >= 2


def test_classifier_sets_include_document_tools():
    """Classifier tool-sets with semantic search also offer document tools."""
    from esdc.chat.query_classifier import (
        QueryClassification,
        QueryType,
        get_tools_for_classification,
    )
    from esdc.chat.tools import read_document, search_documents, semantic_search

    seen_semantic = False
    for qtype in QueryType:
        classification = QueryClassification(
            query_type=qtype,
            confidence=0.9,
            detected_entities={},
            suggested_table=None,
            suggested_columns=[],
            reason="test",
        )
        tools = get_tools_for_classification(classification)
        if semantic_search.name in tools:
            seen_semantic = True
            assert search_documents.name in tools
            assert read_document.name in tools

    assert seen_semantic, "no classification exposes semantic_search at all"


def test_search_documents_description_carries_schema_context():
    """Assert the schema-derived field guide is appended to `.description`.

    search_documents is a langchain StructuredTool; the LLM-facing text is
    `.description` (baked from the function docstring at decoration time),
    not `.__doc__` (which on a StructuredTool instance is the wrapper
    class's own docstring, not the wrapped function's) — see
    esdc/chat/openterminal.py's `run_command.description = ...` for the
    established pattern of appending to `.description` post-decoration.
    """
    from esdc.chat.tools import search_documents

    desc = search_documents.description
    assert "sender" in desc
    assert "organization" in desc.lower()
    assert "doc_level" in desc


def test_search_documents_reuses_embedder(tool_env, monkeypatch):
    """EmbeddingManager should be reused across corpus tool calls."""
    import esdc.chat.tools as tools_mod

    instantiations = []

    class _CountingEmbedder:
        model = "fake"

        def __init__(self):
            instantiations.append(1)

        def generate_embedding(self, text):
            return [0.0] * 4

        def generate_embeddings_batch(self, texts):
            return [[0.0] * 4 for _ in texts]

    monkeypatch.setattr(tools_mod, "_corpus_embedder", None)
    monkeypatch.setattr(
        "esdc.corpus.embedder.InternalEmbedder", _CountingEmbedder
    )
    # invalidate the tool cache so both calls hit the store
    tools_mod.invalidate_tool_cache()

    tools_mod.search_documents.invoke({"query": "query one"})
    tools_mod.search_documents.invoke({"query": "query two"})

    assert sum(instantiations) <= 1


class TestSemanticSearchCorpusFanOut:
    """Tests for semantic_search's fan-out to the document corpus.

    See docs/plans/2026-07-13-improve-document-search-usage.md. It must
    always return a `{"remarks": ..., "documents": ...}` envelope, and a
    corpus failure must never break the remarks section.
    """

    def test_fanout_returns_both_sections_with_documents_hit(
        self, populated, monkeypatch
    ):
        """Corpus has a matching doc -> documents section carries the hit."""
        from unittest.mock import Mock, patch

        import esdc.chat.tools as tools_mod
        from esdc.chat.tools import semantic_search

        # Force a fresh embedder lookup so it picks up tool_env's FakeEmbedder
        # instead of a global singleton some other test may have cached
        # (e.g. test_search_documents_reuses_embedder's _CountingEmbedder).
        monkeypatch.setattr(tools_mod, "_corpus_embedder", None)

        with patch("esdc.search.semantic_resolver.SemanticResolver") as MockResolver:
            mock_resolver = Mock()
            mock_resolver.hybrid_search.return_value = {
                "status": "success",
                "count": 1,
                "results": [{"project_id": "P1", "similarity": 0.9}],
            }
            mock_resolver.close = Mock()
            MockResolver.return_value = mock_resolver

            result = json.loads(
                semantic_search.invoke({"query": "persetujuan POD Duri"})
            )

        assert result["remarks"]["status"] == "success"
        assert result["remarks"]["count"] == 1
        assert result["documents"]["status"] == "success"
        assert result["documents"]["results"][0]["doc_id"] == "abc123"

    def test_fanout_corpus_error_never_breaks_remarks(self, tool_env, monkeypatch):
        """A corpus exception must not affect the remarks section at all."""
        import importlib
        from unittest.mock import Mock, patch

        from esdc.chat.tools import semantic_search

        store_mod = importlib.import_module("esdc.corpus.store")
        monkeypatch.setattr(store_mod, "CorpusStore", ExplodingStore)

        with patch("esdc.search.semantic_resolver.SemanticResolver") as MockResolver:
            mock_resolver = Mock()
            mock_resolver.hybrid_search.return_value = {
                "status": "success",
                "count": 2,
                "results": [
                    {"project_id": "P1", "similarity": 0.95},
                    {"project_id": "P2", "similarity": 0.89},
                ],
            }
            mock_resolver.close = Mock()
            MockResolver.return_value = mock_resolver

            result = json.loads(
                semantic_search.invoke({"query": "kendala teknis"})
            )

        assert result["remarks"]["status"] == "success"
        assert result["remarks"]["count"] == 2
        assert result["documents"]["status"] == "error"
        assert "store exploded" in result["documents"]["message"]

    def test_fanout_empty_corpus_reports_not_available(self, tool_env):
        """No corpus ingested -> documents section is not_available, remarks intact."""
        from unittest.mock import Mock, patch

        from esdc.chat.tools import semantic_search

        with patch("esdc.search.semantic_resolver.SemanticResolver") as MockResolver:
            mock_resolver = Mock()
            mock_resolver.hybrid_search.return_value = {
                "status": "no_results",
                "count": 0,
                "results": [],
            }
            mock_resolver.close = Mock()
            MockResolver.return_value = mock_resolver

            result = json.loads(semantic_search.invoke({"query": "apa saja"}))

        assert result["remarks"]["status"] == "no_results"
        assert result["documents"]["status"] == "not_available"

    def test_fanout_filters_mapped_to_corpus_schema(self, tool_env, monkeypatch):
        """report_year -> year; field_name/wk_name pass through; others dropped."""
        from unittest.mock import Mock, patch

        import esdc.chat.tools as tools_mod
        from esdc.chat.tools import semantic_search

        captured_filters = {}

        class _SpyStore:
            def __init__(self, *args, **kwargs):
                pass

            def search(self, query, limit, filters):
                captured_filters.update(filters or {})
                return {"status": "no_results", "count": 0, "results": []}

            def close(self):
                pass

        monkeypatch.setattr(
            "esdc.corpus.store.CorpusStore", _SpyStore
        )
        tools_mod.invalidate_tool_cache()

        with patch("esdc.search.semantic_resolver.SemanticResolver") as MockResolver:
            mock_resolver = Mock()
            mock_resolver.hybrid_search.return_value = {
                "status": "success",
                "count": 0,
                "results": [],
            }
            mock_resolver.close = Mock()
            MockResolver.return_value = mock_resolver

            semantic_search.invoke(
                {
                    "query": "kendala teknis",
                    "report_year": 2024,
                    "field_name": "%Duri%",
                    "wk_name": "%Rokan%",
                    "province": "%Riau%",
                }
            )

        assert captured_filters == {
            "year": 2024,
            "field_name": "%Duri%",
            "wk_name": "%Rokan%",
        }

    def test_fanout_not_cached_when_documents_not_available(self, tool_env):
        """documents=not_available must not be cached.

        The tool cache is only invalidated on `esdc reload`, not on
        `esdc corpus commit` — caching a not_available documents section
        would freeze it even after a corpus is ingested later. So the
        search must re-run on every call until the corpus exists.
        """
        from unittest.mock import Mock, patch

        from esdc.chat.tools import semantic_search

        with patch("esdc.search.semantic_resolver.SemanticResolver") as MockResolver:
            mock_resolver = Mock()
            mock_resolver.hybrid_search.return_value = {
                "status": "success",
                "count": 0,
                "results": [],
            }
            mock_resolver.close = Mock()
            MockResolver.return_value = mock_resolver

            first = json.loads(semantic_search.invoke({"query": "kendala unik"}))
            second = json.loads(semantic_search.invoke({"query": "kendala unik"}))

        assert first["documents"]["status"] == "not_available"
        assert second["documents"]["status"] == "not_available"
        # No cache hit: both calls reached the (mocked) remarks search.
        assert mock_resolver.hybrid_search.call_count == 2

    def test_fanout_cached_when_both_sections_definitive(
        self, populated, monkeypatch
    ):
        """Both sections success/no_results -> second call is a cache hit."""
        from unittest.mock import Mock, patch

        import esdc.chat.tools as tools_mod
        from esdc.chat.tools import semantic_search

        monkeypatch.setattr(tools_mod, "_corpus_embedder", None)

        with patch("esdc.search.semantic_resolver.SemanticResolver") as MockResolver:
            mock_resolver = Mock()
            mock_resolver.hybrid_search.return_value = {
                "status": "success",
                "count": 1,
                "results": [{"project_id": "P1", "similarity": 0.9}],
            }
            mock_resolver.close = Mock()
            MockResolver.return_value = mock_resolver

            first = json.loads(
                semantic_search.invoke({"query": "persetujuan POD Duri"})
            )
            second = json.loads(
                semantic_search.invoke({"query": "persetujuan POD Duri"})
            )

        assert first["documents"]["status"] == "success"
        assert second == first
        # Cache hit: the second call never reached the remarks search.
        assert mock_resolver.hybrid_search.call_count == 1
