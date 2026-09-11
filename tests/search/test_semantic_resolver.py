"""Tests for SemanticResolver."""

from unittest.mock import MagicMock

import duckdb
import pytest

from esdc.search.semantic_resolver import SemanticResolver


def test_semantic_resolver_initialization():
    """Test SemanticResolver can be initialized."""
    resolver = SemanticResolver(embedder=MagicMock())
    assert resolver is not None


def test_search_by_text():
    """Test semantic search by text query."""
    mock_emb = MagicMock()
    mock_emb.generate_embedding.return_value = [0.1] * 1024

    resolver = SemanticResolver(embedder=mock_emb)

    # Mock database connection - need to handle multiple execute calls
    resolver._get_connection = MagicMock()
    mock_conn = MagicMock()

    # First two calls: COUNT check (once in search_by_text's
    # availability check, once again inside search_by_embedding),
    # third call: actual search
    mock_cursor1 = MagicMock()
    mock_cursor1.fetchone.return_value = [1]  # Count > 0

    mock_cursor2 = MagicMock()
    # Result needs 18 columns matching semantic_resolver.py implementation:
    # project_id, report_year, field_name, project_name, pod_name, wk_name,
    # province, basin128, project_class, project_stage, project_level,
    # operator_name, operator_group, wk_subgroup, wk_regionisasi_ngi,
    # wk_area_perwakilan_skkmigas, project_remarks, similarity
    mock_cursor2.fetchall.return_value = [
        (
            "uuid-1",  # project_id
            2024,  # report_year
            "Field name",  # field_name
            "Project name",  # project_name
            "POD name",  # pod_name
            "WK name",  # wk_name
            "Province",  # province
            "Basin128",  # basin128
            "Class",  # project_class
            "Stage",  # project_stage
            "Level",  # project_level
            "Operator",  # operator_name
            "Group",  # operator_group
            "Subgroup",  # wk_subgroup
            "Region",  # wk_regionisasi_ngi
            "Area",  # wk_area_perwakilan_skkmigas
            "Project remarks text",  # project_remarks
            0.95,  # similarity
        ),
    ]

    mock_conn.execute.side_effect = [mock_cursor1, mock_cursor1, mock_cursor2]
    resolver._get_connection.return_value = mock_conn
    resolver._ensure_semantic_meta = MagicMock(return_value=None)

    result = resolver.search_by_text("proyek masalah teknis", limit=5)

    assert result["status"] == "success"
    assert len(result["results"]) == 1
    assert result["results"][0]["project_id"] == "uuid-1"


def test_search_by_embedding():
    """Test semantic search by pre-computed embedding."""
    resolver = SemanticResolver(embedder=MagicMock())
    resolver._get_connection = MagicMock()
    mock_conn = MagicMock()

    # Mock count check first (returns 1 document available)
    mock_cursor1 = MagicMock()
    mock_cursor1.fetchone.return_value = [1]

    # Mock search results (empty)
    mock_cursor2 = MagicMock()
    mock_cursor2.fetchall.return_value = []

    mock_conn.execute.side_effect = [mock_cursor1, mock_cursor2]
    resolver._get_connection.return_value = mock_conn

    query_embedding = [0.1] * 4096
    result = resolver.search_by_embedding(query_embedding, limit=10)

    assert result["status"] == "no_results"


def test_build_embeddings_table():
    """Test building embeddings table."""
    mock_emb = MagicMock()
    mock_emb.generate_embedding.return_value = [0.0] * 1024

    resolver = SemanticResolver(embedder=mock_emb, read_only=False)
    resolver._get_connection = MagicMock()
    mock_conn = MagicMock()
    resolver._get_connection.return_value = mock_conn

    result = resolver.build_embeddings_table()

    assert result is True


def test_search_not_available():
    """Test search when embeddings not available."""
    mock_emb = MagicMock()
    mock_emb.generate_embedding.return_value = [0.1] * 4096

    resolver = SemanticResolver(embedder=mock_emb)
    resolver._get_connection = MagicMock()
    mock_conn = MagicMock()
    # Return 0 for count check
    mock_conn.execute.return_value.fetchall.return_value = []
    mock_conn.execute.return_value.fetchone.return_value = [0]
    resolver._get_connection.return_value = mock_conn

    result = resolver.search_by_text("test query")

    assert result["status"] == "not_available"


def test_search_by_text_no_embeddings_skips_query_embedding():
    """search_by_text must return not_available without calling the embedder.

    When the store has no embeddings, no embedder call should be needed.
    """

    def _boom(text):
        raise AssertionError("embedder must not be called")

    mock_emb = MagicMock()
    mock_emb.generate_embedding.side_effect = _boom

    resolver = SemanticResolver(embedder=mock_emb)
    resolver._get_connection = MagicMock()
    mock_conn = MagicMock()
    # Count check returns 0 -> no embeddings
    mock_conn.execute.return_value.fetchone.return_value = [0]
    resolver._get_connection.return_value = mock_conn

    result = resolver.search_by_text("test query")

    assert result["status"] == "not_available"


def test_readonly_flag_is_recorded():
    resolver = SemanticResolver(embedder=MagicMock(), read_only=True)
    assert resolver._read_only is True
    assert SemanticResolver(embedder=MagicMock())._read_only is True


def test_hybrid_search_missing_db_degrades_not_raises(tmp_path):
    """A missing DB file must degrade hybrid_search to not_available.

    A read-only resolver cannot create the file; the serving tool keys its
    FTS fallback off ``not_available``, so a raised IOException (mapped to a
    status="error" envelope) would silently skip the fallback.
    """
    mock_emb = MagicMock()
    resolver = SemanticResolver(
        db_path=tmp_path / "sub" / "missing.duckdb",
        embedder=mock_emb,
        read_only=True,
    )

    result = resolver.hybrid_search("kendala teknis")

    assert result["status"] == "not_available"
    assert result["results"] == []
    mock_emb.generate_embedding.assert_not_called()
    resolver.close()


def test_search_by_embedding_missing_db_degrades_not_raises(tmp_path):
    """search_by_embedding must check availability before connecting."""
    resolver = SemanticResolver(
        db_path=tmp_path / "sub" / "missing.duckdb",
        embedder=MagicMock(),
        read_only=True,
    )

    result = resolver.search_by_embedding([0.1] * 8, limit=5)

    assert result["status"] == "not_available"
    assert result["results"] == []
    resolver.close()


def test_readonly_validation_does_not_create_semantic_meta(tmp_path):
    """A read-only resolver must never seed semantic_meta on the read path."""
    db_path = tmp_path / "empty.duckdb"
    duckdb.connect(str(db_path)).close()

    mock_emb = MagicMock()
    resolver = SemanticResolver(db_path=db_path, embedder=mock_emb, read_only=True)
    out = resolver._ensure_semantic_meta()
    assert out is not None
    assert out["status"] == "not_available"
    # The read path neither created nor wrote semantic_meta.
    columns = resolver._get_connection().execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'semantic_meta'"
    ).fetchall()
    assert columns == []
    mock_emb.generate_embedding.assert_not_called()
    resolver.close()


def test_readonly_mutator_guard_precedes_embedder(tmp_path):
    """build_embeddings_table rejects a reader before touching the embedder."""
    mock_emb = MagicMock()
    resolver = SemanticResolver(
        db_path=tmp_path / "ro.duckdb", embedder=mock_emb, read_only=True
    )
    with pytest.raises(PermissionError, match="read_only=False"):
        resolver.build_embeddings_table()
    mock_emb.generate_embedding.assert_not_called()
    assert resolver._conn is None
    resolver.close()
