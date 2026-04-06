import pytest
from unittest.mock import MagicMock, patch

from esdc.knowledge_graph.entity_resolver import EntityResolver


def test_resolve_field_entity():
    """Test resolving field name from ESDC DB."""
    mock_db = MagicMock()
    mock_cursor = MagicMock()

    def mock_fetchone():
        call_count = mock_cursor.execute.call_count
        if call_count == 1:
            return ("Abadi",)
        return None

    mock_cursor.fetchone.side_effect = mock_fetchone
    mock_cursor.execute.return_value = None
    mock_db.cursor.return_value = mock_cursor

    resolver = EntityResolver(db_connection=mock_db)
    result = resolver.resolve("Abadi")

    assert result["type"] == "field"
    assert result["name"] == "Abadi"
    assert result["source"] == "esdc_db"


def test_resolve_project_entity():
    """Test resolving project name from ESDC DB."""
    mock_db = MagicMock()
    mock_cursor = MagicMock()

    def mock_fetchone():
        call_count = mock_cursor.execute.call_count
        if call_count == 2:
            return ("Corridor",)
        return None

    mock_cursor.fetchone.side_effect = mock_fetchone
    mock_cursor.execute.return_value = None
    mock_db.cursor.return_value = mock_cursor

    resolver = EntityResolver(db_connection=mock_db)
    result = resolver.resolve("Corridor")

    assert result["type"] == "project"
    assert result["name"] == "Corridor"
    assert result["source"] == "esdc_db"


def test_resolve_unknown_entity():
    """Test resolving unknown entity."""
    mock_db = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None
    mock_cursor.execute.return_value = None
    mock_db.cursor.return_value = mock_cursor

    resolver = EntityResolver(db_connection=mock_db)
    result = resolver.resolve("UnknownField")

    assert result["type"] == "custom"
    assert result["name"] == "UnknownField"
    assert result["source"] == "custom"


def test_resolve_wk_entity():
    """Test resolving work area name."""
    mock_db = MagicMock()
    mock_cursor = MagicMock()

    def mock_fetchone():
        call_count = mock_cursor.execute.call_count
        if call_count == 3:
            return ("Rokan",)
        return None

    mock_cursor.fetchone.side_effect = mock_fetchone
    mock_cursor.execute.return_value = None
    mock_db.cursor.return_value = mock_cursor

    resolver = EntityResolver(db_connection=mock_db)
    result = resolver.resolve("Rokan")

    assert result["type"] == "wk"
    assert result["source"] == "esdc_db"


def test_resolve_operator_entity():
    """Test resolving operator name."""
    mock_db = MagicMock()
    mock_cursor = MagicMock()

    def mock_fetchone():
        call_count = mock_cursor.execute.call_count
        if call_count == 4:
            return ("Pertamina",)
        return None

    mock_cursor.fetchone.side_effect = mock_fetchone
    mock_cursor.execute.return_value = None
    mock_db.cursor.return_value = mock_cursor

    resolver = EntityResolver(db_connection=mock_db)
    result = resolver.resolve("Pertamina")

    assert result["type"] == "operator"
    assert result["source"] == "esdc_db"


def test_resolve_pod_entity():
    """Test resolving POD name."""
    mock_db = MagicMock()
    mock_cursor = MagicMock()

    def mock_fetchone():
        call_count = mock_cursor.execute.call_count
        if call_count == 5:
            return ("POD-A",)
        return None

    mock_cursor.fetchone.side_effect = mock_fetchone
    mock_cursor.execute.return_value = None
    mock_db.cursor.return_value = mock_cursor

    resolver = EntityResolver(db_connection=mock_db)
    result = resolver.resolve("POD-A")

    assert result["type"] == "pod"
    assert result["source"] == "esdc_db"


def test_resolve_case_insensitive():
    """Test that entity resolution is case-insensitive."""
    mock_db = MagicMock()
    mock_cursor = MagicMock()

    def mock_fetchone():
        call_count = mock_cursor.execute.call_count
        if call_count == 1:
            return ("Abadi",)
        return None

    mock_cursor.fetchone.side_effect = mock_fetchone
    mock_cursor.execute.return_value = None
    mock_db.cursor.return_value = mock_cursor

    resolver = EntityResolver(db_connection=mock_db)
    result = resolver.resolve("ABADI")

    assert result["type"] == "field"
    assert result["name"] == "ABADI"


def test_close():
    """Test that close() closes connection if owned."""
    mock_db = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None
    mock_cursor.execute.return_value = None
    mock_db.cursor.return_value = mock_cursor

    resolver = EntityResolver(db_connection=mock_db)
    resolver.close()

    # Connection should NOT be closed because it was provided externally
    mock_db.close.assert_not_called()


def test_close_owned_connection():
    """Test that close() closes connection if owned by resolver."""
    with patch("esdc.knowledge_graph.entity_resolver.sqlite3.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_cursor.execute.return_value = None
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        resolver = EntityResolver()
        resolver.close()

        # Connection should be closed because it was created by resolver
        mock_conn.close.assert_called_once()


def test_resolve_returns_esdc_id():
    """Test that resolve returns esdc_id when found."""
    mock_db = MagicMock()
    mock_cursor = MagicMock()

    def mock_fetchone():
        call_count = mock_cursor.execute.call_count
        if call_count == 1:
            return ("Abadi",)
        return None

    mock_cursor.fetchone.side_effect = mock_fetchone
    mock_cursor.execute.return_value = None
    mock_db.cursor.return_value = mock_cursor

    resolver = EntityResolver(db_connection=mock_db)
    result = resolver.resolve("Abadi")

    assert "esdc_id" in result
    assert result["esdc_id"] == "Abadi"
