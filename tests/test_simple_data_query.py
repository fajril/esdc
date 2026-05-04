"""Unit tests for simple_data_query tool."""

import json
from unittest.mock import MagicMock, patch

import pytest

from esdc.chat.smart_query import simple_data_query


def _make_conn(rows, desc):
    """Build a mocked DuckDB connection."""
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = rows
    mock_cursor.description = desc

    mock_conn = MagicMock()
    # conn.execute(sql, params) -> cursor
    mock_conn.execute.return_value = mock_cursor
    # Set description directly on connection for DuckDB API
    mock_conn.description = desc
    return mock_conn


class TestSimpleDataQuery:
    """Tests for the simple_data_query LangChain tool."""

    @patch("esdc.chat.smart_query.Config.get_db_file")
    @patch("esdc.chat.smart_query.get_duckdb_connection")
    def test_reserves_query(self, mock_get_conn, mock_get_db):
        mock_db = MagicMock()
        mock_db.exists.return_value = True
        mock_get_db.return_value = mock_db

        desc = [("report_year",), ("reserves_mstb",), ("reserves_bscf",)]
        mock_get_conn.return_value = _make_conn([(2024, 1000.0, 500.0)], desc)

        result = simple_data_query.invoke(
            {
                "query_type": "reserves",
                "entity_level": "work_area",
                "entity_name": "Rokan",
            }
        )
        data = json.loads(result)
        assert data["query_type"] == "reserves"
        assert data["entity"] == "Rokan"
        assert data["entity_level"] == "work_area"
        assert "1,000.00" in data["summary"]
        mock_get_conn.return_value.close.assert_called_once()

    @patch("esdc.chat.smart_query.Config.get_db_file")
    @patch("esdc.chat.smart_query.get_duckdb_connection")
    def test_resources_query(self, mock_get_conn, mock_get_db):
        mock_db = MagicMock()
        mock_db.exists.return_value = True
        mock_get_db.return_value = mock_db

        desc = [
            ("project_class",),
            ("project_stage",),
            ("resources_mstb",),
            ("resources_bscf",),
            ("reserves_mstb",),
            ("reserves_bscf",),
        ]
        mock_get_conn.return_value = _make_conn(
            [("1. Reserves & GRR", "Exploration", 1000.0, 500.0, 800.0, 400.0)],
            desc,
        )

        result = simple_data_query.invoke(
            {
                "query_type": "resources",
                "entity_level": "work_area",
                "entity_name": "Rokan",
            }
        )
        data = json.loads(result)
        assert data["query_type"] == "resources"
        assert "Potensi" in data["summary"]

    @patch("esdc.chat.smart_query.Config.get_db_file")
    def test_database_not_found(self, mock_get_db):
        mock_db = MagicMock()
        mock_db.exists.return_value = False
        mock_get_db.return_value = mock_db

        result = simple_data_query.invoke(
            {
                "query_type": "reserves",
                "entity_level": "national",
            }
        )
        data = json.loads(result)
        assert "error" in data
        assert "Database not found" in data["error"]

    def test_invalid_query_type(self):
        with pytest.raises(ValueError, match="Invalid query_type"):
            simple_data_query.invoke(
                {
                    "query_type": "invalid",
                    "entity_level": "work_area",
                }
            )

    def test_invalid_entity_level(self):
        with pytest.raises(ValueError, match="Invalid entity_level"):
            simple_data_query.invoke(
                {
                    "query_type": "reserves",
                    "entity_level": "invalid_level",
                }
            )

    @patch("esdc.chat.smart_query.Config.get_db_file")
    @patch("esdc.chat.smart_query.get_duckdb_connection")
    def test_comparison_mode(self, mock_get_conn, mock_get_db):
        mock_db = MagicMock()
        mock_db.exists.return_value = True
        mock_get_db.return_value = mock_db

        desc = [("report_year",), ("reserves_mstb",), ("reserves_bscf",)]
        mock_get_conn.return_value = _make_conn(
            [
                (2023, 900.0, 450.0),
                (2024, 1000.0, 500.0),
            ],
            desc,
        )

        result = simple_data_query.invoke(
            {
                "query_type": "reserves",
                "entity_level": "work_area",
                "entity_name": "Rokan",
                "report_year": [2023, 2024],
            }
        )
        data = json.loads(result)
        # summary should contain both years
        assert "2023" in data["summary"] or "trend" in data["summary"]
        mock_get_conn.return_value.close.assert_called_once()
