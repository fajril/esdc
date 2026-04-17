"""Tests for spatial resolver WK-scoped queries."""

from unittest.mock import MagicMock, patch

from esdc.knowledge_graph.spatial_resolver import SpatialResolver


class TestSpatialResolverWKScoped:
    """Test that wk_name parameter filters results correctly."""

    @patch("esdc.knowledge_graph.spatial_resolver.SpatialResolver._get_connection")
    def test_find_fields_near_field_accepts_wk_name(self, mock_conn):
        """find_fields_near_field should accept wk_name parameter."""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.return_value.execute.return_value = mock_cursor
        mock_conn.return_value.close = MagicMock()

        resolver = SpatialResolver()
        resolver._conn = mock_conn.return_value

        result = resolver.find_fields_near_field(
            field_name="Tambora",
            radius_km=20,
            wk_name="Mahakam",
        )

        assert result["status"] == "no_results"
        call_args = mock_conn.return_value.execute.call_args
        sql = call_args[0][0]
        assert "wk_name" in sql

    @patch("esdc.knowledge_graph.spatial_resolver.SpatialResolver._get_connection")
    def test_find_fields_near_field_without_wk_name(self, mock_conn):
        """find_fields_near_field should work without wk_name (backward compat)."""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.return_value.execute.return_value = mock_cursor
        mock_conn.return_value.close = MagicMock()

        resolver = SpatialResolver()
        resolver._conn = mock_conn.return_value

        result = resolver.find_fields_near_field(
            field_name="Duri",
            radius_km=20,
        )

        assert result["status"] == "no_results"

    @patch("esdc.knowledge_graph.spatial_resolver.SpatialResolver._get_connection")
    def test_calculate_distance_accepts_wk_name(self, mock_conn):
        """calculate_distance should accept wk_name parameter."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (
            "Tambora",
            1.23,
            116.45,
            "Handil",
            1.25,
            116.50,
            5.67,
        )
        mock_conn.return_value.execute.return_value = mock_cursor
        mock_conn.return_value.close = MagicMock()

        resolver = SpatialResolver()
        resolver._conn = mock_conn.return_value

        result = resolver.calculate_distance(
            from_field="Tambora",
            to_field="Handil",
            wk_name="Mahakam",
        )

        assert result["status"] == "success"
        assert result["wk_name"] == "Mahakam"
        call_args = mock_conn.return_value.execute.call_args
        sql = call_args[0][0]
        assert "wk_name" in sql

    @patch("esdc.knowledge_graph.spatial_resolver.SpatialResolver._get_connection")
    def test_calculate_distance_without_wk_name(self, mock_conn):
        """calculate_distance should work without wk_name (backward compat)."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (
            "Duri",
            1.23,
            116.45,
            "Rokan",
            0.5,
            100.0,
            150.0,
        )
        mock_conn.return_value.execute.return_value = mock_cursor
        mock_conn.return_value.close = MagicMock()

        resolver = SpatialResolver()
        resolver._conn = mock_conn.return_value

        result = resolver.calculate_distance(
            from_field="Duri",
            to_field="Rokan",
        )

        assert result["status"] == "success"
        assert "wk_name" not in result

    @patch("esdc.knowledge_graph.spatial_resolver.SpatialResolver._get_connection")
    def test_get_field_coordinates_accepts_wk_name(self, mock_conn):
        """get_field_coordinates should accept wk_name parameter."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (
            "field-1",
            "Tambora",
            1.23,
            116.45,
            "Mahakam",
        )
        mock_conn.return_value.execute.return_value = mock_cursor
        mock_conn.return_value.close = MagicMock()

        resolver = SpatialResolver()
        resolver._conn = mock_conn.return_value

        result = resolver.get_field_coordinates(
            field_name="Tambora",
            wk_name="Mahakam",
        )

        assert result["status"] == "success"
        call_args = mock_conn.return_value.execute.call_args
        sql = call_args[0][0]
        assert "wk_name" in sql

    @patch("esdc.knowledge_graph.spatial_resolver.SpatialResolver._get_connection")
    def test_get_field_coordinates_without_wk_name(self, mock_conn):
        """get_field_coordinates should work without wk_name (backward compat)."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("field-1", "Duri", 1.23, 116.45, "Rokan")
        mock_conn.return_value.execute.return_value = mock_cursor
        mock_conn.return_value.close = MagicMock()

        resolver = SpatialResolver()
        resolver._conn = mock_conn.return_value

        result = resolver.get_field_coordinates(field_name="Duri")

        assert result["status"] == "success"


class TestResolveSpatialToolWKParam:
    """Test that resolve_spatial tool accepts wk_name parameter."""

    def test_resolve_spatial_has_wk_name_parameter(self):
        """resolve_spatial tool should accept wk_name parameter."""
        from esdc.chat.tools import resolve_spatial

        args_schema = resolve_spatial.args_schema
        assert args_schema is not None, "Tool has no args_schema"
        properties = args_schema.model_fields
        assert "wk_name" in properties, (
            f"wk_name not in args_schema fields: {list(properties.keys())}"
        )
