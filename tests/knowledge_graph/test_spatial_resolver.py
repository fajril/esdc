"""Tests for SpatialResolver proximity queries."""

from __future__ import annotations

import pytest
import duckdb
from pathlib import Path

from esdc.knowledge_graph.spatial_resolver import SpatialResolver


@pytest.fixture
def mock_spatial_db(tmp_path: Path) -> Path:
    """Create a mock DuckDB with spatial data for testing."""
    db_path = tmp_path / "test_spatial.db"
    conn = duckdb.connect(str(db_path))
    
    # Load spatial extension
    try:
        conn.execute("INSTALL spatial")
    except Exception:
        pass
    conn.execute("LOAD spatial")
    
    # Create table with test data
    # Using real Indonesian oil field coordinates
    conn.execute("""
        CREATE TABLE project_resources (
            field_id TEXT PRIMARY KEY,
            field_name TEXT,
            field_lat DOUBLE,
            field_long DOUBLE,
            wk_id TEXT,
            wk_name TEXT,
            wk_lat DOUBLE,
            wk_long DOUBLE
        )
    """)
    
    # Insert test data with known coordinates
    # Duri field (Riau, Indonesia) and nearby fields
    test_data = [
        # field_id, field_name, field_lat, field_long, wk_id, wk_name, wk_lat, wk_long
        ("DURI", "Duri", 1.7167, 101.4500, "WK_ROKAN", "Rokan", 1.7500, 101.4700),
        ("ROKAN", "Rokan", 1.8000, 101.5000, "WK_ROKAN", "Rokan", 1.7500, 101.4700),
        ("BELANAK", "Belanak", 1.6500, 101.4000, "WK_ROKAN", "Rokan", 1.7500, 101.4700),
        ("PELANGKEN", "Pelangken", 1.7200, 101.4600, "WK_ROKAN", "Rokan", 1.7500, 101.4700),
        ("DUMAI", "Dumai", 1.6667, 101.4500, "WK_ROKAN", "Rokan", 1.7500, 101.4700),
        # Another working area nearby
        ("MINAS", "Minas", 1.6000, 101.3500, "WK_MINAS", "Minas WK", 1.5800, 101.3300),
        ("LIrik", "Lirik", 1.5500, 101.3000, "WK_MINAS", "Minas WK", 1.5800, 101.3300),
        # Jakarta area (far from Duri - ~900km)
        ("JAKARTA", "Jakarta Field", -6.2000, 106.8167, "WK_JAKARTA", "Jakarta WK", -6.1800, 106.8000),
    ]
    
    conn.executemany(
        "INSERT INTO project_resources VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        test_data
    )
    conn.close()
    return db_path


class TestSpatialResolverProximity:
    """Tests for proximity query functionality."""
    
    def test_find_fields_near_field_returns_nonzero_distance(
        self, mock_spatial_db: Path
    ) -> None:
        """Test that proximity query returns non-zero distances."""
        resolver = SpatialResolver(db_path=mock_spatial_db)
        
        result = resolver.find_fields_near_field(
            field_name="Duri",
            radius_km=50.0,
            limit=10
        )
        
        assert result["status"] == "success"
        assert len(result["nearby_fields"]) > 0
        
        # All distances should be non-zero and positive
        for field in result["nearby_fields"]:
            assert field["distance_km"] > 0, (
                f"Field {field['field_name']} has distance {field['distance_km']}, "
                "expected non-zero value"
            )
    
    def test_find_fields_near_field_distance_accuracy(
        self, mock_spatial_db: Path
    ) -> None:
        """Test that distance calculation is reasonably accurate."""
        resolver = SpatialResolver(db_path=mock_spatial_db)
        
        result = resolver.find_fields_near_field(
            field_name="Duri",
            radius_km=100.0,
            limit=10
        )
        
        assert result["status"] == "success"
        
        # Find Rokan field (should be ~11km from Duri based on coordinates)
        rokan_field = next(
            (f for f in result["nearby_fields"] if f["field_name"] == "Rokan"),
            None
        )
        
        if rokan_field:
            # Distance should be reasonable (within 5-20km range)
            distance = rokan_field["distance_km"]
            assert 5.0 <= distance <= 20.0, (
                f"Distance from Duri to Rokan is {distance}km, "
                "expected roughly 11km"
            )
    
    def test_find_fields_near_field_respects_radius(
        self, mock_spatial_db: Path
    ) -> None:
        """Test that query respects the radius parameter."""
        resolver = SpatialResolver(db_path=mock_spatial_db)
        
        # Small radius - only very close fields
        result_small = resolver.find_fields_near_field(
            field_name="Duri",
            radius_km=5.0,
            limit=10
        )
        
        # Large radius - more fields
        result_large = resolver.find_fields_near_field(
            field_name="Duri",
            radius_km=100.0,
            limit=10
        )
        
        # Both should succeed
        assert result_small["status"] == "success"
        assert result_large["status"] == "success"
        
        # Small radius should return fewer or equal results
        assert len(result_small["nearby_fields"]) <= len(result_large["nearby_fields"])
        
        # All distances in small radius should be <= 5km
        for field in result_small["nearby_fields"]:
            assert field["distance_km"] <= 5.0, (
                f"Field {field['field_name']} at {field['distance_km']}km "
                "exceeds radius of 5km"
            )
    
    def test_find_fields_near_field_excludes_reference(
        self, mock_spatial_db: Path
    ) -> None:
        """Test that reference field is not in results."""
        resolver = SpatialResolver(db_path=mock_spatial_db)
        
        result = resolver.find_fields_near_field(
            field_name="Duri",
            radius_km=100.0,
            limit=10
        )
        
        assert result["status"] == "success"
        
        # Duri should not appear in its own nearby fields
        field_names = [f["field_name"] for f in result["nearby_fields"]]
        assert "Duri" not in field_names, "Reference field should not be in results"
    
    def test_find_fields_near_field_no_results(self, mock_spatial_db: Path) -> None:
        """Test query with very small radius returns no results."""
        resolver = SpatialResolver(db_path=mock_spatial_db)
        
        result = resolver.find_fields_near_field(
            field_name="Duri",
            radius_km=0.1,
            limit=10
        )
        
        assert result["status"] == "no_results"
        assert len(result["nearby_fields"]) == 0
    
    def test_find_fields_near_field_invalid_field(self, mock_spatial_db: Path) -> None:
        """Test query with non-existent field."""
        resolver = SpatialResolver(db_path=mock_spatial_db)
        
        result = resolver.find_fields_near_field(
            field_name="NonExistentField",
            radius_km=50.0,
            limit=10
        )
        
        assert result["status"] == "no_results"
        assert len(result["nearby_fields"]) == 0


class TestFindNearestFromCoordinates:
    """Tests for find_nearest_from_coordinates method."""
    
    def test_find_nearest_from_coordinates_field(self, mock_spatial_db: Path) -> None:
        """Test finding nearest fields from arbitrary coordinates."""
        resolver = SpatialResolver(db_path=mock_spatial_db)
        
        # Coordinates near Duri
        result = resolver.find_nearest_from_coordinates(
            lat=1.72,
            long=101.45,
            entity_type="field",
            radius_km=20.0,
            limit=10
        )
        
        assert result["status"] == "success"
        assert len(result["nearby_entities"]) > 0
        
        # Duri should be the closest
        first_field = result["nearby_entities"][0]
        assert first_field["field_name"] == "Duri"
        assert first_field["distance_km"] < 1.0  # Should be very close
    
    def test_find_nearest_from_coordinates_working_area(self, mock_spatial_db: Path) -> None:
        """Test finding nearest working areas from arbitrary coordinates."""
        resolver = SpatialResolver(db_path=mock_spatial_db)
        
        result = resolver.find_nearest_from_coordinates(
            lat=1.75,
            long=101.47,
            entity_type="working_area",
            radius_km=50.0,
            limit=10
        )
        
        assert result["status"] == "success"
        assert len(result["nearby_entities"]) > 0
        
        # Should find Rokan WK
        wk_names = [e["wk_name"] for e in result["nearby_entities"]]
        assert "Rokan" in wk_names
    
    def test_find_nearest_from_coordinates_invalid_entity_type(self, mock_spatial_db: Path) -> None:
        """Test invalid entity type returns error."""
        resolver = SpatialResolver(db_path=mock_spatial_db)
        
        result = resolver.find_nearest_from_coordinates(
            lat=1.0,
            long=101.0,
            entity_type="invalid_type",
            radius_km=20.0,
        )
        
        assert result["status"] == "error"
        assert "entity_type" in result["message"].lower()
    
    def test_find_nearest_from_coordinates_no_results(self, mock_spatial_db: Path) -> None:
        """Test coordinates with no nearby entities."""
        resolver = SpatialResolver(db_path=mock_spatial_db)
        
        result = resolver.find_nearest_from_coordinates(
            lat=0.0,
            long=0.0,  # Ocean, far from test data
            entity_type="field",
            radius_km=1.0,
        )
        
        assert result["status"] == "no_results"
        assert len(result["nearby_entities"]) == 0


class TestFindFieldClusters:
    """Tests for find_field_clusters method."""
    
    def test_find_field_clusters_basic(self, mock_spatial_db: Path) -> None:
        """Test basic clustering functionality."""
        resolver = SpatialResolver(db_path=mock_spatial_db)
        
        result = resolver.find_field_clusters(
            max_distance_km=25.0,
            min_cluster_size=2
        )
        
        assert result["status"] == "success"
        # Should find at least one cluster with Rokan fields
        assert result["cluster_count"] >= 1
    
    def test_find_field_clusters_respects_min_size(self, mock_spatial_db: Path) -> None:
        """Test that min_cluster_size is respected."""
        resolver = SpatialResolver(db_path=mock_spatial_db)
        
        result = resolver.find_field_clusters(
            max_distance_km=100.0,
            min_cluster_size=3
        )
        
        assert result["status"] == "success"
        # All clusters should have at least min_cluster_size fields
        for cluster in result["clusters"]:
            assert cluster["field_count"] >= 3
    
    def test_find_field_clusters_unclustered(self, mock_spatial_db: Path) -> None:
        """Test that distant fields are marked as unclustered."""
        resolver = SpatialResolver(db_path=mock_spatial_db)
        
        result = resolver.find_field_clusters(
            max_distance_km=10.0,  # Small distance
            min_cluster_size=2
        )
        
        assert result["status"] == "success"
        # Jakarta Field should be unclustered (far from others)
        assert "Jakarta Field" in result["unclustered"]


class TestFindAdjacentWorkingAreas:
    """Tests for find_adjacent_working_areas method."""
    
    def test_find_adjacent_working_areas_basic(self, mock_spatial_db: Path) -> None:
        """Test finding adjacent working areas."""
        resolver = SpatialResolver(db_path=mock_spatial_db)
        
        result = resolver.find_adjacent_working_areas(
            wk_name="Rokan",
            max_distance_km=50.0,
            limit=10
        )
        
        assert result["status"] == "success"
        # Should find Minas WK as adjacent
        wk_names = [wk["wk_name"] for wk in result["adjacent_working_areas"]]
        assert "Minas WK" in wk_names
    
    def test_find_adjacent_working_areas_excludes_reference(self, mock_spatial_db: Path) -> None:
        """Test that reference WK is not in results."""
        resolver = SpatialResolver(db_path=mock_spatial_db)
        
        result = resolver.find_adjacent_working_areas(
            wk_name="Rokan",
            max_distance_km=100.0,
        )
        
        assert result["status"] == "success"
        # Rokan should not appear in its own adjacent list
        wk_names = [wk["wk_name"] for wk in result["adjacent_working_areas"]]
        assert "Rokan" not in wk_names
    
    def test_find_adjacent_working_areas_no_results(self, mock_spatial_db: Path) -> None:
        """Test with small radius returns no results."""
        resolver = SpatialResolver(db_path=mock_spatial_db)
        
        result = resolver.find_adjacent_working_areas(
            wk_name="Rokan",
            max_distance_km=1.0,
        )
        
        assert result["status"] == "no_results"
        assert len(result["adjacent_working_areas"]) == 0


class TestCalculateAverageDistance:
    """Tests for calculate_average_distance method."""
    
    def test_calculate_average_distance_basic(self, mock_spatial_db: Path) -> None:
        """Test basic average distance calculation."""
        resolver = SpatialResolver(db_path=mock_spatial_db)
        
        result = resolver.calculate_average_distance(
            field_names=["Duri", "Rokan", "Belanak"]
        )
        
        assert result["status"] == "success"
        assert result["field_count"] == 3
        assert len(result["pairwise_distances"]) == 3  # 3 choose 2
        
        # Check statistics
        stats = result["statistics"]
        assert stats["average_distance_km"] > 0
        assert stats["min_distance_km"] <= stats["max_distance_km"]
        assert stats["pair_count"] == 3
    
    def test_calculate_average_distance_single_field(self, mock_spatial_db: Path) -> None:
        """Test with single field returns error."""
        resolver = SpatialResolver(db_path=mock_spatial_db)
        
        result = resolver.calculate_average_distance(
            field_names=["Duri"]
        )
        
        assert result["status"] == "error"
        assert "at least 2" in result["message"].lower()
    
    def test_calculate_average_distance_invalid_fields(self, mock_spatial_db: Path) -> None:
        """Test with non-existent fields."""
        resolver = SpatialResolver(db_path=mock_spatial_db)
        
        result = resolver.calculate_average_distance(
            field_names=["NonExistent1", "NonExistent2"]
        )
        
        assert result["status"] == "not_found"