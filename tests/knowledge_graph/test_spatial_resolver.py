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
            wk_name TEXT
        )
    """)
    
    # Insert test data with known coordinates
    # Duri field (Riau, Indonesia) and nearby fields
    test_data = [
        # field_id, field_name, field_lat, field_long, wk_name
        ("DURI", "Duri", 1.7167, 101.4500, "Rokan"),
        ("ROKAN", "Rokan", 1.8000, 101.5000, "Rokan"),
        ("BELANAK", "Belanak", 1.6500, 101.4000, "Rokan"),
        ("PELANGKEN", "Pelangken", 1.7200, 101.4600, "Rokan"),
        ("DUMAI", "Dumai", 1.6667, 101.4500, "Rokan"),
        # Jakarta area (far from Duri - ~900km)
        ("JAKARTA", "Jakarta Field", -6.2000, 106.8167, "North Java"),
    ]
    
    conn.executemany(
        "INSERT INTO project_resources VALUES (?, ?, ?, ?, ?)",
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
