"""Spatial query resolver for ESDC using DuckDB spatial extension.

Provides spatial proximity queries and graph traversal using DuckDB's
native spatial capabilities without requiring LadybugDB.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
from sklearn.cluster import DBSCAN

from esdc.configs import Config

logger = logging.getLogger(__name__)


class SpatialResolver:
    """Resolve spatial queries using DuckDB spatial extension.

    Provides:
    - Field proximity queries (find fields within radius)
    - Working area containment queries
    - Distance calculations between entities
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        """Initialize with DuckDB connection."""
        if db_path is None:
            db_path = Config.get_db_file()
        self._db_path = Path(db_path)
        self._conn: duckdb.DuckDBPyConnection | None = None

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get or create DuckDB connection with spatial extension loaded."""
        if self._conn is None:
            self._conn = duckdb.connect(str(self._db_path), read_only=True)
            # Load spatial extension
            try:
                self._conn.execute("INSTALL spatial")
            except Exception:
                pass  # Already installed
            self._conn.execute("LOAD spatial")
            logger.debug(
                "[Spatial] DuckDB connection established with spatial extension"
            )
        return self._conn

    def find_fields_near_field(
        self,
        field_name: str,
        radius_km: float = 20.0,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Find fields within specified radius of a target field.

        Args:
            field_name: Name of the reference field
            radius_km: Search radius in kilometers (default: 20)
            limit: Maximum results (default: 10)

        Returns:
            Dict with status and list of nearby fields with distances
        """
        conn = self._get_connection()

        query = """
            WITH reference AS (
                SELECT DISTINCT
                    field_id,
                    field_name,
                    field_lat,
                    field_long,
                    ST_Point(field_lat, field_long)::POINT_2D as geom
                FROM project_resources
                WHERE field_name ILIKE ?
                    AND field_lat IS NOT NULL
                    AND field_long IS NOT NULL
                LIMIT 1
            ),
            candidates AS (
                SELECT DISTINCT
                    p.field_id,
                    p.field_name,
                    p.field_lat,
                    p.field_long,
                    ST_Point(p.field_lat, p.field_long)::POINT_2D as geom
                FROM project_resources p
                WHERE p.field_lat IS NOT NULL
                    AND p.field_long IS NOT NULL
                    AND p.field_id NOT IN (SELECT field_id FROM reference)
            )
            SELECT
                c.field_id,
                c.field_name,
                ST_Distance_Spheroid(
                    r.geom,
                    c.geom
                ) / 1000.0 as distance_km  -- Convert meters to km
            FROM reference r
            CROSS JOIN candidates c
            WHERE ST_Distance_Spheroid(r.geom, c.geom) / 1000.0 <= ?
            ORDER BY distance_km
            LIMIT ?
        """

        try:
            result = conn.execute(
                query, [f"%{field_name}%", radius_km, limit]
            ).fetchall()

            if not result:
                return {
                    "status": "no_results",
                    "query": {"field": field_name, "radius_km": radius_km},
                    "nearby_fields": [],
                }

            nearby = [
                {
                    "field_id": row[0],
                    "field_name": row[1],
                    "distance_km": round(row[2], 2),
                }
                for row in result
            ]

            return {
                "status": "success",
                "query": {"field": field_name, "radius_km": radius_km},
                "nearby_fields": nearby,
                "count": len(nearby),
            }

        except Exception as e:
            logger.error("[Spatial] proximity_query_failed | error=%s", e)
            return {
                "status": "error",
                "message": str(e),
                "query": {"field": field_name, "radius_km": radius_km},
            }

    def find_fields_in_working_area(
        self,
        wk_name: str,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Find all fields within a working area.

        Args:
            wk_name: Name of the working area
            limit: Maximum results

        Returns:
            Dict with status and list of fields
        """
        conn = self._get_connection()

        query = """
            SELECT DISTINCT
                field_id,
                field_name,
                field_lat,
                field_long
            FROM project_resources
            WHERE wk_name ILIKE ?
                AND field_id IS NOT NULL
            ORDER BY field_name
            LIMIT ?
        """

        try:
            result = conn.execute(query, [f"%{wk_name}%", limit]).fetchall()

            fields = [
                {
                    "field_id": row[0],
                    "field_name": row[1],
                    "lat": row[2],
                    "long": row[3],
                }
                for row in result
            ]

            return {
                "status": "success" if fields else "no_results",
                "query": {"working_area": wk_name},
                "fields": fields,
                "count": len(fields),
            }

        except Exception as e:
            logger.error("[Spatial] working_area_query_failed | error=%s", e)
            return {
                "status": "error",
                "message": str(e),
            }

    def calculate_distance(
        self,
        from_field: str,
        to_field: str,
    ) -> dict[str, Any]:
        """Calculate distance between two fields.

        Args:
            from_field: Name of origin field
            to_field: Name of destination field

        Returns:
            Dict with distance in km
        """
        conn = self._get_connection()

        query = """
            WITH coords1 AS (
                SELECT DISTINCT
                    field_name,
                    field_lat,
                    field_long
                FROM project_resources
                WHERE field_name ILIKE ?
                    AND field_lat IS NOT NULL
                    AND field_long IS NOT NULL
                LIMIT 1
            ),
            coords2 AS (
                SELECT DISTINCT
                    field_name,
                    field_lat,
                    field_long
                FROM project_resources
                WHERE field_name ILIKE ?
                    AND field_lat IS NOT NULL
                    AND field_long IS NOT NULL
                LIMIT 1
            )
            SELECT
                c1.field_name as from_name,
                c1.field_lat as from_lat,
                c1.field_long as from_long,
                c2.field_name as to_name,
                c2.field_lat as to_lat,
                c2.field_long as to_long,
                -- Haversine formula for distance calculation
                6371.0 * 2 * ASIN(SQRT(
                    POWER(SIN(RADIANS(c2.field_lat - c1.field_lat) / 2), 2) +
                    COS(RADIANS(c1.field_lat)) * COS(RADIANS(c2.field_lat)) *
                    POWER(SIN(RADIANS(c2.field_long - c1.field_long) / 2), 2)
                )) as distance_km
            FROM coords1 c1
            CROSS JOIN coords2 c2
        """

        try:
            result = conn.execute(
                query, [f"%{from_field}%", f"%{to_field}%"]
            ).fetchone()

            if not result or result[0] is None or result[3] is None:
                return {
                    "status": "not_found",
                    "message": f"Could not find both fields: {from_field}, {to_field}",
                }

            return {
                "status": "success",
                "from_field": result[0],
                "from_lat": result[1],
                "from_long": result[2],
                "to_field": result[3],
                "to_lat": result[4],
                "to_long": result[5],
                "distance_km": round(result[6], 2),
            }

        except Exception as e:
            logger.error("[Spatial] distance_calculation_failed | error=%s", e)
            return {
                "status": "error",
                "message": str(e),
            }

    def get_field_coordinates(
        self,
        field_name: str,
    ) -> dict[str, Any]:
        """Get coordinates for a field.

        Args:
            field_name: Name of the field

        Returns:
            Dict with lat/long coordinates
        """
        conn = self._get_connection()

        query = """
            SELECT DISTINCT
                field_id,
                field_name,
                field_lat,
                field_long,
                wk_name
            FROM project_resources
            WHERE field_name ILIKE ?
                AND field_lat IS NOT NULL
                AND field_long IS NOT NULL
            LIMIT 1
        """

        try:
            result = conn.execute(query, [f"%{field_name}%"]).fetchone()

            if not result:
                return {
                    "status": "not_found",
                    "message": f"Field not found: {field_name}",
                }

            return {
                "status": "success",
                "field_id": result[0],
                "field_name": result[1],
                "latitude": result[2],
                "longitude": result[3],
                "working_area": result[4],
            }

        except Exception as e:
            logger.error("[Spatial] coordinate_lookup_failed | error=%s", e)
            return {
                "status": "error",
                "message": str(e),
            }

    def find_nearest_from_coordinates(
        self,
        lat: float,
        long: float,
        entity_type: str = "field",
        radius_km: float = 20.0,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Find nearest entities from arbitrary coordinates.

        Args:
            lat: Latitude coordinate
            long: Longitude coordinate
            entity_type: Type of entity to search ("field" or "working_area")
            radius_km: Search radius in kilometers (default: 20)
            limit: Maximum results (default: 10)

        Returns:
            Dict with status and list of nearby entities with distances
        """
        conn = self._get_connection()

        if entity_type == "field":
            query = """
                WITH reference AS (
                    SELECT ST_Point(?, ?)::POINT_2D as geom
                ),
                candidates AS (
                    SELECT DISTINCT
                        field_id,
                        field_name,
                        field_lat,
                        field_long,
                        wk_name,
                        ST_Point(field_lat, field_long)::POINT_2D as geom
                    FROM project_resources
                    WHERE field_lat IS NOT NULL
                        AND field_long IS NOT NULL
                )
                SELECT
                    c.field_id,
                    c.field_name,
                    c.wk_name,
                    c.field_lat,
                    c.field_long,
                    ST_Distance_Spheroid(r.geom, c.geom) / 1000.0 as distance_km
                FROM reference r
                CROSS JOIN candidates c
                WHERE ST_Distance_Spheroid(r.geom, c.geom) / 1000.0 <= ?
                ORDER BY distance_km
                LIMIT ?
            """
            params = [lat, long, radius_km, limit]
        elif entity_type == "working_area":
            query = """
                WITH reference AS (
                    SELECT ST_Point(?, ?)::POINT_2D as geom
                ),
                candidates AS (
                    SELECT DISTINCT
                        wk_id,
                        wk_name,
                        wk_lat,
                        wk_long,
                        ST_Point(wk_lat, wk_long)::POINT_2D as geom
                    FROM project_resources
                    WHERE wk_lat IS NOT NULL
                        AND wk_long IS NOT NULL
                )
                SELECT
                    c.wk_id,
                    c.wk_name,
                    c.wk_lat,
                    c.wk_long,
                    ST_Distance_Spheroid(r.geom, c.geom) / 1000.0 as distance_km
                FROM reference r
                CROSS JOIN candidates c
                WHERE ST_Distance_Spheroid(r.geom, c.geom) / 1000.0 <= ?
                ORDER BY distance_km
                LIMIT ?
            """
            params = [lat, long, radius_km, limit]
        else:
            return {
                "status": "error",
                "message": f"Invalid entity_type: {entity_type}. Use 'field' or 'working_area'",
            }

        try:
            result = conn.execute(query, params).fetchall()

            if not result:
                return {
                    "status": "no_results",
                    "query": {
                        "lat": lat,
                        "long": long,
                        "entity_type": entity_type,
                        "radius_km": radius_km,
                    },
                    "nearby_entities": [],
                }

            if entity_type == "field":
                nearby = [
                    {
                        "field_id": row[0],
                        "field_name": row[1],
                        "wk_name": row[2],
                        "latitude": row[3],
                        "longitude": row[4],
                        "distance_km": round(row[5], 2),
                    }
                    for row in result
                ]
            else:
                nearby = [
                    {
                        "wk_id": row[0],
                        "wk_name": row[1],
                        "latitude": row[2],
                        "longitude": row[3],
                        "distance_km": round(row[4], 2),
                    }
                    for row in result
                ]

            return {
                "status": "success",
                "query": {
                    "lat": lat,
                    "long": long,
                    "entity_type": entity_type,
                    "radius_km": radius_km,
                },
                "nearby_entities": nearby,
                "count": len(nearby),
            }

        except Exception as e:
            logger.error("[Spatial] nearest_from_coords_failed | error=%s", e)
            return {
                "status": "error",
                "message": str(e),
                "query": {
                    "lat": lat,
                    "long": long,
                    "entity_type": entity_type,
                    "radius_km": radius_km,
                },
            }

    def find_field_clusters(
        self,
        max_distance_km: float = 20.0,
        min_cluster_size: int = 2,
    ) -> dict[str, Any]:
        """Find clusters of fields based on proximity.

        Uses simple distance-based clustering where fields within max_distance_km
        of each other are grouped together.

        Args:
            max_distance_km: Maximum distance between fields in same cluster (default: 20)
            min_cluster_size: Minimum number of fields to form a cluster (default: 2)

        Returns:
            Dict with status and list of clusters
        """
        conn = self._get_connection()

        # Get all fields with coordinates
        query = """
            SELECT DISTINCT
                field_id,
                field_name,
                field_lat,
                field_long
            FROM project_resources
            WHERE field_lat IS NOT NULL
                AND field_long IS NOT NULL
            ORDER BY field_id
        """

        try:
            result = conn.execute(query).fetchall()

            if not result or len(result) < min_cluster_size:
                return {
                    "status": "no_results",
                    "message": f"Insufficient fields with coordinates (found {len(result)})",
                    "clusters": [],
                    "unclustered": [],
                }

            # Limit check - prevent memory issues with very large datasets
            MAX_FIELDS = 10000
            if len(result) > MAX_FIELDS:
                return {
                    "status": "error",
                    "message": f"Too many fields ({len(result)}). Maximum supported: {MAX_FIELDS}",
                }

            # Build field data
            fields_data = [
                {"field_id": row[0], "field_name": row[1], "lat": row[2], "long": row[3]}
                for row in result
            ]

            # Convert to numpy arrays for sklearn
            # Coordinates in radians for haversine metric
            coords_rad = np.array([[f["lat"], f["long"]] for f in fields_data]) * np.pi / 180.0

            # Convert max_distance_km to radians (eps parameter for DBSCAN)
            # Earth's radius = 6371 km
            eps_radians = max_distance_km / 6371.0

            # Apply DBSCAN clustering
            # haversine metric is optimized for geographic coordinates
            clustering = DBSCAN(
                eps=eps_radians,
                min_samples=min_cluster_size,
                metric='haversine',
                algorithm='ball_tree'
            )
            cluster_labels = clustering.fit_predict(coords_rad)

            # Process results
            clusters_dict = {}
            unclustered = []

            for i, label in enumerate(cluster_labels):
                field = fields_data[i]
                if label == -1:
                    # Noise point (unclustered)
                    unclustered.append(field["field_name"])
                else:
                    if label not in clusters_dict:
                        clusters_dict[label] = []
                    clusters_dict[label].append(field)

            # Format clusters
            clusters = []
            for label, cluster_fields in clusters_dict.items():
                # Calculate cluster center
                avg_lat = sum(f["lat"] for f in cluster_fields) / len(cluster_fields)
                avg_long = sum(f["long"] for f in cluster_fields) / len(cluster_fields)

                clusters.append({
                    "cluster_id": int(label) + 1,
                    "fields": [f["field_name"] for f in cluster_fields],
                    "field_count": len(cluster_fields),
                    "center_lat": round(avg_lat, 4),
                    "center_long": round(avg_long, 4),
                })

            # Sort clusters by field count (largest first)
            clusters.sort(key=lambda x: x["field_count"], reverse=True)

            # Reassign cluster IDs after sorting
            for i, cluster in enumerate(clusters):
                cluster["cluster_id"] = i + 1

            return {
                "status": "success",
                "query": {
                    "max_distance_km": max_distance_km,
                    "min_cluster_size": min_cluster_size,
                },
                "cluster_count": len(clusters),
                "clusters": clusters,
                "unclustered": unclustered,
            }

        except Exception as e:
            logger.error("[Spatial] clustering_failed | error=%s", e)
            return {
                "status": "error",
                "message": str(e),
            }

    def find_adjacent_working_areas(
        self,
        wk_name: str,
        max_distance_km: float = 20.0,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Find working areas adjacent to a target working area.

        Args:
            wk_name: Name of the reference working area
            max_distance_km: Maximum distance to consider adjacent (default: 20)
            limit: Maximum results (default: 10)

        Returns:
            Dict with status and list of adjacent working areas with distances
        """
        conn = self._get_connection()

        query = """
            WITH reference AS (
                SELECT DISTINCT
                    wk_id,
                    wk_name,
                    wk_lat,
                    wk_long,
                    ST_Point(wk_lat, wk_long)::POINT_2D as geom
                FROM project_resources
                WHERE wk_name ILIKE ?
                    AND wk_lat IS NOT NULL
                    AND wk_long IS NOT NULL
                LIMIT 1
            ),
            candidates AS (
                SELECT DISTINCT
                    wk_id,
                    wk_name,
                    wk_lat,
                    wk_long,
                    ST_Point(wk_lat, wk_long)::POINT_2D as geom
                FROM project_resources
                WHERE wk_lat IS NOT NULL
                    AND wk_long IS NOT NULL
                    AND wk_id NOT IN (SELECT wk_id FROM reference)
            )
            SELECT
                c.wk_id,
                c.wk_name,
                c.wk_lat,
                c.wk_long,
                ST_Distance_Spheroid(r.geom, c.geom) / 1000.0 as distance_km
            FROM reference r
            CROSS JOIN candidates c
            WHERE ST_Distance_Spheroid(r.geom, c.geom) / 1000.0 <= ?
            ORDER BY distance_km
            LIMIT ?
        """

        try:
            result = conn.execute(
                query, [f"%{wk_name}%", max_distance_km, limit]
            ).fetchall()

            if not result:
                return {
                    "status": "no_results",
                    "query": {"wk_name": wk_name, "max_distance_km": max_distance_km},
                    "adjacent_working_areas": [],
                }

            adjacent = [
                {
                    "wk_id": row[0],
                    "wk_name": row[1],
                    "latitude": row[2],
                    "longitude": row[3],
                    "distance_km": round(row[4], 2),
                }
                for row in result
            ]

            return {
                "status": "success",
                "query": {"wk_name": wk_name, "max_distance_km": max_distance_km},
                "adjacent_working_areas": adjacent,
                "count": len(adjacent),
            }

        except Exception as e:
            logger.error("[Spatial] adjacent_wk_query_failed | error=%s", e)
            return {
                "status": "error",
                "message": str(e),
                "query": {"wk_name": wk_name, "max_distance_km": max_distance_km},
            }

    def calculate_average_distance(
        self,
        field_names: list[str],
    ) -> dict[str, Any]:
        """Calculate average distance between multiple fields.

        Computes pairwise distances between all fields and returns
        statistics (average, min, max).

        Args:
            field_names: List of field names to calculate distances between

        Returns:
            Dict with pairwise distances and statistics
        """
        if len(field_names) < 2:
            return {
                "status": "error",
                "message": "At least 2 field names are required",
            }

        conn = self._get_connection()

        # Get coordinates for all fields
        placeholders = ", ".join(["?"] * len(field_names))
        query = f"""
            SELECT DISTINCT
                field_name,
                field_lat,
                field_long
            FROM project_resources
            WHERE field_name ILIKE ANY (ARRAY[{placeholders}])
                AND field_lat IS NOT NULL
                AND field_long IS NOT NULL
        """

        try:
            result = conn.execute(query, field_names).fetchall()

            if len(result) < 2:
                return {
                    "status": "not_found",
                    "message": f"Could not find at least 2 fields from: {field_names}",
                }

            fields = [
                {"field_name": row[0], "lat": row[1], "long": row[2]}
                for row in result
            ]

            # Calculate pairwise distances
            pairwise_distances = []
            for i, field1 in enumerate(fields):
                for field2 in fields[i + 1 :]:
                    dist_query = """
                        SELECT ST_Distance_Spheroid(
                            ST_Point(?, ?)::POINT_2D,
                            ST_Point(?, ?)::POINT_2D
                        ) / 1000.0 as distance_km
                    """
                    dist_result = conn.execute(
                        dist_query,
                        [field1["lat"], field1["long"], field2["lat"], field2["long"]],
                    ).fetchone()

                    if dist_result:
                        distance = round(dist_result[0], 2)
                        pairwise_distances.append({
                            "from": field1["field_name"],
                            "to": field2["field_name"],
                            "distance_km": distance,
                        })

            if not pairwise_distances:
                return {
                    "status": "error",
                    "message": "Could not calculate distances between fields",
                }

            # Calculate statistics
            distances = [d["distance_km"] for d in pairwise_distances]
            avg_distance = sum(distances) / len(distances)
            min_distance = min(distances)
            max_distance = max(distances)

            return {
                "status": "success",
                "field_count": len(fields),
                "fields": [f["field_name"] for f in fields],
                "pairwise_distances": pairwise_distances,
                "statistics": {
                    "average_distance_km": round(avg_distance, 2),
                    "min_distance_km": min_distance,
                    "max_distance_km": max_distance,
                    "pair_count": len(pairwise_distances),
                },
            }

        except Exception as e:
            logger.error("[Spatial] average_distance_calculation_failed | error=%s", e)
            return {
                "status": "error",
                "message": str(e),
            }

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
