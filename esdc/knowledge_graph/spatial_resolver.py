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

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
