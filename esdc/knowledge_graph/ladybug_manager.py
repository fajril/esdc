"""LadybugDB graph database manager for ESDC.

Provides zero-copy integration with DuckDB via ATTACH,
graph schema creation, data loading, and Cypher query execution.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import real_ladybug as lb

from esdc.configs import Config

logger = logging.getLogger(__name__)


class LadybugDBManager:
    """Manage LadybugDB graph database lifecycle.

    Supports:
    - Zero-copy ATTACH to existing DuckDB database
    - Graph schema creation from YAML definitions
    - Cypher query execution
    - Graceful fallback when LadybugDB is unavailable
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        """Initialize the LadybugDB knowledge graph manager."""
        if db_path is None:
            kg_dir = Config.get_cache_dir() / "knowledge_graph"
            kg_dir.mkdir(parents=True, exist_ok=True)
            db_path = kg_dir / "esdc_kg.lbug"
        self._db_path = Path(db_path)
        self._db: lb.Database | None = None
        self._conn: lb.Connection | None = None
        self._initialized = False

    @property
    def is_available(self) -> bool:
        """Check if the knowledge graph is initialized and ready."""
        return self._initialized and self._conn is not None

    def initialize(self) -> bool:
        """Initialize or open the LadybugDB graph database.

        Returns True if initialization succeeded, False otherwise.
        """
        if self._initialized:
            return True

        try:
            if self._db_path.exists():
                logger.info("[KG] opening_existing | path=%s", self._db_path)
                self._db = lb.Database(str(self._db_path))
            else:
                logger.info("[KG] creating_new | path=%s", self._db_path)
                self._db = lb.Database(str(self._db_path))

            self._conn = lb.Connection(self._db)
            self._initialized = True
            logger.info("[KG] initialized_successfully")
            return True

        except Exception as e:
            logger.error("[KG] initialization_failed | error=%s", e)
            self._cleanup()
            return False

    def build_graph(self, duckdb_path: Path | str) -> bool:
        """Build graph schema and load data from DuckDB.

        Uses ATTACH for zero-copy access to DuckDB tables,
        then creates node/relationship tables and copies data.
        """
        if not self.is_available and not self.initialize():
            return False

        assert self._conn is not None

        try:
            duckdb_path = Path(duckdb_path)
            if not duckdb_path.exists():
                logger.error("[KG] duckdb_not_found | path=%s", duckdb_path)
                return False

            self._create_schema()
            self._attach_duckdb(duckdb_path)
            self._load_data()
            logger.info("[KG] graph_built_successfully")
            return True

        except Exception as e:
            logger.error("[KG] graph_build_failed | error=%s", e)
            return False

    def execute_cypher(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query and return results as list of dicts.

        Args:
            query: Cypher query string (may contain $param placeholders)
            params: Optional parameters for prepared statements

        Returns:
            List of result rows as dictionaries
        """
        if not self.is_available:
            logger.warning("[KG] not_initialized | cannot_execute_cypher")
            return []

        assert self._conn is not None

        try:
            if params:
                prepared = self._conn.prepare(query)  # type: ignore[union-attr]
                for key, value in params.items():
                    if isinstance(value, int):
                        prepared.bind(key, int(value))  # type: ignore[union-attr]
                    elif isinstance(value, float):
                        prepared.bind(key, float(value))  # type: ignore[union-attr]
                    else:
                        prepared.bind(key, str(value))  # type: ignore[union-attr]
                result = prepared.execute()  # type: ignore[union-attr]
            else:
                result = self._conn.execute(query)

            rows: list[dict[str, Any]] = []
            while result.has_next():  # type: ignore[union-attr]
                row = result.get_next()  # type: ignore[union-attr]
                row_dict: dict[str, Any] = {}
                for i in range(len(row)):
                    row_dict[str(i)] = row[i]  # type: ignore[index]
                rows.append(row_dict)

            logger.debug("[KG] cypher_executed | rows=%d", len(rows))
            return rows

        except Exception as e:
            logger.error("[KG] cypher_error | query=%s error=%s", query[:80], e)
            return []

    def get_schema_info(self) -> dict[str, Any]:
        """Return information about the current graph schema."""
        if not self.is_available:
            return {"status": "not_initialized"}

        assert self._conn is not None

        try:
            tables_result = self._conn.execute("SHOW TABLES")
            tables: list[str] = []
            while tables_result.has_next():  # type: ignore[union-attr]
                row = tables_result.get_next()  # type: ignore[union-attr]
                tables.append(str(row[0]))  # type: ignore[index]
            return {
                "status": "ok",
                "tables": tables,
                "db_path": str(self._db_path),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _create_schema(self) -> None:
        """Create node and relationship tables from schema definitions."""
        assert self._conn is not None

        schema_statements = self._get_schema_statements()
        for stmt in schema_statements:
            try:
                self._conn.execute(stmt)
                logger.debug("[KG] schema_created | stmt=%s", stmt[:60])
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.debug("[KG] table_exists | skipping")
                else:
                    logger.warning("[KG] schema_error | stmt=%s error=%s", stmt[:60], e)

    def _attach_duckdb(self, duckdb_path: Path) -> None:
        """ATTACH DuckDB database for zero-copy access."""
        assert self._conn is not None

        try:
            # Install and load DuckDB extension for ATTACH support
            self._conn.execute("install duckdb")
            self._conn.execute("load extension duckdb")
            logger.info("[KG] duckdb_extension_loaded")
            self._conn.execute(f"ATTACH '{duckdb_path}' AS esdc (dbtype duckdb)")
            logger.info("[KG] duckdb_attached | path=%s", duckdb_path)
        except Exception as e:
            if "already attached" in str(e).lower():
                logger.debug("[KG] duckdb_already_attached")
            else:
                logger.error("[KG] duckdb_attach_failed | error=%s", e)
                raise

    def _load_data(self) -> None:
        """Load data from DuckDB tables into graph node/relationship tables."""
        assert self._conn is not None

        data_loading = self._get_data_loading_statements()
        for stmt in data_loading:
            try:
                self._conn.execute(stmt)
                logger.debug("[KG] data_loaded | stmt=%s", stmt[:80])
            except Exception as e:
                logger.warning("[KG] data_load_error | stmt=%s error=%s", stmt[:80], e)

    def _get_schema_statements(self) -> list[str]:
        """Return CREATE NODE/REL TABLE statements for graph schema."""
        return [
            "CREATE NODE TABLE IF NOT EXISTS Field ("
            "field_id STRING PRIMARY KEY, field_name STRING, "
            "field_lat DOUBLE, field_long DOUBLE, field_area DOUBLE"
            ")",
            "CREATE NODE TABLE IF NOT EXISTS WorkingArea ("
            "wk_id STRING PRIMARY KEY, wk_name STRING, "
            "wk_lat DOUBLE, wk_long DOUBLE, wk_area DOUBLE"
            ")",
            "CREATE NODE TABLE IF NOT EXISTS Project ("
            "project_id STRING PRIMARY KEY, project_name STRING, "
            "field_id STRING, wk_id STRING"
            ")",
            "CREATE NODE TABLE IF NOT EXISTS Operator ("
            "operator_name STRING PRIMARY KEY, operator_group STRING"
            ")",
            "CREATE NODE TABLE IF NOT EXISTS Province (province STRING PRIMARY KEY)",
            "CREATE NODE TABLE IF NOT EXISTS Basin (basin128 STRING PRIMARY KEY)",
            "CREATE NODE TABLE IF NOT EXISTS Report ("
            "report_id STRING PRIMARY KEY, "
            "report_year INT64, report_status STRING, "
            "project_class STRING, project_stage STRING, "
            "uncert_level STRING, onstream_year INT64, "
            "project_remarks STRING, vol_remarks STRING"
            ")",
            "CREATE REL TABLE IF NOT EXISTS PROJECT_BELONGS_TO_FIELD ("
            "FROM Project TO Field"
            ")",
            "CREATE REL TABLE IF NOT EXISTS PROJECT_IN_WORKING_AREA ("
            "FROM Project TO WorkingArea"
            ")",
            "CREATE REL TABLE IF NOT EXISTS REPORT_BELONGS_TO_PROJECT ("
            "FROM Report TO Project"
            ")",
            "CREATE REL TABLE IF NOT EXISTS OPERATED_BY (FROM Project TO Operator)",
            "CREATE REL TABLE IF NOT EXISTS FIELD_IN_WORKING_AREA ("
            "FROM Field TO WorkingArea"
            ")",
            "CREATE REL TABLE IF NOT EXISTS FIELD_IN_BASIN (FROM Field TO Basin)",
            "CREATE REL TABLE IF NOT EXISTS FIELD_IN_PROVINCE (FROM Field TO Province)",
            "CREATE REL TABLE IF NOT EXISTS LOCATED_NEAR ("
            "FROM Field TO Field, distance_km DOUBLE"
            ")",
        ]

    def _get_data_loading_statements(self) -> list[str]:
        """Return COPY FROM statements to load data from attached DuckDB.

        Uses LadybugDB LOAD FROM syntax:
        COPY table FROM (LOAD FROM alias.table RETURN cols)
        """
        return [
            "COPY Field FROM ("
            "LOAD FROM esdc.project_resources "
            "RETURN DISTINCT field_id, field_name, "
            "field_lat AS field_lat, field_long AS field_long, "
            "0.0 AS field_area"
            ")",
            "COPY WorkingArea FROM ("
            "LOAD FROM esdc.project_resources "
            "RETURN DISTINCT wk_id, wk_name, "
            "0.0 AS wk_lat, 0.0 AS wk_long, 0.0 AS wk_area"
            ")",
            "COPY Project FROM ("
            "LOAD FROM esdc.project_resources "
            "RETURN DISTINCT project_id, project_name, field_id, wk_id"
            ")",
            "COPY Operator FROM ("
            "LOAD FROM esdc.project_resources "
            "RETURN DISTINCT operator_name, '' AS operator_group"
            ")",
            "COPY Province FROM ("
            "LOAD FROM esdc.project_resources RETURN DISTINCT province"
            ")",
            "COPY Basin FROM ("
            "LOAD FROM esdc.project_resources RETURN DISTINCT basin128"
            ")",
            "COPY Report FROM ("
            "LOAD FROM esdc.project_resources "
            "RETURN uuid AS report_id, report_year, report_status, "
            "project_class, project_stage, uncert_level, "
            "onstream_year AS onstream_year, "
            "project_remarks AS project_remarks, vol_remarks AS vol_remarks"
            ")",
            "COPY PROJECT_BELONGS_TO_FIELD FROM ("
            "LOAD FROM esdc.project_resources "
            "RETURN DISTINCT project_id, field_id"
            ")",
            "COPY PROJECT_IN_WORKING_AREA FROM ("
            "LOAD FROM esdc.project_resources "
            "RETURN DISTINCT project_id, wk_id"
            ")",
            "COPY REPORT_BELONGS_TO_PROJECT FROM ("
            "LOAD FROM esdc.project_resources "
            "RETURN DISTINCT uuid, project_id"
            ")",
            "COPY OPERATED_BY FROM ("
            "LOAD FROM esdc.project_resources "
            "RETURN DISTINCT project_id, operator_name"
            ")",
            "COPY FIELD_IN_WORKING_AREA FROM ("
            "LOAD FROM esdc.project_resources "
            "RETURN DISTINCT field_id, wk_id"
            ")",
            "COPY FIELD_IN_BASIN FROM ("
            "LOAD FROM esdc.project_resources "
            "RETURN DISTINCT field_id, basin128"
            ")",
            "COPY FIELD_IN_PROVINCE FROM ("
            "LOAD FROM esdc.project_resources "
            "RETURN DISTINCT field_id, province"
            ")",
        ]

    def _cleanup(self) -> None:
        """Clean up database connections."""
        import contextlib

        if self._conn is not None:
            with contextlib.suppress(Exception):
                self._conn.close()
            self._conn = None
        if self._db is not None:
            with contextlib.suppress(Exception):
                self._db.close()
            self._db = None
        self._initialized = False

    def close(self) -> None:
        """Close the database connections."""
        self._cleanup()
        logger.info("[KG] closed")
