# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2026-04-22

### Added

- **OpenWebUI v0.9.0 Compatibility Features**:
  - Source context metadata in `function_call_output` items — enables OpenWebUI inline citations for tool results
  - Incomplete status handling — emits `response.incomplete` with partial output on interrupted/timed-out responses
  - Reasoning content preservation in multi-turn input — preserves `reasoning_content` from previous assistant turns
  - Responses citation visibility — populates `annotations` in `output_text` content parts for source citations
  - Hybrid search (vector + BM25 with RRF) — combines semantic similarity with keyword scoring for improved recall
  - Richer tool result content types — JSON-serializes dict/list tool results instead of Python repr

### Changed

- `semantic_search` tool now uses hybrid search by default instead of pure vector similarity
- Error handlers in Responses API now emit `incomplete` status when partial output exists

## [0.5.4] - 2026-04-17

### Added

- **Domain Knowledge Corrections**:
  - Clarified R=GRR suffix in 1R/2R/3R uncertainty level descriptions and system prompt
  - Corrected LTP from "Letter to President" to "Long Term Plan" in schema_definitions
  - Added WAP (Waktu Acuan Pelaporan) concept, synonyms, and system prompt definition
  - Added document type concepts: POD (Plan of Development), POFD (Plan of Further Development), OPL (Optimasi Pengembangan Lapangan), OPLL (Optimasi Pengembangan Lapangan - Lapangan), POP (Put on Production), POD I (first POD in a working area), PSE (Penentuan Status Eksplorasi)
  - Enriched schema descriptions for `pod_name`, `is_pod_approved`, `is_pse_approved` columns
  - Added synonyms for POD, POFD, OPL, OPLL, POP, POD I, PSE, LTP, WAP

- **Domain Knowledge Architecture**:
  - Added `document_types` section to `DOMAIN_CONCEPTS` with `resolve_concept()` support
  - Added `report_terms` section to `DOMAIN_CONCEPTS` with `resolve_concept()` support
  - System prompt now includes Report Timing (WAP) section and POD/PSE/LTP in Indonesian Terms table

### Fixed

- **Test Suite Stability**: Full `pytest` suite now completes in ~6s (previously hung indefinitely)
  - Added autouse `_mock_provider_config` fixture to prevent accidental real LLM calls
  - Added `allow_provider_config` marker for tests that intentionally test provider config
  - Fixed unmocked `generate_streaming_response()` test that triggered real LLM calls
  - Moved module-level `TestClient` into pytest fixture in `test_routes_nonstreaming.py`
  - Extracted dead code from `test_generate_streaming_response_mock` into proper test
  - Updated `test_thinking_wrapped_with_tags` to match current behavior (thinking tags removed)
- **Linting (Ruff F/E/W)**: Resolved all 272 violations across production and test code
  - E501 (line too long): 231 fixed via formatting and string wrapping
  - F841 (unused variables): 12 fixed via removal or underscore-prefix
  - E712 (== True/False): 6 fixed via direct bool checks
  - E402 (import not at top): 8 fixed via `# noqa: E402` for circular import workarounds
  - F401, F405, F541, F601, W291, W293: all resolved
- **Linting (Ruff SIM/B/N)**: Resolved 14 production code violations
  - SIM102 (collapsible if): 3 fixed
  - SIM103 (return bool directly): 5 fixed
  - SIM108 (ternary): 1 fixed
  - B904 (raise...from): 1 fixed in `configs.py`
  - B905 (zip strict=): 1 fixed in `semantic_resolver.py`
  - N806 (uppercase vars): 3 marked `# noqa: N806` for deliberate constants
- **Type Checking (Basedpyright)**: Resolved all 19 type errors in test files
  - `test_context_manager.py`: Added full `AgentState` dicts (7 occurrences)
  - `test_query_classifier.py`: Added default empty string to `.get()` calls
  - `test_wizard.py`: Added `# type: ignore[assignment]` for mock overrides
  - Integration tests: Added None-guards for `fetchone()` results
  - `test_responses_api.py`, `test_spatial_wk_scoping.py`, `test_responses_input.py`: Type fixes

### Changed

- System prompt uncertainty levels table now explains suffix letters (P/R/C/U)

## [0.5.3] - 2025-04-14

### Added

- **Spatial Enhancement**: Added 4 new spatial query features
  - `find_nearest_from_coordinates()`: Find nearest fields/WK from arbitrary lat/long coordinates
    - Supports both "field" and "working_area" entity types
    - Returns distance in kilometers
  - `find_field_clusters()`: Cluster fields based on proximity
    - Simple distance-based clustering algorithm
    - Configurable max_distance_km and min_cluster_size
    - Returns cluster centers and unclustered fields
  - `find_adjacent_working_areas()`: Find WK adjacent to a target WK
    - Uses WK centroid coordinates (wk_lat, wk_long)
    - Returns adjacent WK with distances
  - `calculate_average_distance()`: Calculate average distance between multiple fields
    - Computes pairwise distances for all combinations
    - Returns statistics: average, min, max, pair count
  - Updated `resolve_spatial()` tool with new query types:
    - "nearest_from_coords": JSON with lat, long, entity_type, radius_km
    - "field_clusters": JSON with max_distance_km, min_cluster_size
    - "adjacent_wk": JSON with wk_name, max_distance_km
    - "average_distance": JSON with field_names array

### Technical Details

- Uses DuckDB Spatial Extension functions:
  - `ST_Point(lat, long)::POINT_2D` for point creation
  - `ST_Distance_Spheroid()` for accurate geodesic distance calculation
- Default radius: 20 km for all proximity queries
- All distances returned in kilometers
- Added comprehensive test suite (13 new tests)

## [0.5.2] - 2025-04-14

### Bug Fixes

- **Spatial Query**: Fixed proximity query distance calculation
  - Changed `ST_Distance()` to `ST_Distance_Spheroid()` for accurate geographic distance
  - Fixed coordinate order from `ST_Point(long, lat)` to `ST_Point(lat, long)::POINT_2D`
  - Resolves issue where `distance_km` always returned 0.0
  - Added 6 new tests for proximity distance calculation accuracy

## [0.5.1] - 2025-04-13

### Added

- **DuckDB Migration**: Migrate from SQLite to DuckDB for better analytical query performance
  - Automatic migration from existing SQLite databases on reload
  - Columnar storage for faster analytical queries
  - Improved query optimization and parallel execution

- **SQL Result Caching**: Implement diskcache for caching SQL query results
  - Automatic cache invalidation via `invalidate_sql_cache()` during `esdc reload`
  - Uses permanent storage (no TTL) since data only changes on reload
  - Cache location: `~/.esdc/cache/sql_results/`

- **FTS Optimization**: Add Full Text Search (FTS) rewrite for ILIKE queries
  - Automatic rewrite of `ILIKE '%keyword%'` to `match_bm25()` for indexed search
  - Preserves original ILIKE as fallback filter
  - Significant performance improvement for text search queries

- **Security Enhancements**: Add query classification and validation
  - SQL sanitization with parameterized queries
  - Query classification for optimal tool selection
  - Enhanced error handling for database operations

- **Semantic Search**: Full semantic search implementation for project_remarks
  - EmbeddingManager: Ollama-based embedding generation (default: qwen3-embedding:0.6b)
  - SemanticResolver: DuckDB VSS with HNSW index for fast similarity search
  - semantic_search tool: LLM-accessible tool for concept-based queries
  - Integrated with esdc reload: automatic embedding generation on data reload
  - Commands: esdc reload --no-embeddings, esdc reload --embeddings-only
  - Configurable via embedding_model in ~/.esdc/config.yaml
  - 37 tests for semantic search components

### Changed

- Database engine migrated from SQLite to DuckDB for improved analytical query performance
- Query execution now uses columnar storage for better performance
- Text search queries automatically optimized with FTS indexes

### Deprecated

- SQLite database support (automatically migrated to DuckDB on reload)

### Fixed

- **Performance**: 50-60% faster query execution through caching and FTS optimization

### Configuration

- Added `tool_format` config option to config.yaml
  - Supports "native" (default), "markdown", or "auto" values
  - Priority: Environment variable `ESDC_TOOL_FORMAT` > config.yaml > default
  - Controls tool output format (native OpenAI format vs markdown)

## [0.5.0] - 2025-04-06

### Added

- **IRIS Rebranding**: System renamed from "esdc-agent" to "IRIS" (Intelligent Reservoir Inference System)
  - Updated model identity, server responses, and UI branding
  - Model name hidden from users - shows only "IRIS"
  - Professional/formal tone in all responses

- **Intelligent Domain Knowledge System**:
  - Column selection rules: Combined columns (res_oc/res_an) by default
  - Specific columns when user mentions substance ("minyak"/"oil"/"gas")
  - Helper functions: `get_volume_columns()`, `detect_substance_from_query()`, `detect_volume_type_from_query()`, `should_use_risked_columns()`, `get_project_stage_filter()`
  - Report year fallback with automatic year detection: `detect_report_year_from_query()`, `get_available_report_year()`, `build_report_year_filter()`

- **Domain Definitions**:
  - GRR (Government of Indonesia Recoverable Resources) definition added to system prompt
  - ESDC (Elektronik Sumber Daya dan Cadangan) definition added to system prompt
  - Sales Potential definition (GRR - Reserves)
  - Clarified: GRR = Reserves + Sales Potential, NOT geological resources

### Fixed

- **Potensi Definition**: Corrected to mean ALL classified resources (rec_* columns), not just Prospective Resources
  - `rec_*` columns for all project_class values
  - `rec_*_risked` columns only for Prospective Resources
  - For Reserves & GRR and Contingent Resources: rec_* = rec_*_risked (same values)

- **Database Path Resolution**: Added absolute path resolution with `.resolve()` and file existence check
  - Helpful error messages for missing database file
  - FileNotFoundError with guidance on running 'esdc fetch'

- **KeyError in System Prompt**: Escaped curly braces in system prompt to prevent format string errors

### Changed

- **Version**: Bumped to 0.5.0 in pyproject.toml
- **Documentation**: Updated README.md for IRIS branding

### Tests

- Added 52 tests for volume column detection and selection
- Added 27 tests for report year fallback functionality
- Total: 199 tests passing (172 existing + 79 new)

## [0.4.1] - 2025-04-03

### Added

- **Domain Knowledge Functions**: Added missing `get_volume_columns()`, `build_aggregate_query()`, and `get_recommended_table()` functions to support resource aggregation queries
- **Type Stubs**: Added `types-requests` and `types-tabulate` to dev dependencies for better type checking
- **Changelog**: Added this CHANGELOG.md file to track version history

### Fixed

- **Type Checking**: Resolved all basedpyright errors in standard mode
  - Fixed provider method signatures to include `**kwargs: Any` for compatibility with base class
  - Fixed DataFrame type compatibility with `tabulate()` by converting to dict format
  - Fixed unbound variable warnings in `get_timeseries_columns()`
- **Linting**: Fixed critical ruff errors (E, F, W, I)
  - Removed wildcard imports from `__init__.py`
  - Fixed whitespace and line length issues
  - Organized import blocks
- **Streaming**: Fixed content ordering in streaming responses - content now emits BEFORE tool_calls
- **Tests**: Fixed failing tests in `test_domain_knowledge.py`
  - Updated expected aggregation levels from 4 to 8 (includes timeseries tables)
  - All 432 tests now passing

### Changed

- **Ruff Configuration**: Updated line-length from 100 to 88 (Black standard)
- **PyProject**: Migrated deprecated ruff top-level settings to `[tool.ruff.lint]` section

## [0.4.0] - 2025-03-15

### Added

- **Web Server**: OpenAI-compatible API server for multi-user support
  - FastAPI-based server with SSE streaming
  - Native tool calling support
  - ESDC_TOOL_FORMAT configuration option
  - Interleaved thinking with tool calls
  - Smart buffering for streaming responses
- **Server Features**:
  - Buffered streaming with SmartBuffer
  - Markdown formatting with thinking and tool sections
  - ThinkingState for interleaved thinking preservation
  - Preserved thinking support in StreamingBuffer
  - Native thinking support with OpenWebUI compatible tags
- **Tests**: Comprehensive server API integration tests

## [0.3.1] - 2024-12-15

### Fixed

- Include SQL files in package data for proper distribution
- Updated package metadata and dependencies

## [0.3.0] - 2024-12-01

### Added

- **Chat TUI**: Interactive chat interface with agentic text-to-SQL
  - Multi-line chat input with auto-expansion
  - Chat panel with results panel and split view
  - Message handling and streaming responses
  - Collapsible panels for SQL and results
  - Dynamic panel mounting during query processing
  - Context panel with session information
  - Thinking steps visualization
  - Orange styling for UI components
- **Domain Knowledge**: Comprehensive domain knowledge system
  - Split domain_knowledge into modular structure
  - Table hierarchies and column definitions
  - Terminology mappings for Indonesian oil & gas data
  - Timeseries support integration
  - Project timeseries detail support with smart table selection
  - Auto-create timeseries views on fetch
  - Extended table mappings with prefer_aggregation parameter
- **Text-to-SQL Engine**: Natural language to SQL conversion
  - Schema loader for context-aware queries
  - Multi-provider configuration management
- **Provider System**: Support for multiple LLM providers
  - Provider base class and configuration
  - Ollama provider implementation
  - OpenAI provider implementation
  - OpenAI-Compatible provider support

### Changed

- Moved dev dependencies to [dependency-groups] in pyproject.toml
- Upgraded to Python 3.10+ typing syntax consistently
- Added basedpyright configuration to pyproject.toml
- Added pandas-stubs for type checking

## [0.2.0] - 2024-10-20

### Added

- **Provider Management**: CLI commands for managing LLM providers
- **Chat Command**: Interactive chat interface foundation
- **Configuration System**: Multi-provider config with OAuth support
  - OAuth flow implementation
  - Config file management
- **KSMI ID Generator**: Utility for generating KSMI-compliant IDs
- **CLI Improvements**: Enhanced command-line interface

### Fixed

- Resolved type errors in chat input handling
- Fixed forecast types in DOMAIN_CONCEPTS to prevent model hallucination
- Removed duplicate COLUMN_METADATA keys
- Fixed SQL column mappings based on latest API changes

## [0.1.0] - 2024-09-10

### Added

- **CLI Framework**: Complete CLI application with Typer
  - Show command with filtering and export to Excel
  - Fetch command with progress bar and incremental updates
  - Database info command
  - Where clause support for queries
- **Database Layer**: SQLite database operations
  - Database manager for CRUD operations
  - SQL query files for standard operations
  - Timeseries views and documentation
- **Validation Engine**: Rule-based validation system (RE001-RE0022)
  - Validation rules for data integrity
  - Rule engine with proper error handling
- **ESDC Downloader**: Base API integration
  - Authentication handling
  - Data fetching with progress tracking
  - JSON parsing and data transformation
- **Configuration**: Environment-based configuration
  - Support for .env files
  - Config file switching to YAML format
  - User prompt for credentials

### Changed

- Switched from Click to Typer for CLI framework
- Migrated from dotenv to config.yaml
- Removed platformdirs dependency
- Cleaned up documentation structure

## [0.0.1] - 2024-08-15

### Added

- **Initial Release**: Base ESDC module functionality
  - ESDC downloader with API integration
  - Basic show and fetch commands
  - SQLite database storage
  - Progress bar visualization
  - Database location in user directory
  - Basic error handling and logging
  - Initial project structure and documentation

