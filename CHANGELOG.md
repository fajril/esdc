# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Configuration**: Added `tool_format` config option to config.yaml
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

