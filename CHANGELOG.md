# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

- Initial release with CLI interface for ESDC database
- Support for fetching and querying oil & gas reserves data
- SQLite database storage
- Chat agent with LangChain/LangGraph integration
- Web server with FastAPI and SSE streaming
- Provider support: Ollama, OpenAI, OpenAI-Compatible
