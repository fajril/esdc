# ESDC (Elektronik Sumber Daya dan Cadangan)

A data management system for Indonesian oil & gas reserves data from ESDC (https://esdc.skkmigas.go.id).

## Features

### Chat TUI
Interactive terminal-based chat interface to query oil & gas reserves data using natural language.

- **Natural Language Queries**: Ask questions about reserves, production, and fields
- **Real-time Streaming**: AI responses stream in real-time
- **Context Panel**: Shows current model, provider, token usage, and conversation thread
- **Collapsible Sections**: Toggle session info, context usage, and tool status
- **Provider Support**: Works with Ollama models (kimi-k2.5:cloud, etc.)

### Data Management CLI
Command-line interface for fetching and managing ESDC data.

- **Fetch Data**: Download data from ESDC API (CSV, JSON, ZIP formats)
- **Load Data**: Import data into SQLite database
- **Query Data**: Display and filter data with various options

## Installation

```bash
# Clone repository
git clone https://github.com/fajril/esdc.git
cd esdc

# Install with uv
uv add -e .
```

## Quick Start

### Chat TUI

```bash
# Start interactive chat
esdc chat

# Example queries:
# "What are the top 10 oil reserves in 2023?"
# "Show all fields in North Sumatra basin"
# "Compare gas reserves between 2020 and 2023"
```

### Data CLI

```bash
# Fetch latest data
esdc fetch --filetype csv --save

# Reload into database
esdc reload --filetype csv

# Show data with filters
esdc show project_resources --year 2023 --save
```

## Configuration

Configuration is stored in `~/.esdc/`:

```
~/.esdc/
├── config.yaml    # Provider and model settings
└── esdc.db        # SQLite database
```

### Chat TUI Configuration

The chat TUI uses the default provider from `config.yaml`:

```yaml
default_provider: ollama
providers:
  ollama:
    model: kimi-k2.5:cloud
    base_url: http://localhost:11434
```

### Environment Variables

Set credentials for data fetching:

```bash
export ESDC_USER="your_username"
export ESDC_PASS="your_password"
```

| Variable | Purpose |
|----------|---------|
| `ESDC_USER` | ESDC API username |
| `ESDC_PASS` | ESDC API password |
| `ESDC_URL` | API URL (default: https://esdc.skkmigas.go.id/) |
| `ESDC_DB_FILE` | Database file path |

## Architecture

```
esdc/
├── chat/                    # Chat TUI application
│   ├── app.py              # Textual TUI app
│   ├── agent.py            # LangGraph agent with tools
│   ├── tools.py            # SQL execution & schema tools
│   ├── prompts.py          # System prompt template
│   └── schema_loader.py    # Database schema loader
├── commands/                # CLI commands
├── configs.py               # Configuration management
└── db/                      # Database operations
```

## Documentation

- **Knowledge Base**: `docs/reference/` - Architecture and conventions
- **Database Schema**: `docs/reference/schema/esdc-database-schema.md`
- **Active Work**: `docs/active/` - Current development
- **Completed Work**: `docs/completed/` - Archived plans
- **For AI Agents**: See `AGENTS.md`

## CLI Commands

### `fetch`
Download data from ESDC API.

```bash
esdc fetch --filetype csv --save
```

Options:
- `--filetype`: Output format (csv, json, zip). Default: json
- `--save`: Save fetched data to file

### `reload`
Import existing data files into database.

```bash
esdc reload --filetype csv
```

### `show`
Display data with filters.

```bash
esdc show project_resources --year 2023 --columns project_name res_oil
```

Arguments:
- `table`: Table name to query
- `--where`: Column to filter
- `--search`: Search keyword
- `--year`: Filter by year
- `--output`: Detail level (0=summary, 1=detail)
- `--save`: Save output to file
- `--columns`: Columns to display

## Tech Stack

- **Python 3.11+**
- **Textual** - Terminal UI framework
- **LangChain & LangGraph** - AI agent framework
- **SQLite** - Local database
- **Typer** - CLI framework

## Development

```bash
# Run tests
pytest tests/

# Run chat TUI
esdc chat

# Format code
ruff check --fix esdc/
```

## License

Apache Software License. See `LICENSE` file.

## Contact

fambia@skkmigas.go.id or fajril@ambia.id
