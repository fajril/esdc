# IRIS - Intelligent Reservoir Inference System

**IRIS** (Intelligent Reservoir Inference System) is an AI-powered data analyst for Indonesian oil & gas reserves and resources, built on top of ESDC (Elektronik Sumber Daya dan Cadangan) data from [SKK Migas](https://esdc.skkmigas.go.id).

## Features

### Natural Language Interface
Ask questions about Indonesian oil & gas data in English or Bahasa Indonesia.

- **Intelligent Column Selection**: Automatically uses combined columns (res_oc/res_an) unless user specifies a substance
- **Volume Type Detection**: Distinguishes between reserves (cadangan), resources (sumberdaya), and prospective resources (potensi)
- **Report Year Fallback**: Automatically finds most recent data if requested year unavailable
- **Real-time Streaming**: AI responses stream in real-time
- **Context Panel**: Shows session info, token usage, and tool status

### Supported Query Types

| Query Type | Example |
|------------|---------|
| Field-level reserves | "Berapa cadangan lapangan Duri?" |
| Work area totals | "Potensi wilayah kerja Rokan?" |
| National statistics | "Total cadangan minyak Indonesia?" |
| Specific substance | "Cadangan minyak lapangan Duri?" |
| Year-specific | "Cadangan lapangan Duri tahun 2024?" |
| Prospective resources | "Potensi eksplorasi lapangan Duri?" |

### Data Management CLI
Fetch and manage ESDC data from the command line.

- **Fetch Data**: Download from ESDC API (CSV, JSON, ZIP formats)
- **Load Data**: Import into SQLite database
- **Query Data**: Display with filters

## Installation

```bash
# Clone repository
git clone https://github.com/fajril/esdc.git
cd esdc

# Install with uv
uv sync

# Or install editable
uv pip install -e .
```

## Quick Start

### Chat Interface

```bash
# Start interactive chat
esdc chat

# Example queries:
# "Berapa cadangan lapangan Duri?"
# "Top 5 lapangan minyak terbesar di Indonesia?"
# "Potensi eksplorasi wilayah kerja Rokan tahun 2024?"
# "Cadangan gas di lapangan Duri?"
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

### Provider Configuration

Configure your AI provider in `~/.esdc/config.yaml`:

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

## Domain Knowledge

### Volume Types and Columns

| Query Term | Volume Type | Column Prefix | Risked? |
|------------|-------------|---------------|---------|
| cadangan | Reserves | res_* | No |
| sumberdaya/grr | Resources | rec_* | No |
| potensi | All classified | rec_* | Varies |
| potensi eksplorasi | Prospective | rec_*_risked | Yes |
| potensi contingent | Contingent | rec_* | No |

### Column Naming

- **Combined columns** (default): `res_oc`, `res_an`, `rec_oc`, `rec_an`
- **Specific substance**: `res_oil`, `res_con` (minyak) or `res_ga`, `res_gn` (gas)

### Report Year Fallback

IRIS automatically handles missing data years:
- If year 2024 requested but unavailable → falls back to 2023, 2022, etc.
- SQL uses `MAX(report_year) WHERE report_year <= {requested_year}`

## Architecture

```
esdc/
├── chat/                    # Chat interface
│   ├── app.py              # Textual TUI app
│   ├── agent.py            # LangGraph agent
│   ├── tools.py            # SQL execution tools
│   ├── prompts.py          # System prompt
│   ├── domain_knowledge/   # Volume/column logic
│   │   ├── functions.py    # Helper functions
│   │   ├── synonyms.py     # Indonesian/English terms
│   │   └── tables.py       # Entity→Table mapping
│   └── schema_loader.py    # Database schema
├── commands/                # CLI commands
├── configs.py               # Configuration
└── db/                      # Database operations
```

## Documentation

- **Knowledge Base**: `docs/reference/` - Architecture and conventions
- **Database Schema**: `docs/reference/schema/esdc-database-schema.md`
- **For AI Agents**: See `AGENTS.md`

## CLI Commands

### `chat`
Start interactive chat with IRIS.

```bash
esdc chat
```

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

## Version History

### v0.5.0 (Current)
- **IRIS rebranding**: Model renamed from "esdc-agent" to "iris"
- **Intelligent column selection**: Combined columns by default, specific when user mentions substance
- **Correct "potensi" handling**: All classified resources, with risked columns for prospective only
- **Report year fallback**: Automatic fallback to most recent available year
- **Anti-reveal instructions**: IRIS maintains identity without revealing underlying model

### v0.4.0
- Initial column selection improvement
- Domain knowledge helpers

## Development

```bash
# Run tests
pytest tests/

# Run chat TUI
esdc chat

# Format code
ruff check --fix esdc/
ruff format esdc/

# Type check
basedpyright esdc/
```

## License

Apache Software License. See `LICENSE` file.

## Contact

fambia at skkmigas.go.id or fajril at ambia.id