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
- **Load Data**: Import into DuckDB database
- **Query Data**: Display with filters

## Installation

```bash
# Clone repository
git clone https://github.com/fajril/esdc.git
cd esdc

# Install with uv (core only — lighter, no observability)
uv sync

# Or install editable
uv pip install -e .

# With Phoenix observability and evaluation
uv pip install -e ".[phoenix]"

# Dev environment (includes everything + test tools)
uv sync --dev
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
└── esdc.duckdb        # DuckDB database
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

DeepSeek can be configured as a first-class provider:

```yaml
default_provider: deepseek
provider_order:
  - deepseek
  - openai
providers:
  deepseek:
    provider_type: deepseek
    api_key: sk-...
    model: deepseek-v4-flash
    reasoning_effort: high
  openai:
    provider_type: openai
    api_key: sk-...
    model: gpt-4o-mini
```

Use `reasoning_effort: none` for non-thinking mode, or `high` / `max`
for DeepSeek thinking mode. `provider_order` controls failover priority; ESDC
uses `default_provider` first, then tries the remaining providers in order if
the current provider fails.

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
| `PHOENIX_ENABLED` | Enable Phoenix tracing (true/false) |
| `PHOENIX_COLLECTOR_ENDPOINT` | Phoenix OTLP endpoint |
| `PHOENIX_PROJECT_NAME` | Phoenix project name |

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
├── server/                  # OpenAI-compatible API
│   ├── app.py              # FastAPI application
│   └── routes.py           # API endpoints
├── phoenix/                 # Observability (optional)
│   ├── phoenix_tracing.py  # OpenTelemetry setup
│   ├── phoenix_evals.py    # LLM evaluation
│   └── phoenix_config.py   # Phoenix configuration
├── validate/                # Data validation rules
│   ├── rule_re0.py         # Volumetric rules
│   ├── rule_re1.py         # Reserves rules
│   ├── rule_re2.py         # Resources rules
│   └── rule_re5.py         # Transition rules
├── esdc.py                  # CLI entry point
├── configs.py               # Configuration
└── dbmanager.py             # Database operations
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

### `status`
Show database status and index integrity.

```bash
esdc status --verify
```

### `validate`
Validate data against business rules (RE0-RE5).

```bash
# Default: summary per group
esdc validate

# Per-rule detail
esdc validate -v

# Full violation detail
esdc validate -vv

# Specific year
esdc validate --year 2024
```

### `configs`
Interactive configuration wizard.

```bash
esdc configs
```

### `serve`
Run OpenAI-compatible API server.

```bash
esdc serve --host 0.0.0.0 --port 3334
```

## Tech Stack

- **Python 3.10+**
- **Textual** - Terminal UI framework
- **LangChain & LangGraph** - AI agent framework
- **DuckDB** - Analytics database (local)
- **FastAPI + Uvicorn** - OpenAI-compatible API server
- **Typer** - CLI framework
- **Arize Phoenix** (optional) - Observability & evaluation
- **OpenTelemetry** (optional) - Distributed tracing

## Version History

### v0.7.0 (Current)
- **Phoenix/OTel dependencies optional**: Install with `pip install esdc[phoenix]` — core install ~19 packages lighter
- **RE0 volumetric validation**: 66 field-level rules for reserves vs production, cumulative vs sales, year-over-year comparison
- **Verbosity levels for `validate`**: 0=group summary, 1=per-rule detail, 2=full violation values
- **GCF exact comparisons**: Replaced tolerance with exact value comparisons for GCF rules
- **FTS auto-reindex**: Full-text search indexes rebuilt automatically after data fetch
- **Multi-provider support**: Anthropic Claude, Google Gemini, Azure OpenAI, Groq, Ollama (local + cloud)
- **OpenAI-compatible API server**: `esdc serve` with streaming and tool calling

### v0.6.0
- **Arize Phoenix observability**: Tracing and evaluation integration
- **Hybrid external tool passthrough**: External tools routed transparently through LangGraph
- **OpenWebUI v0.9.0 compatibility**: Streaming, reasoning, citations, source metadata
- **New providers**: Claude, Gemini, Azure, Groq, Ollama Cloud

### v0.5.0
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
