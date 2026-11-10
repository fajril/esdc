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

#### Fetch and publish

`esdc fetch` prepares and validates the complete response set before changing
the databases. Resource and timeseries rows are committed to their canonical
SQLite source snapshots first. DuckDB then publishes both analytical
projections and their exact revisions, row counts, and source hashes together
in one transaction. If DuckDB publication fails after SQLite commits, the new
SQLite revisions remain available for recovery and are marked stale; the
previous analytical resource/timeseries pair and lineage remain intact. Cache
clearing, checkpointing, and index rebuilding run after publication. A
maintenance failure is reported as a failed command even though the committed
data is retained.

For selected resource or timeseries years, a valid nonempty response replaces
the rows for each selected year; an empty or nonmatching slice preserves that
year's existing rows. An entirely empty full resource response is a no-op.
Selective updates require a published, matching
SQLite/DuckDB baseline. A first unfiltered fetch publishes a new baseline
rather than importing legacy DuckDB timeseries implicitly. If a legacy
`project_timeseries` table must be preserved, inspect it first with the
default-dry-run command, then explicitly apply the bootstrap:

```bash
esdc bootstrap-timeseries-source
esdc bootstrap-timeseries-source --apply
esdc fetch
```

The first preserving full fetch after a verified bootstrap establishes the
current API baseline. `--no-reload` validates and saves downloads without
database, cache, or index mutation. `--no-reindex` skips the extra final
reindex pass; indexes required during publication are still maintained. Use
`esdc fetch --help` for the supported year, file type, and reload options.

`esdc rebuild-api-analytics` repairs the resource and timeseries analytical
projections together, offline, from canonical SQLite snapshots using installed
extensions only. It
requires a verified baseline and refuses incomplete or inconsistent source
state. The older `esdc rebuild-project-resources` command repairs only the
resource projection and refuses when timeseries state is pending or
inconsistent. Neither command rebuilds unrelated database tables.

The analytical `project_resources` projection has no geometry columns. Optional
legacy `field_geom` WKT remains unchanged in canonical SQLite for history,
replay, and source-hash compatibility; it is validated separately for the
`invalid_geometry_count` metadata. Use Spatial Resolver and published spatial
catalogs for geometry when available. Rebuild analytics to replace an existing
projection with the current schema.

For an offline closed-clone rehearsal, prepare a baseline directory containing
`esdc.sqlite` and `esdc.duckdb`, plus a complete fixture manifest, then run:

```bash
esdc verify-analytics-migration \
  --baseline-dir /path/to/closed-clone \
  --fixture-manifest /path/to/fixture-manifest.json \
  --output-dir /path/to/new-rehearsal-output
```

This writes a machine-readable `report.json` under the new output directory;
it does not perform a live cutover. The legacy
`scripts/verify_api_resource_migration.py` resource rehearsal remains
available for existing integrations. A passing small-fixture rehearsal does
not establish representative full-size rehearsal, real database migration,
live API behavior, or production readiness.

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

### Corpus embedding and rerank (HTTP service)

Embedding and reranking run on an HTTP service: Ollama, omlx, llama-server,
LM Studio, vLLM, or TEI. localhost is allowed, so a service on the same machine
is fine. ESDC does not download a model, start a daemon, or carry a native
inference library; storage, vector search, BM25 and RRF fusion stay in ESDC.

That means semantic search needs a reachable service. A cold model load can add
seconds to a query, and `esdc corpus warmup` is a service check rather than a
download: it asks each role's endpoint with a synthetic input and reports what
answered. Credentials and document text are never printed.

### Embeddings

Embeddings are assigned to a connection and model like every other task. The
connection type decides the transport: an Ollama connection uses Ollama's native
embedding API, and an `openai_compatible` connection uses any
`/v1/embeddings` server (LM Studio, llama-server, vLLM, TEI). The wire model id
is explicit; there is no implicit localhost service or pinned UI tag.

Configure it under `models.embedding` in `config.yaml`, or choose
`Models by task > Embeddings` in `esdc configs`:

```yaml
connections:
  embed-server:
    type: openai_compatible
    base_url: http://localhost:8889/v1
    api_key: ""
models:
  embedding:
    mode: model
    connection: embed-server
    model: Qwen3-Embedding-0.6B-8bit
```

Each vector space stores a probe vector. If a service produces vectors that
disagree with the stored ones by more than a small tolerance — a different
quantization, pooling mode, or model — writes are rejected and searches
report that embeddings need rebuilding. Measured cosine between services
running the same weights (Ollama `qwen3-embedding:0.6b` and an MLX 8-bit build)
is ≥ 0.9986, so the probe, not the model name, is what proves the space.

Per-run override: `--embed-backend ollama|openai` on
`esdc corpus commit`, `esdc corpus reembed` and `esdc reload`. The override must
match the assigned connection type; it never switches destination silently.

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

### Connections and models by task

One connection registry owns transport and authentication. Six task assignments
own their own model: Chat, Embeddings, Reranking, OCR, Metadata extraction and
Text cleanup. A connection does not own a hidden chat model.

```yaml
model_settings_version: 1
connections:
  gateway:
    type: openai_compatible
    base_url: https://gateway.example/v1
    api_key: sk-...
  desktop:
    type: ollama
    base_url: http://localhost:11434
models:
  chat:
    mode: model
    connection: gateway
    model: gpt-4o-mini
    options:
      reasoning_effort: low
    fallbacks:
      - connection: desktop
        model: llama3.2
  embedding:
    mode: model
    connection: desktop
    model: qwen3-embedding:0.6b
  rerank:
    mode: unset
    enabled: false
  ocr:
    mode: model
    connection: desktop
    model: glm-ocr
  metadata:
    mode: chat
  cleanup:
    mode: chat
```

Task modes are `model`, `chat`, `ocr`, `off` and `unset`:

| Task | Permitted modes / controls |
|---|---|
| Chat | explicit connection/model, ordered fallbacks, per-target reasoning options |
| Embeddings | explicit connection/model; changing vector space may require rebuilding embeddings |
| Reranking | explicit connection/model and a separate Enabled switch |
| OCR | explicit connection/model; positively non-vision models are refused |
| Metadata extraction | explicit model, `chat`, or `ocr` (use the OCR result) |
| Text cleanup | explicit model, `chat`, or `off` |

`unset` means not configured; it is never an implicit localhost service. Fresh
settings have no connections, Chat/Embeddings/OCR unset, Reranking unset and
disabled, and Metadata/Cleanup following Chat.

`esdc configs` provides one searchable `Connection / Model` list for every task.
It includes discovered models, the current model when listing cannot verify it,
`Enter model ID…`, `Add connection…`, `Refresh models` and `Back`. A connection
added from a task picker stays in that draft until Save; Cancel discards it.
Chat fallbacks are managed inside the Chat task. Connections themselves never
store a model or reasoning effort.

Reranking keeps choosing/testing separate from enabling: a successful test never
turns the task on. OCR uses the same image request as runtime and refuses only a
model positively reported as non-vision; unknown capability remains testable.

### Converting existing model settings

Older files with `providers`, `default_provider`, `provider_order`, direct
embedding fields or `corpus.*_model/provider` keys are not interpreted at
runtime. Convert them once, explicitly:

```bash
# Masked preview; no writes, no network.
esdc configs convert-model-settings --input ~/.esdc/config.yaml

# Apply after reviewing the preview; the backup must not exist yet.
esdc configs convert-model-settings \
  --input ~/.esdc/config.yaml \
  --apply \
  --backup ~/.esdc/config.pre-unified.yaml
```

The apply creates a private complete backup, rechecks the source, and replaces
it atomically. An already-canonical file is a no-op. The interactive wizard also
shows `Update model settings…` while the saved file lacks the canonical marker;
Cancel writes nothing.

Non-interactive Chat changes use connection/model pairs:

```bash
esdc configs --set-chat-primary gateway:gpt-4o-mini
esdc configs --set-chat-fallbacks desktop:llama3.2,backup:gpt-4o-mini
```

`--set-default-provider` and `--set-provider-order` are retired; they print the
replacement syntax and write nothing.

`esdc configs --reset-key models.chat` resets only that task assignment;
connections and other tasks are untouched. `--show` prints canonical nested
settings with API/OAuth secrets and URL credentials masked.

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

### POD data and portal

The canonical bundled contract is `esdc/schemas/pod.yaml`, matching dictionary 0.4.0. Five physical SQLite inputs (`m_project`, `m_pod`, `pod_project`, `pod_monitoring`, `pod_forecast`) feed two regenerated physical outputs (`pod_monitoring_api`, `pod_forecast_api`), plus three readonly SQL-seeded references (`r_institution`, `r_pod_type`, `r_pod_stage`). The last successful full API or manual load replaces facts; no parallel old schema remains. Strict schema, value, and relationship validation runs before replacement.

```text
esdc schema --schema-pod --output pod-template.xlsx
esdc load --from-excel pod-template.xlsx --schema-pod --dry-run
esdc load --from-excel pod-template.xlsx --schema-pod
esdc load --from-sqlite source.sqlite --dry-run
esdc load --from-sqlite source.sqlite
esdc pod-upgrade-economics --database path/to/esdc.sqlite
esdc pod-upgrade-economics --database path/to/esdc.sqlite --apply --backup path/to/pre-upgrade.sqlite
esdc fetch
esdc portal
```

A full XLSX import requires all five input sheets. `--pod-registry` replaces only the three master tables and preserves monitoring and forecast inputs. Old schema needs explicit validated full replacement; existing document links require reviewed `--pod-crosswalk` JSON, never guessed. Portal recalculates from local resources/timeseries; missing basis remains NULL with diagnostics. Download workbook produces a POD business backup with permanent identity reservations and read-only reference and result sheets. It excludes raw resources, timeseries, and corpus documents. Restoring it regenerates the outputs.

Registries created before the economics column rename advertise contract 1. Ordinary reads and writes refuse them with a recovery message; they never rename columns implicitly. Inspect with `esdc pod-upgrade-economics --database <path>`, then apply once with `--apply --backup <new-file>`: the backup is a validated SQLite copy of the pre-upgrade database, the upgrade is transactional and name-only, and the application must restart or reopen the database afterwards. Applying to an already-current registry is a no-op and creates no backup.

Corpus and POD changes invalidate derived knowledge. Run `esdc corpus learn` to rebuild the knowledge graph and `esdc corpus reembed --stale` to update stale document embeddings. Ordinary metadata saves do not call models. Source commit survives cache failure; portal retry only republishes committed source; `esdc rebuild-api-analytics` provides offline recovery. Writers use WAL/FK ON and validate patched SQLite runtime; GET does not initialize or migrate.

### `chat`
Start interactive chat with IRIS.

```bash
esdc chat
```

Requires a configured provider (`esdc configs`).

Layout: full-width chat column; right panel with conversation title, live tool
timeline, collapsible SQL/results, and query history; one-line status bar
showing version, model, thread, and context usage (turns red at 75%). Model
reasoning streams into a thinking indicator above the answer. Plots generated
via OpenTerminal are surfaced as image links.

Slash commands:
- `/new` — start a new conversation (fresh thread, cleared panels)
- `/help` — list commands

Keybindings:
| Key | Action |
|-----|--------|
| `ctrl+h` | Toggle right panel |
| `ctrl+l` | Toggle SQL section |
| `ctrl+r` | Toggle results section |
| `ctrl+e` | Toggle SQL + results together |
| `ctrl+o` | Open last image in browser |
| `ctrl+shift+s` | Save screenshot |
| `escape` | Cancel current query |

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

The home menu is Models by task, Connections, Application settings, Advanced,
View settings, and Exit. Back returns one level, Escape at the top level
exits, and Control+C ends the run without losing an earlier explicit save.
When the saved file still needs the one-time model-settings conversion, an
`Update model settings…` row appears before Exit.

Models by task holds the six tasks — Chat, Embeddings, Reranking, OCR,
Metadata extraction, Text cleanup — and each one opens a single searchable
`connection / model` list. It includes discovered models, the current model
marked `saved, unverified` when listing cannot confirm it, `Enter model ID…`,
`Add connection…`, `Refresh models`, and `Back`. Manual entry asks for an
eligible saved or draft connection and a model ID, then returns to the same
draft. Selecting a model never writes; Save is explicit.

Connections owns only transport and authentication: type, endpoint, API key,
API version and transport-specific fields. Every connection can have a custom
name, including multiple Ollama or OpenAI-compatible entries. It has no model,
reasoning, default-provider or failover control. Editing a shared endpoint or
key lists the tasks that use it before Save. Renaming updates every task
reference in one write. Removal is refused while a task or Chat fallback still
uses the connection, with those uses listed.

Chat owns its ordered fallbacks. Each fallback has its own connection, model and
options; changing the primary preserves distinct fallbacks. Reranking keeps
Enabled as its own switch: choosing or testing a model never turns it on.
Metadata extraction offers explicit model, `Use chat model`, or `Use OCR
result`. Text cleanup offers explicit model, `Use chat model`, or `Off`.
Embeddings warns that changing the vector space may require rebuilding
embeddings. OCR refuses a model only when metadata positively says it is not
vision-capable; an unknown capability remains selectable and testable.

A task draft has Save, Test, model options where relevant, and Cancel. Test
resolves the draft through the same resolver as runtime and calls the role's own
protocol — embedding request, rerank scores, image OCR, or text inference. It
never saves, enables reranking, rebuilds an index, refreshes OAuth, or changes
another task. An expired credential is reported for reauthentication. Secrets
are masked in the wizard and in `configs --show`, including URL userinfo/query
credentials.

Application settings and Advanced edit only non-model application, logging,
cache, tracing and corpus-processing controls. View settings prints the flat,
masked view. `--reset-key models.<role>` resets only that task assignment;
connections and other tasks are untouched.

Running ESDC services may need restart to read the saved config; the wizard
states this after a save.

### `serve`
Run OpenAI-compatible API server.

```bash
esdc serve --host 0.0.0.0 --port 3334
```

## Document Corpus

`esdc corpus` ingests official PDFs (POD approvals, MoM minutes, BA documents, etc.) into a searchable local corpus that IRIS can query in chat via the `search_documents` and `read_document` tools.

### Prerequisites

Configure an OCR connection and model in `esdc configs` → Models by task → OCR.
For a local Ollama connection, pull the model first:

```bash
ollama pull glm-ocr
```

`glm-ocr` (zai-org/GLM-OCR, 0.9B) OCRs scanned/image-only pages. Pages with a usable native text layer are extracted directly and never sent to the model.

### Configuration

Corpus processing controls live under `corpus.*` in `~/.esdc/config.yaml`,
merged over these defaults. Task models, connections and rerank enablement are
canonical `models.*` settings and are edited by `esdc configs`, not here.

| Key | Default | Description |
|-----|---------|-------------|
| `chunk_size` | `3000` | Max characters per chunk (~750 tokens) |
| `chunk_overlap` | `300` | Characters carried over between consecutive chunks |
| `rerank_pool` | `30` | Candidates scored when the Reranking task is enabled |
| `max_chunks_per_doc` | `2` | Max chunks of one document in search results (`0` = no cap) |
| `query_instruct` | `true` | Prefix queries with Qwen3-Embedding's instruction template |
| `ocr_dpi` | `200` | Page render resolution for OCR; raise to 300 if OCR quality is poor |
| `num_ctx` | `16384` | Ollama context window; `glm-ocr` fails on page images below this |
| `min_chars_per_page` | `50` | Text-layer character threshold below which a page counts as scanned |
| `min_image_area` | `0.05` | Embedded-image area below which images are ignored |
| `extract_max_tokens` | `8192` | Bound on corpus-knowledge extraction output tokens (`0` = no cap) |
| `extract_timeout_seconds` | `300` | Bound on corpus-knowledge extraction calls (`0` = no timeout) |

Embeddings, OCR, reranking, metadata and cleanup models live under `models.*`
as shown in Connections and models by task. A task change never enables
reranking or rebuilds an index; re-embed separately when the vector space
changes.

### doc_type vs doc_topic, and doc_level rules

`doc_type` is a document's **form** (uu, permen, letter, mom, ba, contract, book, ...); `doc_topic` is what business object it's **about** (pod, pofd, wpnb, psc, gsa, ...) — a POD approval book has `doc_type: book` and `doc_topic: [pod]`. `doc_level` gains a `regulation` value alongside `wk`/`field`/`project`/`unknown` for national/ministerial instruments not tied to any wk/field/project.

`doc_level` is partly automatic: a regulatory `doc_type` (uu, perpu, mk, pp, permen, kepmen, ptk, sop) always forces `doc_level: regulation` and clears any wk_name/field_name/project_name; certain `doc_topic` values (e.g. `pod`, `pofd`, `afe` -> `project`; `wpnb`, `psc` -> `wk`) set the level when every topic on the document implies the same one. These rules apply at `commit` and at `meta` (a stale sidecar self-heals — and warns — the moment `meta` touches it), and reject contradictory flags up front (e.g. `--level regulation` with an entity flag, or an explicit `--level` that conflicts with what `--doc-type`/`--topic` implies).

Legacy `doc_type` values `psc`/`gsa`/`pod` are remapped automatically (`psc`/`gsa` -> `doc_type: contract`, `pod` -> `doc_type: book`, seeding the corresponding `doc_topic`). **Already-committed documents keep their old values until recommitted with `commit --force`** — there's no automatic store migration.

### Workflow

Ingestion is two steps with a human review gate in between — nothing reaches the searchable corpus unreviewed:

1. **Extract** — parse sources (`.pdf`, `.docx`, `.md`) into reviewable `.corpus.md` sidecar files next to the source, with LLM-prefilled (unreviewed) metadata:
   ```bash
   esdc corpus extract path/to/document.pdf
   esdc corpus extract path/to/folder/          # batch: pdf + docx + md
   esdc corpus extract path/to/document.pdf --topic pofd   # set doc_topic (single value)
   ```
   Every page is wrapped in a `<!-- page N: native -->` or `<!-- page N: llm_ocr -->` marker.

2. **Review** — open the `.corpus.md` file in an editor, check the `llm_ocr` pages against the source PDF, correct the prefilled frontmatter (doc_type, doc_topic, dates, entities, etc.), then flip `reviewed: false` to `reviewed: true`. Sidecars still marked `reviewed: false` are skipped on commit. Metadata can also be bulk-edited across many sidecars at once instead of hand-editing each file:
   ```bash
   esdc corpus meta path/to/folder/ --wk-name "Rokan"   # bulk-set, persists to frontmatter
   esdc corpus meta path/to/folder/ --topic wpnb        # set doc_topic (single value)
   esdc corpus meta path/to/folder/                     # no flags: show current metadata
   esdc corpus meta path/to/folder/ --regenerate        # re-run LLM metadata analysis on the existing body
   ```
   `--regenerate` re-runs metadata extraction over each sidecar's already-extracted markdown (no re-parse/OCR) using the Metadata extraction task — useful after improving the prompt or model. It requires a reachable text Metadata extraction task, replaces the LLM-owned fields (doc_type, doc_topic, dates, entities, ...), resets `reviewed: false` unless `--reviewed`/`--no-reviewed` is also passed, and any explicit flag in the same invocation wins over the regenerated value.

3. **Commit** — ingest reviewed sidecars into the DuckDB-backed corpus (chunked, embedded, hybrid-indexed):
   ```bash
   esdc corpus commit path/to/document.corpus.md
   esdc corpus commit path/to/folder/            # batch
   ```

### Management

```bash
esdc corpus status path/to/folder/   # where each PDF/sidecar sits in extract -> review -> commit
esdc corpus meta path/to/folder/     # show sidecar metadata (incl. topic column); add flags to bulk-set
esdc corpus list                     # documents committed to the corpus
esdc corpus remove <doc_id>...       # remove document(s) (files on disk untouched)
esdc corpus clear --yes              # delete the entire corpus
esdc corpus reembed                  # rebuild embeddings after an embedding-model change
```

### Rename sources to a canonical structure

```bash
esdc corpus rename <file|folder>...            # preview (dry-run)
esdc corpus rename <file|folder>... --yes      # apply
esdc corpus rename report.pdf --doc-type letter --yes
```

Renames each source (and its `.corpus.md` sidecar, if present) to
`DOC_TYPE - YYYY.MM.DD - title.<ext>`. The three parts are resolved from an
existing sidecar, the committed corpus database, or — as a fallback —
LLM/OCR inference with the filename passed as a hint. `--doc-type` overrides
the detected type. Dry-run is the default; pass `--yes` to rename on disk.
Files whose date/type/title cannot be resolved are skipped, not renamed.

### Learn the knowledge graph

```bash
esdc corpus learn --dry-run          # report what would be processed
esdc corpus learn                    # process every new/changed document
esdc corpus learn --limit 3          # process at most N documents (smoke runs)
esdc corpus learn --force            # reprocess every document and rebuild all dossiers
esdc corpus learn --init-guideline   # draft ~/.esdc/guideline.yaml from the corpus, then exit
esdc corpus proposals                # list schema proposals discovered along the way
```

`esdc corpus learn` reconstructs a knowledge graph over the committed corpus,
eagerly, so chat never needs an LLM to serve learned knowledge. Four phases,
incremental per document (a per-doc hash of file content + guideline content
decides whether it needs relearning):

1. **Deterministic linking** — registry-backed edges (letter-number matches,
   `suggested_pod_ids`, field/WK/project metadata) with no LLM; exact
   letter-number matches auto-promote `pod_document` links.
2. **Guideline-driven extraction** — an LLM reads each new/changed document
   against `~/.esdc/guideline.yaml` (falling back to the packaged default)
   and proposes entities, claims, and any types missing from the guideline.
3. **Registry-backed resolution** — extracted mentions are resolved to
   canonical POD/project/field/WK entities; unresolved mentions are
   discarded, not guessed at.
4. **Dossier synthesis** — one Markdown case file per POD with at least one
   linked document, cached by source hash so unaffected PODs are skipped on
   rerun.

Types the LLM proposes that aren't in the guideline yet (new claim types,
entity types, etc.) are queued as schema proposals rather than silently
accepted — review them with `esdc corpus proposals` and add accepted ones to
`~/.esdc/guideline.yaml` by hand.

In chat, the `explore_entity` tool traverses the resulting graph (a
disposable in-memory LadybugDB instance graph rebuilt from `esdc.sqlite`) to
answer broad questions about one POD/field/project/WK — its dossier, related
documents/projects/revisions, and extracted claims.

### Evaluating retrieval quality

Generate a statistically-sized, stratified query set (one synthesized query
per sampled document, allocated across `doc_type`), then score retrieval
Pass@k and latency:

```bash
esdc corpus eval --init              # auto sample size (95% CI, ±5%)
esdc corpus eval --init --samples 100   # fixed 100 samples
esdc corpus eval --init --margin 0.10   # cheaper, ±10% margin (auto-size)
esdc corpus eval                     # score existing set
esdc corpus eval --refresh           # incremental sync after corpus changes
```

The query set lives at `~/.esdc/corpus_queries.jsonl`. If the corpus changes,
`esdc corpus eval` refuses to run (exit 1, reporting new/removed documents) and
prompts you to rerun with `--refresh` (incremental) or `--init` (regenerate).
Query synthesis uses the configured chat LLM provider.

### Remote Ollama

`OLLAMA_HOST` is honored for OCR, so extraction can run against a remote Ollama server (e.g. a GPU host) instead of localhost:

```bash
export OLLAMA_HOST=https://your-ollama-host:11434
esdc corpus extract path/to/document.pdf
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
