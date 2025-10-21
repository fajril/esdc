# CLI ID Interface Design

This document outlines the proposed command-line interface for managing KSMI identifiers in the `esdc` tool. The goal is to provide a consistent, extensible command hierarchy that keeps generation, verification, and export tasks easy to reason about.

---

## ID Generation Command Tree

```
esdc
|- id
   |- field     Generate field identifiers
   |- project   Generate project identifiers bound to a field
   |- zone      Generate zone identifiers bound to a field
```

All subcommands share the following global options:

| Option | Description |
| --- | --- |
| `-o, --output PATH` | Optional destination file. When omitted, IDs are printed to stdout. |
| `--format {plain,json,csv}` | Output format. Defaults to `plain` (one ID per line). |
| `--dry-run` | Perform validation and display a summary without writing IDs. |
| `--seed INTEGER` | Optional random seed to make generation deterministic. |

---

### Common Input Patterns

Many users already maintain files with previously generated identifiers. To keep commands predictable, the CLI accepts input via:

- Piped stdin (when `--from -` is supplied). Stdin always overrides other inputs.
- Text files containing one identifier per line.
- Inline IDs via `--id` (repeatable) when validating or seeding generation.
- Direct grid selectors via `--grid` or similar flags (for generation workflows).

Input precedence is: `stdin > file > inline flags`.

#### Deduplication & Validation

Every subcommand validates the provided identifiers (using the helpers in `esdc.idgen`) and de-duplicates them before generation. Invalid IDs abort the command with a descriptive error unless `--skip-invalid` is supplied to ignore bad lines.

---

### `esdc id field`

Generate field IDs for one or more grids.

| Option | Required | Description |
| --- | --- | --- |
| `--grid GRID` | Yes (unless `--from-field` or stdin is supplied) | One or more two-character grid codes, e.g., `2C`. Repeatable. |
| `--from-field PATH` | No | Path to a file containing existing field IDs. Use `-` to read from stdin. |
| `--id ID` | No | Inline field ID input (typically paired with `--verify-only`); repeat for multiple IDs. |
| `--count INTEGER` | No | Number of new IDs to generate. Defaults to `1`. |
| `--display-with-separators` | No | Output IDs using the dashed form `F-XX-XXXX-X`. Without the flag, IDs are compact. |

Behaviour notes:

- When both grids and existing IDs are provided, grids take precedence and existing IDs are used to avoid suffix collisions.
- If only existing IDs are supplied, the command infers grids from those records and continues the sequence.
- The command fails with exit code `2` if no grid context is available.

#### Examples

Generate five new IDs for grids `2C` and `2D`, using any existing IDs in a file:

```bash
esdc id field --grid 2C --grid 2D --from-field data/field_ids.txt --count 5
```

Continue generating IDs purely from existing data (stdin in this case):

```bash
cat data/field_ids.txt | esdc id field --from-field - --count 10
```

---

### `esdc id project`

Generate project IDs tied to a single field.

| Option | Required | Description |
| --- | --- | --- |
| `--field-id FIELD_ID` | Yes | The parent field ID (compact or dashed). |
| `--from-project PATH` | No | File/stdin providing existing project IDs for the field. |
| `--id ID` | No | Inline project ID input (used with `--verify-only`); repeat as needed. |
| `--count INTEGER` | No | Number of IDs to generate. Defaults to `1`. |
| `--start INTEGER` | No | Override the next sequence number. Useful for manual corrections. |

Behaviour notes:

- The command validates that every existing project ID belongs to the same field as `--field-id`.
- When `--start` is provided, sequence allocation resumes from that value, ensuring the CLI can recover after manual edits.

#### Examples

```bash
esdc id project --field-id F-2C-0123-B --from-project data/projects.txt --count 3
```

---

### `esdc id zone`

Generate zone IDs tied to a single field.

| Option | Required | Description |
| --- | --- | --- |
| `--field-id FIELD_ID` | Yes | The parent field ID (compact or dashed). |
| `--from-zone PATH` | No | File/stdin with existing zone IDs for the field. |
| `--id ID` | No | Inline zone ID input (used with `--verify-only`); repeat as needed. |
| `--count INTEGER` | No | Number of zone IDs to generate. Defaults to `1`. |

Behaviour notes:

- Zone suffix allocation honours the "unique digit" rule per field.
- The command exits with code `3` if all suffix combinations are exhausted.

#### Examples

```bash
esdc id zone --field-id F-2C-0123-B --from-zone data/zones.txt --count 2
```

---

### Verification Mode

Every generator supports a `--verify-only` flag that runs checksum validation on input IDs without creating new ones. Inline IDs can be passed directly via `--id`, enabling quick keyboard-driven validation without intermediate files.

- For `field`, verification relies on `verify_field_id` to ensure grid, suffix, and checksum integrity before re-use.
- For `project`, verification uses the underlying project parser to confirm the field payload and sequence are coherent.
- For `zone`, verification uses `verify_zone_id` to confirm the zone suffix uniqueness and checksum against its parent field payload.

The option can be combined with `--format json` to emit structured reports:

```bash
esdc id field --id F-2C-0123-B --verify-only --format json
```
```bash
esdc id project --id P-2C0123B-01 --verify-only
```
```bash
esdc id zone --id Z-2C0123B-012-B --verify-only --format json
```

The JSON payload contains:

```json
{
  "totals": {
    "processed": 120,
    "valid": 118,
    "invalid": 2
  },
  "invalid": [
    {"value": "F-2C-0000-0", "error": "suffix digits must be unique"},
    {"value": "F-99-9999-Z", "error": "checksum mismatch"}
  ]
}
```


---

### Exit Codes

| Code | Meaning |
| --- | --- |
| `0` | Success. |
| `1` | Unexpected internal error. |
| `2` | Missing or invalid grid context. |
| `3` | Exhausted suffix space for the given field. |
| `4` | Input validation failures when `--skip-invalid` is not set. |

---

### Pattern Considerations & Future Work

- The command tree mirrors the domain objects (field, project, zone) to keep the UX intuitive and make future additions (e.g., `esdc id batch`) straightforward.
- Shared options are factored out at the `id` level for consistency.
- Validation-only workflows are supported without creating separate commands.
