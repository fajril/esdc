# AGENTS.md

> **Instructions for AI agents working on this codebase**

## Resources

- **Knowledge Base:** `docs/reference/` - Architecture, conventions, troubleshooting
- **Schema:** `docs/reference/schema/esdc-database-schema.md` - Database schema reference
- **Active Work:** `docs/active/` - Current work in progress
- **Completed Work:** `docs/completed/YYYY-MM/` - Finished and archived plans
- **Session Notes:** `docs/sessions/` - Historical session records

## Current Focus

See `docs/active/README.md` (if exists) for current active work.

## Key Context

### Project: ESDC (Elektronik Sumber Daya dan Cadangan)
- Indonesian oil & gas reserves data management
- Tech stack: Python, SQLite, Textual TUI, LangChain, LangGraph
- Repository: https://github.com/fajril/esdc

### Architecture
- TUI: `esdc/chat/app.py` - Textual-based chat interface
- Agent: `esdc/chat/agent.py` - LangGraph agent with tools
- Tools: `esdc/chat/tools.py` - SQL execution, schema retrieval
- Schema: Loaded from `docs/reference/schema/esdc-database-schema.md`

### Testing
```bash
# Run tests
pytest tests/

# Run chat TUI
esdc chat
```

## Important Files

| File | Purpose |
|------|---------|
| `esdc/chat/app.py` | Main TUI application |
| `esdc/chat/agent.py` | LangGraph agent logic |
| `esdc/chat/tools.py` | Database tools (execute_sql, get_schema, list_tables) |
| `esdc/chat/prompts.py` | System prompt template |
| `esdc/chat/schema_loader.py` | Database schema loader |
| `esdc/configs.py` | Configuration management |

## Workflow

1. **Starting Work:** Check `docs/active/` for current tasks
2. **During Work:** Update session notes in `docs/sessions/`
3. **Completing Work:** Move plans to `docs/completed/YYYY-MM/`
4. **Reference:** Use `docs/reference/` for architecture docs

## Notes

- Session notes are historical records (read-only after creation)
- Plans in `docs/completed/` are archived (no longer modified)
- Use `docs/reference/` for persistent documentation