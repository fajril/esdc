# ESDC UI Layout

## Left Panel (Chat Panel - 75% width)
- **Chat messages** (scrollable container)
- **Dynamic collapsibles** (mounted during query processing):
  - Thinking indicator (appears first, shows AI reasoning steps)
  - SQL Query (appears when SQL is executed)
  - Query Results (appears after SQL results return)
  - AI Answer (appears last, thinking indicator is removed)

## Right Panel (Context Panel - 25% width)
- **Session Info** (static, not scrollable)
  - Provider name
  - Model name
  - Thread ID

## Chat Flow

When a user asks a question, collapsibles appear in the chat flow in this order:

```
┌─────────────────────────────────────────┐
│ [User Question]                         │
│                                         │
│ ▼ ▶ Thinking... (2 steps)               │
│ │   • Running: execute_sql              │
│ │   • Running: get_schema               │
│                                         │
│ ▼ 📝 SQL Query                          │
│ │ SELECT * FROM projects...             │
│                                         │
│ ▼ 📊 Query Results                      │
│ │ [table data]                          │
│                                         │
│ Hello! Here are the results...          │
│ [AI answer message]                     │
└─────────────────────────────────────────┘
```

**Key behaviors:**
1. Thinking indicator appears immediately when query starts
2. SQL collapsible appears when `tool_result` with SQL is received
3. Results collapsible appears after SQL collapsible
4. Thinking indicator is **removed** when AI answer arrives
5. Order reflects actual execution flow (not fixed)

## Key Widgets

### ThinkingIndicator
- Collapsible that shows AI reasoning steps
- Created dynamically for each query
- Removed when AI answer arrives
- Title shows step count: "▶ Thinking... (3 steps)"

### SQLPanel
- Collapsible showing generated SQL
- Created dynamically when SQL is executed
- Uses Markdown formatting for syntax highlighting
- Auto-expands when mounted

### ResultsPanel
- Collapsible showing query results
- Created dynamically when results arrive
- Shows raw results text
- Auto-expands when mounted

### ContextPanel
- Static sidebar with minimal session info
- No scrolling - fits in visible area
- Updated via `update_session_info()`