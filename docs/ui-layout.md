# ESDC UI Layout

## Left Panel (75% width)
- **Chat messages** (scrollable)
- **Thinking indicator** (collapsible, auto-expands during processing)
- **SQL Query** (collapsible, auto-expands when SQL generated)
- **Query Results** (collapsible, auto-expands when results ready)

## Right Panel (25% width)
- **Session Info** (static, not scrollable)
  - Provider name
  - Model name
  - Thread ID

## Key Widgets

### ThinkingIndicator
- Collapsible widget that shows AI reasoning steps
- Auto-expands when steps are added
- Title shows step count: "▶ Thinking... (3 steps)"

### SQLPanel
- Collapsible widget showing generated SQL
- Auto-expands when new SQL is generated
- Uses Markdown formatting for syntax highlighting

### ResultsPanel
- Collapsible widget showing query results
- Auto-expands when results arrive
- Shows raw results text

### ContextPanel
- Static sidebar with minimal session info
- No scrolling needed - fits in visible area
- Updates session info via `update_session_info()`