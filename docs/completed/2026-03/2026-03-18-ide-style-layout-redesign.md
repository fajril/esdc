# IDE-Style Layout Redesign Design Document

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Redesign the ESDC chat app to follow modern IDE/terminal layout patterns with a collapsible context panel showing session information, token usage, tools, SQL preview, results, schema browser, and query history.

**Architecture:** Convert the current 60/40 horizontal split with SQL/Results on the right to a 70/30 split with a rich context panel. The status bar moves to a true bottom position, and SQL/Results become collapsible sections within the context panel. This follows the design pattern seen in modern IDEs with information density and visual hierarchy.

**Tech Stack:** Python, Textual (TUI framework), existing chat infrastructure (LangGraph agent, LangChain tools)

---

## Design Overview

### Layout Structure

```
┌─────────────────────────────────────────────────────────────────┐
│  CHAT AREA (70%)                    │  CONTEXT PANEL (30%)     │
│  - Message history                   │  - Session Info ✓        │
│  - User messages                     │  - Token Usage ✓         │
│  - AI responses                      │  - Tools Available ✓     │
│  - Thinking indicator                │  - Last SQL Query ▸      │
│                                      │  - Query Results ▸       │
│                                      │  - Schema Browser ▸       │
│                                      │  - Query History ▸       │
├─────────────────────────────────────┴───────────────────────────┤
│  > Input field (full width)                                      │
├─────────────────────────────────────────────────────────────────┤
│  ollama | llama3.2 | 5,432 tokens (34%) | thread: abc123       │
└─────────────────────────────────────────────────────────────────┘

✓ = Expanded by default
▸ = Collapsed by default
```

### Components

#### New Components

1. **`ContextSection`** - Reusable collapsible section widget
   - Header with expand/collapse icon (▸/▾)
   - Optional badge/count indicator
   - Content area (scrollable if needed)
   - Maintains expanded/collapsed state

2. **`TokenUsageWidget`** - Token usage display
   - Shows: `5,432 / 16,384 (34%)`
   - Optional progress bar visualization
   - Updates in real-time

3. **`ToolStatusList`** - Shows available tools
   - Lists: execute_sql, get_schema, list_tables
   - Green checkmark for available
   - Optional: highlight tools used in last query

4. **`QueryPreview`** - SQL and Results preview
   - Collapsible sections
   - Shows truncated preview (50 chars for SQL, 5 rows for results)
   - Expand to see full content

5. **`SchemaBrowser`** - Table information
   - Lists available tables
   - Informational only (click to insert future enhancement)

6. **`QueryHistory`** - Session-only query history
   - Last 5 queries
   - Truncated preview
   - Does NOT persist across sessions

#### Modified Components

1. **`ESDCChatApp`** - Main app
   - Change split from 60/40 to 70/30
   - Move status bar to true bottom
   - Full-width input field
   - Add context panel state management

2. **`ChatPanel`** - Simplified
   - Remove inner footer
   - Message container takes full height

3. **`StatusBar`** - New position and format
   - Position: true bottom of app
   - Format: `provider | model | tokens (percent) | thread_id`

4. **`RightPanel`** - Replaced by `ContextPanel`

#### Removed Components

1. **`SQLPanel`** - Replaced by section in context panel
2. **`ResultsPanel`** - Replaced by section in context panel
3. **`Footer`** - Merged into main app layout

### State Management

```python
# New state in ESDCChatApp
self._context_panel_visible: bool = True
self._section_expanded: dict[str, bool] = {
    "session_info": True,
    "token_usage": True,
    "tools": True,
    "sql": False,
    "results": False,
    "schema": False,
    "history": False,
}
self._query_history: list[str] = []  # Session-only, max 5
self._tools_used_last_query: list[str] = []
```

### Bindings

| Binding | Action | Description |
|---------|--------|-------------|
| `ctrl+h` | `toggle_context_panel` | Show/hide context panel |
| `ctrl+l` | `toggle_sql_section` | Expand/collapse SQL section |
| `ctrl+r` | `toggle_results_section` | Expand/collapse results section |
| `ctrl+e` | `toggle_all_sections` | Expand/collapse all sections |
| `ctrl+s` | `screenshot` | Save screenshot (existing) |
| `escape` | `cancel_query` | Cancel query (existing) |
| `ctrl+b` | `toggle_split` | Toggle split (existing - may be removed) |

### CSS Changes

```css
/* Main layout */
Screen {
    layout: vertical;
}

#main-content {
    layout: horizontal;
    height: 1fr;
}

#chat-area {
    width: 70%;
    border: solid $primary;
}

#context-panel {
    width: 30%;
    border: solid $accent;
    overflow-y: auto;
}

/* Input area */
#input-area {
    height: auto;
    border: solid $surface;
}

/* Status bar */
StatusBar {
    height: 1;
    padding: 0 1;
    color: $text-muted;
    background: $surface;
}

/* Context sections */
ContextSection {
    border: solid $surface;
    margin: 0 1;
}

ContextSection-header {
    background: $surface;
    padding: 0 1;
}

/* Token usage bar */
TokenUsageWidget {
    height: auto;
    padding: 1;
}

/* Tool list */
ToolStatusList {
    height: auto;
    padding: 0 1;
}
```

### File Changes

| File | Action | Description |
|------|--------|-------------|
| `esdc/chat/app.py` | Major refactor | New layout, components, bindings |
| `tests/test_chat_app.py` | Update tests | Test new components and layout |

### Behavior Details

#### Context Panel Visibility

- Default: visible
- Toggle with `ctrl+h`
- When hidden, chat expands to full width
- State persists during session only

#### Section Expansion States

- Default expanded: session_info, token_usage, tools
- Default collapsed: sql, results, schema, history
- User can toggle individual sections
- State resets on app restart (no persistence)

#### Query History

- Stores last 5 queries (truncated to 50 chars)
- Session-only (cleared on restart)
- Display only (click to re-run is future enhancement)

#### SQL Query Display

- Auto-expand on query execution
- Show truncated SQL (first 50 chars)
- Expand to see full query
- Collapse manually or stays expanded

#### Results Display

- Auto-collapse on new query
- Show row count
- Expand to see preview (first 5 rows)
- Full results not stored in context panel

#### Token Usage

- Real-time updates from agent streaming
- Shows: current tokens / context window (percentage)
- Visual progress bar (optional)
- Only for current conversation (no persistence)

#### Schema Browser

- Lists tables: project_resources, field_resources, etc.
- Informational display only
- Future: click to insert table name into input
- Collapsed by default

### Testing Strategy

1. Unit tests for new components (ContextSection, TokenUsageWidget, etc.)
2. Integration tests for ESDCChatApp layout
3. Test all bindings and toggles
4. Test state management
5. Visual smoke test

### Trade-offs

**Pros:**
- Modern IDE-like interface
- Better information density
- Collapsible sections reduce clutter
- More screen space for chat

**Cons:**
- SQL/Results less visible by default (mitigated by auto-expand on query)
- More complex component hierarchy
- Requires more state management

### Future Enhancements (Out of Scope)

- Resizable split panels (drag to resize)
- Query history persistence across sessions
- Schema browser click-to-insert
- Query bookmarks/favorites
- Export query results to file
- Customizable section order

---

## Implementation Notes

### Implementation Order

1. Create `ContextSection` component (reusable base)
2. Create `TokenUsageWidget` (standalone widget)
3. Create `ToolStatusList` (standalone widget)
4. Create `QueryPreview` (combines SQL + Results)
5. Create `SchemaBrowser` (standalone widget)
6. Create `QueryHistory` (standalone widget)
7. Create `ContextPanel` (combines all sections)
8. Refactor `ESDCChatApp` layout
9. Update `StatusBar` format
10. Add new bindings
11. Update tests

### Key Considerations

- **Backward compatibility:** Ensure existing chat functionality works
- **Performance:** Don't re-render context panel on every keystroke
- **Memory:** Query history capped at 5 items
- **UX:** Smooth transitions when toggling sections

### Success Criteria

- [ ] Layout matches design specification
- [ ] All components render correctly
- [ ] Collapsible sections work
- [ ] Token usage updates in real-time
- [ ] All bindings work as expected
- [ ] Tests pass
- [ ] Type checking passes
- [ ] Visual smoke test passes