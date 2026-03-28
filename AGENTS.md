---
## Goal

Fix the ESDC TUI chat application's streaming and indicator issues:
1. **Real-time token-by-token streaming** - Character-by-character display like ChatGPT
2. **Inline indicator visibility** - Show "🔍 Querying database..." during tool calls, keep it visible after completion
3. **Fix text duplication** - Prevent content from being appended twice
4. **Show SQL query** - Display executed SQL query in code block below indicator for ALL tool calls
5. **Fix UI freeze** - Make UI responsive during tool execution
6. **Auto-focus input** - Cursor should be in chat box when app starts
7. **Tool status indicator** - Show query status in right panel
8. **Right panel restructure** - Clean layout with conversation title, collapsible sections, and token usage

## Instructions

- User uses iTerm2 terminal (has emoji support)
- User prefers emoji 🔍 for indicator (not ASCII text)
- Don't remove indicator after tool_result - keep it visible
- When there are multiple approaches, choose the simpler one
- Always commit with clear, descriptive messages
- Test with `esdc chat` command

## Discoveries

### Critical Bug Fixed - Token Events Dropped
- **Root Cause**: `_stream_response()` method in `app.py` was NOT forwarding `token` events from agent to UI
- **Fix**: Added forwarding of token events
- **Evidence**: Agent logs showed `TOKEN_EVENT #100`, but UI never received them

### Text Duplication Issue
- **Root Cause**: Both `token` events AND `message` events were appending to `accumulated_content`
- **Fix**: Skip `message` event if `accumulated_content` already has content from tokens

### Indicator Disappearing Issue - FIXED
- **Root Cause**: Indicator was added to UI but NOT to `accumulated_content` variable
- **Fix**: Update `accumulated_content` when adding indicator
- **Result**: Indicator now persists through token streaming

### SQL Query Passing Issue - FIXED (Multiple Bugs)

**Bug 1-4**: Multiple fixes for SQL query passing and formatting
- Agent passed empty dict, _stream_response dropped args
- Code block format issues with closing backticks
- Token concatenation with closing backticks

### Multiple Tool Calls - FIXED
- **Root Cause**: Skip logic prevented showing indicator if emoji already existed
- **Fix**: Removed the skip check - always show indicator for each tool call
- **Result**: All SQL queries now displayed

### Auto-Focus Input - FIXED
- **Issue**: User had to click on input box to start typing
- **Fix**: Add `self.user_input.focus()` in `on_mount()` method

### Tool Status Indicator - IMPLEMENTED
- **Feature**: Visual feedback in right panel during tool execution
- **Implementation**: 
  - Added `ToolStatus` widget to `ContextPanel`
  - Display states: `🔍 Idle` / `⏳ Querying database...` / `✅ Query completed`
  - Auto-reset to idle after streaming completes
  - CSS styling with color-coded states

### UI Freeze During Tool Execution - FIXED
- **Root Cause**: `execute_sql` was synchronous and blocking the event loop
- **Fix**: 
  1. Renamed function to `_execute_sql_sync` (sync implementation)
  2. Created `execute_sql` as async function
  3. Uses `asyncio.run_in_executor()` to run sync code in thread pool
  4. Event loop remains free to handle UI updates
- **Result**: UI stays responsive during database queries

### Streaming Architecture
- Agent uses `astream_events(version="v2")` for token-level streaming
- ChatOllama may emit both `on_chat_model_stream` and `on_llm_stream` events
- Need to handle both event types

## Accomplished

### Completed:
1. ✅ Implemented `astream_events(version="v2")` for token-level streaming
2. ✅ Added dual event handling (`on_chat_model_stream`, `on_llm_stream`)
3. ✅ Fixed token event forwarding in `_stream_response()`
4. ✅ Fixed indicator condition (removed `accumulated_content` requirement)
5. ✅ Fixed text duplication (skip `message` if already have token content)
6. ✅ Removed code that deletes indicator after tool_result
7. ✅ Added comprehensive debug logging
8. ✅ Added content snapshot logging to track indicator disappearance
9. ✅ **Fixed indicator disappearing** - update `accumulated_content` when adding indicator
10. ✅ **Fixed SQL query passing** - multiple bug fixes to pass args correctly
11. ✅ **Fixed code block format** - correct markdown code block syntax
12. ✅ **Added auto-focus** - cursor now in chat input on app start
13. ✅ **Fixed token concatenation** - add newline after SQL code block closing backticks
14. ✅ **Show all SQL queries** - display indicator + SQL for each tool call (not just first)
15. ✅ **Added tool status indicator** - real-time status display in right panel
16. ✅ **Made tool execution non-blocking** - async wrapper with thread pool executor
17. ✅ **Increased recursion limit** - from 25 to 50 in LangGraph config
18. ✅ **Renamed TokenUsageWidget to ContextUsageWidget** - with color coding (green/yellow/red)
19. ✅ **Added ConversationTitle widget** - shows AI-generated summary from first query
20. ✅ **Restructured ContextPanel** - clean layout with collapsible sections
21. ✅ **Added generate_conversation_title** - AI generates title from first user query
22. ✅ **Integrated token display** - updates in real-time in Context section
23. ✅ **Added current directory** - displayed in Session Info section

### Commits Made:
```
86846e6 - debug: add comprehensive logging for token streaming and tool calls
f58a895 - fix: forward token events and enable indicator without content requirement
786fb82 - fix: prevent text duplication and keep indicator visible
a15d869 - debug: add content snapshot logging to track indicator disappearance
1c21297 - fix: indicator stays visible by updating accumulated_content
552da79 - debug: add logging to track tool_call args structure
1f5dc07 - fix: pass parsed args with query to app instead of empty dict
fddfe20 - fix: forward args in tool_call from _stream_response to app
2dd9cb6 - fix: correct code block format and add auto-focus to chat input
fe708b6 - fix: add newline after SQL code block to prevent token concatenation
85553e2 - feat: show SQL query for each tool call (multiple queries)
2b281e8 - feat: add tool status indicator and non-blocking SQL execution
```

## Relevant Files / Directories

### Primary Files:
- **`esdc/chat/app.py`** (~1404 lines) - Main TUI application
  - Lines 291-389: `ContextPanel` class with ToolStatus widget
  - Lines 996: Auto-focus user input on mount
  - Lines 1157-1164: Token handler - append to `accumulated_content`, update UI
  - Lines 1176-1188: Message handler - SKIPPED if already have content
  - Lines 1195-1267: Tool call handler - extract SQL, add indicator, update tool status
  - Lines 1247-1267: Tool result handler - update tool status to "✅ Query completed"
  - Lines 1281-1302: Error/completion handlers - reset tool status
  - Lines 1295-1331: `_stream_response()` - forward token, message, and tool_call events

- **`esdc/chat/agent.py`** (~385 lines) - LangGraph agent with streaming
  - Lines 153-157: `astream_events(version="v2")`
  - Lines 163-178: Token streaming - handles both `on_chat_model_stream` and `on_llm_stream`
  - Lines 213-242: Tool call handling - passes parsed `args` with query
  - Lines 272-289: Tool result handling

- **`esdc/chat/tools.py`** (~236 lines) - Tool implementations
  - Lines 43-116: `execute_sql` async function with thread pool executor
  - Lines 118-163: `_execute_sql_sync` - synchronous database operations

### Test Files:
- `tests/test_chat_dom.py` - All tests passing
- `tests/test_chat_mounting.py` - All tests passing

### Log File:
- `esdc_chat.log` - Debug logs with emoji markers (🚀📝🛠️💡✅❌🔔📥📊🔍)

## Next Step

**User Action Required**: Test with `esdc chat` and verify:

1. ✅ Indicator "🔍 Querying database..." appears for EACH tool call
2. ✅ All SQL queries displayed in proper code block format
3. ✅ All indicators + SQL stay visible after queries complete
4. ✅ Cursor auto-focus in input box on app start
5. ✅ Tool status indicator in right panel shows: `🔍 Idle` → `⏳ Querying database...` → `✅ Query completed` → `🔍 Idle`
6. ✅ UI remains responsive during tool execution (can scroll)
7. ✅ No freeze during query execution

**Expected Output for Multiple Tool Calls:**
```
[Right Panel - Context]
Provider: ollama
Model: kimi-k2.5:cloud
Thread: 87851788...

⏳ Querying database...  ← Status indicator

[Chat Panel - AI message streaming...]

🔍 Querying database...

```sql
SELECT MAX(report_year) as latest_year FROM project_resources
```

🔍 Querying database...

```sql
SELECT * FROM reserves WHERE working_area = 'WK Wain'
```

[Results + continued AI message...]
```

**Architecture Changes:**
- Tool execution now runs in thread pool (`asyncio.run_in_executor`)
- Event loop remains free for UI updates
- Status widget updates in real-time during query lifecycle