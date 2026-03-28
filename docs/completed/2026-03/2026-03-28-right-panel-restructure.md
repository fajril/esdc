# Right Panel Restructure Implementation Plan

**Goal:** Restructure right panel to have: Conversation Title (static top), Session Info (collapsible with provider/model/thread/directory), Context (collapsible with token usage), and Tool Status (separate static widget).

---

## Target Layout

```
Cadangan Lapangan Karamba - Wilayah Kerja Wain
▼ Session Info (expanded)
  Provider: ollama
  Model: kimi-k2.5:cloud
  Thread: 87851788...
  Dir: /Users/fajril/Projects/esdc

▼ Context (collapsed by default)
  12,345 / 128,000 (9%)

🔍 Idle
```

---

## Implementation Tasks

### Task 1: Create ContextUsageWidget (rename TokenUsageWidget)
- **File:** `esdc/chat/app.py:115-166`
- Rename TokenUsageWidget to ContextUsageWidget
- Add color coding based on percentage (warning >75%, danger >90%)
- Update CSS classes for rich text styling

### Task 2: Add Conversation Title Widget
- **File:** `esdc/chat/app.py`
- Create ConversationTitle widget (static, top of panel)
- Add update method to ContextPanel

### Task 3: Generate Title on First Query
- **File:** `esdc/chat/agent.py`
- Add `generate_conversation_title()` async function
- Use LLM to generate summary from first user query
- Fallback to first 50 chars of query

### Task 4: Restructure ContextPanel
- **File:** `esdc/chat/app.py:289-405`
- Reorganize layout order:
  1. Conversation Title (static)
  2. Session Info (ContextSection, expanded)
  3. Context (ContextSection, collapsed, contains ContextUsageWidget)
  4. Tool Status (static)
- Add current directory using os.getcwd()

### Task 5: Update Session Info Content
- **File:** `esdc/chat/app.py`
- Move Provider/Model/Thread into Session Info collapsible
- Add Dir: current working directory
- Update methods to refresh all sections

### Task 6: Integrate Token Display
- **File:** `esdc/chat/app.py`
- Wire token count updates to ContextUsageWidget
- Initialize with context_length from provider

---

## Commands to Run

```bash
# After each file edit
ruff check esdc/chat/app.py && basedpyright esdc/chat/app.py

# After all changes
pytest tests/ -v

# Test manually
esdc chat
```

---

## Summary

Changes will create a cleaner right panel with:
- Generated conversation title from first query
- Session info hidden in collapsible section (expanded by default)
- Context usage hidden in collapsible section (collapsed by default)  
- Tool status always visible at bottom
- Current working directory displayed in session info
