# Fix Right Panel Issues Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix three issues: (1) context length not fetched from model API, (2) conversation title not updating, (3) collapsible sections not working.

**Architecture:** Update ContextSection click handling, fix ConversationTitle update mechanism, ensure context_length is fetched from provider API on mount.

**Tech Stack:** Textual TUI, LangChain, Provider API context length lookup

---

## Issues Analysis

### Issue 1: Context Length Shows 4096 (Default)
**Current:** Token count shows `5,935 / 4,096 (144%)` - kimi-k2.5:cloud should be ~128k
**Root Cause:** `on_mount()` in ChatApp initializes `_context_length = 4096` default, but kimi-k2.5:cloud is not in CONTEXT_LENGTHS mapping
**Fix:** 
- Add "kimi-k2.5" to OllamaProvider.CONTEXT_LENGTHS with 128000
- Ensure context length is fetched BEFORE initializing ContextPanel

### Issue 2: Conversation Title Not Updating
**Current:** Still shows "New Conversation" after first query about "WK Masela"
**Root Cause:** `ConversationTitle.compose()` yields a Static widget, but `update_conversation_title()` tries to call `title_widget.update()` which doesn't work on Static children
**Fix:**
- Make ConversationTitle extend Static and call self.update() directly
- Or use a proper widget structure that supports updates

### Issue 3: Collapsible Sections Not Working
**Current:** Session Info and Context sections show arrow (▾) but clicking doesn't collapse
**Root Cause:** ContextSection.on_click() toggles class but compose() yields Static header that doesn't update the arrow icon
**Fix:**
- Track header widget reference
- Update header text when toggling (change ▾ to ▸)
- Add cursor pointer styling

---

## Task 1: Add kimi-k2.5 to Context Lengths

**Files:**
- Modify: `esdc/providers/ollama.py:13-34`

**Step 1: Add kimi-k2.5 context length mapping**

Add "kimi-k2.5" entry to CONTEXT_LENGTHS dict:

```python
CONTEXT_LENGTHS = {
    "llama3.2": 128000,
    "llama3.1": 128000,
    # ... existing entries ...
    "kimi-k2.5": 128000,  # Add this line
    "kimi": 128000,        # Also add generic kimi for matching
}
```

**Step 2: Run lint/typecheck**

Run: `ruff check esdc/providers/ollama.py && basedpyright esdc/providers/ollama.py`
Expected: PASS

**Step 3: Commit**

```bash
git add esdc/providers/ollama.py
git commit -m "fix: add kimi-k2.5 context length mapping (128k)"
```

---

## Task 2: Fix ConversationTitle Update Mechanism

**Files:**
- Modify: `esdc/chat/app.py:305-330`

**Current broken code:**
```python
class ConversationTitle(Static):
    def compose(self) -> ComposeResult:
        yield Static(self._title if self._title else "New Conversation")
        
    def update_conversation_title(self, title: str) -> None:
        title_widget = self.query_one("#conversation-title", ConversationTitle)
        title_widget._title = title
        title_widget.update(title)  # This doesn't work on Static!
```

**Step 1: Fix ConversationTitle class**

Change from:
```python
class ConversationTitle(Static):
    def __init__(self, title: str = "", id: str | None = None):
        super().__init__(id=id)
        self._title = title
        
    def compose(self) -> ComposeResult:
        from textual.widgets import Static
        yield Static(self._title if self._title else "New Conversation")
```

To:
```python
class ConversationTitle(Static):
    def __init__(self, title: str = "", id: str | None = None):
        super().__init__(
            title if title else "New Conversation",
            id=id
        )
        self._title = title
        
    def set_title(self, title: str) -> None:
        """Update the conversation title."""
        self._title = title
        self.update(title)
```

**Step 2: Update ContextPanel.compose()**

Change:
```python
yield ConversationTitle(
    self._conversation_title,
    id="conversation-title",
)
```

To:
```python
yield ConversationTitle(
    self._conversation_title,
    id="conversation-title",
)
```
(same, just verify it's correct)

**Step 3: Update ContextPanel.update_conversation_title()**

Change from:
```python
def update_conversation_title(self, title: str) -> None:
    self._conversation_title = title
    try:
        title_widget = self.query_one("#conversation-title", ConversationTitle)
        title_widget._title = title
        title_widget.update(title)
    except Exception:
        pass
    self.refresh()
```

To:
```python
def update_conversation_title(self, title: str) -> None:
    """Update the conversation title."""
    self._conversation_title = title
    try:
        title_widget = self.query_one("#conversation-title", ConversationTitle)
        title_widget.set_title(title)
    except Exception as e:
        logger.debug(f"Failed to update conversation title: {e}")
```

**Step 4: Run lint/typecheck**

Run: `ruff check esdc/chat/app.py && basedpyright esdc/chat/app.py`
Expected: PASS

**Step 5: Commit**

```bash
git add esdc/chat/app.py
git commit -m "fix: correct ConversationTitle update mechanism"
```

---

## Task 3: Fix Collapsible Sections

**Files:**
- Modify: `esdc/chat/app.py:46-112`

**Step 1: Add click cursor styling**

Add to ContextSection DEFAULT_CSS:
```python
ContextSection .header {
    background: transparent;
    padding: 1 1;
    border-bottom: solid $primary-background;
    cursor: pointer;  /* Add this line */
}
```

**Step 2: Fix ContextSection to track and update header**

Change ContextSection to properly update the header text when toggled:

```python
class ContextSection(Container):
    """Collapsible section widget for context panel."""

    DEFAULT_CSS = """
    ContextSection {
        margin: 0;
        border: none;
    }

    ContextSection .header {
        background: transparent;
        padding: 1 1;
        border-bottom: solid $primary-background;
        cursor: pointer;
    }

    ContextSection .title {
        color: $text;
        text-style: bold;
    }

    ContextSection .content {
        padding: 1 1;
        background: transparent;
    }

    ContextSection.collapsed .content {
        display: none;
    }
    """

    def __init__(
        self,
        title: str,
        expanded: bool = True,  # Change default to True
        badge: str = "",
        id: str | None = None,
    ):
        super().__init__(id=id)
        self.section_title = title  # Use different name to avoid conflict
        self.expanded = expanded
        self.badge = badge
        self._header: Static | None = None  # Track header widget

    def compose(self) -> ComposeResult:
        """Compose the section."""
        icon = "▼" if self.expanded else "▶"
        title_text = f"{icon} {self.section_title}"
        if self.badge:
            title_text += f" [{self.badge}]"

        self._header = Static(title_text, classes="header")
        yield self._header

    def on_mount(self) -> None:
        """Set initial expanded state."""
        if not self.expanded:
            self.add_class("collapsed")

    def on_click(self) -> None:
        """Handle click to toggle."""
        self.toggle()
        
    def toggle(self) -> None:
        """Toggle expanded state and update header."""
        self.expanded = not self.expanded
        
        if self.expanded:
            self.remove_class("collapsed")
        else:
            self.add_class("collapsed")
        
        # Update header text with new icon
        if self._header:
            icon = "▼" if self.expanded else "▶"
            title_text = f"{icon} {self.section_title}"
            if self.badge:
                title_text += f" [{self.badge}]"
            self._header.update(title_text)
```

**Step 3: Update ContextPanel to use proper defaults**

Session Info should be expanded (True), Context should be collapsed (False).

**Step 4: Run lint/typecheck**

Run: `ruff check esdc/chat/app.py && basedpyright esdc/chat/app.py`
Expected: PASS

**Step 5: Commit**

```bash
git add esdc/chat/app.py
git commit -m "fix: make ContextSection properly collapsible with click handling"
```

---

## Task 4: Ensure Context Length is Fetched Before Panel Init

**Files:**
- Modify: `esdc/chat/app.py:1150-1185`

**Current issue:** Context length is fetched in `on_mount()` but `_context_panel` is already composed by then.

**Step 1: Update on_mount to set context length BEFORE panel init**

Find the on_mount section around line 1150 and ensure:

```python
def on_mount(self) -> None:
    # ... setup code ...
    
    # Get context length from provider FIRST
    provider_config = Config.get_provider_config()
    if provider_config and provider_config.get("model"):
        from esdc.providers import get_provider
        
        provider_type = provider_config.get("provider_type", "ollama")
        provider = get_provider(provider_type)
        if provider:
            self._context_length = provider.get_context_length(
                provider_config.get("model", "")
            )
    
    # Update context panel with all info
    if self._context_panel:
        self._context_panel.update_session_info(
            self._provider_name,
            self._model_name,
            self._thread_id,
        )
        # Update context usage with correct length
        self._context_panel.update_context_usage(
            self._token_count,
            self._context_length,
        )
```

**Step 2: Run lint/typecheck**

Run: `ruff check esdc/chat/app.py && basedpyright esdc/chat/app.py`
Expected: PASS

**Step 3: Commit**

```bash
git add esdc/chat/app.py
git commit -m "fix: ensure context length fetched from provider API before panel init"
```

---

## Task 5: Integration Test

**Files:**
- Test: Manual with `esdc chat`

**Step 1: Run manual test**

```bash
esdc chat
```

**Verify:**
1. [ ] Right panel shows "New Conversation" initially
2. [ ] Session Info is expanded (▼) with Provider, Model, Thread, Dir
3. [ ] Context is collapsed (▶) or empty
4. [ ] After first query, title updates to AI-generated summary
5. [ ] Token usage shows correct context length (128,000 for kimi-k2.5)
6. [ ] Clicking "Session Info" header collapses it (▶)
7. [ ] Clicking "Context" header expands it (▼) and shows token usage

**Step 2: Run all tests**

```bash
pytest tests/ -v
```
Expected: All tests pass (or same as before)

**Step 3: Final commit if needed**

```bash
git add -A
git commit -m "fix: all right panel issues resolved"
```

---

## Summary

Fixes:
1. ✅ **Context length** - Added kimi-k2.5 to CONTEXT_LENGTHS (128k), ensure fetched before panel init
2. ✅ **Title update** - Fixed ConversationTitle.update() to work on Static widget
3. ✅ **Collapsible sections** - Fixed ContextSection header tracking and toggle() method

Expected right panel after fix:
```
Cadangan Lapangan Karamba Wilayah Kerja  ← Generated title
▼ Session Info                              ← Click to collapse
  Provider: ollama
  Model: kimi-k2.5:cloud
  Thread: 87851788...
  Dir: /Users/fajril/Documents/GitHub/esdc

▶ Context                                   ← Click to expand
🔍 Idle
```

After expanding Context:
```
▼ Context
  12,345 / 128,000 (9%)                     ← Correct context length
🔍 Idle
```
