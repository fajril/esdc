# Fix Right Panel Issues - Dynamic Context Length

**Goal:** Fix three issues: (1) context length fetched dynamically from API, (2) conversation title updates, (3) collapsible sections work.

**Architecture:** Use Ollama API to dynamically fetch context_length from model_info, fix ConversationTitle update mechanism, fix ContextSection click handling.

---

## Dynamic Context Length Solution

**Discovery:** Ollama API has `POST /api/show` endpoint returning:
```json
{
  "model_info": {
    "gemma3.context_length": 131072,
    "gemma3.embedding_length": 2560,
    ...
  },
  "parameters": "temperature 0.7\nnum_ctx 2048"
}
```

**Implementation:** Use Ollama Python library `show()` method to get model_info dynamically.

---

## Task 1: Add Dynamic Context Length Fetching

**Files:**
- Modify: `esdc/providers/ollama.py`

**Step 1: Add method to fetch context length from API**

Add to `OllamaProvider` class:

```python
    @classmethod
    def get_context_length_from_api(cls, model: str, base_url: str | None = None) -> int:
        """Fetch context length dynamically from Ollama API.
        
        Args:
            model: Model name (e.g., "kimi-k2.5:cloud")
            base_url: Optional base URL for API
            
        Returns:
            Context length from model_info, or fallback to hardcoded mapping
        """
        try:
            import ollama
            
            client = ollama.Client(host=base_url or cls.DEFAULT_BASE_URL)
            info = client.show(model)
            
            # model_info contains "{model_name}.context_length"
            model_info = info.get("model_info", {})
            
            # Try to find context_length field
            # Format is usually "{architecture}.context_length"
            for key, value in model_info.items():
                if "context_length" in key and isinstance(value, (int, float)):
                    return int(value)
                    
        except Exception as e:
            logger.debug(f"Failed to fetch context length from API: {e}")
            
        # Fallback to hardcoded mapping
        return cls.get_context_length(model)
```

**Step 2: Import logger at top of file**

Add after imports:
```python
import logging
logger = logging.getLogger(__name__)
```

**Step 3: Run lint/typecheck**

```bash
ruff check esdc/providers/ollama.py && basedpyright esdc/providers/ollama.py
```

**Step 4: Commit**

```bash
git add esdc/providers/ollama.py
git commit -m "feat: add dynamic context length fetching from Ollama API"
```

---

## Task 2: Update ChatApp to Use Dynamic Context Length

**Files:**
- Modify: `esdc/chat/app.py`

**Step 1: Update on_mount to call dynamic method**

Around line 1155-1165, change:

```python
# Get context length from provider
provider_config = Config.get_provider_config()
if provider_config and provider_config.get("model"):
    from esdc.providers import get_provider
    
    provider_type = provider_config.get("provider_type", "ollama")
    provider = get_provider(provider_type)
    if provider:
        # Use dynamic API fetching for Ollama
        if provider_type == "ollama":
            from esdc.providers.ollama import OllamaProvider
            self._context_length = OllamaProvider.get_context_length_from_api(
                provider_config.get("model", ""),
                provider_config.get("base_url")
            )
        else:
            self._context_length = provider.get_context_length(
                provider_config.get("model", "")
            )
```

**Step 2: Run lint/typecheck**

```bash
ruff check esdc/chat/app.py && basedpyright esdc/chat/app.py
```

**Step 3: Commit**

```bash
git add esdc/chat/app.py
git commit -m "feat: use dynamic context length from Ollama API"
```

---

## Task 3: Fix ConversationTitle Update

**Files:**
- Modify: `esdc/chat/app.py:305-330`

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

**Step 2: Update ContextPanel.update_conversation_title**

Change:
```python
def update_conversation_title(self, title: str) -> None:
    self._conversation_title = title
    try:
        title_widget = self.query_one("#conversation-title", ConversationTitle)
        title_widget._title = title
        title_widget.update(title)
    except Exception:
        pass
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

**Step 3: Commit**

```bash
git add esdc/chat/app.py
git commit -m "fix: correct ConversationTitle update mechanism using set_title"
```

---

## Task 4: Fix Collapsible Sections

**Files:**
- Modify: `esdc/chat/app.py:46-112`

**Step 1: Fix ContextSection toggle and styling**

Update ContextSection:

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

    ContextSection .header:hover {
        color: $accent;
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
        expanded: bool = True,
        badge: str = "",
        id: str | None = None,
    ):
        super().__init__(id=id)
        self.section_title = title
        self.expanded = expanded
        self.badge = badge
        self._header: Static | None = None

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

**Step 2: Commit**

```bash
git add esdc/chat/app.py
git commit -m "fix: make ContextSection properly collapsible with working toggle"
```

---

## Task 5: Integration Test

**Step 1: Manual test**

```bash
esdc chat
```

Verify:
1. [ ] Context length shows correct value (128000 for kimi-k2.5:cloud, not 4096)
2. [ ] First query updates title to AI-generated summary
3. [ ] Session Info is expanded (▼), clicking collapses (▶)
4. [ ] Context is collapsed (▶), clicking expands (▼) and shows token usage

**Step 2: Run tests**

```bash
pytest tests/ -v
```

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: dynamic context length and fixed right panel issues"
```

---

## Summary

**Fixes:**
1. ✅ **Dynamic context length** - Uses Ollama API to fetch real context_length from model_info
2. ✅ **Title updates** - Fixed using set_title() method on Static widget
3. ✅ **Collapsible sections** - Fixed toggle() to update header icon and added cursor pointer

**Expected:**
```
Cadangan WK Masela Reserves
▼ Session Info
  Provider: ollama
  Model: kimi-k2.5:cloud  
  Thread: 87851788...
  Dir: /Users/fajril/...
▶ Context
🔍 Idle
```

Expanding Context:
```
▼ Context
  5,935 / 128,000 (4%)  ← Correct length!
🔍 Idle
```
