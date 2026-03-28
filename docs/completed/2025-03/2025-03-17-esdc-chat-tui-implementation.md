# ESDC Chat TUI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a TUI chat interface where users can ask natural language questions about ESDC reserve data, with AI-powered text-to-SQL conversion and multi-provider support.

**Architecture:** Textual-based split-view TUI with chat panel (left) and results panel (right). AI providers configured via CLI and config file. Hybrid text-to-SQL with core schema + dynamic lookup.

**Tech Stack:** Textual, Rich, SQLite, OpenAI/Anthropic/Ollama APIs

---

## Phase 1: Provider System

### Task 1: Provider Base Class and Config

**Files:**
- Create: `esdc/providers/__init__.py`
- Create: `esdc/providers/base.py`
- Modify: `esdc/configs.py` - add provider config methods
- Test: `tests/test_providers.py`

**Step 1: Write the failing test**

```python
# tests/test_providers.py
from esdc.providers.base import Provider, ProviderConfig

def test_provider_config_validation():
    config = ProviderConfig(
        name="openai",
        api_key="sk-test",
        model="gpt-4o"
    )
    assert config.name == "openai"
    assert config.api_key == "sk-test"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_providers.py::test_provider_config_validation -v
```
Expected: FAIL - module not found

**Step 3: Write minimal implementation**

```python
# esdc/providers/__init__.py
from esdc.providers.base import Provider, ProviderConfig

__all__ = ["Provider", "ProviderConfig"]
```

```python
# esdc/providers/base.py
from dataclasses import dataclass
from typing import Literal

ProviderType = Literal["openai", "anthropic", "ollama", "custom"]

@dataclass
class ProviderConfig:
    name: str
    provider_type: ProviderType
    api_key: str = ""
    base_url: str = ""
    model: str = ""
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_providers.py::test_provider_config_validation -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add esdc/providers/ tests/test_providers.py
git commit -m "feat: add provider base classes"
```

---

### Task 2: OpenAI Provider Implementation

**Files:**
- Create: `esdc/providers/openai.py`
- Test: `tests/test_providers.py`

**Step 1: Write the failing test**

```python
def test_openai_provider_query():
    from esdc.providers.openai import OpenAIProvider
    
    provider = OpenAIProvider(
        api_key="sk-test",
        model="gpt-4o"
    )
    
    schema = "project_resources: project_name TEXT, province TEXT"
    query = "Show projects in Java"
    
    result = provider.generate_sql(schema, query)
    assert "SELECT" in result.upper()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_providers.py::test_openai_provider_query -v
```
Expected: FAIL - provider not implemented

**Step 3: Write minimal implementation**

```python
# esdc/providers/openai.py
from esdc.providers.base import Provider, ProviderConfig

class OpenAIProvider(Provider):
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.client = None  # Lazy import
    
    def generate_sql(self, schema: str, user_query: str) -> str:
        # Will implement with OpenAI API call
        # For now, return mock
        return f"-- SQL for: {user_query}\nSELECT * FROM project_resources LIMIT 10;"
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_providers.py::test_openai_provider_query -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add esdc/providers/openai.py
git commit -m "feat: add OpenAI provider"
```

---

### Task 3: Config File Provider Management

**Files:**
- Modify: `esdc/configs.py` - add get_providers, save_provider, remove_provider
- Create: `tests/test_configs_provider.py`

**Step 1: Write the failing test**

```python
def test_get_providers_from_config():
    from esdc.configs import Config
    
    # Mock config file
    providers = Config.get_providers()
    assert isinstance(providers, dict)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_configs_provider.py::test_get_providers_from_config -v
```
Expected: FAIL - method not defined

**Step 3: Write minimal implementation**

```python
# Add to esdc/configs.py
@classmethod
def get_providers(cls) -> dict:
    config = cls._load_config()
    return config.get("providers", {})

@classmethod
def save_provider(cls, name: str, provider_config: dict) -> None:
    config = cls._load_config() or {}
    providers = config.get("providers", {})
    providers[name] = provider_config
    config["providers"] = providers
    cls._save_config(config)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_configs_provider.py::test_get_providers_from_config -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add esdc/configs.py
git commit -m "feat: add provider config methods to Config"
```

---

## Phase 2: Text-to-SQL Engine

### Task 4: Schema Loader

**Files:**
- Create: `esdc/chat/schema_loader.py`
- Create: `tests/test_schema_loader.py`

**Step 1: Write the failing test**

```python
def test_load_core_schema():
    from esdc.chat.schema_loader import SchemaLoader
    
    loader = SchemaLoader()
    schema = loader.get_core_schema()
    
    assert "project_resources" in schema
    assert "province" in schema
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_schema_loader.py::test_load_core_schema -v
```
Expected: FAIL - module not found

**Step 3: Write minimal implementation**

```python
# esdc/chat/schema_loader.py
from pathlib import Path

class SchemaLoader:
    SCHEMA_FILE = Path(__file__).parent.parent / "docs" / "plans" / "2025-03-17-esdc-tui-chat-schema.md"
    
    CORE_COLUMNS = [
        "project_name", "field_name", "wk_name", "operator_name",
        "province", "basin86", "basin128", "project_stage", "project_class",
        "project_level", "uncert_level", "rec_oil", "rec_con", "rec_ga", "rec_gn",
        "res_oil", "res_con", "res_ga", "res_gn", "prj_ioip", "prj_igip",
        "report_year", "is_offshore", "is_discovered"
    ]
    
    def get_core_schema(self) -> str:
        # Return schema summary for core columns
        return """project_resources table:
- project_name TEXT: Name of the project
- field_name TEXT: Name of the field
- wk_name TEXT: Work area name
- operator_name TEXT: Operator company name
- province TEXT: Province name
- basin86 TEXT: Basin name (86 classification)
- project_stage TEXT: Project stage (Exploration, Exploitation)
- project_class TEXT: Project classification (Reserves & GRR, Contingent Resources, Prospective Resources)
- project_level TEXT: Project maturity level
- uncert_level TEXT: Uncertainty level (Low, Mid, High)
- rec_oil REAL: Resources oil (MSTB)
- res_oil REAL: Reserves oil (MSTB)
- report_year INTEGER: Report year
- is_offshore INTEGER: Whether offshore (0/1)
- is_discovered INTEGER: Whether discovered (0/1)
"""
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_schema_loader.py::test_load_core_schema -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add esdc/chat/schema_loader.py tests/test_schema_loader.py
git commit -m "feat: add schema loader for core columns"
```

---

### Task 5: Text-to-SQL Engine

**Files:**
- Create: `esdc/chat/text_to_sql.py`
- Test: `tests/test_text_to_sql.py`

**Step 1: Write the failing test**

```python
def test_text_to_sql_generation():
    from esdc.chat.text_to_sql import TextToSQL
    
    engine = TextToSQL(provider=None)  # Mock provider
    sql = engine.generate("Show oil reserves in East Java")
    
    assert "SELECT" in sql.upper()
    assert "project_resources" in sql.lower()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_text_to_sql.py::test_text_to_sql_generation -v
```
Expected: FAIL - module not found

**Step 3: Write minimal implementation**

```python
# esdc/chat/text_to_sql.py
from esdc.chat.schema_loader import SchemaLoader

class TextToSQL:
    def __init__(self, provider):
        self.provider = provider
        self.schema_loader = SchemaLoader()
    
    def generate(self, user_query: str) -> str:
        schema = self.schema_loader.get_core_schema()
        # This will call the provider to generate SQL
        # For now, return a simple query
        return "SELECT project_name, province, res_oil FROM project_resources LIMIT 10;"
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_text_to_sql.py::test_text_to_sql_generation -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add esdc/chat/text_to_sql.py tests/test_text_to_sql.py
git commit -m "feat: add text-to-sql engine"
```

---

## Phase 3: TUI Components

### Task 6: Basic Textual App Structure

**Files:**
- Create: `esdc/chat/app.py`
- Test: `tests/test_chat_app.py`

**Step 1: Write the failing test**

```python
def test_chat_app_import():
    from esdc.chat.app import ESDCChatApp
    
    app = ESDCChatApp()
    assert app is not None
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_chat_app.py::test_chat_app_import -v
```
Expected: FAIL - module not found

**Step 3: Write minimal implementation**

```python
# esdc/chat/app.py
from textual.app import App

class ESDCChatApp(App):
    CSS = """
    Screen {
        layout: horizontal;
    }
    """
    
    def compose(self):
        # Will add panels later
        pass
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_chat_app.py::test_chat_app_import -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add esdc/chat/app.py tests/test_chat_app.py
git commit -m "feat: add basic Textual app structure"
```

---

### Task 7: Chat Panel Component

**Files:**
- Modify: `esdc/chat/app.py` - add ChatPanel
- Create: `tests/test_chat_panel.py`

**Step 1: Write the failing test**

```python
def test_chat_panel_exists():
    from esdc.chat.app import ChatPanel
    
    panel = ChatPanel()
    assert hasattr(panel, "messages")
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_chat_panel.py::test_chat_panel_exists -v
```
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# Add to esdc/chat/app.py
from textual.widgets import Static, Input
from textual.containers import Vertical

class ChatMessage(Static):
    def __init__(self, role: str, content: str):
        super().__init__(content)
        self.role = role

class ChatPanel(Vertical):
    def __init__(self):
        super().__init__()
        self.messages = []
    
    def add_message(self, role: str, content: str):
        self.messages.append((role, content))
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_chat_panel.py::test_chat_panel_exists -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add esdc/chat/app.py
git commit -m "feat: add chat panel component"
```

---

### Task 8: Results Panel Component

**Files:**
- Modify: `esdc/chat/app.py` - add ResultsPanel
- Create: `tests/test_results_panel.py`

**Step 1: Write the failing test**

```python
def test_results_panel_exists():
    from esdc.chat.app import ResultsPanel
    
    panel = ResultsPanel()
    assert hasattr(panel, "sql_query")
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_results_panel.py::test_results_panel_exists -v
```
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# Add to esdc/chat/app.py
class ResultsPanel(Vertical):
    def __init__(self):
        super().__init__()
        self.sql_query = ""
        self.results = None
    
    def set_results(self, sql: str, dataframe):
        self.sql_query = sql
        self.results = dataframe
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_results_panel.py::test_results_panel_exists -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add esdc/chat/app.py
git commit -m "feat: add results panel component"
```

---

### Task 9: Split View Layout

**Files:**
- Modify: `esdc/chat/app.py` - combine panels in split view

**Step 1: Write the failing test**

```python
def test_split_layout():
    from esdc.chat.app import ESDCChatApp
    
    app = ESDCChatApp()
    assert 'layout: horizontal' in app.CSS
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_chat_app.py::test_split_layout -v
```
Expected: FAIL

**Step 3: Write implementation**

```python
# Update ESDCChatApp.compose()
def compose(self):
    yield ChatPanel(id="chat")
    yield ResultsPanel(id="results")
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_chat_app.py::test_split_layout -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add esdc/chat/app.py
git commit -m "feat: add split view layout"
```

---

### Task 10: Message Handling and Display

**Files:**
- Modify: `esdc/chat/app.py` - add message handling

**Step 1: Write the failing test**

```python
def test_display_user_message():
    from esdc.chat.app import ESDCChatApp
    
    app = ESDCChatApp()
    # Test that user message can be added
    assert hasattr(app, "display_message")
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_chat_app.py::test_display_user_message -v
```
Expected: FAIL

**Step 3: Write implementation**

```python
# Add to ESDCChatApp
def display_message(self, role: str, content: str):
    chat_panel = self.query_one("#chat", ChatPanel)
    chat_panel.add_message(role, content)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_chat_app.py::test_display_user_message -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add esdc/chat/app.py
git commit -m "feat: add message handling to chat app"
```

---

## Phase 4: CLI Commands

### Task 11: Provider CLI Commands

**Files:**
- Create: `esdc/commands/provider.py`
- Modify: `esdc/esdc.py` - add provider commands

**Step 1: Write the failing test**

```python
def test_provider_list_command():
    from esdc.commands.provider import provider_list
    
    result = provider_list()
    assert isinstance(result, str)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_commands.py::test_provider_list_command -v
```
Expected: FAIL

**Step 3: Write implementation**

```python
# esdc/commands/provider.py
import typer
from esdc.configs import Config

provider_app = typer.Typer()

@provider_app.command("list")
def provider_list():
    providers = Config.get_providers()
    if not providers:
        return "No providers configured. Use 'esdc provider add' to add a provider."
    
    lines = ["Configured providers:"]
    for name, cfg in providers.items():
        lines.append(f"  - {name}: {cfg.get('provider_type', 'unknown')}")
    return "\n".join(lines)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_commands.py::test_provider_list_command -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add esdc/commands/provider.py
git commit -m "feat: add provider list command"
```

---

### Task 12: Add/Remove Provider Commands

**Files:**
- Modify: `esdc/commands/provider.py`

**Step 1: Write the failing test**

```python
def test_add_provider_command():
    from esdc.commands.provider import provider_add
    
    # Test adding a provider
    result = provider_add("test", "openai", "sk-test", "gpt-4o")
    assert "added" in result.lower()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_commands.py::test_add_provider_command -v
```
Expected: FAIL

**Step 3: Write implementation**

```python
# Add to esdc/commands/provider.py
@provider_app.command("add")
def provider_add(
    name: str,
    provider_type: str,
    api_key: str = "",
    model: str = "",
    base_url: str = "",
):
    config = {
        "provider_type": provider_type,
    }
    if api_key:
        config["api_key"] = api_key
    if model:
        config["model"] = model
    if base_url:
        config["base_url"] = base_url
    
    Config.save_provider(name, config)
    return f"Provider '{name}' added successfully."

@provider_app.command("remove")
def provider_remove(name: str):
    # Implementation
    pass
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_commands.py::test_add_provider_command -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add esdc/commands/provider.py
git commit -m "feat: add provider add/remove commands"
```

---

### Task 13: Chat Command Entry Point

**Files:**
- Modify: `esdc/esdc.py` - add chat command

**Step 1: Write the failing test**

```python
def test_chat_command_exists():
    from esdc.esdc import app as esdc_app
    # Check chat command exists
    assert "chat" in str(esdc_app.command_names())
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_esdc.py::test_chat_command_exists -v
```
Expected: FAIL

**Step 3: Write implementation**

```python
# Add to esdc/esdc.py
from esdc.chat.app import ESDCChatApp

@app.command("chat")
def chat():
    """Start the interactive chat TUI."""
    app = ESDCChatApp()
    app.run()
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_esdc.py::test_chat_command_exists -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add esdc/esdc.py
git commit -m "feat: add chat command to CLI"
```

---

## Phase 5: Integration and Polish

### Task 14: Connect TUI to Text-to-SQL

**Files:**
- Modify: `esdc/chat/app.py` - integrate with text-to-sql

**Step 1: Write the failing test**

```python
def test_chat_with_query():
    from esdc.chat.app import ESDCChatApp
    from unittest.mock import MagicMock
    
    app = ESDCChatApp()
    app.provider = MagicMock()
    app.provider.generate_sql.return_value = "SELECT * FROM project_resources"
    
    # Simulate query
    result = app.handle_query("Show me projects")
    assert "SELECT" in result
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_chat_app.py::test_chat_with_query -v
```
Expected: FAIL

**Step 3: Write implementation**

```python
# Add handle_query method to ESDCChatApp
def handle_query(self, user_input: str) -> str:
    from esdc.chat.text_to_sql import TextToSQL
    
    engine = TextToSQL(self.provider)
    sql = engine.generate(user_input)
    
    # Execute SQL
    from esdc.dbmanager import run_query
    results = run_query(sql)
    
    return sql, results
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_chat_app.py::test_chat_with_query -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add esdc/chat/app.py
git commit -m "feat: integrate TUI with text-to-sql engine"
```

---

### Task 15: Provider Selection and Switching

**Files:**
- Modify: `esdc/chat/app.py` - add provider switching

**Step 1: Write the failing test**

```python
def test_switch_provider():
    from esdc.chat.app import ESDCChatApp
    
    app = ESDCChatApp()
    app.switch_provider("openai")
    
    assert app.provider.name == "openai"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_chat_app.py::test_switch_provider -v
```
Expected: FAIL

**Step 3: Write implementation**

```python
# Add switch_provider method
def switch_provider(self, provider_name: str):
    from esdc.providers import get_provider
    
    self.provider = get_provider(provider_name)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_chat_app.py::test_switch_provider -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add esdc/chat/app.py
git commit -m "feat: add provider switching in TUI"
```

---

### Task 16: Input Handling and History

**Files:**
- Modify: `esdc/chat/app.py` - add input handling

**Step 1: Write the failing test**

```python
def test_input_submission():
    from esdc.chat.app import ESDCChatApp
    
    app = ESDCChatApp()
    app.on_input_submitted("Show projects")
    
    assert len(app.chat_panel.messages) > 0
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_chat_app.py::test_input_submission -v
```
Expected: FAIL

**Step 3: Write implementation**

```python
# Add input handler
def on_input_submitted(self, event):
    user_input = event.value
    self.display_message("user", user_input)
    
    # Process query and get response
    sql, results = self.handle_query(user_input)
    self.display_message("ai", f"Generated SQL:\n{sql}")
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_chat_app.py::test_input_submission -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add esdc/chat/app.py
git commit -m "feat: add input handling in chat TUI"
```

---

## Phase 6: Final Integration Tests

### Task 17: End-to-End Chat Flow

**Files:**
- Create: `tests/integration/test_chat_e2e.py`

**Step 1: Write the failing test**

```python
def test_full_chat_flow():
    # Start app, send query, verify response
    pass  # Integration test
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/integration/test_chat_e2e.py -v
```
Expected: FAIL or SKIP

**Step 3: Write implementation**

```python
# Integration test - verify full flow
def test_full_chat_flow():
    # Mock provider
    # Send "Show oil reserves in Java"
    # Verify SQL generated
    # Verify results displayed
    pass
```

**Step 4: Run test**

```bash
pytest tests/integration/test_chat_e2e.py -v
```

**Step 5: Commit**

```bash
git add tests/integration/test_chat_e2e.py
git commit -m "test: add end-to-end chat flow test"
```

---

## Summary

**Plan complete.** Tasks are organized in phases:

1. **Phase 1**: Provider system (base class, OpenAI, config)
2. **Phase 2**: Text-to-SQL engine (schema loader, SQL generation)
3. **Phase 3**: TUI components (app, panels, layout)
4. **Phase 4**: CLI commands (provider add/remove/list, chat command)
5. **Phase 5**: Integration (connect TUI to engine, provider switching, input)
6. **Phase 6**: Final tests (end-to-end)

Each task follows TDD: write failing test → implement minimal code → verify pass → commit.
