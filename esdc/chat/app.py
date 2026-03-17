# esdc/chat/app.py
from typing import Any
from textual.app import App
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Static, Input, Button
from textual.events import Key


class ChatMessage(Static):
    def __init__(self, role: str, content: str):
        super().__init__(content)
        self.role = role


class ChatPanel(Vertical):
    def __init__(self):
        super().__init__()
        self.messages: list[tuple[str, str]] = []

    def add_message(self, role: str, content: str):
        self.messages.append((role, content))
        self.mount(ChatMessage(role, content))


class ResultsPanel(Vertical):
    def __init__(self):
        super().__init__()
        self.sql_content = ""
        self.results_content = ""

    def set_results(self, sql: str, results: str):
        self.sql_content = sql
        self.results_content = results
        self.update_display()

    def update_display(self):
        content = f"[bold]SQL Query:[/bold]\n{self.sql_content}\n\n[bold]Results:[/bold]\n{self.results_content}"
        self.remove_children()
        self.mount(Static(content))


class ESDCChatApp(App):
    CSS = """
    Screen {
        layout: horizontal;
    }
    
    #chat {
        width: 50%;
        border: solid green;
    }
    
    #results {
        width: 50%;
        border: solid blue;
    }
    
    ChatPanel {
        height: 80%;
        overflow-y: auto;
    }
    
    Input {
        dock: bottom;
        height: 3;
    }
    
    ResultsPanel {
        height: 100%;
    }
    """

    def __init__(self):
        super().__init__()
        self.chat_panel: ChatPanel | None = None
        self.results_panel: ResultsPanel | None = None
        self.user_input: Input | None = None

    def compose(self):
        yield Horizontal(
            Vertical(id="chat", classes="panel"),
            Vertical(id="results", classes="panel"),
        )

    def on_mount(self) -> None:
        chat_container = self.query_one("#chat", Vertical)
        results_container = self.query_one("#results", Vertical)

        self.chat_panel = ChatPanel()
        self.results_panel = ResultsPanel()

        chat_container.mount(self.chat_panel)
        results_container.mount(self.results_panel)

        self.user_input = Input(placeholder="Ask about your data...", id="user_input")
        self.chat_panel.mount(self.user_input)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input submission."""
        user_input = event.value.strip()
        if not user_input:
            return

        self.display_message("user", user_input)

        # Clear input
        event.input.value = ""

        # Process query
        sql, results = self.handle_query(user_input)

        # Display results
        self.set_results(sql, results)
        self.display_message("ai", f"Generated SQL:\n{sql}")

    def display_message(self, role: str, content: str) -> None:
        if self.chat_panel:
            self.chat_panel.add_message(role, content)

    def set_results(self, sql: str, results: str) -> None:
        if self.results_panel:
            self.results_panel.set_results(sql, results)

    def handle_query(self, user_input: str) -> tuple[str, str]:
        from esdc.chat.text_to_sql import TextToSQL
        from esdc.providers.openai import OpenAIProvider
        from esdc.configs import Config

        api_key = Config.get_provider_api_key()
        model = Config.get_provider_model()

        provider = OpenAIProvider(api_key=api_key, model=model)
        engine = TextToSQL(provider=provider)
        sql = engine.generate(user_input)

        return sql, "Results will be displayed here"
