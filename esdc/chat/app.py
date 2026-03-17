# esdc/chat/app.py
from typing import Any
from textual.app import App
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Static, Input, Button


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


class SQLPanel(Vertical):
    def __init__(self):
        super().__init__()
        self.sql_query = ""
        self.query_result: Any = None

    def set_query(self, sql: str, result: Any):
        self.sql_query = sql
        self.query_result = result


class ResultsPanel(Vertical):
    def __init__(self):
        super().__init__()
        self.sql_content = ""
        self.results_content = ""

    def set_results(self, sql: str, results: str):
        self.sql_content = sql
        self.results_content = results


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

        self.chat_panel.mount(
            Input(placeholder="Ask about your data...", id="user_input")
        )

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
