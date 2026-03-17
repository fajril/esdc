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
