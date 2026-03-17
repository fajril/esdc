def test_chat_app_import():
    from esdc.chat.app import ESDCChatApp

    app = ESDCChatApp()
    assert app is not None
