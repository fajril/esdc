"""Tests for route signatures."""


class TestRoutes:
    """Test suite for route signature validation."""

    def test_chat_completions_endpoint_accepts_request_obj(self):
        """Verify the chat_completions endpoint has request_obj parameter."""
        import inspect

        from esdc.server.routes import chat_completions

        sig = inspect.signature(chat_completions)
        params = list(sig.parameters.keys())
        assert "request_obj" in params
