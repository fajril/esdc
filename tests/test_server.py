"""Tests for ESDC server API."""

# Standard library
import json

# Third-party
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

# Local
from esdc.server.app import create_app
from esdc.server.models import Message


@pytest.fixture
def client():
    """Create FastAPI test client."""
    app = create_app()
    return TestClient(app)


class TestServerEndpoints:
    """Test suite for server endpoints."""

    def test_list_models(self, client):
        """Test GET /v1/models endpoint."""
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "esdc-agent"
        assert data["data"][0]["owned_by"] == "esdc"

    def test_docs_endpoint(self, client):
        """Test API documentation endpoint."""
        response = client.get("/docs")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestServerConfiguration:
    """Test server configuration and setup."""

    def test_app_creation(self):
        """Test FastAPI app creation."""
        app = create_app()

        assert app.title == "ESDC API"
        assert app.version == "0.4.0"

    def test_cors_middleware_configured(self):
        """Test CORS middleware is configured."""
        from fastapi.middleware.cors import CORSMiddleware

        app = create_app()

        # Check if CORS middleware is in the stack
        has_cors = any(
            isinstance(middleware.cls, type)
            and issubclass(middleware.cls, CORSMiddleware)
            for middleware in app.user_middleware
        )
        assert has_cors


class TestServerErrorHandling:
    """Test error handling scenarios."""

    def test_404_response(self, client):
        """Test 404 for non-existent endpoint."""
        response = client.get("/nonexistent")

        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """Test method not allowed."""
        response = client.put("/v1/models")

        assert response.status_code == 405


class TestChatCompletionsValidation:
    """Test chat completions request validation."""

    def test_invalid_request_body(self, client):
        """Test handling of invalid request body."""
        response = client.post(
            "/v1/chat/completions",
            json={"invalid": "data"},
        )

        assert response.status_code == 422

    def test_missing_messages(self, client):
        """Test request without messages field."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "esdc-agent",
            },
        )

        assert response.status_code == 422


class TestServerModels:
    """Test server models and Pydantic schemas."""

    def test_message_model_creation(self):
        """Test Message model can be created."""
        msg = Message(role="user", content="Hello")

        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_chat_completion_request_model(self):
        """Test ChatCompletionRequest model."""
        from esdc.server.models import ChatCompletionRequest

        request = ChatCompletionRequest(
            model="esdc-agent",
            messages=[Message(role="user", content="Hello")],
            stream=False,
        )

        assert request.model == "esdc-agent"
        assert len(request.messages) == 1
        assert request.stream is False

    def test_chat_completion_response_model(self):
        """Test ChatCompletionResponse model."""
        from esdc.server.models import ChatCompletionResponse, Choice

        response = ChatCompletionResponse(
            id="chatcmpl-test123",
            created=1234567890,
            model="esdc-agent",
            choices=[
                Choice(
                    message=Message(role="assistant", content="Hello"),
                    finish_reason="stop",
                )
            ],
        )

        assert response.id == "chatcmpl-test123"
        assert response.model == "esdc-agent"
        assert len(response.choices) == 1
        assert response.choices[0].message.content == "Hello"
