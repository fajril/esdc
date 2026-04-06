"""Regression tests for non-streaming response handling.

Tests for KeyError: 'role' issue when using native format with non-streaming requests.
"""

# Third-party
from fastapi.testclient import TestClient

# Local
from esdc.server.app import create_app

client = TestClient(create_app())


def test_nonstreaming_native_format():
    """Test that non-streaming requests with native format don't cause KeyError."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "iris",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
        headers={"Accept": "application/json"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert data["choices"][0]["message"]["content"] is not None


def test_nonstreaming_response_structure():
    """Test that non-streaming response has correct OpenAI-compatible structure."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "iris",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
        headers={"Accept": "application/json"},
    )
    assert response.status_code == 200
    data = response.json()

    # Check OpenAI-compatible response structure
    assert "id" in data
    assert "object" in data
    assert data["object"] == "chat.completion"
    assert "created" in data
    assert "model" in data
    assert "choices" in data
    assert isinstance(data["choices"], list)
    assert len(data["choices"]) > 0

    # Check choice structure
    choice = data["choices"][0]
    assert "index" in choice
    assert "message" in choice
    assert "finish_reason" in choice

    # Check message structure
    message = choice["message"]
    assert "role" in message
    assert message["role"] == "assistant"
    assert "content" in message
