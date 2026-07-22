"""Tests for PKCE generation and OAuth HTTP hygiene."""

import base64
import hashlib
from unittest.mock import patch

import pytest

from esdc.auth.oauth import CallbackHandler, _record_callback, generate_pkce_pair


def test_pkce_challenge_is_local_s256_of_verifier():
    """Challenge must be BASE64URL(SHA256(verifier)) computed locally.

    No network call is allowed: the verifier is a secret.
    """
    with patch("esdc.auth.oauth.requests") as mock_requests:
        verifier, challenge = generate_pkce_pair()

    mock_requests.post.assert_not_called()
    mock_requests.get.assert_not_called()

    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    assert challenge == expected
    assert 43 <= len(verifier) <= 128  # RFC 7636 §4.1


def test_token_requests_have_timeout():
    """exchange_code_for_tokens / refresh_access_token must not hang forever."""
    from esdc.auth import oauth

    with patch("esdc.auth.oauth.requests.post") as mock_post:
        mock_post.return_value.json.return_value = {}
        mock_post.return_value.raise_for_status.return_value = None

        oauth.refresh_access_token("dummy-refresh-token")
        _, kwargs = mock_post.call_args
        assert kwargs.get("timeout"), "refresh_access_token missing timeout"

        oauth.exchange_code_for_tokens("dummy-code", "dummy-verifier")
        _, kwargs = mock_post.call_args
        assert kwargs.get("timeout"), "exchange_code_for_tokens missing timeout"


def test_record_callback_captures_code_and_state():
    CallbackHandler.auth_code = None
    CallbackHandler.state = None
    CallbackHandler.error = None

    _record_callback({"code": ["abc123"], "state": ["xyz789"]})
    assert CallbackHandler.auth_code == "abc123"
    assert CallbackHandler.state == "xyz789"


def test_start_oauth_flow_rejects_state_mismatch(monkeypatch):
    from esdc.auth import oauth

    class _FakeServer:
        def handle_request(self):
            CallbackHandler.auth_code = "attacker-code"
            CallbackHandler.state = "attacker-state"  # != generated state

    monkeypatch.setattr(oauth, "start_callback_server", lambda: _FakeServer())
    monkeypatch.setattr(oauth.webbrowser, "open", lambda url: True)

    with pytest.raises(RuntimeError, match="state"):
        oauth.start_oauth_flow()


def test_error_page_escapes_html():
    """Reflected error param must be HTML-escaped (XSS)."""
    import html as html_mod

    from esdc.auth.oauth import _render_error_page

    page = _render_error_page("<script>alert(1)</script>")
    assert "<script>" not in page
    assert html_mod.escape("<script>alert(1)</script>") in page


def test_tokens_do_not_contain_code_verifier(monkeypatch):
    from esdc.auth import oauth

    class _FakeServer:
        expected_state: str | None = None

        def handle_request(self):
            CallbackHandler.auth_code = "good-code"
            CallbackHandler.state = _FakeServer.expected_state

    real_urlsafe = oauth.secrets.token_urlsafe

    def capture_state(n=16):
        val = real_urlsafe(n)
        if n == 16:
            _FakeServer.expected_state = val
        return val

    monkeypatch.setattr(oauth.secrets, "token_urlsafe", capture_state)
    monkeypatch.setattr(oauth, "start_callback_server", lambda: _FakeServer())
    monkeypatch.setattr(oauth.webbrowser, "open", lambda url: True)
    monkeypatch.setattr(
        oauth,
        "exchange_code_for_tokens",
        lambda code, verifier: {"access_token": "t", "expires_in": 3600},
    )

    tokens = oauth.start_oauth_flow()
    assert "code_verifier" not in tokens
