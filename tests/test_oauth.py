"""Tests for PKCE generation and OAuth HTTP hygiene."""

import base64
import hashlib
from unittest.mock import patch

from esdc.auth.oauth import generate_pkce_pair


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
