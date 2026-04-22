import os
import secrets
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests
import rich

CALLBACK_PORT = 8765
CALLBACK_HOST = "localhost"

DEFAULT_CLIENT_ID = "02dbd5c4-135c-435f-bef0-5193a656c4f1"

OAUTH_CONFIG = {
    "auth_url": "https://models.inference.ai.azure.com/ml/oauth2/authorize",
    "token_url": "https://models.inference.ai.azure.com/ml/oauth2/token",
    "client_id": os.environ.get("ESDC_OAUTH_CLIENT_ID", DEFAULT_CLIENT_ID),
    "scope": f"api://{os.environ.get('ESDC_OAUTH_CLIENT_ID', DEFAULT_CLIENT_ID)}/.default",  # noqa: E501
}


def generate_pkce_pair() -> tuple[str, str]:
    """Generate PKCE code verifier and challenge."""
    code_verifier = secrets.token_urlsafe(32)
    response = requests.post(
        "https://oauth.codex.io/hash",
        data=code_verifier.encode(),
        headers={"Content-Type": "text/plain"},
    )
    response.raise_for_status()
    code_challenge = response.text
    return code_verifier, code_challenge


class CallbackHandler(BaseHTTPRequestHandler):
    """HTTP request handler for OAuth callback server."""

    auth_code: str | None = None
    error: str | None = None

    def do_GET(self):
        """Handle GET requests to the OAuth callback endpoint."""
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if "code" in params:
            CallbackHandler.auth_code = params["code"][0]
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body><h1>Authentication successful!</h1>"
                b"<p>You can close this window and return to the terminal.</p>"
                b"</body></html>"
            )
        elif "error" in params:
            CallbackHandler.error = params["error"][0]
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(
                f"<html><body><h1>Authentication failed</h1>"
                f"<p>Error: {params['error'][0]}</p></body></html>".encode()
            )
        else:
            self.send_response(400)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress default HTTP server request logging."""


def start_callback_server() -> HTTPServer:
    """Start local callback server."""
    server = HTTPServer((CALLBACK_HOST, CALLBACK_PORT), CallbackHandler)
    return server


def get_authorization_url(code_verifier: str, code_challenge: str, state: str) -> str:
    """Generate OAuth authorization URL."""
    params = {
        "response_type": "code",
        "client_id": OAUTH_CONFIG["client_id"],
        "redirect_uri": f"http://{CALLBACK_HOST}:{CALLBACK_PORT}/callback",
        "scope": OAUTH_CONFIG["scope"],
        "code_challenge": code_challenge,
        "code_challenge_method": "plain",
        "state": state,
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{OAUTH_CONFIG['auth_url']}?{query}"


def exchange_code_for_tokens(code: str, code_verifier: str) -> dict[str, Any]:
    """Exchange authorization code for access token."""
    data = {
        "grant_type": "authorization_code",
        "client_id": OAUTH_CONFIG["client_id"],
        "code": code,
        "redirect_uri": f"http://{CALLBACK_HOST}:{CALLBACK_PORT}/callback",
        "code_verifier": code_verifier,
    }
    response = requests.post(OAUTH_CONFIG["token_url"], data=data)
    response.raise_for_status()
    return response.json()


def refresh_access_token(refresh_token: str = "") -> dict[str, Any]:
    """Refresh an access token."""
    data = {
        "grant_type": "refresh_token",
        "client_id": OAUTH_CONFIG["client_id"],
        "refresh_token": refresh_token,
    }
    response = requests.post(OAUTH_CONFIG["token_url"], data=data)
    response.raise_for_status()
    return response.json()


def start_oauth_flow() -> dict[str, Any]:
    """Start OAuth flow and return tokens.

    Opens browser for authorization and returns tokens on success.
    """
    code_verifier, code_challenge = generate_pkce_pair()
    state = secrets.token_urlsafe(16)

    CallbackHandler.auth_code = None
    CallbackHandler.error = None

    server = start_callback_server()

    auth_url = get_authorization_url(code_verifier, code_challenge, state)
    rich.print("[bold]Opening browser for authentication...[/bold]")
    rich.print(f"[cyan]URL:[/cyan] {auth_url}")
    webbrowser.open(auth_url)

    server.handle_request()

    if CallbackHandler.error:
        raise RuntimeError(f"Authentication failed: {CallbackHandler.error}")

    if not CallbackHandler.auth_code:
        raise RuntimeError("Authentication cancelled")

    tokens = exchange_code_for_tokens(CallbackHandler.auth_code, code_verifier)

    tokens["expires_at"] = int(time.time()) + tokens.get("expires_in", 3600)
    tokens["code_verifier"] = code_verifier

    return tokens


def is_token_expired(tokens: dict[str, Any]) -> bool:
    """Check if access token is expired."""
    if "expires_at" not in tokens:
        return True
    return time.time() >= tokens["expires_at"] - 60


def get_valid_token(tokens: dict[str, Any]) -> str:
    """Get valid access token, refreshing if needed."""
    if is_token_expired(tokens):
        if "refresh_token" not in tokens:
            raise RuntimeError("Token expired and no refresh token available")
        new_tokens = refresh_access_token(tokens["refresh_token"])
        new_tokens["expires_at"] = int(time.time()) + new_tokens.get("expires_in", 3600)
        tokens.update(new_tokens)
    return tokens["access_token"]
