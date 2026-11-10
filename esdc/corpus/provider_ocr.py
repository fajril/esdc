"""OCR through an assigned OpenAI-compatible connection.

One chat request per page image: the text prompt plus a PNG data URL, the OCR
task's model, and the connection credential. Ollama-only options (`num_ctx`,
`options`) are not sent to a gateway.

Vision support is never assumed from a model name: a gateway without reliable
metadata is still used as selected, and a failed request reports OCR
unavailable instead of switching providers. The credential and the response
body never appear in a message or a log line.

The health probe reads the optional model-list route only; an absent route is
not taken as proof that image chat is unavailable.
"""

from __future__ import annotations

import base64
import logging
from collections.abc import Mapping
from typing import Any

import requests

from esdc.corpus.ocr import OCR_PROMPT
from esdc.model_catalog import api_endpoint

logger = logging.getLogger(__name__)

# Request bound: short connect probe, generous read for slow OCR pages.
CONNECT_TIMEOUT_SECONDS = 10.0
MIN_TIMEOUT_SECONDS = 0.1

# Health check bound (D3): 3-second connect, 5-second read.
HEALTH_CONNECT_TIMEOUT_SECONDS = 3.0
HEALTH_READ_TIMEOUT_SECONDS = 5.0

# Model listing is optional on a gateway, so an absent route on the health probe is
# not proof that image chat is unavailable: these two statuses are accepted
# optimistically. Every other status stays unavailable.
HEALTH_OPTIMISTIC_STATUSES = frozenset({404, 405})


class OpenAICompatibleOcr:
    """OCR client for one saved OpenAI-compatible provider and its model."""

    def __init__(
        self,
        provider: Mapping[str, Any],
        model: str,
        *,
        timeout_seconds: float = 300.0,
    ) -> None:
        base_url = str(provider.get("base_url") or "").strip()
        model = str(model or "").strip()
        if not base_url:
            raise ValueError("OpenAICompatibleOcr requires a base URL")
        if not model:
            raise ValueError("OpenAICompatibleOcr requires a model")

        self.provider = provider
        self.model = model
        self.timeout_seconds = float(timeout_seconds)
        self.base_url = base_url

    def query_image(self, png_bytes: bytes, prompt: str) -> str:
        """Send one page image + prompt as an image-chat request, return the text."""
        encoded = base64.b64encode(png_bytes).decode("ascii")
        data_url = f"data:image/png;base64,{encoded}"
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "temperature": 0,
        }

        try:
            response = requests.post(
                api_endpoint(self.base_url, "chat/completions"),
                headers=self._headers(),
                json=body,
                timeout=(
                    CONNECT_TIMEOUT_SECONDS,
                    max(MIN_TIMEOUT_SECONDS, self.timeout_seconds),
                ),
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as error:  # one attempt, no retry
            raise RuntimeError(_failure_reason(error)) from error

        return _first_content(payload)

    def ocr_page(self, png_bytes: bytes) -> str:
        """OCR a page with the default OCR_PROMPT."""
        return self.query_image(png_bytes, OCR_PROMPT)

    def health_check(self) -> bool:
        """True when the connection answers 2xx, or the model-list route is absent.

        Model listing is optional on a gateway: 404 or 405 here means the route is not
        implemented, not that image chat is unavailable, so those two statuses are
        accepted optimistically and the selection is still used. 401/403, every other
        status, a timeout, or a connection failure means unavailable.

        Availability only: the selected model is never required to appear in a list, the
        response body is never read, no image is submitted, and nothing is downloaded.
        """
        try:
            response = requests.get(
                api_endpoint(self.base_url, "models"),
                headers=self._headers(),
                timeout=(HEALTH_CONNECT_TIMEOUT_SECONDS, HEALTH_READ_TIMEOUT_SECONDS),
            )
        except Exception as error:
            logger.warning(
                "[Corpus] OCR provider unavailable: %s", _failure_reason(error)
            )
            return False

        status = _status_code(response)
        if status is not None and 200 <= status < 300:
            return True
        if status in HEALTH_OPTIMISTIC_STATUSES:
            logger.info(
                "[Corpus] OCR model listing is absent (HTTP %s); the selected "
                "connection is used as chosen",
                status,
            )
            return True

        logger.warning("[Corpus] OCR provider unavailable: HTTP %s", status)
        return False

    def _credential(self) -> str:
        """The saved connection credential, or an empty string (no auth header)."""
        oauth = self.provider.get("oauth")
        token = oauth.get("access_token") if isinstance(oauth, Mapping) else ""
        return str(self.provider.get("api_key") or token or "")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        credential = self._credential()
        if credential:
            headers["Authorization"] = f"Bearer {credential}"
        return headers


def _first_content(payload: Any) -> str:
    """`choices[0].message.content`; a bad shape or an empty answer is an error."""
    choices = payload.get("choices") if isinstance(payload, Mapping) else None
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("OCR returned an unexpected response shape")

    first = choices[0]
    message = first.get("message") if isinstance(first, Mapping) else None
    if not isinstance(message, Mapping):
        raise RuntimeError("OCR returned an unexpected response shape")

    content = message.get("content")
    if isinstance(content, str):
        if content:
            return content
        raise RuntimeError("OCR returned an empty response")
    raise RuntimeError("OCR returned an unexpected response shape")


def _status_code(response: Any) -> int | None:
    """The response status as an int, or None when the response carries none."""
    status = getattr(response, "status_code", None)
    if status is None:
        return None
    try:
        return int(status)
    except (TypeError, ValueError):
        return None


def _failure_reason(error: Exception) -> str:
    """Short, credential-free reason for a failed request (no response body)."""
    if isinstance(error, requests.exceptions.HTTPError):
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
        return f"HTTP {status}" if status is not None else "HTTP error"
    if isinstance(error, requests.exceptions.Timeout):
        return "timeout"
    if isinstance(error, requests.exceptions.ConnectionError):
        return "connection error"
    if isinstance(error, ValueError):
        return "invalid response"
    if isinstance(error, requests.exceptions.RequestException):
        return "request failed"

    return type(error).__name__
