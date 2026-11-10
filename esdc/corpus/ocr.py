"""Scanned-page OCR via the assigned image-chat connection.

temperature=0 and a verbatim-numbers prompt minimize hallucination on official
documents. num_ctx must be large (16384): glm-ocr fails on page images at
Ollama's default 2048. Ollama speaks its native API for vision requests;
OpenAI-compatible connections use the existing provider image adapter.

`create_ocr_client()` is the single factory the pipeline and rename use. It
reads the resolved OCR task only: the assigned connection and explicit OCR
model. An unusable or unset selection comes back as `UnavailableOcr` rather
than an exception at construction.
"""

import logging
from collections.abc import Mapping
from typing import Any, Protocol

import httpx
import ollama

from esdc.configs import Config
from esdc.model_runtime import ResolvedTask
from esdc.model_settings import ModelConfigurationRequired

logger = logging.getLogger(__name__)

# ollama-python's own default is timeout=None (httpx timeouts disabled), which
# hangs forever on an unreachable remote host. Always carry a bounded timeout:
# short connect probe for health checks, generous read for slow OCR pages.
DEFAULT_TIMEOUT = httpx.Timeout(300.0, connect=10.0)

OCR_PROMPT = (
    "Convert this scanned Indonesian official document page to clean markdown. "
    "Rules: keep every number, date, document number and name EXACTLY as written "
    "(do not correct or normalize them); render tables as markdown tables; "
    "mark illegible text as [tidak terbaca]; output only the markdown, no commentary."
)


class OcrClient(Protocol):
    """One page image in, model text out, plus a non-mutating availability probe."""

    def query_image(self, png_bytes: bytes, prompt: str) -> str: ...

    def ocr_page(self, png_bytes: bytes) -> str: ...

    def health_check(self) -> bool: ...


class OllamaVisionOcr:
    """OCR client for a local Ollama vision model (default glm-ocr via config)."""

    def __init__(
        self,
        model: str,
        client: ollama.Client | None = None,
        num_ctx: int = 16384,
        host: str | None = None,
        timeout: httpx.Timeout | float | None = None,
    ) -> None:
        self.model = model
        self._client = client or ollama.Client(
            host=host, timeout=DEFAULT_TIMEOUT if timeout is None else timeout
        )
        self._options = {"temperature": 0, "num_ctx": num_ctx}

    def query_image(self, png_bytes: bytes, prompt: str) -> str:
        """Send one page image + prompt, return raw model output."""
        response = self._client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt, "images": [png_bytes]}],
            options=self._options,
        )
        return response["message"]["content"]

    def ocr_page(self, png_bytes: bytes) -> str:
        """OCR a page with the default OCR_PROMPT."""
        return self.query_image(png_bytes, OCR_PROMPT)

    def health_check(self) -> bool:
        """True when Ollama is up and the model is available."""
        try:
            self._client.show(self.model)
            return True
        except Exception as e:
            logger.warning("[Corpus] OCR model unavailable: %s", e)
            return False


class UnavailableOcr:
    """An OCR selection that cannot be used, reported instead of raised.

    `health_check()` is False so the pipeline skips OCR the same way it skips an
    unhealthy client; a request raises the stored reason.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def query_image(self, png_bytes: bytes, prompt: str) -> str:
        raise RuntimeError(self.reason)

    def ocr_page(self, png_bytes: bytes) -> str:
        raise RuntimeError(self.reason)

    def health_check(self) -> bool:
        return False


def create_ocr_client(
    config: Mapping[str, Any],
    *,
    resolved: ResolvedTask | None = None,
    timeout_seconds: float | None = None,
) -> OcrClient:
    """Build the OCR client from the resolved OCR task, best effort.

    Processing limits (``extract_timeout_seconds``, ``num_ctx``) still come
    from the corpus config. The connection address, credential and wire model
    come only from ``models.ocr``; old ``ocr_model``/``ocr_provider``/
    ``ollama_host`` keys are never interpreted here. An unset task, a missing
    connection, an unsupported type or an empty model comes back as
    `UnavailableOcr`: this factory never raises for an invalid or inactive
    selection.
    """
    read_timeout = float(
        timeout_seconds
        if timeout_seconds is not None
        else config.get("extract_timeout_seconds") or 300
    )
    timeout = httpx.Timeout(read_timeout, connect=10.0)
    num_ctx = config.get("num_ctx", 16384)

    if resolved is None:
        try:
            resolved = Config.resolve_model_task("ocr")
        except ModelConfigurationRequired as exc:
            return UnavailableOcr(str(exc))

    if resolved.mode != "model" or resolved.primary is None:
        return UnavailableOcr(
            "OCR is not configured. Choose a connection and model in the OCR task."
        )

    provider = resolved.primary.provider
    model = resolved.primary.model
    if not model:
        return UnavailableOcr("OCR model is not configured")

    provider_type = str(
        provider.get("provider_type") or provider.get("type") or ""
    ).strip().lower()

    if provider_type == "ollama":
        # Only the assigned connection's own address is used. An empty address
        # means the Ollama client default, never a stale corpus host.
        host = str(provider.get("base_url") or "").strip() or None
        return OllamaVisionOcr(
            model,
            num_ctx=num_ctx,
            host=host,
            timeout=timeout,
        )

    if provider_type == "openai_compatible":
        base_url = str(provider.get("base_url") or "").strip()
        if not base_url:
            return UnavailableOcr(
                f"OCR connection {resolved.primary.connection!r} has no base URL"
            )

        # Imported here so the two modules can share OCR_PROMPT without a cycle.
        from esdc.corpus.provider_ocr import OpenAICompatibleOcr

        return OpenAICompatibleOcr(provider, model, timeout_seconds=read_timeout)

    name = provider_type or "unknown"
    return UnavailableOcr(f"OCR is not supported for provider type {name}")
