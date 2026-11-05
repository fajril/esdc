"""Scanned-page OCR via a local Ollama model (default: glm-ocr).

Same local-only stack as EmbeddingManager — document content never
leaves the machine. temperature=0 and a verbatim-numbers prompt to
minimize hallucination on official documents. num_ctx must be large
(16384): glm-ocr fails on page images at Ollama's default 2048.
The ollama python client talks to Ollama's native API — required,
because Ollama's OpenAI-compatible API is limited for vision requests.

`create_ocr_client()` is the single factory the pipeline and rename use: an
empty `corpus.ocr_provider` keeps this Ollama client, a saved provider selects
its own adapter, and an unusable selection comes back as `UnavailableOcr`
rather than an exception at construction.
"""

import logging
from collections.abc import Mapping
from typing import Any, Protocol

import httpx
import ollama

from esdc.configs import Config

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
    providers: Mapping[str, Mapping[str, Any]] | None = None,
    timeout_seconds: float | None = None,
) -> OcrClient:
    """Build the OCR client for one corpus config, best effort.

    An empty `ocr_provider` keeps the existing local Ollama client and its
    `corpus.ollama_host`. A named provider uses its saved address, credential, and
    the explicit `ocr_model`; the provider's chat model is never used. A named Ollama
    provider with no saved address uses the Ollama client default, not
    `corpus.ollama_host`: that value belongs to the un-referenced source, so reusing
    it could send document images to a different endpoint. A missing provider, an
    unsupported type, or an empty model comes back as `UnavailableOcr`: this factory
    never raises for an invalid or inactive selection.
    """
    read_timeout = float(
        timeout_seconds
        if timeout_seconds is not None
        else config.get("extract_timeout_seconds") or 300
    )
    timeout = httpx.Timeout(read_timeout, connect=10.0)
    num_ctx = config.get("num_ctx", 16384)
    model = str(config.get("ocr_model") or "").strip()
    reference = str(config.get("ocr_provider") or "").strip()

    if not model:
        return UnavailableOcr("OCR model is not configured")

    if not reference:
        return OllamaVisionOcr(
            model,
            num_ctx=num_ctx,
            host=config.get("ollama_host") or None,
            timeout=timeout,
        )

    provider_map = providers if providers is not None else Config.get_providers()
    provider = (
        provider_map.get(reference) if isinstance(provider_map, Mapping) else None
    )
    if not isinstance(provider, Mapping):
        return UnavailableOcr(f"OCR provider '{reference}' is not configured")

    provider_type = str(
        provider.get("provider_type") or provider.get("type") or ""
    ).strip().lower()

    if provider_type == "ollama":
        # A named provider is only its own saved address. With no saved address, the
        # Ollama client default is used; `corpus.ollama_host` is never reused here,
        # because it belongs to the un-referenced source.
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
            return UnavailableOcr(f"OCR provider '{reference}' has no base URL")

        # Imported here so the two modules can share OCR_PROMPT without a cycle.
        from esdc.corpus.provider_ocr import OpenAICompatibleOcr

        return OpenAICompatibleOcr(provider, model, timeout_seconds=read_timeout)

    name = provider_type or "unknown"
    return UnavailableOcr(f"OCR is not supported for provider type {name}")
