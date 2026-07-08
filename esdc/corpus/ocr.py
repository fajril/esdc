"""Scanned-page OCR via a local Ollama model (default: glm-ocr).

Same local-only stack as EmbeddingManager — document content never
leaves the machine. temperature=0 and a verbatim-numbers prompt to
minimize hallucination on official documents. num_ctx must be large
(16384): glm-ocr fails on page images at Ollama's default 2048.
The ollama python client talks to Ollama's native API — required,
because Ollama's OpenAI-compatible API is limited for vision requests.
"""

import logging

import ollama

logger = logging.getLogger(__name__)

OCR_PROMPT = (
    "Convert this scanned Indonesian official document page to clean markdown. "
    "Rules: keep every number, date, document number and name EXACTLY as written "
    "(do not correct or normalize them); render tables as markdown tables; "
    "mark illegible text as [tidak terbaca]; output only the markdown, no commentary."
)


class OllamaVisionOcr:
    def __init__(
        self,
        model: str,
        client: ollama.Client | None = None,
        num_ctx: int = 16384,
    ):
        self.model = model
        self._client = client or ollama.Client()
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
        return self.query_image(png_bytes, OCR_PROMPT)

    def health_check(self) -> bool:
        """True when Ollama is up and the model is available."""
        try:
            self._client.show(self.model)
            return True
        except Exception as e:
            logger.warning("[Corpus] OCR model unavailable: %s", e)
            return False
