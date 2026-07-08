from esdc.corpus.ocr import OCR_PROMPT, OllamaVisionOcr


class FakeOllamaClient:
    def __init__(self, content="# Surat\nNomor: SRT-001"):
        self.calls = []
        self._content = content

    def chat(self, model, messages, options):
        self.calls.append((model, messages, options))
        return {"message": {"content": self._content}}

    def show(self, model):
        return {"details": {}}


def test_ocr_page_returns_markdown():
    fake = FakeOllamaClient()
    ocr = OllamaVisionOcr(model="glm-ocr", client=fake, num_ctx=16384)
    result = ocr.ocr_page(b"\x89PNG")
    assert "SRT-001" in result
    model, messages, options = fake.calls[0]
    assert model == "glm-ocr"
    assert options == {"temperature": 0, "num_ctx": 16384}
    assert messages[0]["images"] == [b"\x89PNG"]
    assert messages[0]["content"] == OCR_PROMPT


def test_query_image_sends_custom_prompt():
    fake = FakeOllamaClient(content='{"doc_type": "surat"}')
    ocr = OllamaVisionOcr(model="glm-ocr", client=fake)
    result = ocr.query_image(b"\x89PNG", "extract metadata as JSON")
    assert result == '{"doc_type": "surat"}'
    assert fake.calls[0][1][0]["content"] == "extract metadata as JSON"


def test_health_check_true():
    assert OllamaVisionOcr(model="m", client=FakeOllamaClient()).health_check() is True


def test_health_check_false():
    class Broken:
        def show(self, model):
            raise ConnectionError("ollama down")

    assert OllamaVisionOcr(model="m", client=Broken()).health_check() is False
