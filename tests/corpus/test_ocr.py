"""Tests for OCR-based document ingestion."""

from typing import cast

import ollama
import pytest

from esdc.corpus.ocr import OCR_PROMPT, OllamaVisionOcr
from esdc.corpus.provider_ocr import OpenAICompatibleOcr


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
    ocr = OllamaVisionOcr(
        model="glm-ocr", client=cast(ollama.Client, fake), num_ctx=16384
    )
    result = ocr.ocr_page(b"\x89PNG")
    assert "SRT-001" in result
    model, messages, options = fake.calls[0]
    assert model == "glm-ocr"
    assert options == {"temperature": 0, "num_ctx": 16384}
    assert messages[0]["images"] == [b"\x89PNG"]
    assert messages[0]["content"] == OCR_PROMPT


def test_query_image_sends_custom_prompt():
    fake = FakeOllamaClient(content='{"doc_type": "surat"}')
    ocr = OllamaVisionOcr(model="glm-ocr", client=cast(ollama.Client, fake))
    result = ocr.query_image(b"\x89PNG", "extract metadata as JSON")
    assert result == '{"doc_type": "surat"}'
    assert fake.calls[0][1][0]["content"] == "extract metadata as JSON"


def test_health_check_true():
    assert (
        OllamaVisionOcr(
            model="m", client=cast(ollama.Client, FakeOllamaClient())
        ).health_check()
        is True
    )


def test_health_check_false():
    class Broken:
        def show(self, model):
            raise ConnectionError("ollama down")

    assert (
        OllamaVisionOcr(model="m", client=cast(ollama.Client, Broken())).health_check()
        is False
    )


def test_client_gets_bounded_timeout(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, host=None, **kwargs):
            captured["host"] = host
            captured["kwargs"] = kwargs

    import esdc.corpus.ocr as ocr_mod

    monkeypatch.setattr(ocr_mod.ollama, "Client", FakeClient)
    OllamaVisionOcr(model="glm-ocr")
    timeout = captured["kwargs"]["timeout"]
    # ollama's default timeout is None (no timeout -> hang forever on a
    # remote host); the client must always carry a bounded one.
    assert timeout is not None
    assert timeout.connect < timeout.read


def test_remote_host_passed_to_client(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, host=None, **kwargs):
            captured["host"] = host
            captured["kwargs"] = kwargs

    import esdc.corpus.ocr as ocr_mod

    monkeypatch.setattr(ocr_mod.ollama, "Client", FakeClient)
    OllamaVisionOcr(
        model="glm-ocr", host="http://gpu-box:11434"
    )
    assert captured["host"] == "http://gpu-box:11434"
    assert captured["kwargs"]["timeout"] is not None


def test_explicit_timeout_overrides_default(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, host=None, **kwargs):
            captured["kwargs"] = kwargs

    import esdc.corpus.ocr as ocr_mod

    monkeypatch.setattr(ocr_mod.ollama, "Client", FakeClient)
    OllamaVisionOcr(model="glm-ocr", timeout=5.0)
    assert captured["kwargs"]["timeout"] == 5.0


def test_no_host_uses_default_client(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, host=None, **kwargs):
            captured["host"] = host
            captured["kwargs"] = kwargs

    import esdc.corpus.ocr as ocr_mod

    monkeypatch.setattr(ocr_mod.ollama, "Client", FakeClient)
    OllamaVisionOcr(model="glm-ocr")
    assert captured["host"] is None


# --------------------------------------------------------------------------
# create_ocr_client: the single OCR selection factory
# --------------------------------------------------------------------------

SAVED_PROVIDERS = {
    "engine": {
        "name": "engine",
        "provider_type": "openai_compatible",
        "base_url": "http://engine:4000/v1",
        "api_key": "fixture-key",
        "model": "chat-a",  # the saved Chat model, never the OCR model
    },
    "gpu-box": {
        "name": "gpu-box",
        "provider_type": "ollama",
        "base_url": "http://gpu-box:11434",
        "model": "chat-a",
    },
    "local-daemon": {
        "name": "local-daemon",
        "provider_type": "ollama",
        "base_url": "",  # no saved address: the Ollama client default applies
        "model": "chat-a",
    },
    "other": {"provider_type": "anthropic", "model": "claude-x"},
    "no-address": {"provider_type": "openai_compatible", "model": "chat-a"},
}


def _factory_config(**overrides) -> dict:
    cfg = {"ocr_model": "glm-ocr", "ocr_provider": "", "ollama_host": None}
    cfg.update(overrides)
    return cfg


def _fake_ollama_transport(monkeypatch):
    """Record Ollama client construction and chat calls; no socket is opened."""
    import esdc.corpus.ocr as ocr_mod

    inits: list[dict] = []
    chats: list[tuple] = []

    class FakeClient:
        def __init__(self, host=None, **kwargs):
            inits.append({"host": host, "kwargs": kwargs})

        def chat(self, model, messages, options):
            chats.append((model, options))
            return {"message": {"content": "ok"}}

        def show(self, model):
            return {"details": {}}

    monkeypatch.setattr(ocr_mod.ollama, "Client", FakeClient)
    return inits, chats


def test_factory_empty_reference_keeps_the_ollama_client(monkeypatch):
    import esdc.corpus.ocr as ocr_mod

    inits, chats = _fake_ollama_transport(monkeypatch)
    client = ocr_mod.create_ocr_client(
        _factory_config(ollama_host="http://gpu-box:11434"),
        providers=SAVED_PROVIDERS,
    )

    assert isinstance(client, OllamaVisionOcr)
    assert client.health_check() is True
    client.ocr_page(b"\x89PNG")
    assert inits[0]["host"] == "http://gpu-box:11434"
    # the factory always carries a bounded timeout: short connect, full read
    assert (inits[0]["kwargs"]["timeout"].connect, inits[0]["kwargs"]["timeout"].read) == (
        10.0,
        300.0,
    )
    assert chats[0][0] == "glm-ocr"
    assert chats[0][1] == {"temperature": 0, "num_ctx": 16384}


def test_factory_named_ollama_provider_uses_its_own_host(monkeypatch):
    import esdc.corpus.ocr as ocr_mod

    inits, chats = _fake_ollama_transport(monkeypatch)
    client = ocr_mod.create_ocr_client(
        _factory_config(
            ocr_provider="gpu-box",
            num_ctx=20000,
            extract_timeout_seconds=120,
        ),
        providers=SAVED_PROVIDERS,
    )

    assert isinstance(client, OllamaVisionOcr)
    client.ocr_page(b"\x89PNG")
    assert inits[0]["host"] == "http://gpu-box:11434"
    timeout = inits[0]["kwargs"]["timeout"]
    assert (timeout.connect, timeout.read) == (10.0, 120.0)
    # the explicit corpus.ocr_model wins over the provider's chat model
    assert chats[0][0] == "glm-ocr"
    assert chats[0][1] == {"temperature": 0, "num_ctx": 20000}


def test_factory_named_ollama_provider_prefers_its_own_address(monkeypatch):
    import esdc.corpus.ocr as ocr_mod

    legacy = "http://legacy-remote:11434"
    inits, chats = _fake_ollama_transport(monkeypatch)
    client = ocr_mod.create_ocr_client(
        _factory_config(ocr_provider="gpu-box", ollama_host=legacy),
        providers=SAVED_PROVIDERS,
    )

    assert isinstance(client, OllamaVisionOcr)
    client.ocr_page(b"\x89PNG")
    assert inits[0]["host"] == "http://gpu-box:11434"
    assert chats[0][0] == "glm-ocr"


@pytest.mark.parametrize("blank_address", ["", "   "])
def test_factory_named_ollama_provider_without_an_address_uses_the_client_default(
    monkeypatch, blank_address
):
    """A named provider with no saved address is never sent to `corpus.ollama_host`."""
    import esdc.corpus.ocr as ocr_mod

    providers = {
        **SAVED_PROVIDERS,
        "local-daemon": {**SAVED_PROVIDERS["local-daemon"], "base_url": blank_address},
    }
    legacy = "http://legacy-remote:11434"
    inits, chats = _fake_ollama_transport(monkeypatch)
    client = ocr_mod.create_ocr_client(
        _factory_config(ocr_provider="local-daemon", ollama_host=legacy),
        providers=providers,
    )

    assert isinstance(client, OllamaVisionOcr)
    assert client.health_check() is True
    client.ocr_page(b"\x89PNG")
    # host=None is the Ollama client default, never the legacy corpus URL.
    assert inits[0]["host"] is None
    assert chats[0][0] == "glm-ocr"


def test_factory_legacy_host_still_applies_to_the_un_referenced_source(monkeypatch):
    """Only an empty `ocr_provider` uses `corpus.ollama_host`."""
    import esdc.corpus.ocr as ocr_mod

    inits, chats = _fake_ollama_transport(monkeypatch)
    legacy = "http://legacy-remote:11434"

    un_referenced = ocr_mod.create_ocr_client(
        _factory_config(ollama_host=legacy), providers=SAVED_PROVIDERS
    )
    assert isinstance(un_referenced, OllamaVisionOcr)
    un_referenced.ocr_page(b"\x89PNG")
    assert inits[0]["host"] == legacy

    named = ocr_mod.create_ocr_client(
        _factory_config(ocr_provider="local-daemon", ollama_host=legacy),
        providers=SAVED_PROVIDERS,
    )
    assert isinstance(named, OllamaVisionOcr)
    named.ocr_page(b"\x89PNG")
    # The named provider names no address, so it uses the client default instead.
    assert inits[1]["host"] is None
    assert [chat[0] for chat in chats] == ["glm-ocr", "glm-ocr"]


def test_factory_openai_compatible_provider_gets_its_own_adapter(monkeypatch):
    import esdc.corpus.ocr as ocr_mod

    client = ocr_mod.create_ocr_client(
        _factory_config(ocr_provider="engine", extract_timeout_seconds=45),
        providers=SAVED_PROVIDERS,
    )

    assert isinstance(client, OpenAICompatibleOcr)
    assert client.model == "glm-ocr"
    assert client.base_url == "http://engine:4000/v1"
    assert client.timeout_seconds == 45.0
    assert client.model != SAVED_PROVIDERS["engine"]["model"]


def test_factory_uses_config_when_no_providers_are_injected(monkeypatch):
    import esdc.corpus.ocr as ocr_mod

    monkeypatch.setattr(
        ocr_mod.Config, "get_providers", classmethod(lambda cls: SAVED_PROVIDERS)
    )
    client = ocr_mod.create_ocr_client(_factory_config(ocr_provider="engine"))
    assert isinstance(client, OpenAICompatibleOcr)


def test_factory_injected_providers_do_not_read_config(monkeypatch):
    import esdc.corpus.ocr as ocr_mod

    def forbidden(cls):
        raise AssertionError("the factory must not read global config")

    monkeypatch.setattr(ocr_mod.Config, "get_providers", classmethod(forbidden))
    client = ocr_mod.create_ocr_client(
        _factory_config(ocr_provider="engine"), providers=SAVED_PROVIDERS
    )
    assert isinstance(client, OpenAICompatibleOcr)


@pytest.mark.parametrize(
    ("provider_name", "phrase"),
    [
        ("nope", "not configured"),
        ("other", "not supported for provider type anthropic"),
        ("no-address", "has no base URL"),
    ],
)
def test_factory_returns_the_unavailable_sentinel(monkeypatch, provider_name, phrase):
    import esdc.corpus.ocr as ocr_mod

    client = ocr_mod.create_ocr_client(
        _factory_config(ocr_provider=provider_name), providers=SAVED_PROVIDERS
    )

    assert isinstance(client, ocr_mod.UnavailableOcr)
    assert phrase in client.reason
    assert client.health_check() is False
    with pytest.raises(RuntimeError, match=phrase):
        client.query_image(b"\x89PNG", "prompt")
    with pytest.raises(RuntimeError, match=phrase):
        client.ocr_page(b"\x89PNG")


def test_factory_empty_model_returns_the_sentinel(monkeypatch):
    import esdc.corpus.ocr as ocr_mod

    client = ocr_mod.create_ocr_client(
        _factory_config(ocr_model="", ocr_provider="engine"), providers=SAVED_PROVIDERS
    )

    assert isinstance(client, ocr_mod.UnavailableOcr)
    assert client.health_check() is False
    with pytest.raises(RuntimeError, match="model is not configured"):
        client.query_image(b"\x89PNG", "prompt")
