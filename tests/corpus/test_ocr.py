"""Tests for OCR-based document ingestion."""

from typing import cast

import ollama
import pytest

from esdc.corpus.ocr import OCR_PROMPT, OllamaVisionOcr
from esdc.corpus.provider_ocr import OpenAICompatibleOcr
from esdc.model_runtime import ResolvedTarget, ResolvedTask


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
    cfg = {"extract_timeout_seconds": None, "num_ctx": 16384}
    cfg.update(overrides)
    return cfg


def _resolved_ocr(
    provider_name: str | None = "gpu-box", model: str = "glm-ocr"
) -> ResolvedTask:
    if provider_name is None:
        return ResolvedTask(role="ocr", mode="unset", enabled=True)
    entry = SAVED_PROVIDERS[provider_name]
    provider_type = str(entry.get("provider_type") or entry.get("type") or "")
    provider = {
        "name": provider_name,
        "provider_type": provider_type,
        "type": provider_type,
        "base_url": entry.get("base_url", ""),
        "api_key": entry.get("api_key", ""),
        "model": model,
    }
    return ResolvedTask(
        role="ocr",
        mode="model",
        enabled=True,
        primary=ResolvedTarget(
            connection=provider_name,
            provider=provider,
            model=model,
            options={},
        ),
    )


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


def test_factory_ollama_connection_uses_its_assigned_host(monkeypatch):
    import esdc.corpus.ocr as ocr_mod

    inits, chats = _fake_ollama_transport(monkeypatch)
    client = ocr_mod.create_ocr_client(
        _factory_config(), resolved=_resolved_ocr("gpu-box")
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


def test_factory_uses_processing_limits_with_the_assigned_connection(monkeypatch):
    import esdc.corpus.ocr as ocr_mod

    inits, chats = _fake_ollama_transport(monkeypatch)
    client = ocr_mod.create_ocr_client(
        _factory_config(num_ctx=20000, extract_timeout_seconds=120),
        resolved=_resolved_ocr("gpu-box"),
    )

    assert isinstance(client, OllamaVisionOcr)
    client.ocr_page(b"\x89PNG")
    assert inits[0]["host"] == "http://gpu-box:11434"
    timeout = inits[0]["kwargs"]["timeout"]
    assert (timeout.connect, timeout.read) == (10.0, 120.0)
    assert chats[0][0] == "glm-ocr"
    assert chats[0][1] == {"temperature": 0, "num_ctx": 20000}


def test_factory_does_not_borrow_another_source_address(monkeypatch):
    import esdc.corpus.ocr as ocr_mod

    inits, chats = _fake_ollama_transport(monkeypatch)
    first = ocr_mod.create_ocr_client(
        _factory_config(), resolved=_resolved_ocr("gpu-box")
    )
    assert isinstance(first, OllamaVisionOcr)
    first.ocr_page(b"\x89PNG")

    second = ocr_mod.create_ocr_client(
        _factory_config(), resolved=_resolved_ocr("local-daemon")
    )
    assert isinstance(second, OllamaVisionOcr)
    second.ocr_page(b"\x89PNG")

    assert inits[0]["host"] == "http://gpu-box:11434"
    # The second connection has no address; the first connection's host is
    # never substituted.
    assert inits[1]["host"] is None
    assert [chat[0] for chat in chats] == ["glm-ocr", "glm-ocr"]


@pytest.mark.parametrize("blank_address", ["", "   "])
def test_factory_ollama_connection_without_address_uses_client_default(
    monkeypatch, blank_address
):
    import esdc.corpus.ocr as ocr_mod

    providers = {
        **SAVED_PROVIDERS,
        "local-daemon": {**SAVED_PROVIDERS["local-daemon"], "base_url": blank_address},
    }
    entry = providers["local-daemon"]
    resolved = ResolvedTask(
        role="ocr",
        mode="model",
        enabled=True,
        primary=ResolvedTarget(
            connection="local-daemon",
            provider={
                "name": "local-daemon",
                "provider_type": "ollama",
                "type": "ollama",
                "base_url": entry.get("base_url", ""),
            },
            model="glm-ocr",
            options={},
        ),
    )
    inits, chats = _fake_ollama_transport(monkeypatch)
    client = ocr_mod.create_ocr_client(_factory_config(), resolved=resolved)

    assert isinstance(client, OllamaVisionOcr)
    assert client.health_check() is True
    client.ocr_page(b"\x89PNG")
    assert inits[0]["host"] is None
    assert chats[0][0] == "glm-ocr"


def test_factory_openai_compatible_provider_gets_its_own_adapter(monkeypatch):
    import esdc.corpus.ocr as ocr_mod

    client = ocr_mod.create_ocr_client(
        _factory_config(extract_timeout_seconds=45),
        resolved=_resolved_ocr("engine"),
    )

    assert isinstance(client, OpenAICompatibleOcr)
    assert client.model == "glm-ocr"
    assert client.base_url == "http://engine:4000/v1"
    assert client.timeout_seconds == 45.0
    assert client.model != SAVED_PROVIDERS["engine"]["model"]


def test_factory_reads_resolved_config_when_not_injected(monkeypatch):
    import esdc.corpus.ocr as ocr_mod

    monkeypatch.setattr(
        ocr_mod.Config,
        "resolve_model_task",
        classmethod(lambda cls, role: _resolved_ocr("engine")),
    )
    client = ocr_mod.create_ocr_client(_factory_config())
    assert isinstance(client, OpenAICompatibleOcr)


def test_factory_injected_task_does_not_read_global_config(monkeypatch):
    import esdc.corpus.ocr as ocr_mod

    def forbidden(cls, role):
        raise AssertionError("the factory must not read global config")

    monkeypatch.setattr(ocr_mod.Config, "resolve_model_task", classmethod(forbidden))
    client = ocr_mod.create_ocr_client(
        _factory_config(), resolved=_resolved_ocr("engine")
    )
    assert isinstance(client, OpenAICompatibleOcr)


@pytest.mark.parametrize(
    ("resolved", "phrase"),
    [
        (_resolved_ocr(None), "not configured"),
        (
            ResolvedTask(
                role="ocr",
                mode="model",
                enabled=True,
                primary=ResolvedTarget(
                    connection="other",
                    provider={"name": "other", "provider_type": "anthropic", "type": "anthropic"},
                    model="glm-ocr",
                    options={},
                ),
            ),
            "not supported for provider type anthropic",
        ),
        (_resolved_ocr("no-address"), "has no base URL"),
    ],
)
def test_factory_returns_the_unavailable_sentinel(resolved, phrase):
    import esdc.corpus.ocr as ocr_mod

    client = ocr_mod.create_ocr_client(_factory_config(), resolved=resolved)

    assert isinstance(client, ocr_mod.UnavailableOcr)
    assert phrase in client.reason
    assert client.health_check() is False
    with pytest.raises(RuntimeError, match=phrase):
        client.query_image(b"\x89PNG", "prompt")
    with pytest.raises(RuntimeError, match=phrase):
        client.ocr_page(b"\x89PNG")


def test_factory_empty_model_returns_the_sentinel():
    import esdc.corpus.ocr as ocr_mod

    client = ocr_mod.create_ocr_client(
        _factory_config(), resolved=_resolved_ocr("engine", model="")
    )

    assert isinstance(client, ocr_mod.UnavailableOcr)
    assert client.health_check() is False
    with pytest.raises(RuntimeError, match="model is not configured"):
        client.query_image(b"\x89PNG", "prompt")
