"""Tests for OCR through a saved OpenAI-compatible provider connection.

Every request here is mocked; no test opens a socket, and every credential is
synthetic. The audited gateway is never contacted and no saved secret is used.
Health is probed on `GET /v1/models`, where an absent route (404/405) is accepted
optimistically because model listing is optional on a gateway.
"""

import base64
from typing import Any

import fitz
import pytest
import requests

import esdc.corpus.pipeline as pipeline
import esdc.corpus.provider_ocr as provider_ocr
from esdc.corpus.extractor import ExtractionResult
from esdc.corpus.ocr import OCR_PROMPT
from esdc.corpus.provider_ocr import OpenAICompatibleOcr
from esdc.corpus.sidecar import read_sidecar
from esdc.model_runtime import ResolvedTarget, ResolvedTask

BASE = "http://engine:4000/v1"
CHAT_URL = "http://engine:4000/v1/chat/completions"
MODELS_URL = "http://engine:4000/v1/models"

# Synthetic saved provider. Its `model` is the Chat selection and must never be
# used as the OCR model.
PROVIDER = {
    "name": "engine",
    "provider_type": "openai_compatible",
    "base_url": BASE,
    "api_key": "fixture-key",
    "model": "chat-a",
}

PNG = b"\x89PNG\r\n\x1a\nfake-page"


class _FakeResponse:
    """Stand-in for a requests.Response, so no connection is ever made."""

    def __init__(self, status: int, payload: object) -> None:
        self.status_code = status
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.exceptions.HTTPError(f"HTTP {self.status_code}")
            error.response = self
            raise error

    def json(self) -> object:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _fake_http(monkeypatch, post_route: Any = None, get_route: Any = None) -> dict:
    """Route the mocked POST/GET and record each call."""
    calls = {"post": [], "get": []}

    def fake_post(url, headers=None, json=None, timeout=None):
        calls["post"].append(
            {"url": url, "headers": dict(headers or {}), "json": json, "timeout": timeout}
        )
        if isinstance(post_route, Exception):
            raise post_route
        status, payload = post_route
        return _FakeResponse(status, payload)

    def fake_get(url, headers=None, timeout=None):
        calls["get"].append({"url": url, "headers": dict(headers or {}), "timeout": timeout})
        if isinstance(get_route, Exception):
            raise get_route
        status, payload = get_route
        return _FakeResponse(status, payload)

    monkeypatch.setattr(provider_ocr.requests, "post", fake_post)
    monkeypatch.setattr(provider_ocr.requests, "get", fake_get)
    return calls


def _ocr(**overrides) -> OpenAICompatibleOcr:
    kwargs = {"timeout_seconds": 300.0}
    kwargs.update(overrides)
    return OpenAICompatibleOcr(PROVIDER, "ocr-model", **kwargs)


def test_requires_a_base_url_and_a_model():
    with pytest.raises(ValueError, match="base URL"):
        OpenAICompatibleOcr({"api_key": "fixture-key"}, "ocr-model")
    with pytest.raises(ValueError, match="model"):
        OpenAICompatibleOcr(PROVIDER, "")


def test_request_shape_is_the_chat_completion_contract(monkeypatch):
    calls = _fake_http(monkeypatch, (200, {"choices": [{"message": {"content": "# Page"}}]}))

    assert _ocr().query_image(PNG, "extract metadata as JSON") == "# Page"

    call = calls["post"][0]
    assert call["url"] == CHAT_URL
    assert call["json"]["model"] == "ocr-model"
    assert call["json"]["temperature"] == 0

    parts = call["json"]["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "extract metadata as JSON"}
    expected_url = f"data:image/png;base64,{base64.b64encode(PNG).decode('ascii')}"
    assert parts[1] == {"type": "image_url", "image_url": {"url": expected_url}}

    # No Ollama-only options reach a gateway.
    assert "options" not in call["json"]
    assert "num_ctx" not in call["json"]
    assert call["headers"]["Content-Type"] == "application/json"
    assert call["headers"]["Authorization"] == "Bearer fixture-key"
    assert call["timeout"] == (10.0, 300.0)


def test_timeout_follows_the_requested_read_timeout(monkeypatch):
    calls = _fake_http(monkeypatch, (200, {"choices": [{"message": {"content": "ok"}}]}))
    _ocr(timeout_seconds=45.0).query_image(PNG, "prompt")
    assert calls["post"][0]["timeout"] == (10.0, 45.0)

    _ocr(timeout_seconds=0.05).query_image(PNG, "prompt")
    assert calls["post"][1]["timeout"] == (10.0, 0.1)


def test_no_authorization_header_when_the_credential_is_empty(monkeypatch):
    calls = _fake_http(monkeypatch, (200, {"choices": [{"message": {"content": "ok"}}]}))
    _ocr().query_image(PNG, "prompt")

    provider = {**PROVIDER, "api_key": ""}
    OpenAICompatibleOcr(provider, "ocr-model").query_image(PNG, "prompt")
    assert calls["post"][1]["headers"] == {"Content-Type": "application/json"}


def test_oauth_token_is_used_when_no_api_key(monkeypatch):
    calls = _fake_http(monkeypatch, (200, {"choices": [{"message": {"content": "ok"}}]}))
    provider = {**PROVIDER, "api_key": "", "oauth": {"access_token": "tok"}}
    OpenAICompatibleOcr(provider, "ocr-model").query_image(PNG, "prompt")
    assert calls["post"][0]["headers"]["Authorization"] == "Bearer tok"


def test_saved_address_prefix_is_preserved(monkeypatch):
    calls = _fake_http(monkeypatch, (200, {"choices": [{"message": {"content": "ok"}}]}))
    OpenAICompatibleOcr(
        {"base_url": "http://engine:4000/proxy/v1", "api_key": "k"}, "ocr-model"
    ).query_image(PNG, "prompt")
    assert calls["post"][0]["url"] == "http://engine:4000/proxy/v1/chat/completions"


def test_ocr_page_uses_the_default_prompt(monkeypatch):
    calls = _fake_http(monkeypatch, (200, {"choices": [{"message": {"content": "markdown"}}]}))
    assert _ocr().ocr_page(PNG) == "markdown"
    assert calls["post"][0]["json"]["messages"][0]["content"][0]["text"] == OCR_PROMPT


def test_one_attempt_with_no_retry(monkeypatch):
    calls = _fake_http(
        monkeypatch, requests.exceptions.ConnectionError("connection refused")
    )
    with pytest.raises(RuntimeError):
        _ocr().query_image(PNG, "prompt")
    assert len(calls["post"]) == 1


@pytest.mark.parametrize(
    ("route", "phrase"),
    [
        ((500, {}), "HTTP 500"),
        ((401, {}), "HTTP 401"),
        (requests.exceptions.Timeout("read timed out"), "timeout"),
        (requests.exceptions.ConnectionError("connection refused"), "connection error"),
        ((200, ValueError("Unexpected EOF")), "invalid response"),
        ((200, {"results": ["no choices"]}), "unexpected response shape"),
        ((200, {"choices": [{"message": {"content": ""}}]}), "empty response"),
        ((200, {"choices": [{"message": {"content": None}}]}), "unexpected response shape"),
    ],
)
def test_failures_are_runtime_errors_with_short_reasons(monkeypatch, route, phrase):
    calls = _fake_http(monkeypatch, route)
    with pytest.raises(RuntimeError) as exc_info:
        _ocr().query_image(PNG, "prompt")

    assert phrase in str(exc_info.value)
    # no credential, and no response body, reaches the message
    assert "fixture-key" not in str(exc_info.value)
    assert len(calls["post"]) == 1


def test_health_check_accepts_any_2xx_without_the_model(monkeypatch):
    calls = _fake_http(monkeypatch, get_route=(200, {"data": [{"id": "some-other-model"}]}))
    assert _ocr().health_check() is True

    # The selected model is not required to appear, and no image is submitted.
    assert calls["get"][0]["url"] == MODELS_URL
    assert calls["get"][0]["timeout"] == (3.0, 5.0)
    assert calls["get"][0]["headers"]["Authorization"] == "Bearer fixture-key"
    assert calls["post"] == []


@pytest.mark.parametrize(
    "route",
    [
        (401, {}),
        (403, {}),
        (302, {}),
        (418, {}),
        (500, {}),
        requests.exceptions.Timeout("connect timed out"),
        requests.exceptions.ConnectionError("connection refused"),
    ],
)
def test_health_check_reports_unavailable(monkeypatch, route):
    calls = _fake_http(monkeypatch, get_route=route)
    assert _ocr().health_check() is False
    assert len(calls["get"]) == 1
    assert calls["post"] == []


@pytest.mark.parametrize("status", [404, 405])
def test_health_check_accepts_an_absent_model_list_optimistically(monkeypatch, status):
    """A gateway can serve image chat without implementing model listing.

    An absent route is not proof that image chat is unavailable, so these two statuses
    pass the gate. The body here raises when read, so the test also proves the response
    body is never parsed.
    """
    calls = _fake_http(
        monkeypatch,
        get_route=(status, RuntimeError("the health probe must not read the body")),
    )
    assert _ocr().health_check() is True

    # Presence in a list cannot be checked against an absent route, so the selected
    # model is not required here either.
    assert len(calls["get"]) == 1
    assert calls["get"][0]["url"] == MODELS_URL
    assert calls["get"][0]["timeout"] == (3.0, 5.0)
    assert calls["get"][0]["headers"]["Authorization"] == "Bearer fixture-key"
    assert calls["post"] == []


def test_health_check_never_submits_an_image(monkeypatch):
    """Health is a probe: no image payload, no chat request, no model download."""
    calls = _fake_http(monkeypatch, get_route=(404, {}))

    def forbidden(*args, **kwargs):
        raise AssertionError("health_check must not submit an image")

    monkeypatch.setattr(OpenAICompatibleOcr, "query_image", forbidden)
    assert _ocr().health_check() is True
    assert calls["post"] == []


def test_health_check_without_a_credential_sends_no_header(monkeypatch):
    calls = _fake_http(
        monkeypatch, get_route=(200, {"data": []})
    )
    provider = {**PROVIDER, "api_key": ""}
    assert OpenAICompatibleOcr(provider, "ocr-model").health_check() is True
    assert calls["get"][0]["headers"] == {"Content-Type": "application/json"}


def test_gateway_without_model_listing_is_not_disabled_by_the_pipeline_gate(
    tmp_path, monkeypatch
):
    """Factory + pipeline health gate, with every request mocked.

    The gateway answers 404 for `GET /v1/models` and answers the image-chat request.
    `run_extract` applies the health gate, so the factory's client must survive it and
    the first-page image must reach the chat route through that same client, instead of
    OCR being disabled before it is tried.
    """
    calls = _fake_http(
        monkeypatch,
        post_route=(
            200,
            {"choices": [{"message": {"content": '{"doc_type": "letter"}'}}]},
        ),
        get_route=(404, {"data": []}),
    )

    cfg = {
        "chunk_size": 500,
        "chunk_overlap": 50,
        "ocr_dpi": 200,
        "num_ctx": 16384,
        "min_chars_per_page": 50,
        "min_image_area": 0.05,
    }
    monkeypatch.setattr(pipeline.Config, "get_corpus_config", lambda: cfg)

    ocr_task = ResolvedTask(
        role="ocr",
        mode="model",
        enabled=True,
        primary=ResolvedTarget(
            connection="engine",
            provider={**PROVIDER, "model": "ocr-model"},
            model="ocr-model",
            options={},
        ),
    )

    def resolve(role):
        if role == "ocr":
            return ocr_task
        if role == "metadata":
            return ResolvedTask(role="metadata", mode="ocr", enabled=True)
        return ResolvedTask(role=role, mode="off", enabled=True)  # type: ignore[arg-type]

    monkeypatch.setattr(
        pipeline.Config,
        "resolve_model_task",
        classmethod(lambda cls, role: resolve(role)),
    )

    doc = fitz.open()
    doc.new_page()
    pdf = tmp_path / "surat.pdf"
    doc.save(pdf)
    doc.close()

    seen = {}

    def fake_extract(path, ocr_client, min_chars, dpi, min_image_area=0.05):
        seen["ocr_client"] = ocr_client
        return ExtractionResult("body text", "native", 1, 1, 0)

    monkeypatch.setattr(pipeline, "extract_pdf", fake_extract)

    report = pipeline.run_extract([pdf])

    # The gate kept the client: an absent model list is not a reason to disable OCR.
    assert isinstance(seen.get("ocr_client"), OpenAICompatibleOcr)
    assert not any("pre-fill skipped" in warning for warning in report.warnings)

    # One probe, then the image through the same client, and its result stored.
    assert len(calls["get"]) == 1 and calls["get"][0]["url"] == MODELS_URL
    assert len(calls["post"]) == 1 and calls["post"][0]["url"] == CHAT_URL
    meta, _body = read_sidecar(tmp_path / "surat.corpus.md")
    assert meta["doc_type"] == "letter"
    assert report.failed == {}
