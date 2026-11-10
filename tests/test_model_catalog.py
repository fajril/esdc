# tests/test_model_catalog.py
"""Tests for read-only model discovery in esdc.model_catalog.

Every request here is mocked; no test opens a socket, and every credential is
synthetic.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

import esdc.model_catalog as model_catalog
from esdc.model_catalog import ModelCatalog, ModelEntry, api_endpoint, discover_models

BASE = "http://engine:4000/v1"
MODELS_URL = "http://engine:4000/v1/models"
INFO_URL = "http://engine:4000/model/info"
DISCOVERED_IDS = ["chat-a", "embed-b", "rank-c"]

# Synthetic provider fixture. Its `model` is the saved Chat model and must never
# appear as a discovered entry.
PROVIDER = {
    "name": "engine",
    "provider_type": "openai_compatible",
    "base_url": BASE,
    "api_key": "fixture-key",
    "model": "chat-a",
}


class _FakeResponseError(Exception):
    """Stand-in for ``ollama.ResponseError``: its text is the server error body."""

    def __init__(self, body: str, status_code: int) -> None:
        super().__init__(body)
        self.status_code = status_code


class _SecretError(Exception):
    """An unknown failure whose message holds a synthetic secret."""


class _FakeResponse:
    """Stand-in for a requests.Response, so no connection is ever made."""

    def __init__(self, status: int, payload: Any) -> None:
        self.status_code = status
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.exceptions.HTTPError(f"HTTP {self.status_code}")
            error.response = self
            raise error

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _fake_http(monkeypatch, routes: dict[str, Any]) -> list[dict[str, Any]]:
    """Route mocked GETs by URL and record each call.

    A route value may be ``(status, payload)`` or an exception, which the fake
    transport raises as a network failure.
    """
    calls: list[dict[str, Any]] = []

    def fake_get(url: str, headers: Any = None, timeout: Any = None) -> _FakeResponse:
        calls.append({"url": url, "headers": dict(headers or {}), "timeout": timeout})
        route = routes[url]
        if isinstance(route, Exception):
            raise route
        status, payload = route
        return _FakeResponse(status, payload)

    monkeypatch.setattr(model_catalog.requests, "get", fake_get)
    return calls


def _entry(catalog: ModelCatalog, model_id: str) -> ModelEntry:
    for entry in catalog.entries:
        if entry.model_id == model_id:
            return entry
    raise AssertionError(f"{model_id} is not in the catalog")


class TestApiEndpoint:
    """One terminal /v1, prefix preserved, no /v1/v1."""

    @pytest.mark.parametrize(
        ("base_url", "resource", "root", "expected"),
        [
            ("http://h:4000", "models", False, "http://h:4000/v1/models"),
            ("http://h:4000/", "models", False, "http://h:4000/v1/models"),
            ("http://h:4000/v1", "models", False, "http://h:4000/v1/models"),
            ("http://h:4000/v1/", "models", False, "http://h:4000/v1/models"),
            ("  http://h:4000/v1/  ", "models", False, "http://h:4000/v1/models"),
            ("http://h:4000/proxy/v1", "models", False, "http://h:4000/proxy/v1/models"),
            ("http://h:4000/proxy", "models", False, "http://h:4000/proxy/v1/models"),
            ("http://h:4000/v1", "model/info", True, "http://h:4000/model/info"),
            ("http://h:4000/v1/", "model/info", True, "http://h:4000/model/info"),
            ("http://h:4000", "model/info", True, "http://h:4000/model/info"),
            ("http://h:4000/proxy/v1", "model/info", True, "http://h:4000/proxy/model/info"),
            ("http://h:4000/proxy", "model/info", True, "http://h:4000/proxy/model/info"),
            ("http://h:4000/v1", "/rerank", False, "http://h:4000/v1/rerank"),
        ],
    )
    def test_url_forms(self, base_url, resource, root, expected):
        assert api_endpoint(base_url, resource, root=root) == expected

    def test_plain_base_never_produces_double_v1(self):
        assert api_endpoint("http://h:4000", "models").count("/v1") == 1

    def test_empty_base_is_a_caller_error(self):
        with pytest.raises(ValueError, match="base URL"):
            api_endpoint("   ", "models")


class TestOpenAICompatibleDiscovery:
    """Standard model IDs plus best-effort role metadata."""

    def _routes(self, monkeypatch, metadata=None, ids=None):
        routes = {
            MODELS_URL: (200, {"data": [{"id": model_id} for model_id in ids or DISCOVERED_IDS]}),
            INFO_URL: (200, metadata if metadata is not None else {"data": []}),
        }
        return _fake_http(monkeypatch, routes)

    def test_modes_and_vision_from_metadata(self, monkeypatch):
        metadata = {
            "data": [
                {
                    "model_name": "chat-a",
                    "id": "alias-1",
                    "model_info": {"mode": "chat", "supports_vision": True},
                },
                {
                    "model_name": "embed-b",
                    "id": "alias-2",
                    "model_info": {"mode": "embedding", "supports_vision": False},
                },
                {"model_name": "rank-c", "model_info": {}},
            ]
        }
        self._routes(monkeypatch, metadata)
        catalog = discover_models(PROVIDER)

        assert catalog.warning is None
        assert [entry.model_id for entry in catalog.entries] == DISCOVERED_IDS
        assert _entry(catalog, "chat-a").roles == frozenset({"chat"})
        assert _entry(catalog, "embed-b").roles == frozenset({"embedding"})
        # No mode means unknown, not chat.
        assert _entry(catalog, "rank-c").roles == frozenset()
        assert _entry(catalog, "rank-c").supports_vision is None

    def test_explicit_rerank_mode_adds_rerank(self, monkeypatch):
        self._routes(
            monkeypatch,
            {"data": [{"model_name": "rank-c", "model_info": {"mode": "  RERANK "}}]},
        )
        catalog = discover_models(PROVIDER)
        assert _entry(catalog, "rank-c").roles == frozenset({"rerank"})

    def test_missing_mode_is_unknown_and_never_chat(self, monkeypatch):
        self._routes(
            monkeypatch,
            {"data": [{"model_name": "embed-b", "model_info": {"mode": None}}]},
        )
        catalog = discover_models(PROVIDER)
        assert _entry(catalog, "embed-b").roles == frozenset()
        assert _entry(catalog, "chat-a").roles == frozenset()

    def test_non_string_mode_is_unknown(self, monkeypatch):
        self._routes(
            monkeypatch,
            {"data": [{"model_name": "chat-a", "model_info": {"mode": 7}}]},
        )
        catalog = discover_models(PROVIDER)
        assert _entry(catalog, "chat-a").roles == frozenset()

    def test_vision_only_from_a_json_boolean(self, monkeypatch):
        self._routes(
            monkeypatch,
            {
                "data": [
                    {"model_name": "chat-a", "model_info": {"supports_vision": True}},
                    {"model_name": "embed-b", "model_info": {"supports_vision": False}},
                    {"model_name": "rank-c", "model_info": {"supports_vision": "true"}},
                ]
            },
        )
        catalog = discover_models(PROVIDER)
        assert _entry(catalog, "chat-a").supports_vision is True
        assert _entry(catalog, "embed-b").supports_vision is False
        # A string value is unknown, not a refusal.
        assert _entry(catalog, "rank-c").supports_vision is None

    def test_vision_is_never_inferred_from_name_or_mode(self, monkeypatch):
        """A model named like a vision model stays unknown without explicit metadata."""
        self._routes(
            monkeypatch,
            {"data": [{"model_name": "vision-model", "model_info": {"mode": "chat"}}]},
            ids=["vision-model"],
        )
        catalog = discover_models(PROVIDER)
        entry = _entry(catalog, "vision-model")
        assert entry.roles == frozenset({"chat"})
        assert entry.supports_vision is None

    def test_chat_mode_does_not_mean_vision(self, monkeypatch):
        self._routes(
            monkeypatch,
            {"data": [{"model_name": "chat-a", "model_info": {"mode": "chat"}}]},
        )
        catalog = discover_models(PROVIDER)
        assert _entry(catalog, "chat-a").supports_vision is None

    def test_match_by_model_name_then_id(self, monkeypatch):
        self._routes(
            monkeypatch,
            {
                "data": [
                    {
                        "model_name": "embed-b",
                        "id": "other",
                        "model_info": {"mode": "embedding"},
                    },
                    {"id": "chat-a", "model_info": {"mode": "chat"}},
                    {"model_name": "unmatched-alias", "model_info": {"mode": "rerank"}},
                ]
            },
        )
        catalog = discover_models(PROVIDER)
        assert _entry(catalog, "embed-b").roles == frozenset({"embedding"})
        assert _entry(catalog, "chat-a").roles == frozenset({"chat"})
        # An unmatched metadata entry is ignored, never added as an entry.
        assert [entry.model_id for entry in catalog.entries] == DISCOVERED_IDS
        assert catalog.warning is None

    def test_metadata_403_keeps_the_standard_list(self, monkeypatch):
        routes = {
            MODELS_URL: (200, {"data": [{"id": model_id} for model_id in DISCOVERED_IDS]}),
            INFO_URL: (403, {}),
        }
        calls = _fake_http(monkeypatch, routes)
        catalog = discover_models(PROVIDER)

        assert [entry.model_id for entry in catalog.entries] == DISCOVERED_IDS
        assert all(entry.roles == frozenset() for entry in catalog.entries)
        assert catalog.warning == (
            "Role metadata unavailable (HTTP 403); models are unverified."
        )
        assert calls[1]["url"] == INFO_URL

    def test_metadata_404_and_timeout_keep_the_standard_list(self, monkeypatch):
        for route, phrase in [
            ((404, {}), "HTTP 404"),
            (requests.exceptions.Timeout("read timed out"), "timeout"),
        ]:
            routes = {MODELS_URL: (200, {"data": [{"id": "chat-a"}]}), INFO_URL: route}
            _fake_http(monkeypatch, routes)
            catalog = discover_models(PROVIDER)

            assert [entry.model_id for entry in catalog.entries] == ["chat-a"]
            assert catalog.entries[0].roles == frozenset()
            warning = catalog.warning or ""
            assert warning.startswith("Role metadata unavailable")
            assert phrase in warning
            assert "unverified" in warning

    def test_malformed_metadata_warns_but_keeps_ids(self, monkeypatch):
        routes = {
            MODELS_URL: (200, {"data": [{"id": "chat-a"}]}),
            INFO_URL: (200, {"result": [{"model_name": "chat-a", "mode": "chat"}]}),
        }
        _fake_http(monkeypatch, routes)
        catalog = discover_models(PROVIDER)

        assert catalog.entries[0].model_id == "chat-a"
        assert catalog.entries[0].roles == frozenset()
        assert "unexpected response shape" in (catalog.warning or "")

    def test_metadata_json_error_warns(self, monkeypatch):
        routes = {
            MODELS_URL: (200, {"data": [{"id": "chat-a"}]}),
            INFO_URL: (200, ValueError("Unexpected EOF")),
        }
        _fake_http(monkeypatch, routes)
        catalog = discover_models(PROVIDER)

        assert catalog.entries[0].roles == frozenset()
        assert "invalid JSON" in (catalog.warning or "")

    def test_metadata_with_no_usable_entries_warns_nothing(self, monkeypatch):
        self._routes(monkeypatch, {"data": [{"model_name": "", "model_info": {}}]})
        catalog = discover_models(PROVIDER)
        assert [entry.model_id for entry in catalog.entries] == DISCOVERED_IDS
        assert catalog.warning is None

    def test_failed_standard_lookup_returns_warning_catalog(self, monkeypatch):
        for route, phrase in [
            ((500, {}), "HTTP 500"),
            (requests.exceptions.ConnectionError("connection refused"), "connection error"),
            (requests.exceptions.Timeout("connect timed out"), "timeout"),
            (ValueError("not JSON"), "invalid JSON"),
        ]:
            _fake_http(monkeypatch, {MODELS_URL: route, INFO_URL: (200, {"data": []})})
            catalog = discover_models(PROVIDER)

            assert catalog.entries == ()
            warning = catalog.warning or ""
            assert warning.startswith("Model discovery failed")
            assert phrase in warning

    def test_standard_lookup_shape_failure_warns(self, monkeypatch):
        _fake_http(monkeypatch, {MODELS_URL: (200, {"models": [{"id": "chat-a"}]})})
        catalog = discover_models(PROVIDER)
        assert catalog.entries == ()
        assert "unexpected response shape" in (catalog.warning or "")

    def test_top_level_list_and_deduplicate_ids(self, monkeypatch):
        _fake_http(
            monkeypatch,
            {
                MODELS_URL: (
                    200,
                    [
                        {"id": "chat-a"},
                        {"id": "chat-a"},
                        "not-an-object",
                        {"id": ""},
                        {"model_name": "no-id"},
                        {"id": "embed-b"},
                    ],
                ),
                INFO_URL: (200, []),
            },
        )
        catalog = discover_models(PROVIDER)
        assert [entry.model_id for entry in catalog.entries] == ["chat-a", "embed-b"]

    def test_saved_model_is_not_invented_by_discovery(self, monkeypatch):
        """A saved model missing from the list is not added here; the picker shows it."""
        self._routes(
            monkeypatch,
            {"data": [{"model_name": "chat-a", "model_info": {"mode": "chat"}}]},
            ids=["chat-a"],
        )
        catalog = discover_models(PROVIDER)
        assert [entry.model_id for entry in catalog.entries] == ["chat-a"]

    def test_entries_are_frozen_with_frozenset_roles(self, monkeypatch):
        _fake_http(monkeypatch, {MODELS_URL: (200, {"data": [{"id": "chat-a"}]})})
        catalog = discover_models(PROVIDER)
        entry = catalog.entries[0]
        assert isinstance(entry.roles, frozenset)
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(entry, "model_id", "changed")  # noqa: B010

    def test_discovery_never_reads_or_writes_config(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ESDC_CONFIG_DIR", str(tmp_path))
        _fake_http(monkeypatch, {MODELS_URL: (200, {"data": [{"id": "chat-a"}]})})
        catalog = discover_models(PROVIDER)

        assert catalog.entries
        assert not list(tmp_path.iterdir())


class TestCredentialsAndTimeouts:
    """Only the saved connection credential is sent, and only when it exists."""

    def test_bearer_header_when_a_key_exists(self, monkeypatch):
        calls = _fake_http(
            monkeypatch,
            {MODELS_URL: (200, {"data": [{"id": "chat-a"}]}), INFO_URL: (200, {"data": []})},
        )
        discover_models(PROVIDER)
        for call in calls:
            assert call["headers"]["Authorization"] == "Bearer fixture-key"

    def test_no_header_and_no_placeholder_key_when_empty(self, monkeypatch):
        calls = _fake_http(
            monkeypatch,
            {MODELS_URL: (200, {"data": [{"id": "chat-a"}]}), INFO_URL: (200, {"data": []})},
        )
        discover_models({**PROVIDER, "api_key": ""})
        for call in calls:
            assert call["headers"] == {}

    def test_oauth_token_is_used_when_no_api_key(self, monkeypatch):
        calls = _fake_http(
            monkeypatch,
            {MODELS_URL: (200, {"data": [{"id": "chat-a"}]}), INFO_URL: (200, {"data": []})},
        )
        discover_models({"provider_type": "openai", "api_key": "", "oauth": {"access_token": "tok"}})
        assert calls[0]["headers"]["Authorization"] == "Bearer tok"

    def test_timeout_formula(self, monkeypatch):
        calls = _fake_http(
            monkeypatch,
            {MODELS_URL: (200, {"data": [{"id": "chat-a"}]}), INFO_URL: (200, {"data": []})},
        )

        discover_models(PROVIDER)
        assert [call["timeout"] for call in calls] == [(5.0, 8.0), (5.0, 8.0)]

        calls.clear()
        discover_models(PROVIDER, timeout_seconds=3.0)
        assert calls[0]["timeout"] == (3.0, 3.0)

        calls.clear()
        discover_models(PROVIDER, timeout_seconds=0.05)
        assert calls[0]["timeout"] == (0.1, 0.1)

    def test_urls_use_the_saved_address_and_the_root_helper(self, monkeypatch):
        calls = _fake_http(
            monkeypatch,
            {
                "http://h:4000/proxy/v1/models": (200, {"data": [{"id": "chat-a"}]}),
                "http://h:4000/proxy/model/info": (404, {}),
            },
        )
        catalog = discover_models(
            {"provider_type": "openai_compatible", "base_url": "http://h:4000/proxy"}
        )

        assert [call["url"] for call in calls] == [
            "http://h:4000/proxy/v1/models",
            "http://h:4000/proxy/model/info",
        ]
        assert "Role metadata unavailable" in (catalog.warning or "")


class TestProviderTypeRouting:
    """Supported types resolve to their own defaults; others get a warning."""

    def test_openai_default_base(self, monkeypatch):
        from esdc.providers.openai import OpenAIProvider

        assert "api.openai.com" not in OpenAIProvider.BASE_URL
        url = f"{OpenAIProvider.BASE_URL.rstrip('/')}/models"
        calls = _fake_http(
            monkeypatch,
            {url: (200, {"data": [{"id": "chat-a"}]})},
        )
        catalog = discover_models({"provider_type": "openai"})

        assert catalog.entries[0].model_id == "chat-a"
        assert calls[0]["url"] == url

    def test_openai_compatible_saved_base_is_used(self, monkeypatch):
        calls = _fake_http(
            monkeypatch,
            {MODELS_URL: (200, {"data": [{"id": "chat-a"}]}), INFO_URL: (200, {"data": []})},
        )
        discover_models(PROVIDER)
        assert calls[0]["url"] == MODELS_URL

    def test_openai_compatible_without_base_url_warns(self, monkeypatch):
        catalog = discover_models({"provider_type": "openai_compatible", "api_key": "fixture-key"})
        assert catalog.entries == ()
        assert "no base URL" in (catalog.warning or "")

    def test_unknown_provider_types_are_not_guessed(self, monkeypatch):
        catalog = discover_models({"provider_type": "mystery", "model": "unknown"})
        assert catalog.entries == ()
        assert "not supported" in (catalog.warning or "")


class TestProviderSDKDiscovery:
    """Other supported text providers reuse their own list_models method."""

    @pytest.mark.parametrize(
        ("provider_type", "connection", "expected_kwargs"),
        [
            ("anthropic", {"api_key": "fixture-key"}, {"api_key": "fixture-key"}),
            ("google", {"api_key": "fixture-key"}, {"api_key": "fixture-key"}),
            (
                "azure_openai",
                {"base_url": "https://azure.example", "api_key": "fixture-key"},
                {"api_key": "fixture-key", "base_url": "https://azure.example"},
            ),
            ("groq", {"api_key": "fixture-key"}, {"api_key": "fixture-key"}),
            ("deepseek", {"api_key": "fixture-key"}, {"api_key": "fixture-key"}),
            (
                "ollama_cloud",
                {"base_url": "https://ollama.example", "api_key": "fixture-key"},
                {"api_key": "fixture-key", "base_url": "https://ollama.example"},
            ),
            (
                "opencode",
                {"base_url": "https://opencode.example/v1", "api_key": "fixture-key"},
                {
                    "api_key": "fixture-key",
                    "base_url": "https://opencode.example/v1",
                },
            ),
        ],
    )
    def test_supported_text_provider_reuses_its_list_models(
        self, monkeypatch, provider_type, connection, expected_kwargs
    ):
        from esdc.providers import PROVIDER_CLASSES

        provider_class = PROVIDER_CLASSES[provider_type]
        listed = MagicMock(return_value=["model-a", "model-b", "model-a"])
        monkeypatch.setattr(provider_class, "list_models", listed)

        catalog = discover_models({"type": provider_type, **connection})

        listed.assert_called_once_with(**expected_kwargs)
        assert [entry.model_id for entry in catalog.entries] == [
            "model-a",
            "model-b",
        ]
        for entry in catalog.entries:
            assert entry.roles == frozenset()
            assert entry.supports_vision is None
        assert catalog.warning is None

    def test_an_sdk_failure_is_an_unverified_warning_without_the_error_text(
        self, monkeypatch
    ):
        from esdc.providers import PROVIDER_CLASSES

        provider_class = PROVIDER_CLASSES["anthropic"]
        monkeypatch.setattr(
            provider_class,
            "list_models",
            MagicMock(side_effect=RuntimeError("rejected secret sk-synthetic")),
        )

        catalog = discover_models({"type": "anthropic", "api_key": "fixture-key"})

        assert catalog.entries == ()
        warning = catalog.warning or ""
        assert "unavailable" in warning
        assert "unverified" in warning
        assert "secret" not in warning
        assert "sk-synthetic" not in warning
        assert "cannot serve" not in warning

    def test_an_empty_sdk_list_is_an_unverified_warning(self, monkeypatch):
        from esdc.providers import PROVIDER_CLASSES

        monkeypatch.setattr(
            PROVIDER_CLASSES["deepseek"], "list_models", MagicMock(return_value=[])
        )

        catalog = discover_models({"type": "deepseek", "api_key": "fixture-key"})

        assert catalog.entries == ()
        assert "unverified" in (catalog.warning or "")

    def test_an_expired_oauth_token_is_passed_as_a_snapshot(self, monkeypatch):
        from esdc.providers import PROVIDER_CLASSES

        listed = MagicMock(return_value=["claude-test"])
        monkeypatch.setattr(PROVIDER_CLASSES["anthropic"], "list_models", listed)
        provider = {
            "type": "anthropic",
            "oauth": {
                "access_token": "expired-synthetic-token",
                "expires_at": "2000-01-01T00:00:00Z",
            },
        }

        catalog = discover_models(provider)

        listed.assert_called_once_with(api_key="expired-synthetic-token")
        assert [entry.model_id for entry in catalog.entries] == ["claude-test"]

    def test_same_connection_name_with_a_new_url_is_not_cached(self, monkeypatch):
        calls = _fake_http(
            monkeypatch,
            {
                "http://first.example/v1/models": (
                    200,
                    {"data": [{"id": "first-model"}]},
                ),
                "http://second.example/v1/models": (
                    200,
                    {"data": [{"id": "second-model"}]},
                ),
                "http://first.example/model/info": (200, {"data": []}),
                "http://second.example/model/info": (200, {"data": []}),
            },
        )
        provider = {
            "name": "engine",
            "type": "openai_compatible",
            "base_url": "http://first.example/v1",
        }

        first = discover_models(provider)
        provider["base_url"] = "http://second.example/v1"
        second = discover_models(provider)

        assert [entry.model_id for entry in first.entries] == ["first-model"]
        assert [entry.model_id for entry in second.entries] == ["second-model"]
        assert calls[0]["url"] == "http://first.example/v1/models"
        assert calls[-2]["url"] == "http://second.example/v1/models"

    def test_sdk_discovery_does_not_read_or_write_config(
        self, monkeypatch, tmp_path
    ):
        from esdc.configs import Config
        from esdc.providers import PROVIDER_CLASSES

        monkeypatch.setenv("ESDC_CONFIG_DIR", str(tmp_path))
        monkeypatch.setattr(
            PROVIDER_CLASSES["anthropic"],
            "list_models",
            MagicMock(return_value=["claude-test"]),
        )
        before = tmp_path.iterdir()
        with patch.object(
            Config, "_save_config", side_effect=AssertionError("must not write")
        ):
            discover_models({"type": "anthropic", "api_key": "fixture-key"})

        assert list(tmp_path.iterdir()) == list(before)


class TestStandardOpenAIDiscovery:
    """The standard provider's catalog follows the same base as its runtime."""

    def test_standard_openai_catalog_uses_runtime_base_and_models_path(
        self, monkeypatch
    ):
        from esdc.providers.openai import OpenAIProvider

        monkeypatch.setattr(OpenAIProvider, "BASE_URL", "https://runtime.example/prefix")
        calls = _fake_http(
            monkeypatch,
            {"https://runtime.example/prefix/models": (200, {"data": [{"id": "chat-a"}]})},
        )
        provider = {
            "provider_type": "openai",
            "api_key": "fixture-key",
            "base_url": "https://saved.example/other",
        }
        before = dict(provider)

        catalog = discover_models(provider)

        assert [entry.model_id for entry in catalog.entries] == ["chat-a"]
        assert catalog.entries[0].roles == frozenset()
        assert catalog.entries[0].supports_vision is None
        assert len(calls) == 1
        assert calls[0]["url"] == "https://runtime.example/prefix/models"
        assert "api.openai.com" not in calls[0]["url"]
        assert provider == before

    def test_standard_openai_catalog_matches_runtime_when_saved_base_is_present(
        self, monkeypatch
    ):
        from esdc.providers.openai import OpenAIProvider

        monkeypatch.setattr(OpenAIProvider, "BASE_URL", "https://runtime.example/prefix")
        calls = _fake_http(
            monkeypatch,
            {"https://runtime.example/prefix/models": (200, {"data": [{"id": "chat-a"}]})},
        )

        discover_models(
            {
                "provider_type": "openai",
                "base_url": "https://saved.example/custom/v1",
            }
        )

        # The standard provider pins its class runtime base; a custom address is
        # the separate `openai_compatible` provider path.
        assert [call["url"] for call in calls] == [
            "https://runtime.example/prefix/models"
        ]

    @pytest.mark.parametrize(
        ("runtime_base", "expected"),
        [
            ("https://runtime.example/prefix/", "https://runtime.example/prefix/models"),
            (
                "https://runtime.example/prefix/v1",
                "https://runtime.example/prefix/v1/models",
            ),
            ("https://runtime.example/v1/", "https://runtime.example/v1/models"),
        ],
    )
    def test_standard_openai_appends_models_once(
        self, monkeypatch, runtime_base: str, expected: str
    ):
        from esdc.providers.openai import OpenAIProvider

        monkeypatch.setattr(OpenAIProvider, "BASE_URL", runtime_base)
        calls = _fake_http(monkeypatch, {expected: (200, {"data": [{"id": "chat-a"}]})})

        discover_models({"provider_type": "openai"})

        assert [call["url"] for call in calls] == [expected]
        assert calls[0]["url"].count("/models") == 1

    @pytest.mark.parametrize(
        ("provider", "expected_header"),
        [
            ({"provider_type": "openai", "api_key": "sk-synthetic-key"}, "Bearer sk-synthetic-key"),
            (
                {"provider_type": "openai", "oauth": {"access_token": "oauth-synthetic"}},
                "Bearer oauth-synthetic",
            ),
        ],
        ids=["api-key", "oauth"],
    )
    def test_standard_openai_credentials_match_discovery_rules(
        self, monkeypatch, provider: dict[str, Any], expected_header: str
    ):
        from esdc.providers.openai import OpenAIProvider

        url = f"{OpenAIProvider.BASE_URL.rstrip('/')}/models"
        calls = _fake_http(monkeypatch, {url: (200, {"data": [{"id": "chat-a"}]})})

        discover_models(provider)

        assert calls[0]["headers"] == {"Authorization": expected_header}

    def test_standard_openai_failure_warning_hides_secrets_and_urls(
        self, monkeypatch
    ):
        from esdc.providers.openai import OpenAIProvider

        url = f"{OpenAIProvider.BASE_URL.rstrip('/')}/models"
        secret = "sk-synthetic-secret"
        error = requests.exceptions.ConnectionError(
            f"failed {secret} https://user:password@runtime.example/v1/models body"
        )
        calls = _fake_http(monkeypatch, {url: error})

        catalog = discover_models({"provider_type": "openai", "api_key": secret})

        assert calls[0]["url"] == url  # mocked transport, no real socket
        warning = catalog.warning or ""
        assert warning == "Model discovery failed (connection error)."
        for forbidden in (secret, "password", "runtime.example", "body"):
            assert forbidden not in warning


    @pytest.mark.parametrize("type_key", ["provider_type", "type"])
    def test_saved_oauth_wins_when_api_key_is_also_present(
        self, monkeypatch, type_key: str
    ):
        from copy import deepcopy

        from esdc.providers.openai import OpenAIProvider

        provider = {
            type_key: "openai",
            "api_key": "api-key-account-a",
            "oauth": {
                "access_token": "oauth-account-b",
                "expires_at": 4_102_444_800,
            },
        }
        before = deepcopy(provider)
        url = f"{OpenAIProvider.BASE_URL.rstrip('/')}/models"
        calls = _fake_http(
            monkeypatch,
            {url: (200, {"data": [{"id": "oauth-model"}]})},
        )

        catalog = discover_models(provider)

        assert len(calls) == 1
        assert calls[0]["headers"] == {"Authorization": "Bearer oauth-account-b"}
        assert [item.model_id for item in catalog.entries] == ["oauth-model"]
        assert provider == before

    def test_standard_openai_discovery_matches_runtime_oauth_identity(
        self, monkeypatch
    ):
        import esdc.providers.openai as openai_module
        from esdc.providers.base import ProviderConfig
        from esdc.providers.openai import OpenAIProvider

        oauth = {
            "access_token": "oauth-account-b",
            "refresh_token": "refresh-account-b",
            "expires_at": 4_102_444_800,
        }
        recorded: dict[str, Any] = {}

        def fake_chat(**kwargs):
            recorded.update(kwargs)
            return MagicMock()

        monkeypatch.setattr(openai_module, "ChatOpenAI", fake_chat)
        monkeypatch.setattr(
            OpenAIProvider, "get_actual_context_length", lambda *args, **kwargs: 0
        )
        OpenAIProvider.create_llm(
            model="gpt-test",
            api_key="api-key-account-a",
            config=ProviderConfig(
                name="synthetic",
                provider_type="openai",
                oauth=dict(oauth),
            ),
        )
        assert recorded["api_key"] == "oauth-account-b"

        provider = {
            "provider_type": "openai",
            "api_key": "api-key-account-a",
            "oauth": dict(oauth),
        }
        url = f"{OpenAIProvider.BASE_URL.rstrip('/')}/models"
        calls = _fake_http(monkeypatch, {url: (200, {"data": [{"id": "m"}]})})
        discover_models(provider)

        assert calls[0]["headers"] == {
            "Authorization": f"Bearer {recorded['api_key']}"
        }

    @pytest.mark.parametrize(
        ("extra", "expected"),
        [
            (
                {"api_key": "api-key-account-a"},
                {"Authorization": "Bearer api-key-account-a"},
            ),
            (
                {"oauth": {"access_token": "oauth-account-b"}},
                {"Authorization": "Bearer oauth-account-b"},
            ),
            (
                {
                    "api_key": "api-key-account-a",
                    "oauth": {"access_token": "oauth-account-b"},
                },
                {"Authorization": "Bearer oauth-account-b"},
            ),
            (
                {
                    "api_key": "api-key-account-a",
                    "oauth": {"access_token": ""},
                },
                {"Authorization": "Bearer api-key-account-a"},
            ),
            ({}, {}),
            (
                {
                    "api_key": "api-key-account-a",
                    "oauth": {"access_token": "oauth-account-b"},
                },
                {"Authorization": "Bearer oauth-account-b"},
            ),
            (
                {
                    "api_key": "api-key-account-a",
                    "oauth": {
                        "access_token": "oauth-account-b",
                        "expires_at": 1,
                    },
                },
                {"Authorization": "Bearer oauth-account-b"},
            ),
            (
                {
                    "api_key": "api-key-account-a",
                    "oauth": "not-a-mapping",
                },
                {"Authorization": "Bearer api-key-account-a"},
            ),
        ],
        ids=[
            "api-key-only",
            "oauth-only",
            "both-oauth-wins",
            "empty-oauth-token-falls-back",
            "no-credentials",
            "missing-expiry",
            "expired-oauth-still-wins",
            "malformed-oauth-falls-back",
        ],
    )
    def test_standard_openai_auth_selection_matrix(
        self, monkeypatch, extra: dict[str, Any], expected: dict[str, str]
    ):
        from esdc.providers.openai import OpenAIProvider

        url = f"{OpenAIProvider.BASE_URL.rstrip('/')}/models"
        calls = _fake_http(
            monkeypatch, {url: (200, {"data": [{"id": "model-a"}]})}
        )
        provider = {"provider_type": "openai", **extra}

        discover_models(provider)

        assert calls[0]["headers"] == expected

    def test_openai_compatible_both_credentials_keep_api_key_precedence(
        self, monkeypatch
    ):
        provider = {
            "provider_type": "openai_compatible",
            "base_url": BASE,
            "api_key": "api-key-account-a",
            "oauth": {"access_token": "oauth-account-b"},
        }
        calls = _fake_http(
            monkeypatch,
            {
                MODELS_URL: (200, {"data": [{"id": "compatible-model"}]}),
                INFO_URL: (200, {"data": []}),
            },
        )

        discover_models(provider)

        assert calls
        assert all(
            call["headers"] == {"Authorization": "Bearer api-key-account-a"}
            for call in calls
        )

    def test_standard_openai_401_does_not_retry_with_api_key_or_refresh(
        self, monkeypatch, tmp_path
    ):
        from copy import deepcopy

        import esdc.auth as auth_module
        import esdc.providers.openai as openai_module
        from esdc.providers.openai import OpenAIProvider

        monkeypatch.setenv("ESDC_CONFIG_DIR", str(tmp_path))
        provider = {
            "provider_type": "openai",
            "api_key": "api-key-account-a",
            "oauth": {
                "access_token": "oauth-account-b",
                "refresh_token": "refresh-account-b",
                "expires_at": 4_102_444_800,
            },
        }
        before = deepcopy(provider)
        url = f"{OpenAIProvider.BASE_URL.rstrip('/')}/models"
        calls = _fake_http(monkeypatch, {url: (401, {})})
        refresh = MagicMock(side_effect=AssertionError("must not refresh"))
        persist = MagicMock(side_effect=AssertionError("must not persist"))
        monkeypatch.setattr(auth_module, "refresh_access_token", refresh)
        monkeypatch.setattr(openai_module, "_persist_refreshed_oauth", persist)

        catalog = discover_models(provider)

        assert len(calls) == 1
        assert calls[0]["headers"] == {"Authorization": "Bearer oauth-account-b"}
        assert catalog.entries == ()
        assert catalog.warning == "Model discovery failed (HTTP 401)."
        refresh.assert_not_called()
        persist.assert_not_called()
        assert provider == before
        assert not list(tmp_path.iterdir())
        assert "oauth-account-b" not in (catalog.warning or "")
        assert "api-key-account-a" not in (catalog.warning or "")


class TestOllamaDiscovery:
    """The selected host is the only host used."""

    class _FakeClient:
        def __init__(self, recorded: dict[str, Any], list_result: Any, shows: dict[str, Any]):
            self.recorded = recorded
            self.list_result = list_result
            self.shows = shows

        def __call__(self, host: str | None = None, **kwargs: Any) -> Any:
            self.recorded["hosts"].append(host)
            self.recorded["timeouts"].append(kwargs.get("timeout"))
            return self

        def list(self) -> Any:
            if isinstance(self.list_result, Exception):
                raise self.list_result
            return self.list_result

        def show(self, model: str) -> Any:
            value = self.shows.get(model)
            if isinstance(value, Exception):
                raise value
            return value

    def _client(self, monkeypatch, list_result, shows=None):
        import ollama

        recorded: dict[str, Any] = {"hosts": [], "timeouts": []}
        monkeypatch.setattr(ollama, "Client", self._FakeClient(recorded, list_result, shows or {}))
        return recorded

    def test_non_default_host_reaches_that_host(self, monkeypatch):
        recorded = self._client(monkeypatch, SimpleNamespace(models=[SimpleNamespace(model="glm-ocr")]))
        catalog = discover_models({"provider_type": "ollama", "base_url": "http://gpu-box:11434"})

        assert recorded["hosts"] == ["http://gpu-box:11434"]
        # Client-level bound: the Ollama client takes no per-call timeout.
        assert recorded["timeouts"] == [8.0]
        assert catalog.entries == (ModelEntry("glm-ocr"),)

    def test_client_timeout_uses_the_requested_value(self, monkeypatch):
        recorded = self._client(monkeypatch, SimpleNamespace(models=[SimpleNamespace(model="glm-ocr")]))
        discover_models({"provider_type": "ollama", "base_url": "http://gpu-box:11434"}, timeout_seconds=3.0)
        assert recorded["timeouts"] == [3.0]

    def test_default_host_when_no_base_url(self, monkeypatch):
        recorded = self._client(monkeypatch, SimpleNamespace(models=[SimpleNamespace(model="llama3.2")]))
        catalog = discover_models({"provider_type": "ollama"})

        assert recorded["hosts"] == ["http://localhost:11434"]
        assert catalog.entries[0].model_id == "llama3.2"

    def test_remote_failure_never_calls_the_local_host(self, monkeypatch):
        recorded = self._client(monkeypatch, ConnectionError("remote host unreachable"))
        catalog = discover_models({"provider_type": "ollama", "base_url": "http://gpu-box:11434"})

        assert catalog.entries == ()
        warning = catalog.warning or ""
        assert warning.startswith("Ollama model list failed")
        # Only the failure kind survives: the message that names the remote host is not
        # repeated in the warning.
        assert "ConnectionError" in warning
        assert "remote host unreachable" not in warning
        assert recorded["hosts"] == ["http://gpu-box:11434"]

    def test_response_error_warns_with_the_status_only(self, monkeypatch):
        self._client(
            monkeypatch, _FakeResponseError("SECRET SERVER BODY", status_code=500)
        )
        catalog = discover_models(
            {"provider_type": "ollama", "base_url": "http://gpu-box:11434"}
        )

        assert catalog.entries == ()
        # An Ollama response error carries the server error body as its message, so the
        # warning keeps only the status code.
        assert catalog.warning == (
            "Ollama model list failed (HTTP 500); models are unverified."
        )
        assert "SECRET SERVER BODY" not in (catalog.warning or "")

    def test_unknown_failure_warns_with_the_type_name_only(self, monkeypatch):
        self._client(monkeypatch, _SecretError("SECRET TOKEN leaked in the message"))
        catalog = discover_models(
            {"provider_type": "ollama", "base_url": "http://gpu-box:11434"}
        )

        assert catalog.entries == ()
        # No status code here, so nothing but the exception type name is kept.
        assert catalog.warning == (
            "Ollama model list failed (_SecretError); models are unverified."
        )
        assert "SECRET TOKEN" not in (catalog.warning or "")

    def test_capabilities_true_false_unknown(self, monkeypatch):
        models = [
            SimpleNamespace(model="vision-a"),
            SimpleNamespace(model="chat-b"),
            SimpleNamespace(model="unknown-c"),
            SimpleNamespace(model="malformed-d"),
            SimpleNamespace(model="failed-e"),
        ]
        shows = {
            "vision-a": SimpleNamespace(capabilities=["vision"]),
            "chat-b": SimpleNamespace(capabilities=["completion"]),
            "unknown-c": SimpleNamespace(capabilities=None),
            "malformed-d": {"capabilities": "vision"},
            "failed-e": RuntimeError("show failed"),
        }
        self._client(monkeypatch, SimpleNamespace(models=models), shows)
        catalog = discover_models({"provider_type": "ollama", "base_url": "http://gpu-box:11434"})

        assert _entry(catalog, "vision-a").supports_vision is True
        assert _entry(catalog, "chat-b").supports_vision is False
        assert _entry(catalog, "unknown-c").supports_vision is None
        assert _entry(catalog, "malformed-d").supports_vision is None
        # One failed show does not fail the catalog; the other entries survive.
        assert _entry(catalog, "failed-e").supports_vision is None
        assert len(catalog.entries) == 5
        assert catalog.warning is None

    def test_dict_shapes_are_tolerated(self, monkeypatch):
        recorded = self._client(
            monkeypatch,
            {"models": [{"model": "glm-ocr"}, {"name": "llama3.2"}]},
            {"glm-ocr": {"capabilities": ["vision"]}},
        )
        catalog = discover_models({"provider_type": "ollama", "base_url": "http://gpu-box:11434"})

        assert [entry.model_id for entry in catalog.entries] == ["glm-ocr", "llama3.2"]
        assert _entry(catalog, "glm-ocr").supports_vision is True
        assert recorded["hosts"] == ["http://gpu-box:11434"]

    def test_ollama_roles_stay_unknown(self, monkeypatch):
        self._client(
            monkeypatch,
            SimpleNamespace(models=[SimpleNamespace(model="embed-tag")]),
            {"embed-tag": SimpleNamespace(capabilities=["embedding"])},
        )
        catalog = discover_models({"provider_type": "ollama", "base_url": "http://gpu-box:11434"})
        assert catalog.entries[0].roles == frozenset()

    def test_ollama_discovery_never_writes_config(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ESDC_CONFIG_DIR", str(tmp_path))
        self._client(monkeypatch, SimpleNamespace(models=[SimpleNamespace(model="glm-ocr")]))
        catalog = discover_models({"provider_type": "ollama", "base_url": "http://gpu-box:11434"})

        assert catalog.entries
        assert not list(tmp_path.iterdir())

    @pytest.mark.parametrize("host", ["http://[", "http://localhost:bad"])
    def test_malformed_ollama_host_returns_unverified_catalog(
        self, monkeypatch, host: str
    ):
        import ollama

        def blocked(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("no list/show call may run for an invalid host")

        # Real client construction; list/show are blocked so no network is possible.
        monkeypatch.setattr(ollama.Client, "list", blocked)
        monkeypatch.setattr(ollama.Client, "show", blocked)

        catalog = discover_models({"provider_type": "ollama", "base_url": host})

        assert catalog.entries == ()
        warning = catalog.warning or ""
        assert "construction failed" in warning
        assert "localhost:11434" not in warning

    def test_ollama_constructor_failure_does_not_expose_secrets(
        self, monkeypatch
    ):
        import ollama

        secret = "sk-synthetic-ollama-secret"

        class _BoomClient:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError(
                    f"failed {secret} https://user:password@host/v1 body"
                )

        monkeypatch.setattr(ollama, "Client", _BoomClient)

        catalog = discover_models(
            {"provider_type": "ollama", "base_url": "http://gpu-box:11434"}
        )

        assert catalog.entries == ()
        warning = catalog.warning or ""
        assert "construction failed" in warning
        for forbidden in (
            secret,
            "password",
            "gpu-box",
            "body",
            "RuntimeError",
        ):
            assert forbidden not in warning
