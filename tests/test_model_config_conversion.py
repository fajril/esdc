"""Tests for the bounded one-time model-settings converter.

Every fixture is synthetic. The converter is pure, so these tests never touch
the user's configuration, a database, a provider client, or the network.
"""

import copy
from typing import Any

import pytest

from esdc.model_config_conversion import (
    STATIC_PROVIDER_DEFAULTS,
    ModelSettingsConversion,
    convert_model_settings,
)
from esdc.model_settings import (
    LEGACY_MODEL_CORPUS_KEYS,
    LEGACY_MODEL_ROOT_KEYS,
    validate_model_settings,
)

LEGACY_GGUF = "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF"


def provider(provider_type: str = "openai_compatible", **fields: Any) -> dict[str, Any]:
    return {"provider_type": provider_type, **fields}


def legacy_base(
    *,
    providers: dict[str, Any] | None = None,
    default_provider: str | None = None,
    provider_order: list[str] | None = None,
    embedding: dict[str, Any] | None = None,
    corpus: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A saved pre-canonical config with explicit old defaults.

    Defaults deliberately avoid implicit embedding/OCR assignments so a test
    can focus on one task. Tests that want those defaults use a minimal source.
    """
    source: dict[str, Any] = {
        "api_url": "https://legacy.example/",
        "database_path": "/tmp/synthetic-esdc.duckdb",
        "custom_root": {"kept": True},
        "embedding_backend": "local",
        "embedding_host": "",
        "embedding_model": "",
        "embedding_api_key": "",
        "embedding_provider": "",
        "corpus": {
            "chunk_size": 3000,
            "custom_corpus": {"kept": True},
            "rerank": False,
            "rerank_backend": "provider",
            "rerank_provider": "",
            "rerank_model": LEGACY_GGUF,
            "ocr_model": "",
            "ocr_provider": "",
            "metadata_model": "main",
            "metadata_provider": "",
            "cleanup_model": "main",
            "cleanup_provider": "",
            "ollama_host": "",
            "n_gpu_layers": -1,
        },
    }
    if providers is not None:
        source["providers"] = providers
    if default_provider is not None:
        source["default_provider"] = default_provider
    if provider_order is not None:
        source["provider_order"] = provider_order
    if embedding:
        source.update(embedding)
    if corpus:
        source["corpus"].update(corpus)
    return source


def canonical_source() -> dict[str, Any]:
    return {
        "model_settings_version": 1,
        "connections": {},
        "models": {
            "chat": {"mode": "unset"},
            "embedding": {"mode": "unset"},
            "rerank": {"mode": "unset", "enabled": False},
            "ocr": {"mode": "unset"},
            "metadata": {"mode": "chat"},
            "cleanup": {"mode": "chat"},
        },
        "unknown_user_key": "kept",
    }


def assert_no_legacy_model_keys(config: dict[str, Any]) -> None:
    assert LEGACY_MODEL_ROOT_KEYS.isdisjoint(config)
    corpus = config.get("corpus", {})
    assert isinstance(corpus, dict)
    assert LEGACY_MODEL_CORPUS_KEYS.isdisjoint(corpus)


def assert_valid(result: ModelSettingsConversion) -> None:
    assert result.blockers == (), result.blockers
    validate_model_settings(result.config)


def connection_for(
    result: ModelSettingsConversion,
    *,
    connection_type: str,
    base_url: str | None = None,
    api_key: str | None = None,
) -> tuple[str, dict[str, Any]]:
    matches: list[tuple[str, dict[str, Any]]] = []
    for name, connection in result.config["connections"].items():
        if connection.get("type") != connection_type:
            continue
        if base_url is not None and connection.get("base_url") != base_url:
            continue
        if api_key is not None and connection.get("api_key") != api_key:
            continue
        matches.append((name, connection))
    assert len(matches) == 1, matches
    return matches[0]


class TestChatConversion:
    def test_chat_primary_and_fallback_models_and_options_survive(self) -> None:
        source = legacy_base(
            providers={
                "gateway.us": provider(
                    base_url="http://gateway.example/v1",
                    api_key="synthetic-gateway",
                    model="chat-main",
                    reasoning_effort="medium",
                    unknown_provider_key="kept",
                ),
                "desk": provider(
                    "ollama",
                    base_url="http://desk.example:11434",
                    model="chat-fallback",
                    reasoning_effort="high",
                ),
                "unused": provider(
                    "ollama",
                    base_url="http://unused.example:11434",
                    model="unused-model",
                    reasoning_effort="low",
                ),
            },
            default_provider="gateway.us",
            provider_order=["desk", "gateway.us"],
        )

        result = convert_model_settings(source)

        assert result.config["models"]["chat"] == {
            "mode": "model",
            "connection": "gateway.us",
            "model": "chat-main",
            "options": {"reasoning_effort": "medium"},
            "fallbacks": [
                {
                    "connection": "desk",
                    "model": "chat-fallback",
                    "options": {"reasoning_effort": "high"},
                }
            ],
        }
        assert set(result.config["connections"]) == {"gateway.us", "desk", "unused"}
        gateway = result.config["connections"]["gateway.us"]
        assert gateway["type"] == "openai_compatible"
        assert gateway["base_url"] == "http://gateway.example/v1"
        assert gateway["api_key"] == "synthetic-gateway"
        assert gateway["unknown_provider_key"] == "kept"
        assert "model" not in gateway
        assert "reasoning_effort" not in gateway
        warnings = "\n".join(result.warnings)
        assert "unused-model" in warnings
        assert "not assigned" in warnings
        assert_valid(result)
        assert_no_legacy_model_keys(result.config)

    def test_unused_provider_is_not_activated_as_a_chat_fallback(self) -> None:
        source = legacy_base(
            providers={
                "primary": provider(
                    base_url="http://primary/v1",
                    api_key="primary-key",
                    model="primary-model",
                ),
                "spare": provider(
                    "ollama",
                    base_url="http://spare:11434",
                    model="spare-model",
                ),
            },
            default_provider="primary",
            provider_order=["primary"],
        )

        result = convert_model_settings(source)

        chat = result.config["models"]["chat"]
        assert chat["connection"] == "primary"
        assert chat["model"] == "primary-model"
        assert "fallbacks" not in chat
        assert "spare-model" in "\n".join(result.warnings)
        assert_valid(result)

    def test_static_provider_default_is_used_only_when_exact(self) -> None:
        source = legacy_base(
            providers={
                "cloud": provider("openai", api_key="synthetic", model=""),
            },
            default_provider="cloud",
            provider_order=["cloud"],
        )

        result = convert_model_settings(source)

        assert result.config["models"]["chat"] == {
            "mode": "model",
            "connection": "cloud",
            "model": "gpt-4o-mini",
        }
        assert "documented default" in "\n".join(result.warnings)
        assert_valid(result)

    def test_unresolved_dynamic_ollama_model_is_visible_and_blocked(self) -> None:
        source = legacy_base(
            providers={
                "local": provider(
                    "ollama", base_url="http://local:11434", model=""
                ),
            },
            default_provider="local",
            provider_order=["local"],
        )

        result = convert_model_settings(source)

        assert result.config["models"]["chat"] == {"mode": "unset"}
        assert result.blockers
        assert any("no exact static default" in blocker for blocker in result.blockers)
        assert result.config["models"]["chat"] == {"mode": "unset"}

    def test_duplicate_fallback_pair_is_skipped(self) -> None:
        source = legacy_base(
            providers={
                "one": provider(
                    base_url="http://one/v1", api_key="key", model="same-model"
                ),
                "two": provider(
                    base_url="http://one/v1", api_key="key", model="same-model"
                ),
            },
            default_provider="one",
            provider_order=["one", "two"],
        )

        result = convert_model_settings(source)

        chat = result.config["models"]["chat"]
        assert chat["connection"] == "one"
        assert chat["model"] == "same-model"
        assert "fallbacks" not in chat
        assert "duplicate" in "\n".join(result.warnings).lower()
        assert_valid(result)


class TestExplicitReferencePrecedence:
    def test_explicit_reference_precedes_main_and_provider_looking_model_ids(
        self,
    ) -> None:
        engine = provider(
            base_url="http://engine/v1",
            api_key="engine-key",
            model="engine-model",
            reasoning_effort="medium",
        )
        source = legacy_base(
            providers={"engine": engine},
            corpus={
                "metadata_provider": "engine",
                "metadata_model": "main",
                "cleanup_provider": "engine",
                "cleanup_model": "provider:foo",
            },
        )

        result = convert_model_settings(source)

        assert result.config["models"]["metadata"] == {
            "mode": "model",
            "connection": "engine",
            "model": "main",
            "options": {"reasoning_effort": "medium"},
        }
        assert result.config["models"]["cleanup"] == {
            "mode": "model",
            "connection": "engine",
            "model": "provider:foo",
            "options": {"reasoning_effort": "medium"},
        }
        assert_valid(result)

        chat_source = legacy_base(
            providers={
                "one": provider(
                    base_url="http://one/v1", api_key="key", model="main"
                ),
                "two": provider(
                    base_url="http://two/v1", api_key="key", model="provider:foo"
                ),
            },
            default_provider="one",
            provider_order=["one", "two"],
        )
        chat_result = convert_model_settings(chat_source)
        chat = chat_result.config["models"]["chat"]
        assert chat["model"] == "main"
        assert chat["fallbacks"][0]["model"] == "provider:foo"
        assert_valid(chat_result)


class TestTextTaskModes:
    @pytest.mark.parametrize(
        ("corpus", "role", "expected"),
        [
            (
                {"metadata_model": "main", "metadata_provider": ""},
                "metadata",
                {"mode": "chat"},
            ),
            (
                {"metadata_model": "", "metadata_provider": ""},
                "metadata",
                {"mode": "ocr"},
            ),
            (
                {"cleanup_model": "", "cleanup_provider": ""},
                "cleanup",
                {"mode": "off"},
            ),
            (
                {"cleanup_model": "main", "cleanup_provider": ""},
                "cleanup",
                {"mode": "chat"},
            ),
        ],
    )
    def test_metadata_and_cleanup_modes_convert_without_guessing(
        self, corpus: dict[str, Any], role: str, expected: dict[str, Any]
    ) -> None:
        source = legacy_base(providers={}, corpus=corpus)

        result = convert_model_settings(source)

        assert result.config["models"][role] == expected
        assert_valid(result)

    def test_provider_model_link_becomes_explicit_pair_with_notice(self) -> None:
        source = legacy_base(
            providers={
                "engine": provider(
                    base_url="http://engine/v1",
                    api_key="engine-key",
                    model="engine-model",
                    reasoning_effort="medium",
                )
            },
            corpus={
                "metadata_model": "provider:engine",
                "metadata_provider": "",
                "cleanup_model": "provider:engine",
                "cleanup_provider": "",
            },
        )

        result = convert_model_settings(source)

        for role in ("metadata", "cleanup"):
            assert result.config["models"][role] == {
                "mode": "model",
                "connection": "engine",
                "model": "engine-model",
                "options": {"reasoning_effort": "medium"},
            }
        warnings = "\n".join(result.warnings)
        assert "provider link" in warnings
        assert "explicit assignment" in warnings
        assert not any(
            "stored model" in warning and "engine" in warning
            for warning in result.warnings
        )
        assert_valid(result)

    def test_local_text_models_share_the_corpus_ollama_connection(self) -> None:
        source = legacy_base(
            providers={},
            corpus={
                "ollama_host": "http://corpus:11434",
                "ocr_model": "vision-model",
                "ocr_provider": "",
                "metadata_model": "llama-meta",
                "metadata_provider": "",
                "cleanup_model": "llama-clean",
                "cleanup_provider": "",
            },
        )

        result = convert_model_settings(source)

        connection_name, connection = connection_for(
            result, connection_type="ollama", base_url="http://corpus:11434"
        )
        assert connection_name == "corpus-ollama"
        for role, model in (
            ("ocr", "vision-model"),
            ("metadata", "llama-meta"),
            ("cleanup", "llama-clean"),
        ):
            assert result.config["models"][role] == {
                "mode": "model",
                "connection": "corpus-ollama",
                "model": model,
            }
        assert_valid(result)

    def test_empty_task_models_become_unset_with_warnings(self) -> None:
        source = legacy_base(
            providers={
                "gateway": provider(
                    base_url="http://gateway/v1",
                    api_key="gateway-key",
                    model="chat-model",
                )
            },
            embedding={
                "embedding_backend": "openai",
                "embedding_provider": "gateway",
                "embedding_host": "http://gateway/v1",
                "embedding_model": "",
                "embedding_api_key": "",
            },
            corpus={
                "ocr_model": "",
                "ocr_provider": "gateway",
                "metadata_model": "",
                "metadata_provider": "gateway",
                "cleanup_model": "",
                "cleanup_provider": "gateway",
                "rerank": True,
                "rerank_backend": "provider",
                "rerank_provider": "gateway",
                "rerank_model": "",
            },
        )

        result = convert_model_settings(source)

        assert result.config["models"]["embedding"] == {"mode": "unset"}
        assert result.config["models"]["ocr"] == {"mode": "unset"}
        assert result.config["models"]["metadata"] == {"mode": "unset"}
        assert result.config["models"]["cleanup"] == {"mode": "unset"}
        assert result.config["models"]["rerank"] == {
            "mode": "unset",
            "enabled": False,
        }
        assert_valid(result)


class TestConnections:
    def test_same_host_different_auth_creates_distinct_connections(self) -> None:
        source = legacy_base(
            providers={
                "a": provider(
                    base_url="http://same/v1", api_key="key-a", model="model-a"
                ),
                "b": provider(
                    base_url="http://same/v1", api_key="key-b", model="model-b"
                ),
                "c": provider(
                    base_url="http://same/v1",
                    api_key="key-a",
                    oauth={"access_token": "synthetic-a"},
                    model="model-c",
                ),
                "d": provider(
                    base_url="http://same/v1",
                    api_key="key-a",
                    api_version="2024-01-01",
                    model="model-d",
                ),
            },
            default_provider="a",
            provider_order=["a", "b", "c", "d"],
        )

        result = convert_model_settings(source)

        assert set(result.config["connections"]) == {"a", "b", "c", "d"}
        chat = result.config["models"]["chat"]
        assert chat["connection"] == "a"
        assert [fallback["connection"] for fallback in chat["fallbacks"]] == [
            "b",
            "c",
            "d",
        ]
        assert_valid(result)

    def test_exact_transport_deduplicates_connections(self) -> None:
        source = legacy_base(
            providers={
                "one": provider(
                    base_url="http://same/v1",
                    api_key="same-key",
                    oauth={"access_token": "same-token"},
                    api_version="2024-01-01",
                    model="model-one",
                    reasoning_effort="low",
                ),
                "two": provider(
                    base_url="http://same/v1",
                    api_key="same-key",
                    oauth={"access_token": "same-token"},
                    api_version="2024-01-01",
                    model="model-two",
                ),
            },
            default_provider="one",
            provider_order=["one", "two"],
        )

        result = convert_model_settings(source)

        assert set(result.config["connections"]) == {"one"}
        chat = result.config["models"]["chat"]
        assert chat["connection"] == "one"
        assert chat["fallbacks"] == [{"connection": "one", "model": "model-two"}]
        assert "shares its transport" in "\n".join(result.warnings)
        assert_valid(result)

    def test_generated_connection_names_are_deterministic_and_collision_free(
        self,
    ) -> None:
        source = legacy_base(
            providers={
                "embedding": provider(
                    base_url="http://provider-one/v1",
                    api_key="key-one",
                    model="chat-one",
                ),
                "embedding-2": provider(
                    base_url="http://provider-two/v1",
                    api_key="key-two",
                    model="chat-two",
                ),
            },
            embedding={
                "embedding_backend": "openai",
                "embedding_provider": "",
                "embedding_host": "http://direct:8888/v1",
                "embedding_model": "embed-model",
                "embedding_api_key": "direct-key",
            },
        )

        first = convert_model_settings(source)
        second = convert_model_settings(source)

        assert first.config["models"]["embedding"]["connection"] == "embedding-3"
        assert set(first.config["connections"]) == {
            "embedding",
            "embedding-2",
            "embedding-3",
        }
        assert first.config == second.config
        assert first.warnings == second.warnings
        assert_valid(first)

    def test_named_ollama_with_empty_host_does_not_inherit_corpus_host(self) -> None:
        source = legacy_base(
            providers={
                "desk": provider("ollama", model="chat-desk"),
            },
            corpus={
                "ollama_host": "http://corpus:11434",
                "ocr_model": "vision-model",
                "ocr_provider": "desk",
                "metadata_model": "llama-meta",
                "metadata_provider": "",
            },
        )

        result = convert_model_settings(source)

        desk = result.config["connections"]["desk"]
        assert desk["type"] == "ollama"
        assert desk.get("base_url") in (None, "")
        assert result.config["models"]["ocr"] == {
            "mode": "model",
            "connection": "desk",
            "model": "vision-model",
        }
        metadata_connection = result.config["models"]["metadata"]["connection"]
        assert metadata_connection != "desk"
        assert (
            result.config["connections"][metadata_connection]["base_url"]
            == "http://corpus:11434"
        )
        assert_valid(result)

    def test_unknown_extension_fields_survive(self) -> None:
        source = legacy_base(
            providers={
                "gateway": provider(
                    base_url="http://gateway/v1",
                    api_key="synthetic-key",
                    model="chat-model",
                    reasoning_effort="medium",
                    oauth={"access_token": "synthetic-access"},
                    api_version="2024-01-01",
                    auth_method="oauth",
                    temperature=0.2,
                    unknown_provider_key={"kept": True},
                )
            },
            default_provider="gateway",
            provider_order=["gateway"],
        )

        result = convert_model_settings(source)

        gateway = result.config["connections"]["gateway"]
        assert gateway["type"] == "openai_compatible"
        assert gateway["oauth"] == {"access_token": "synthetic-access"}
        assert gateway["api_version"] == "2024-01-01"
        assert gateway["auth_method"] == "oauth"
        assert gateway["temperature"] == 0.2
        assert gateway["unknown_provider_key"] == {"kept": True}
        assert "model" not in gateway
        assert "reasoning_effort" not in gateway
        assert "provider_type" not in gateway
        assert result.config["custom_root"] == {"kept": True}
        assert result.config["corpus"]["chunk_size"] == 3000
        assert result.config["corpus"]["custom_corpus"] == {"kept": True}
        assert_no_legacy_model_keys(result.config)
        assert_valid(result)

    def test_dotted_provider_name_is_preserved_as_a_literal_key(self) -> None:
        source = legacy_base(
            providers={
                "gateway.us": provider(
                    base_url="http://gateway/v1",
                    api_key="synthetic-key",
                    model="chat-model",
                )
            },
            default_provider="gateway.us",
            provider_order=["gateway.us"],
        )

        result = convert_model_settings(source)

        assert "gateway.us" in result.config["connections"]
        assert result.config["models"]["chat"]["connection"] == "gateway.us"
        assert_valid(result)


class TestEmbeddingConversion:
    def test_direct_embedding_follows_base_getter_precedence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ESDC_EMBEDDING_HOST", "http://env-only:9999")
        monkeypatch.setenv("ESDC_EMBEDDING_MODEL", "env-model")
        monkeypatch.setenv("ESDC_EMBEDDING_API_KEY", "env-key")
        monkeypatch.setenv("OPENAI_API_KEY", "env-openai-key")

        source = legacy_base(
            providers={},
            embedding={
                "embedding_backend": "openai",
                "embedding_provider": "",
                "embedding_host": "http://direct:8888/v1",
                "embedding_model": "embed-model",
                "embedding_api_key": "direct-key",
            },
        )

        result = convert_model_settings(source)

        assert result.config["models"]["embedding"] == {
            "mode": "model",
            "connection": "embedding",
            "model": "embed-model",
        }
        connection = result.config["connections"]["embedding"]
        assert connection["type"] == "openai_compatible"
        assert connection["base_url"] == "http://direct:8888/v1"
        assert connection["api_key"] == "direct-key"
        text = repr(result.config) + "\n".join(result.warnings)
        assert "env-only" not in text
        assert "env-model" not in text
        assert "env-key" not in text
        assert_valid(result)

    def test_ollama_direct_embedding_uses_its_documented_default_model(self) -> None:
        source = legacy_base(
            providers={},
            embedding={
                "embedding_backend": "ollama",
                "embedding_provider": "",
                "embedding_host": "",
                "embedding_model": "",
                "embedding_api_key": "unused-key",
            },
        )

        result = convert_model_settings(source)

        assert result.config["models"]["embedding"]["model"] == (
            "qwen3-embedding:0.6b"
        )
        connection = result.config["connections"]["embedding"]
        assert connection["type"] == "ollama"
        assert "api_key" not in connection
        assert "documented default" in "\n".join(result.warnings)
        assert_valid(result)

    def test_embedding_provider_credentials_override_direct_embedding_fields(
        self,
    ) -> None:
        source = legacy_base(
            providers={
                "gateway": provider(
                    base_url="http://provider:1234/v1",
                    api_key="provider-key",
                    model="chat-model",
                )
            },
            embedding={
                "embedding_backend": "openai",
                "embedding_provider": "gateway",
                "embedding_host": "http://direct:8888",
                "embedding_model": "embed-model",
                "embedding_api_key": "direct-key",
            },
        )

        result = convert_model_settings(source)

        assert result.config["models"]["embedding"] == {
            "mode": "model",
            "connection": "gateway",
            "model": "embed-model",
        }
        connection = result.config["connections"]["gateway"]
        assert connection["base_url"] == "http://provider:1234/v1"
        assert connection["api_key"] == "provider-key"
        text = repr(result.config)
        assert "direct:8888" not in text
        assert "direct-key" not in text
        assert_valid(result)

    def test_embedding_provider_uses_embedding_host_only_as_address_fallback(
        self,
    ) -> None:
        source = legacy_base(
            providers={
                "gateway": provider(
                    base_url="",
                    api_key="provider-key",
                    model="chat-model",
                )
            },
            embedding={
                "embedding_backend": "openai",
                "embedding_provider": "gateway",
                "embedding_host": "http://fallback:9999/v1",
                "embedding_model": "embed-model",
                "embedding_api_key": "direct-key",
            },
        )

        result = convert_model_settings(source)

        embedding = result.config["models"]["embedding"]
        assert embedding["connection"] != "gateway"
        connection = result.config["connections"][embedding["connection"]]
        assert connection["type"] == "openai_compatible"
        assert connection["base_url"] == "http://fallback:9999/v1"
        assert connection["api_key"] == "provider-key"
        assert "direct-key" not in repr(result.config)
        assert_valid(result)

    def test_retired_local_embedding_stays_unset_with_explanation(self) -> None:
        source = legacy_base(
            providers={},
            embedding={
                "embedding_backend": "local",
                "embedding_provider": "",
                "embedding_host": "http://old:1234",
                "embedding_model": "old-model",
                "embedding_api_key": "old-key",
            },
        )

        result = convert_model_settings(source)

        assert result.config["models"]["embedding"] == {"mode": "unset"}
        assert "retired" in "\n".join(result.warnings).lower()
        assert "http://old:1234" not in repr(result.config)
        assert_valid(result)


class TestRerankConversion:
    def test_disabled_rerank_flag_moves_to_canonical_enabled(self) -> None:
        for enabled in (False, True):
            source = legacy_base(
                providers={
                    "gateway": provider(
                        base_url="http://gateway/v1",
                        api_key="key",
                        model="chat-model",
                    )
                },
                corpus={
                    "rerank": enabled,
                    "rerank_backend": "provider",
                    "rerank_provider": "gateway",
                    "rerank_model": "rerank-model",
                },
            )

            result = convert_model_settings(source)

            assert result.config["models"]["rerank"] == {
                "mode": "model",
                "connection": "gateway",
                "model": "rerank-model",
                "enabled": enabled,
            }
            assert_valid(result)

    def test_explicit_gguf_rerank_id_is_preserved_not_invented(self) -> None:
        source = legacy_base(
            providers={
                "gateway": provider(
                    base_url="http://gateway/v1",
                    api_key="key",
                    model="chat-model",
                )
            },
            corpus={
                "rerank": True,
                "rerank_backend": "provider",
                "rerank_provider": "gateway",
                "rerank_model": LEGACY_GGUF,
            },
        )

        result = convert_model_settings(source)

        assert result.config["models"]["rerank"] == {
            "mode": "model",
            "connection": "gateway",
            "model": LEGACY_GGUF,
            "enabled": True,
        }
        assert_valid(result)

    def test_retired_local_rerank_stays_unset_and_disabled(self) -> None:
        source = legacy_base(
            providers={},
            corpus={
                "rerank": True,
                "rerank_backend": "local",
                "rerank_provider": "",
                "rerank_model": LEGACY_GGUF,
            },
        )

        result = convert_model_settings(source)

        assert result.config["models"]["rerank"] == {
            "mode": "unset",
            "enabled": False,
        }
        assert "retired" in "\n".join(result.warnings).lower()
        assert LEGACY_GGUF not in repr(result.config)
        assert_valid(result)


class TestBlockedConversions:
    @pytest.mark.parametrize(
        ("embedding", "corpus"),
        [
            (
                {
                    "embedding_backend": "ollama",
                    "embedding_provider": "missing",
                    "embedding_host": "",
                    "embedding_model": "embed-model",
                    "embedding_api_key": "",
                },
                {},
            ),
            (
                {},
                {"metadata_provider": "missing", "metadata_model": "model"},
            ),
            (
                {},
                {"cleanup_provider": "missing", "cleanup_model": "model"},
            ),
            (
                {},
                {"ocr_provider": "missing", "ocr_model": "model"},
            ),
            (
                {},
                {
                    "rerank": True,
                    "rerank_backend": "provider",
                    "rerank_provider": "missing",
                    "rerank_model": "model",
                },
            ),
        ],
    )
    def test_missing_named_connection_is_blocked(
        self, embedding: dict[str, Any], corpus: dict[str, Any]
    ) -> None:
        source = legacy_base(providers={}, embedding=embedding, corpus=corpus)

        result = convert_model_settings(source)

        assert result.blockers
        assert any("missing provider" in blocker for blocker in result.blockers)
        assert result.config["model_settings_version"] == 1
        assert "models" in result.config

    def test_missing_provider_link_is_blocked(self) -> None:
        source = legacy_base(
            providers={},
            corpus={"metadata_model": "provider:missing", "metadata_provider": ""},
        )

        result = convert_model_settings(source)

        assert result.blockers
        assert any("missing provider" in blocker for blocker in result.blockers)
        assert result.config["models"]["metadata"] == {"mode": "unset"}

    def test_mixed_old_and_new_settings_are_blocked(self) -> None:
        source = canonical_source()
        source["providers"] = {"engine": provider(api_key="key", model="chat")}
        source["embedding_host"] = "http://old:1"

        result = convert_model_settings(source)

        assert result.blockers
        assert "mix" in "\n".join(result.blockers).lower()
        assert result.config == source

    def test_unsupported_canonical_version_is_blocked(self) -> None:
        source = canonical_source()
        source["model_settings_version"] = 2

        result = convert_model_settings(source)

        assert result.blockers
        assert "version" in "\n".join(result.blockers).lower()
        assert result.config == source

    def test_canonical_fragment_without_marker_is_blocked(self) -> None:
        source = canonical_source()
        del source["model_settings_version"]

        result = convert_model_settings(source)

        assert result.blockers
        assert result.config == source

    def test_invalid_canonical_settings_are_blocked(self) -> None:
        source = canonical_source()
        source["models"]["chat"] = {"mode": "model", "connection": "missing", "model": "x"}

        result = convert_model_settings(source)

        assert result.blockers
        assert result.config == source

    def test_wrong_provider_type_for_ocr_and_rerank_is_explained(self) -> None:
        source = legacy_base(
            providers={
                "cloud": provider("anthropic", api_key="key", model="chat-model")
            },
            corpus={
                "ocr_provider": "cloud",
                "ocr_model": "vision-model",
                "rerank": True,
                "rerank_backend": "provider",
                "rerank_provider": "cloud",
                "rerank_model": "rerank-model",
            },
        )

        result = convert_model_settings(source)

        assert result.config["models"]["ocr"] == {"mode": "unset"}
        assert result.config["models"]["rerank"] == {
            "mode": "unset",
            "enabled": False,
        }
        warnings = "\n".join(result.warnings)
        assert "cannot serve OCR" in warnings
        assert "cannot serve Reranking" in warnings
        assert_valid(result)

    def test_wrong_provider_type_for_embedding_is_blocked(self) -> None:
        source = legacy_base(
            providers={
                "cloud": provider("anthropic", api_key="key", model="chat-model")
            },
            embedding={
                "embedding_backend": "openai",
                "embedding_provider": "cloud",
                "embedding_host": "http://cloud/v1",
                "embedding_model": "embed-model",
                "embedding_api_key": "",
            },
        )

        result = convert_model_settings(source)

        assert result.blockers
        assert any("cannot serve Embeddings" in blocker for blocker in result.blockers)
        assert result.config["models"]["embedding"] == {"mode": "unset"}


class TestNoOpAndSafety:
    def test_already_canonical_candidate_is_a_noop(self) -> None:
        source = canonical_source()

        result = convert_model_settings(source)

        assert result.config == source
        assert result.warnings == ()
        assert result.blockers == ()
        validate_model_settings(result.config)

    def test_source_is_never_mutated(self) -> None:
        source = legacy_base(
            providers={
                "gateway": provider(
                    base_url="http://gateway/v1",
                    api_key="key",
                    model="chat-model",
                    reasoning_effort="medium",
                )
            },
            default_provider="gateway",
            provider_order=["gateway"],
        )
        snapshot = copy.deepcopy(source)

        result = convert_model_settings(source)

        assert source == snapshot
        result.config["connections"]["gateway"]["base_url"] = "http://changed"
        result.config["models"]["chat"]["model"] = "changed"
        assert source == snapshot

    def test_unknown_root_and_corpus_fields_are_preserved(self) -> None:
        source = legacy_base(
            providers={},
            corpus={"custom_corpus": {"kept": True}, "chunk_size": 4321},
        )
        source["custom_root"] = {"kept": True}
        source["another_unknown"] = [1, 2, 3]

        result = convert_model_settings(source)

        assert result.config["custom_root"] == {"kept": True}
        assert result.config["another_unknown"] == [1, 2, 3]
        assert result.config["corpus"]["custom_corpus"] == {"kept": True}
        assert result.config["corpus"]["chunk_size"] == 4321
        assert_no_legacy_model_keys(result.config)
        assert_valid(result)

    def test_no_recognized_old_keys_remain_after_success(self) -> None:
        source = legacy_base(
            providers={
                "gateway": provider(
                    base_url="http://gateway/v1",
                    api_key="key",
                    model="chat-model",
                )
            },
            default_provider="gateway",
            provider_order=["gateway"],
            embedding={
                "embedding_backend": "ollama",
                "embedding_provider": "",
                "embedding_host": "http://embed:11434",
                "embedding_model": "embed-model",
                "embedding_api_key": "",
            },
            corpus={
                "rerank": True,
                "rerank_backend": "provider",
                "rerank_provider": "gateway",
                "rerank_model": "rerank-model",
                "ocr_model": "vision-model",
                "ocr_provider": "gateway",
                "metadata_model": "provider:gateway",
                "metadata_provider": "",
                "cleanup_model": "provider:gateway",
                "cleanup_provider": "",
                "ollama_host": "http://corpus:11434",
            },
        )

        result = convert_model_settings(source)

        assert_no_legacy_model_keys(result.config)
        assert_valid(result)

    def test_absent_old_keys_use_old_runtime_defaults(self) -> None:
        source = {"providers": {}}

        result = convert_model_settings(source)

        assert result.config["models"]["embedding"] == {
            "mode": "model",
            "connection": "embedding",
            "model": "qwen3-embedding:0.6b",
        }
        assert result.config["models"]["ocr"] == {
            "mode": "model",
            "connection": "embedding",
            "model": "glm-ocr",
        }
        assert result.config["models"]["rerank"] == {
            "mode": "unset",
            "enabled": False,
        }
        assert result.config["models"]["metadata"] == {"mode": "chat"}
        assert result.config["models"]["cleanup"] == {"mode": "chat"}
        assert result.config["connections"] == {"embedding": {"type": "ollama"}}
        assert_valid(result)

    def test_warnings_and_repr_contain_no_credentials(self) -> None:
        source = legacy_base(
            providers={
                "gateway": provider(
                    base_url="http://reader:url-password@gateway.example/v1?token=query-secret",
                    api_key="sk-synthetic-secret",
                    model="chat-model",
                    oauth={
                        "access_token": "access-secret",
                        "refresh_token": "refresh-secret",
                    },
                )
            },
            default_provider="gateway",
            provider_order=["gateway"],
        )

        result = convert_model_settings(source)

        text = "\n".join(
            [
                repr(result),
                repr(result.warnings),
                repr(result.blockers),
                *result.warnings,
                *result.blockers,
            ]
        )
        for secret in (
            "sk-synthetic-secret",
            "access-secret",
            "refresh-secret",
            "url-password",
            "query-secret",
        ):
            assert secret not in text
        assert result.config["connections"]["gateway"]["api_key"] == (
            "sk-synthetic-secret"
        )

    def test_static_provider_defaults_match_the_provider_registry(self) -> None:
        from esdc.providers import PROVIDER_CLASSES

        for provider_type, default_model in STATIC_PROVIDER_DEFAULTS.items():
            assert default_model == PROVIDER_CLASSES[provider_type].DEFAULT_MODEL
            assert PROVIDER_CLASSES[provider_type].get_default_model() == default_model

        assert "ollama" not in STATIC_PROVIDER_DEFAULTS
        assert "openai_compatible" not in STATIC_PROVIDER_DEFAULTS
