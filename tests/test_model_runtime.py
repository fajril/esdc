"""Runtime resolution tests for canonical model settings.

All fixtures are synthetic and isolated. These tests never construct a real
provider client, refresh OAuth, open a database, or use the network.
"""

import copy
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from esdc.configs import Config
from esdc.model_runtime import ResolvedTarget, ResolvedTask
from esdc.model_settings import (
    ModelConfigurationRequired,
    ModelTarget,
    TaskAssignment,
    TaskRole,
)
from esdc.providers import create_llm_from_resolved_task

SECRET_API_KEY = "sk-synthetic-secret"


def canonical_config() -> dict[str, Any]:
    return {
        "model_settings_version": 1,
        "connections": {
            "gateway": {
                "type": "openai_compatible",
                "base_url": "http://gateway.example/v1",
                "api_key": SECRET_API_KEY,
                "unknown_extension": {"kept": True},
            },
            "desktop": {
                "type": "ollama",
                "base_url": "http://desktop.example:11434",
            },
            "embed": {
                "type": "openai_compatible",
                "base_url": "http://embed.example/v1",
                "api_key": "embed-saved-key",
            },
        },
        "models": {
            "chat": {
                "mode": "model",
                "connection": "gateway",
                "model": "chat-main",
                "options": {"reasoning_effort": "medium"},
                "fallbacks": [
                    {
                        "connection": "desktop",
                        "model": "chat-fallback",
                        "options": {"reasoning_effort": "low"},
                    }
                ],
            },
            "embedding": {
                "mode": "model",
                "connection": "embed",
                "model": "embed-saved-model",
                "options": {"dimensions": 1024},
            },
            "rerank": {
                "mode": "model",
                "connection": "gateway",
                "model": "rerank-model",
                "enabled": False,
            },
            "ocr": {
                "mode": "model",
                "connection": "desktop",
                "model": "ocr-model",
            },
            "metadata": {"mode": "chat"},
            "cleanup": {"mode": "chat"},
        },
        "unknown_root": {"kept": True},
    }


def write_config(tmp_path: Path, config: dict[str, Any]) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    Config._config_cache = None
    return path


def read_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text())
    assert isinstance(loaded, dict)
    return loaded


@pytest.fixture(autouse=True)
def isolated_config_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    monkeypatch.setenv("ESDC_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("ESDC_DB_DIR", str(tmp_path))
    monkeypatch.setenv("ESDC_DB_FILE", str(tmp_path / "esdc.duckdb"))
    for name in (
        "OPENAI_API_KEY",
        "ESDC_EMBEDDING_BACKEND",
        "ESDC_EMBEDDING_HOST",
        "ESDC_EMBEDDING_MODEL",
        "ESDC_EMBEDDING_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    Config._config_cache = None
    yield tmp_path
    Config._config_cache = None


class TestResolveChat:
    def test_chat_primary_fallback_model_and_options(
        self, isolated_config_dir: Path
    ) -> None:
        write_config(isolated_config_dir, canonical_config())

        resolved = Config.resolve_model_task("chat")

        assert isinstance(resolved, ResolvedTask)
        assert resolved.mode == "model"
        assert resolved.enabled is True
        assert resolved.primary is not None
        assert resolved.primary.connection == "gateway"
        assert resolved.primary.model == "chat-main"
        assert resolved.primary.options == {"reasoning_effort": "medium"}
        assert resolved.primary.provider["provider_type"] == "openai_compatible"
        assert resolved.primary.provider["base_url"] == "http://gateway.example/v1"
        assert resolved.primary.provider["api_key"] == SECRET_API_KEY
        assert resolved.primary.provider["unknown_extension"] == {"kept": True}
        assert resolved.primary.provider["model"] == "chat-main"

        assert len(resolved.fallbacks) == 1
        fallback = resolved.fallbacks[0]
        assert fallback.connection == "desktop"
        assert fallback.model == "chat-fallback"
        assert fallback.options == {"reasoning_effort": "low"}
        assert fallback.provider["provider_type"] == "ollama"
        assert fallback.provider["model"] == "chat-fallback"

        assert SECRET_API_KEY not in repr(resolved)

    def test_switching_primary_does_not_overwrite_other_roles(
        self, isolated_config_dir: Path
    ) -> None:
        write_config(isolated_config_dir, canonical_config())
        roles: tuple[TaskRole, ...] = (
            "embedding",
            "rerank",
            "ocr",
            "metadata",
            "cleanup",
        )
        before: dict[TaskRole, TaskAssignment] = {
            role: Config.get_model_assignment(role) for role in roles
        }

        Config.save_model_assignment(
            "chat",
            TaskAssignment(
                mode="model",
                target=ModelTarget(
                    "desktop", "chat-new", {"reasoning_effort": "high"}
                ),
                enabled=True,
                fallbacks=(ModelTarget("gateway", "chat-fallback", {}),),
            ),
        )

        resolved = Config.resolve_model_task("chat")
        assert resolved.primary is not None
        assert resolved.primary.connection == "desktop"
        assert resolved.primary.model == "chat-new"
        assert resolved.primary.options == {"reasoning_effort": "high"}
        assert [target.connection for target in resolved.fallbacks] == ["gateway"]

        for role, assignment in before.items():
            assert Config.get_model_assignment(role) == assignment

    def test_metadata_and_cleanup_follow_chat_dynamically(
        self, isolated_config_dir: Path
    ) -> None:
        write_config(isolated_config_dir, canonical_config())

        chat = Config.resolve_model_task("chat")
        metadata = Config.resolve_model_task("metadata")
        cleanup = Config.resolve_model_task("cleanup")

        assert metadata.mode == "chat"
        assert metadata.primary is not None
        assert chat.primary is not None
        assert metadata.primary.connection == chat.primary.connection
        assert metadata.primary.model == chat.primary.model
        assert [target.model for target in metadata.fallbacks] == [
            target.model for target in chat.fallbacks
        ]
        assert cleanup.mode == "chat"
        assert cleanup.primary is not None
        assert cleanup.primary.model == chat.primary.model

        Config.save_model_assignment(
            "chat",
            TaskAssignment(
                mode="model",
                target=ModelTarget("desktop", "chat-new", {}),
                enabled=True,
                fallbacks=(),
            ),
        )
        changed = Config.resolve_model_task("chat")
        changed_metadata = Config.resolve_model_task("metadata")
        changed_cleanup = Config.resolve_model_task("cleanup")
        assert changed.primary is not None
        assert changed.primary.model == "chat-new"
        assert changed_metadata.primary is not None
        assert changed_metadata.primary.model == "chat-new"
        assert changed_cleanup.primary is not None
        assert changed_cleanup.primary.model == "chat-new"

        Config.save_model_assignment(
            "chat",
            TaskAssignment(mode="unset", target=None, enabled=True, fallbacks=()),
        )
        unset_metadata = Config.resolve_model_task("metadata")
        unset_cleanup = Config.resolve_model_task("cleanup")
        assert unset_metadata.mode == "chat"
        assert unset_metadata.primary is None
        assert unset_cleanup.mode == "chat"
        assert unset_cleanup.primary is None

    def test_metadata_ocr_cleanup_off_and_unset_are_distinct(
        self, isolated_config_dir: Path
    ) -> None:
        config = canonical_config()
        config["models"]["metadata"] = {"mode": "ocr"}
        config["models"]["cleanup"] = {"mode": "off"}
        config["models"]["rerank"] = {"mode": "unset", "enabled": False}
        config["models"]["embedding"] = {"mode": "unset"}
        write_config(isolated_config_dir, config)

        metadata = Config.resolve_model_task("metadata")
        cleanup = Config.resolve_model_task("cleanup")
        rerank = Config.resolve_model_task("rerank")
        unset_embedding = Config.resolve_model_task("embedding")

        assert metadata.mode == "ocr"
        assert metadata.primary is None
        assert cleanup.mode == "off"
        assert cleanup.primary is None
        assert rerank.mode == "unset"
        assert rerank.enabled is False
        assert rerank.primary is None
        assert unset_embedding.mode == "unset"
        assert unset_embedding.primary is None

    def test_explicit_literal_model_ids_are_not_control_tokens(
        self, isolated_config_dir: Path
    ) -> None:
        config = canonical_config()
        config["models"]["chat"]["model"] = "main"
        config["models"]["chat"]["fallbacks"] = [
            {"connection": "desktop", "model": "provider:foo"}
        ]
        config["models"]["metadata"] = {
            "mode": "model",
            "connection": "gateway",
            "model": "main",
        }
        config["models"]["ocr"] = {
            "mode": "model",
            "connection": "desktop",
            "model": "provider:foo",
        }
        write_config(isolated_config_dir, config)

        chat = Config.resolve_model_task("chat")
        metadata = Config.resolve_model_task("metadata")
        ocr = Config.resolve_model_task("ocr")

        assert chat.primary is not None
        assert chat.primary.model == "main"
        assert chat.fallbacks[0].model == "provider:foo"
        assert metadata.primary is not None
        assert metadata.primary.model == "main"
        assert ocr.primary is not None
        assert ocr.primary.model == "provider:foo"

    def test_missing_configured_task_does_not_construct_a_service(
        self, isolated_config_dir: Path
    ) -> None:
        Config.init_config()

        def fail(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("resolving an unset task constructed a service")

        with (
            patch("esdc.providers.create_llm_from_config", side_effect=fail),
            patch("esdc.providers.create_provider", side_effect=fail),
            patch("esdc.providers._create_single_llm_from_config", side_effect=fail),
        ):
            for role in ("chat", "embedding", "rerank", "ocr"):
                resolved = Config.resolve_model_task(role)
                assert resolved.mode == "unset"
                assert resolved.primary is None
                assert resolved.fallbacks == ()
            assert Config.resolve_model_task("metadata").mode == "chat"
            assert Config.resolve_model_task("metadata").primary is None
            assert Config.resolve_model_task("cleanup").mode == "chat"
            assert Config.resolve_model_task("cleanup").primary is None
            assert Config.get_provider_config() is None


class TestEmbeddingOverrides:
    def test_environment_overrides_are_per_run_effective_values(
        self, isolated_config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_config(isolated_config_dir, canonical_config())
        monkeypatch.setenv("ESDC_EMBEDDING_BACKEND", "openai")
        monkeypatch.setenv("ESDC_EMBEDDING_HOST", "http://override.example/v1")
        monkeypatch.setenv("ESDC_EMBEDDING_MODEL", "override-model")
        monkeypatch.setenv("ESDC_EMBEDDING_API_KEY", "override-key")

        resolved = Config.resolve_model_task("embedding", apply_overrides=True)

        assert resolved.mode == "model"
        assert resolved.primary is not None
        assert resolved.primary.model == "override-model"
        assert resolved.primary.provider["base_url"] == "http://override.example/v1"
        assert resolved.primary.provider["api_key"] == "override-key"
        assert resolved.overrides == (
            "ESDC_EMBEDDING_BACKEND",
            "ESDC_EMBEDDING_HOST",
            "ESDC_EMBEDDING_MODEL",
            "ESDC_EMBEDDING_API_KEY",
        )

        saved = Config.resolve_model_task("embedding", apply_overrides=False)
        assert saved.primary is not None
        assert saved.primary.model == "embed-saved-model"
        assert saved.primary.provider["base_url"] == "http://embed.example/v1"
        assert saved.primary.provider["api_key"] == "embed-saved-key"
        assert saved.overrides == ()

    def test_incompatible_backend_override_is_refused(
        self, isolated_config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_config(isolated_config_dir, canonical_config())
        monkeypatch.setenv("ESDC_EMBEDDING_BACKEND", "ollama")

        with pytest.raises(ValueError, match="does not match"):
            Config.resolve_model_task("embedding")

    def test_unset_embedding_is_not_activated_by_environment(
        self, isolated_config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        Config.init_config()
        monkeypatch.setenv("ESDC_EMBEDDING_MODEL", "env-model")
        monkeypatch.setenv("ESDC_EMBEDDING_HOST", "http://env.example:11434")

        resolved = Config.resolve_model_task("embedding")

        assert resolved.mode == "unset"
        assert resolved.primary is None
        assert resolved.overrides == ()

    def test_embedding_override_does_not_change_chat_or_add_fallbacks(
        self, isolated_config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_config(isolated_config_dir, canonical_config())
        monkeypatch.setenv("ESDC_EMBEDDING_MODEL", "override-model")
        monkeypatch.setenv("ESDC_EMBEDDING_HOST", "http://override.example/v1")

        chat = Config.resolve_model_task("chat")
        embedding = Config.resolve_model_task("embedding")

        assert chat.primary is not None
        assert chat.primary.model == "chat-main"
        assert chat.overrides == ()
        assert len(chat.fallbacks) == 1
        assert embedding.primary is not None
        assert embedding.primary.model == "override-model"


class TestCanonicalAdapters:
    def test_provider_config_adapter_uses_chat_assignments(
        self, isolated_config_dir: Path
    ) -> None:
        write_config(isolated_config_dir, canonical_config())

        provider_config = Config.get_provider_config()

        assert provider_config is not None
        assert provider_config["provider_type"] == "openai_compatible"
        assert provider_config["model"] == "chat-main"
        assert provider_config["base_url"] == "http://gateway.example/v1"
        assert provider_config["api_key"] == SECRET_API_KEY
        assert len(provider_config["fallback_configs"]) == 1
        assert provider_config["fallback_configs"][0]["model"] == "chat-fallback"
        assert Config.get_provider_model() == "chat-main"
        assert Config.get_provider_base_url() == "http://gateway.example/v1"
        assert Config.get_provider_api_key() == SECRET_API_KEY
        assert Config.has_chat_config() is True

    def test_no_legacy_provider_fields_are_read_by_runtime_adapters(
        self, isolated_config_dir: Path
    ) -> None:
        (isolated_config_dir / "config.yaml").write_text(
            """
providers:
  engine:
    provider_type: openai_compatible
    base_url: http://engine/v1
    model: legacy-chat
default_provider: engine
provider_order:
  - engine
"""
        )
        Config._config_cache = None

        for getter in (
            lambda: Config.get_provider_config(),
            lambda: Config.get_provider_model(),
        ):
            with pytest.raises(ModelConfigurationRequired):
                getter()

    def test_unset_chat_adapters_do_not_pretend_a_provider(
        self, isolated_config_dir: Path
    ) -> None:
        Config.init_config()

        assert Config.get_provider_config() is None
        assert Config.get_provider_model() == ""
        assert Config.get_provider_base_url() == ""
        assert Config.has_chat_config() is False


class TestResolverProviderMapping:
    def test_provider_mapping_preserves_oauth_and_extension_fields(
        self, isolated_config_dir: Path
    ) -> None:
        config = canonical_config()
        config["connections"]["oauth"] = {
            "type": "openai",
            "auth_method": "oauth",
            "api_key": "coexisting-key",
            "oauth": {
                "access_token": "synthetic-access",
                "refresh_token": "synthetic-refresh",
            },
            "api_version": "2024-01-01",
            "unknown_extension": {"kept": True},
        }
        config["models"]["chat"] = {
            "mode": "model",
            "connection": "oauth",
            "model": "oauth-model",
        }
        write_config(isolated_config_dir, config)

        resolved = Config.resolve_model_task("chat")

        assert resolved.primary is not None
        provider = resolved.primary.provider
        assert provider["name"] == "oauth"
        assert provider["provider_type"] == "openai"
        assert provider["auth_method"] == "oauth"
        assert provider["api_key"] == "coexisting-key"
        assert provider["oauth"] == {
            "access_token": "synthetic-access",
            "refresh_token": "synthetic-refresh",
        }
        assert provider["api_version"] == "2024-01-01"
        assert provider["unknown_extension"] == {"kept": True}
        assert "synthetic-access" not in repr(resolved)
        assert "synthetic-refresh" not in repr(resolved)


class TestOAuthPersistence:
    def test_updates_only_the_named_canonical_connection(
        self, isolated_config_dir: Path
    ) -> None:
        config = canonical_config()
        config["connections"]["gateway"]["oauth"] = {
            "access_token": "old-access",
            "refresh_token": "old-refresh",
        }
        config["connections"]["desktop"]["oauth"] = {
            "access_token": "desktop-access"
        }
        path = write_config(isolated_config_dir, config)

        new_oauth = {
            "access_token": "new-access",
            "refresh_token": "rotated-refresh",
            "expires_at": 9999999999,
        }
        Config.persist_provider_oauth("gateway", new_oauth)

        raw = read_config(path)
        assert raw["connections"]["gateway"]["oauth"] == new_oauth
        assert raw["connections"]["gateway"]["base_url"] == "http://gateway.example/v1"
        assert raw["connections"]["gateway"]["api_key"] == SECRET_API_KEY
        assert raw["connections"]["desktop"] == {
            "type": "ollama",
            "base_url": "http://desktop.example:11434",
            "oauth": {"access_token": "desktop-access"},
        }
        assert raw["unknown_root"] == {"kept": True}
        assert "providers" not in raw
        assert Config._config_cache is None

    def test_rotates_tokens_on_primary_and_fallback_connections(
        self, isolated_config_dir: Path
    ) -> None:
        config = canonical_config()
        config["connections"]["gateway"]["oauth"] = {"access_token": "primary-old"}
        config["connections"]["desktop"]["oauth"] = {"access_token": "fallback-old"}
        path = write_config(isolated_config_dir, config)

        Config.persist_provider_oauth("gateway", {"access_token": "primary-new"})
        Config.persist_provider_oauth("desktop", {"access_token": "fallback-new"})

        raw = read_config(path)
        assert raw["connections"]["gateway"]["oauth"] == {
            "access_token": "primary-new"
        }
        assert raw["connections"]["desktop"]["oauth"] == {
            "access_token": "fallback-new"
        }

    def test_fresh_disk_read_preserves_unrelated_concurrent_edit(
        self, isolated_config_dir: Path
    ) -> None:
        config = canonical_config()
        config["connections"]["gateway"]["oauth"] = {"access_token": "old"}
        path = write_config(isolated_config_dir, config)
        cache = Config._load_config()
        assert cache is not None

        current = read_config(path)
        current["connections"]["desktop"]["base_url"] = "http://concurrent:11434"
        path.write_text(yaml.safe_dump(current, sort_keys=False))

        Config.persist_provider_oauth("gateway", {"access_token": "new"})

        raw = read_config(path)
        assert raw["connections"]["gateway"]["oauth"] == {"access_token": "new"}
        assert raw["connections"]["desktop"]["base_url"] == "http://concurrent:11434"

    def test_private_file_mode_is_preserved_on_rotation(
        self, isolated_config_dir: Path
    ) -> None:
        config = canonical_config()
        config["connections"]["gateway"]["oauth"] = {"access_token": "old"}
        path = write_config(isolated_config_dir, config)
        path.chmod(0o644)

        Config.persist_provider_oauth("gateway", {"access_token": "new"})

        assert path.stat().st_mode & 0o777 == 0o600

    def test_failed_replace_leaves_file_and_cache_intact(
        self, isolated_config_dir: Path
    ) -> None:
        config = canonical_config()
        config["connections"]["gateway"]["oauth"] = {"access_token": "old"}
        path = write_config(isolated_config_dir, config)
        before = path.read_bytes()
        cache = Config._load_config()
        assert cache is not None

        with (
            patch.object(os, "replace", side_effect=OSError("replace failed")),
            pytest.raises(OSError),
        ):
            Config.persist_provider_oauth("gateway", {"access_token": "new"})

        assert path.read_bytes() == before
        assert Config._config_cache is cache
        assert [p.name for p in isolated_config_dir.iterdir()] == ["config.yaml"]

    def test_missing_connection_is_a_noop(
        self, isolated_config_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = write_config(isolated_config_dir, canonical_config())
        original = path.read_text()

        with caplog.at_level("WARNING"):
            Config.persist_provider_oauth("missing", {"access_token": "new"})

        assert path.read_text() == original
        assert any("missing" in record.message for record in caplog.records)

    def test_old_providers_section_is_never_updated(
        self, isolated_config_dir: Path
    ) -> None:
        path = isolated_config_dir / "config.yaml"
        path.write_text(
            """
providers:
  openai:
    provider_type: openai
    oauth:
      access_token: old
"""
        )
        Config._config_cache = None
        original = path.read_text()

        Config.persist_provider_oauth("openai", {"access_token": "new"})

        assert path.read_text() == original
        assert "connections" not in read_config(path)


class TestResolvedLlmConstruction:
    def test_primary_and_fallbacks_are_passed_with_request_overrides(self) -> None:
        primary = ResolvedTarget(
            connection="gateway",
            provider={
                "name": "gateway",
                "provider_type": "openai_compatible",
                "base_url": "http://gateway/v1",
                "api_key": SECRET_API_KEY,
                "model": "chat-main",
                "reasoning_effort": "medium",
            },
            model="chat-main",
            options={"reasoning_effort": "medium"},
        )
        fallback = ResolvedTarget(
            connection="desktop",
            provider={
                "name": "desktop",
                "provider_type": "ollama",
                "base_url": "http://desktop:11434",
                "model": "chat-fallback",
                "reasoning_effort": "low",
            },
            model="chat-fallback",
            options={"reasoning_effort": "low"},
        )
        task = ResolvedTask(
            role="chat",
            mode="model",
            enabled=True,
            primary=primary,
            fallbacks=(fallback,),
        )
        sentinel = MagicMock()

        with patch(
            "esdc.providers.create_llm_from_config", return_value=sentinel
        ) as mock_create:
            result = create_llm_from_resolved_task(
                task, reasoning_effort="high", temperature=0.2
            )

        assert result is sentinel
        passed = mock_create.call_args.args[0]
        assert passed["model"] == "chat-main"
        assert passed["reasoning_effort"] == "high"
        assert passed["temperature"] == 0.2
        assert passed["fallback_configs"][0]["model"] == "chat-fallback"
        assert passed["fallback_configs"][0]["reasoning_effort"] == "low"
        assert SECRET_API_KEY in copy.deepcopy(passed)["api_key"]
