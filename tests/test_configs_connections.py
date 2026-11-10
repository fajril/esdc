"""Canonical connection CRUD and reference safety.

Every test uses a temporary config directory and synthetic credentials. Nothing
here reads the user's live configuration, opens a database, or uses the network.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from esdc.configs import Config
from esdc.model_settings import ModelConfigurationRequired, TaskRole


def canonical_config() -> dict[str, Any]:
    """A complete canonical file; connection names include a dotted key."""
    return {
        "model_settings_version": 1,
        "connections": {
            "gateway": {
                "type": "openai_compatible",
                "base_url": "http://gateway.example/v1",
                "api_key": "synthetic-gateway-key",
                "oauth": {
                    "access_token": "synthetic-expired-token",
                    "expires_at": "2000-01-01T00:00:00Z",
                },
                "vendor_extension": {"region": "test"},
            },
            "desk.ollama": {
                "type": "ollama",
                "base_url": "http://desk.example:11434",
            },
            "backup": {
                "type": "openai_compatible",
                "base_url": "http://backup.example/v1",
                "api_key": "synthetic-backup-key",
            },
        },
        "models": {
            "chat": {
                "mode": "model",
                "connection": "gateway",
                "model": "chat-primary",
                "options": {"reasoning_effort": "medium"},
                "fallbacks": [
                    {"connection": "desk.ollama", "model": "chat-local"},
                    {
                        "connection": "backup",
                        "model": "chat-backup",
                        "options": {"reasoning_effort": "low"},
                    },
                ],
            },
            "embedding": {
                "mode": "model",
                "connection": "desk.ollama",
                "model": "embed-model",
            },
            "rerank": {
                "mode": "model",
                "connection": "gateway",
                "model": "rerank-model",
                "enabled": False,
            },
            "ocr": {
                "mode": "model",
                "connection": "desk.ollama",
                "model": "ocr-model",
            },
            "metadata": {"mode": "chat"},
            "cleanup": {"mode": "off"},
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
    Config._config_cache = None
    yield tmp_path
    Config._config_cache = None


class TestConnectionReferences:
    def test_references_cover_the_primary_and_each_chat_fallback(
        self, isolated_config_dir: Path
    ) -> None:
        write_config(isolated_config_dir, canonical_config())

        assert Config.get_connection_references("gateway") == [
            "models.chat",
            "models.rerank",
        ]
        assert Config.get_connection_references("desk.ollama") == [
            "models.chat.fallbacks[0]",
            "models.embedding",
            "models.ocr",
        ]
        assert Config.get_connection_references("backup") == [
            "models.chat.fallbacks[1]",
        ]

    def test_a_dotted_connection_name_is_a_literal_mapping_key(
        self, isolated_config_dir: Path
    ) -> None:
        write_config(isolated_config_dir, canonical_config())

        assert Config.get_connection("desk.ollama")["type"] == "ollama"
        assert Config.get_connection_references("desk.ollama")

    def test_an_unreferenced_or_unknown_name_has_no_references(
        self, isolated_config_dir: Path
    ) -> None:
        write_config(isolated_config_dir, canonical_config())

        assert Config.get_connection_references("missing") == []
        assert Config.get_connection_references("") == []


class TestSaveConnection:
    def test_new_connections_keep_insertion_order_and_unknown_fields(
        self, isolated_config_dir: Path
    ) -> None:
        write_config(isolated_config_dir, canonical_config())

        Config.save_connection(
            "second-desk",
            {
                "type": "ollama",
                "base_url": "http://second.example:11434",
                "vendor_extension": {"region": "second"},
            },
        )

        connections = Config.get_connections()
        assert list(connections) == ["gateway", "desk.ollama", "backup", "second-desk"]
        assert connections["second-desk"]["vendor_extension"] == {"region": "second"}
        assert "model" not in connections["second-desk"]
        raw = read_config(isolated_config_dir / "config.yaml")
        assert raw["unknown_root"] == {"kept": True}
        assert "providers" not in raw

    def test_saving_an_existing_name_edits_in_place_without_touching_tasks(
        self, isolated_config_dir: Path
    ) -> None:
        write_config(isolated_config_dir, canonical_config())
        before = Config.get_model_assignment("chat")

        Config.save_connection(
            "gateway",
            {
                "type": "openai_compatible",
                "base_url": "http://new-gateway.example/v1",
                "api_key": "synthetic-new-key",
                "vendor_extension": {"region": "new"},
            },
        )

        assert Config.get_connection("gateway")["base_url"] == (
            "http://new-gateway.example/v1"
        )
        assert Config.get_model_assignment("chat") == before

    def test_a_save_that_breaks_a_role_reference_is_refused_atomically(
        self, isolated_config_dir: Path
    ) -> None:
        path = write_config(isolated_config_dir, canonical_config())
        before = path.read_bytes()

        with pytest.raises(ValueError, match="cannot serve Embeddings"):
            Config.save_connection(
                "desk.ollama",
                {"type": "anthropic", "api_key": "synthetic-key"},
            )

        assert path.read_bytes() == before
        assert Config.get_connection("desk.ollama")["type"] == "ollama"

    def test_legacy_settings_require_conversion_before_any_connection_write(
        self, isolated_config_dir: Path
    ) -> None:
        path = isolated_config_dir / "config.yaml"
        path.write_text("providers:\n  engine:\n    type: openai\n")
        Config._config_cache = None

        with pytest.raises(ModelConfigurationRequired):
            Config.save_connection(
                "desk", {"type": "ollama", "base_url": "http://desk:11434"}
            )


class TestRenameConnection:
    def test_rename_moves_every_reference_in_one_write_without_touching_models(
        self, isolated_config_dir: Path
    ) -> None:
        write_config(isolated_config_dir, canonical_config())
        before = copy_models(Config)
        path = isolated_config_dir / "config.yaml"
        before_text = path.read_text()

        Config.rename_connection("gateway", "cloud.gateway")

        connections = Config.get_connections()
        assert list(connections) == ["cloud.gateway", "desk.ollama", "backup"]
        assert connections["cloud.gateway"]["vendor_extension"] == {"region": "test"}
        assert connections["cloud.gateway"]["oauth"]["access_token"] == (
            "synthetic-expired-token"
        )
        assert Config.get_connection_references("gateway") == []
        assert Config.get_connection_references("cloud.gateway") == [
            "models.chat",
            "models.rerank",
        ]
        after = copy_models(Config)
        assert after["chat"]["connection"] == "cloud.gateway"
        assert after["chat"]["model"] == before["chat"]["model"]
        assert after["chat"]["options"] == before["chat"]["options"]
        assert after["chat"]["fallbacks"] == before["chat"]["fallbacks"]
        assert after["rerank"]["model"] == before["rerank"]["model"]
        # One atomic write: the file is valid YAML immediately after the call.
        raw = read_config(path)
        assert "gateway" not in raw["connections"]
        assert raw["connections"]["cloud.gateway"]["base_url"].startswith("http://")
        assert before_text != path.read_text()

    def test_rename_does_not_rewrite_model_ids_or_task_options_named_like_it(
        self, isolated_config_dir: Path
    ) -> None:
        config = canonical_config()
        config["models"]["chat"]["model"] = "gateway"
        config["models"]["chat"]["options"] = {"tag": "gateway"}
        config["models"]["embedding"]["model"] = "desk.ollama"
        write_config(isolated_config_dir, config)

        Config.rename_connection("gateway", "renamed")

        chat = Config.get_model_assignment("chat")
        assert chat.target is not None
        assert chat.target.model == "gateway"
        assert chat.target.options == {"tag": "gateway"}
        embedding = Config.get_model_assignment("embedding")
        assert embedding.target is not None
        assert embedding.target.model == "desk.ollama"

    def test_rename_applies_the_draft_values_and_leaves_other_connections_alone(
        self, isolated_config_dir: Path
    ) -> None:
        write_config(isolated_config_dir, canonical_config())

        Config.rename_connection(
            "gateway",
            "renamed",
            values={
                "type": "openai_compatible",
                "base_url": "http://renamed.example/v1",
                "api_key": "synthetic-renamed-key",
            },
        )

        renamed = Config.get_connection("renamed")
        assert renamed["base_url"] == "http://renamed.example/v1"
        assert renamed["api_key"] == "synthetic-renamed-key"
        assert Config.get_connection("backup")["base_url"] == (
            "http://backup.example/v1"
        )
        chat = Config.get_model_assignment("chat")
        assert chat.target is not None
        assert chat.target.connection == "renamed"
        assert chat.fallbacks[0].model == "chat-local"

    def test_renaming_to_the_same_name_can_still_apply_values(
        self, isolated_config_dir: Path
    ) -> None:
        write_config(isolated_config_dir, canonical_config())

        Config.rename_connection(
            "backup",
            "backup",
            values={
                "type": "openai_compatible",
                "base_url": "http://changed.example/v1",
                "api_key": "synthetic-changed-key",
            },
        )

        assert Config.get_connection("backup")["base_url"] == (
            "http://changed.example/v1"
        )
        assert Config.get_connection_references("backup") == [
            "models.chat.fallbacks[1]"
        ]

    @pytest.mark.parametrize("target", ["desk.ollama", "   ", ""])
    def test_a_conflicting_or_empty_name_is_rejected_before_writing(
        self, isolated_config_dir: Path, target: str
    ) -> None:
        path = write_config(isolated_config_dir, canonical_config())
        before = path.read_bytes()

        with pytest.raises(ValueError):
            Config.rename_connection("gateway", target)

        assert path.read_bytes() == before

    def test_an_unknown_old_name_is_rejected(self, isolated_config_dir: Path) -> None:
        path = write_config(isolated_config_dir, canonical_config())
        before = path.read_bytes()

        with pytest.raises(ValueError, match="Unknown connection"):
            Config.rename_connection("ghost", "renamed")

        assert path.read_bytes() == before

    def test_a_failed_replace_leaves_file_and_cache_intact(
        self, isolated_config_dir: Path
    ) -> None:
        path = write_config(isolated_config_dir, canonical_config())
        before_text = path.read_text()
        before_cache = Config.get_connections()

        with (
            patch("esdc.configs.os.replace", side_effect=OSError("disk full")),
            pytest.raises(OSError),
        ):
            Config.rename_connection("gateway", "renamed")

        assert path.read_text() == before_text
        assert Config.get_connections() == before_cache


class TestRemoveConnection:
    def test_a_referenced_connection_is_refused_and_every_use_is_named(
        self, isolated_config_dir: Path
    ) -> None:
        path = write_config(isolated_config_dir, canonical_config())
        before = path.read_bytes()

        with pytest.raises(ValueError) as exc_info:
            Config.remove_connection("desk.ollama")

        message = str(exc_info.value)
        assert "models.chat.fallbacks[0]" in message
        assert "models.embedding" in message
        assert "models.ocr" in message
        assert path.read_bytes() == before
        assert "desk.ollama" in Config.get_connections()

    def test_an_unreferenced_connection_is_removed_and_order_is_preserved(
        self, isolated_config_dir: Path
    ) -> None:
        config = canonical_config()
        config["connections"]["spare"] = {
            "type": "ollama",
            "base_url": "http://spare.example:11434",
        }
        write_config(isolated_config_dir, config)

        assert Config.remove_connection("spare") is True

        assert list(Config.get_connections()) == ["gateway", "desk.ollama", "backup"]
        assert Config.remove_connection("spare") is False


def copy_models(config_cls: type[Config]) -> dict[str, Any]:
    from esdc.model_settings import serialize_task_assignment

    roles: tuple[TaskRole, ...] = (
        "chat",
        "embedding",
        "rerank",
        "ocr",
        "metadata",
        "cleanup",
    )
    return {
        role: serialize_task_assignment(role, config_cls.get_model_assignment(role))
        for role in roles
    }


def test_connection_names_with_surrounding_whitespace_are_rejected(
    isolated_config_dir: Path,
) -> None:
    write_config(isolated_config_dir, canonical_config())

    with pytest.raises(ValueError):
        Config.save_connection(" spaced ", {"type": "ollama"})
    with pytest.raises(ValueError):
        Config.save_connection("", {"type": "ollama"})


def test_connection_crud_preserves_other_application_settings(
    isolated_config_dir: Path,
) -> None:
    config = canonical_config()
    config["connections"]["spare"] = {"type": "ollama"}
    config["corpus"] = {"chunk_size": 1234}
    config["phoenix"] = {"enabled": True}
    write_config(isolated_config_dir, config)

    Config.rename_connection("gateway", "renamed")
    Config.remove_connection("spare")
    Config.save_connection("added", {"type": "ollama", "base_url": "http://a:1"})

    raw = read_config(isolated_config_dir / "config.yaml")
    assert raw["corpus"] == {"chunk_size": 1234}
    assert raw["phoenix"] == {"enabled": True}
    assert raw["unknown_root"] == {"kept": True}


def test_saving_a_connection_does_not_write_a_providers_section(
    isolated_config_dir: Path,
) -> None:
    write_config(isolated_config_dir, canonical_config())

    Config.save_connection("added", {"type": "ollama", "base_url": "http://a:1"})

    raw = read_config(isolated_config_dir / "config.yaml")
    assert "providers" not in raw
    assert set(raw["connections"]) == {
        "gateway",
        "desk.ollama",
        "backup",
        "added",
    }


def test_config_file_keeps_its_private_mode(
    isolated_config_dir: Path,
) -> None:
    path = write_config(isolated_config_dir, canonical_config())
    os.chmod(path, 0o600)

    Config.save_connection("added", {"type": "ollama", "base_url": "http://a:1"})

    assert path.stat().st_mode & 0o777 == 0o600


def test_removing_a_connection_does_not_leave_a_backup_file(
    isolated_config_dir: Path,
) -> None:
    config = canonical_config()
    config["connections"]["spare"] = {"type": "ollama"}
    write_config(isolated_config_dir, config)

    Config.remove_connection("spare")

    leftovers = [
        p.name for p in isolated_config_dir.iterdir() if p.name != "config.yaml"
    ]
    assert leftovers == []
