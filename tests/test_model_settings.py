"""Tests for canonical connection and per-task model settings.

These tests use a temporary ``ESDC_CONFIG_DIR`` and synthetic credentials only.
Nothing here reads or writes the user's live configuration, opens a database,
constructs a provider client, or touches the network.
"""

import copy
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest
import yaml

from esdc.configs import Config
from esdc.model_settings import (
    KNOWN_CONNECTION_TYPES,
    ModelConfigurationRequired,
    ModelTarget,
    TaskAssignment,
    TaskRole,
    default_model_settings,
    has_legacy_model_settings,
    parse_task_assignment,
    serialize_task_assignment,
    validate_model_settings,
    with_chat_primary,
)

TASK_ROLES: tuple[TaskRole, ...] = ("chat", "embedding", "rerank", "ocr", "metadata", "cleanup")

LEGACY_ROOT_KEYS = {
    "providers",
    "default_provider",
    "provider_order",
    "embedding_backend",
    "embedding_host",
    "embedding_model",
    "embedding_api_key",
    "embedding_provider",
}

LEGACY_CORPUS_KEYS = {
    "rerank",
    "rerank_backend",
    "rerank_provider",
    "rerank_model",
    "ocr_model",
    "ocr_provider",
    "metadata_model",
    "metadata_provider",
    "cleanup_model",
    "cleanup_provider",
    "ollama_host",
    "n_gpu_layers",
}

SUPPORTED_MODES = {
    "chat": {"model", "unset"},
    "embedding": {"model", "unset"},
    "rerank": {"model", "unset"},
    "ocr": {"model", "unset"},
    "metadata": {"model", "chat", "ocr", "unset"},
    "cleanup": {"model", "chat", "off", "unset"},
}


def canonical_config() -> dict[str, Any]:
    """A complete canonical file using synthetic addresses and credentials."""
    return {
        "model_settings_version": 1,
        "connections": {
            "gateway": {
                "type": "openai_compatible",
                "base_url": "http://gateway.example/v1",
                "api_key": "synthetic-placeholder",
            },
            "desktop": {
                "type": "ollama",
                "base_url": "http://localhost:11434",
            },
        },
        "models": {
            "chat": {
                "mode": "model",
                "connection": "gateway",
                "model": "chat-model",
                "options": {"reasoning_effort": "medium"},
                "fallbacks": [
                    {"connection": "desktop", "model": "chat-fallback"},
                ],
            },
            "embedding": {
                "mode": "model",
                "connection": "desktop",
                "model": "qwen3-embedding:0.6b",
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
                "model": "glm-ocr",
            },
            "metadata": {"mode": "chat"},
            "cleanup": {"mode": "off"},
        },
        "unknown_user_key": "kept",
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


def assert_no_legacy_model_keys(config: dict[str, Any]) -> None:
    assert LEGACY_ROOT_KEYS.isdisjoint(config)
    corpus = config.get("corpus")
    if isinstance(corpus, dict):
        assert LEGACY_CORPUS_KEYS.isdisjoint(corpus)


@pytest.fixture(autouse=True)
def isolated_config_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Every test uses a fresh temporary config and database location."""
    monkeypatch.setenv("ESDC_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("ESDC_DB_DIR", str(tmp_path))
    monkeypatch.setenv("ESDC_DB_FILE", str(tmp_path / "esdc.duckdb"))
    Config._config_cache = None
    yield tmp_path
    Config._config_cache = None


class TestCanonicalDefaults:
    def test_fresh_defaults_match_the_design(self, isolated_config_dir: Path) -> None:
        Config.init_config()

        raw = read_config(isolated_config_dir / "config.yaml")
        assert raw["model_settings_version"] == 1
        assert raw["connections"] == {}
        assert raw["models"] == default_model_settings()
        assert raw["models"] == {
            "chat": {"mode": "unset"},
            "embedding": {"mode": "unset"},
            "rerank": {"mode": "unset", "enabled": False},
            "ocr": {"mode": "unset"},
            "metadata": {"mode": "chat"},
            "cleanup": {"mode": "chat"},
        }
        assert_no_legacy_model_keys(raw)

        assert Config.get_connections() == {}
        default_cases: tuple[tuple[TaskRole, str], ...] = (
            ("chat", "unset"),
            ("embedding", "unset"),
            ("rerank", "unset"),
            ("ocr", "unset"),
            ("metadata", "chat"),
            ("cleanup", "chat"),
        )
        for role, mode in default_cases:
            assignment = Config.get_model_assignment(role)
            assert assignment.mode == mode
            assert assignment.target is None
            assert assignment.fallbacks == ()
            expected_enabled = role != "rerank"
            assert assignment.enabled is expected_enabled

    def test_missing_file_uses_defaults_without_writing(
        self, isolated_config_dir: Path
    ) -> None:
        assert Config.get_connections() == {}
        assert Config.get_model_assignment("chat").mode == "unset"
        assert not (isolated_config_dir / "config.yaml").exists()

    def test_get_defaults_contains_the_canonical_fields(self) -> None:
        defaults = Config.get_defaults()
        assert defaults["model_settings_version"] == 1
        assert defaults["connections"] == {}
        assert defaults["models"] == default_model_settings()


class TestCanonicalRoundTrip:
    def test_canonical_round_trip_has_no_parallel_model_keys(
        self, isolated_config_dir: Path
    ) -> None:
        path = write_config(isolated_config_dir, canonical_config())

        assert Config.get_connections()["gateway"]["base_url"] == (
            "http://gateway.example/v1"
        )
        chat = Config.get_model_assignment("chat")
        assert chat.target == ModelTarget(
            "gateway", "chat-model", {"reasoning_effort": "medium"}
        )
        assert chat.fallbacks == (ModelTarget("desktop", "chat-fallback", {}),)

        for role in TASK_ROLES:
            Config.save_model_assignment(role, Config.get_model_assignment(role))

        raw = read_config(path)
        assert raw["model_settings_version"] == 1
        assert raw["unknown_user_key"] == "kept"
        assert raw["connections"] == canonical_config()["connections"]
        assert raw["models"] == {
            "chat": {
                "mode": "model",
                "connection": "gateway",
                "model": "chat-model",
                "options": {"reasoning_effort": "medium"},
                "fallbacks": [{"connection": "desktop", "model": "chat-fallback"}],
            },
            "embedding": {
                "mode": "model",
                "connection": "desktop",
                "model": "qwen3-embedding:0.6b",
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
                "model": "glm-ocr",
            },
            "metadata": {"mode": "chat"},
            "cleanup": {"mode": "off"},
        }
        assert "target" not in raw["models"]["chat"]
        assert_no_legacy_model_keys(raw)

    def test_serialization_is_plain_yaml_fields(self) -> None:
        assignment = TaskAssignment(
            mode="model",
            target=ModelTarget(
                "gateway", "chat-model", {"reasoning_effort": "medium"}
            ),
            enabled=True,
            fallbacks=(ModelTarget("desktop", "chat-fallback", {}),),
        )
        assert serialize_task_assignment("chat", assignment) == {
            "mode": "model",
            "connection": "gateway",
            "model": "chat-model",
            "options": {"reasoning_effort": "medium"},
            "fallbacks": [{"connection": "desktop", "model": "chat-fallback"}],
        }

    def test_explicit_model_control_words_are_literal(
        self, isolated_config_dir: Path
    ) -> None:
        for model in ("main", "provider:foo", "unset", "off", "chat", "ocr"):
            assignment = parse_task_assignment(
                "chat", {"mode": "model", "connection": "gateway", "model": model}
            )
            assert assignment.mode == "model"
            assert assignment.target is not None
            assert assignment.target.model == model

        config = canonical_config()
        config["models"]["chat"]["model"] = "main"
        config["models"]["chat"]["fallbacks"] = [
            {"connection": "desktop", "model": "provider:foo"}
        ]
        path = write_config(isolated_config_dir, config)

        assignment = Config.get_model_assignment("chat")
        assert assignment.target is not None
        assert assignment.target.model == "main"
        assert assignment.fallbacks[0].model == "provider:foo"

        Config.save_model_assignment("chat", assignment)
        raw = read_config(path)
        assert raw["models"]["chat"]["model"] == "main"
        assert raw["models"]["chat"]["fallbacks"][0]["model"] == "provider:foo"

    def test_assignment_models_and_options_do_not_leak_between_tasks(
        self, isolated_config_dir: Path
    ) -> None:
        config = canonical_config()
        config["models"]["chat"] = {
            "mode": "model",
            "connection": "gateway",
            "model": "chat-a",
            "options": {"reasoning_effort": "high", "nested": {"x": 1}},
        }
        config["models"]["embedding"] = {
            "mode": "model",
            "connection": "desktop",
            "model": "embed-a",
            "options": {"dimensions": 1024},
        }
        config["models"]["metadata"] = {
            "mode": "model",
            "connection": "gateway",
            "model": "meta-a",
            "options": {"temperature": 0.1},
        }
        path = write_config(isolated_config_dir, config)

        chat = Config.get_model_assignment("chat")
        embedding = Config.get_model_assignment("embedding")
        metadata = Config.get_model_assignment("metadata")
        assert chat.target is not None and chat.target.model == "chat-a"
        assert embedding.target is not None and embedding.target.model == "embed-a"
        assert metadata.target is not None and metadata.target.model == "meta-a"
        assert chat.target.options == {"reasoning_effort": "high", "nested": {"x": 1}}
        assert embedding.target.options == {"dimensions": 1024}
        assert metadata.target.options == {"temperature": 0.1}

        # Mutating one returned assignment's nested options must not reach the
        # cache or another task.
        chat.target.options["nested"]["x"] = 999
        embedding_again = Config.get_model_assignment("embedding")
        assert embedding_again.target is not None
        assert embedding_again.target.options == {"dimensions": 1024}

        Config.save_model_assignment(
            "chat",
            TaskAssignment(
                mode="model",
                target=ModelTarget(
                    "gateway", "chat-b", {"reasoning_effort": "low"}
                ),
                enabled=True,
                fallbacks=(),
            ),
        )
        raw = read_config(path)
        assert raw["models"]["chat"]["model"] == "chat-b"
        assert raw["models"]["embedding"]["model"] == "embed-a"
        assert raw["models"]["metadata"]["model"] == "meta-a"
        assert "model" not in raw["connections"]["gateway"]

    def test_parse_deep_copies_nested_options(self) -> None:
        source = {
            "mode": "model",
            "connection": "gateway",
            "model": "chat-a",
            "options": {"nested": {"x": 1}, "items": [1, 2]},
        }
        assignment = parse_task_assignment("chat", source)
        assert assignment.target is not None

        source["options"]["nested"]["x"] = 2
        source["options"]["items"].append(3)

        assert assignment.target.options == {"nested": {"x": 1}, "items": [1, 2]}

    def test_validate_accepts_a_complete_canonical_config(self) -> None:
        assert validate_model_settings(canonical_config()) is None


class TestAtomicNestedSave:
    def test_nested_connection_and_assignment_save_is_atomic(
        self, isolated_config_dir: Path
    ) -> None:
        config = {
            "model_settings_version": 1,
            "connections": {},
            "models": default_model_settings(),
        }
        path = write_config(isolated_config_dir, config)
        assignment = TaskAssignment(
            mode="model",
            target=ModelTarget(
                "gateway", "chat-model", {"reasoning_effort": "medium"}
            ),
            enabled=True,
            fallbacks=(ModelTarget("desktop", "fallback-model", {}),),
        )
        new_connections = {
            "gateway": {
                "type": "openai_compatible",
                "base_url": "http://gateway.example/v1",
                "api_key": "synthetic-placeholder",
            },
            "desktop": {
                "type": "ollama",
                "base_url": "http://localhost:11434",
            },
        }

        replaces: list[tuple[str, str]] = []
        real_replace = os.replace

        def record_replace(src: str, dst: str) -> None:
            replaces.append((src, dst))
            real_replace(src, dst)

        with patch.object(os, "replace", side_effect=record_replace):
            Config.save_model_assignment(
                "chat", assignment, new_connections=new_connections
            )

        assert len(replaces) == 1
        raw = read_config(path)
        assert raw["connections"] == new_connections
        assert raw["models"]["chat"] == {
            "mode": "model",
            "connection": "gateway",
            "model": "chat-model",
            "options": {"reasoning_effort": "medium"},
            "fallbacks": [{"connection": "desktop", "model": "fallback-model"}],
        }
        assert raw["models"]["embedding"] == {"mode": "unset"}
        assert_no_legacy_model_keys(raw)
        assert [p.name for p in isolated_config_dir.iterdir()] == ["config.yaml"]

    def test_failed_replace_leaves_file_and_cache_unchanged(
        self, isolated_config_dir: Path
    ) -> None:
        path = write_config(isolated_config_dir, canonical_config())
        before = path.read_bytes()
        cache_before = Config._load_config()
        assert cache_before is not None
        snapshot = copy.deepcopy(cache_before)

        assignment = TaskAssignment(
            mode="model",
            target=ModelTarget("gateway", "new-chat", {}),
            enabled=True,
            fallbacks=(),
        )
        with (
            patch.object(os, "replace", side_effect=OSError("replace failed")),
            pytest.raises(OSError),
        ):
            Config.save_model_assignment(
                "chat",
                assignment,
                new_connections={"fresh": {"type": "ollama"}},
            )

        assert path.read_bytes() == before
        assert Config._config_cache is cache_before
        assert Config._config_cache == snapshot
        assert [p.name for p in isolated_config_dir.iterdir()] == ["config.yaml"]
        assert "fresh" not in Config.get_connections()

    def test_mutating_the_caller_draft_after_save_changes_nothing(
        self, isolated_config_dir: Path
    ) -> None:
        path = write_config(isolated_config_dir, canonical_config())
        options = {"reasoning_effort": "medium", "nested": {"x": 1}}
        assignment = TaskAssignment(
            mode="model",
            target=ModelTarget("gateway", "chat-new", options),
            enabled=True,
            fallbacks=(),
        )
        new_connections = {
            "fresh": {
                "type": "ollama",
                "base_url": "http://fresh.example:11434",
                "oauth": {"access_token": "synthetic-access"},
            }
        }

        Config.save_model_assignment(
            "chat", assignment, new_connections=new_connections
        )

        options["reasoning_effort"] = "changed"
        options["nested"]["x"] = 999
        new_connections["fresh"]["base_url"] = "http://changed.example:11434"
        new_connections["fresh"]["oauth"]["access_token"] = "changed"

        cache = Config._config_cache
        assert cache is not None
        assert cache["models"]["chat"]["options"] == {
            "reasoning_effort": "medium",
            "nested": {"x": 1},
        }
        assert cache["connections"]["fresh"]["base_url"] == (
            "http://fresh.example:11434"
        )
        assert cache["connections"]["fresh"]["oauth"]["access_token"] == (
            "synthetic-access"
        )

        # And a later unrelated write cannot pick up the mutated draft.
        Config.set_config_value("logging.level", "DEBUG")
        raw = read_config(path)
        assert raw["models"]["chat"]["options"] == {
            "reasoning_effort": "medium",
            "nested": {"x": 1},
        }
        assert raw["connections"]["fresh"]["base_url"] == (
            "http://fresh.example:11434"
        )

    def test_returned_connections_are_detached_copies(
        self, isolated_config_dir: Path
    ) -> None:
        config = canonical_config()
        config["connections"]["gateway"]["oauth"] = {
            "access_token": "synthetic-access"
        }
        write_config(isolated_config_dir, config)

        connections = Config.get_connections()
        connections["gateway"]["base_url"] = "http://changed.example/v1"
        connections["gateway"]["oauth"]["access_token"] = "changed"

        fresh = Config.get_connection("gateway")
        assert fresh["base_url"] == "http://gateway.example/v1"
        assert fresh["oauth"] == {"access_token": "synthetic-access"}

        with pytest.raises(ValueError, match="Unknown connection"):
            Config.get_connection("missing")

    def test_new_model_settings_file_is_private(
        self, isolated_config_dir: Path
    ) -> None:
        Config.save_model_assignment(
            "chat",
            TaskAssignment(
                mode="model",
                target=ModelTarget("gateway", "chat-a", {}),
                enabled=True,
                fallbacks=(),
            ),
            new_connections={
                "gateway": {
                    "type": "openai_compatible",
                    "api_key": "synthetic-placeholder",
                }
            },
        )

        path = isolated_config_dir / "config.yaml"
        assert path.stat().st_mode & 0o077 == 0
        raw = read_config(path)
        assert raw["model_settings_version"] == 1
        assert_no_legacy_model_keys(raw)

    def test_connection_extension_fields_and_dotted_names_survive_saves(
        self, isolated_config_dir: Path
    ) -> None:
        config = canonical_config()
        config["connections"]["gateway.us"] = {
            "type": "openai_compatible",
            "base_url": "http://gateway.example/v1",
            "api_version": "2024-01-01",
            "oauth": {"access_token": "synthetic-access"},
            "unknown_extension": {"kept": True},
        }
        config["models"]["chat"]["connection"] = "gateway.us"
        path = write_config(isolated_config_dir, config)

        assert Config.get_connection("gateway.us")["api_version"] == "2024-01-01"
        Config.save_model_assignment("chat", Config.get_model_assignment("chat"))

        raw = read_config(path)
        assert raw["connections"]["gateway.us"] == config["connections"]["gateway.us"]
        assert raw["models"]["chat"]["connection"] == "gateway.us"

    def test_save_preserves_private_file_mode(self, isolated_config_dir: Path) -> None:
        path = write_config(isolated_config_dir, canonical_config())
        path.chmod(0o600)

        Config.save_model_assignment(
            "chat",
            TaskAssignment(
                mode="model",
                target=ModelTarget("gateway", "chat-new", {}),
                enabled=True,
                fallbacks=(),
            ),
        )

        assert path.stat().st_mode & 0o777 == 0o600


class TestValidation:
    def test_each_role_accepts_only_its_supported_modes(self) -> None:
        for role in TASK_ROLES:
            for mode in ("model", "chat", "ocr", "off", "unset"):
                value: dict[str, Any] = {"mode": mode}
                if mode == "model":
                    value.update(connection="gateway", model=f"{role}-model")
                if role == "rerank":
                    value["enabled"] = False

                if mode in SUPPORTED_MODES[role]:
                    assignment = parse_task_assignment(role, value)
                    assert assignment.mode == mode
                else:
                    with pytest.raises(ValueError):
                        parse_task_assignment(role, value)

        with pytest.raises(ValueError, match="Unknown model task"):
            parse_task_assignment("bogus", {"mode": "unset"})  # type: ignore[arg-type]

    def test_unknown_role_is_rejected(
        self, isolated_config_dir: Path
    ) -> None:
        write_config(isolated_config_dir, canonical_config())
        bogus_role = cast(TaskRole, "bogus")

        with pytest.raises(ValueError, match="Unknown model task"):
            Config.get_model_assignment(bogus_role)

        with pytest.raises(ValueError, match="Unknown model task"):
            Config.save_model_assignment(
                bogus_role,
                TaskAssignment(mode="unset", target=None, enabled=True, fallbacks=()),
            )

        config = canonical_config()
        config["models"]["bogus"] = {"mode": "unset"}
        with pytest.raises(ValueError, match="Unknown model task"):
            validate_model_settings(config)

        missing = canonical_config()
        del missing["models"]["chat"]
        with pytest.raises(ValueError, match="Missing model task"):
            validate_model_settings(missing)

    def test_missing_and_unsupported_versions_are_rejected(
        self, isolated_config_dir: Path
    ) -> None:
        unsupported = canonical_config()
        unsupported["model_settings_version"] = 2
        path = write_config(isolated_config_dir, unsupported)
        with pytest.raises(ValueError, match="Unsupported model settings version"):
            Config.get_connections()
        with pytest.raises(ValueError, match="Unsupported model settings version"):
            Config.get_model_assignment("chat")
        assert path.read_text() == yaml.safe_dump(unsupported, sort_keys=False)

        for version in ("1", 1.0, True):
            config = canonical_config()
            config["model_settings_version"] = version
            with pytest.raises(ValueError, match="Unsupported model settings version"):
                validate_model_settings(config)

        missing_marker = canonical_config()
        del missing_marker["model_settings_version"]
        write_config(isolated_config_dir, missing_marker)
        with pytest.raises(ValueError, match="model_settings_version"):
            Config.get_connections()
        with pytest.raises(ModelConfigurationRequired):
            validate_model_settings(missing_marker)

    def test_obsolete_model_keys_require_conversion_only_for_model_requests(
        self, isolated_config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ESDC_DB_FILE", raising=False)
        monkeypatch.delenv("ESDC_DB_DIR", raising=False)
        legacy_text = """\
api_url: https://legacy.example/
database_path: /tmp/synthetic-esdc.duckdb
providers:
  engine:
    provider_type: openai_compatible
    model: chat-a
"""
        path = isolated_config_dir / "config.yaml"
        path.write_text(legacy_text)
        Config._config_cache = None

        assert Config.get_api_url() == "https://legacy.example/"
        assert Config.get_db_file() == Path("/tmp/synthetic-esdc.duckdb")
        assert has_legacy_model_settings(Config._load_config() or {})

        with pytest.raises(ModelConfigurationRequired, match="not configured"):
            Config.get_connections()
        with pytest.raises(ModelConfigurationRequired, match="not configured"):
            Config.get_model_assignment("chat")
        with pytest.raises(ModelConfigurationRequired, match="not configured"):
            Config.save_model_assignment(
                "chat",
                TaskAssignment(mode="unset", target=None, enabled=True, fallbacks=()),
            )

        assert path.read_text() == legacy_text

    def test_malformed_mappings_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be a mapping"):
            parse_task_assignment("chat", ["model"])  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="options must be a mapping"):
            parse_task_assignment(
                "chat",
                {
                    "mode": "model",
                    "connection": "gateway",
                    "model": "chat-a",
                    "options": ["not", "a", "mapping"],
                },
            )
        with pytest.raises(ValueError, match="fallback"):
            parse_task_assignment(
                "chat",
                {
                    "mode": "model",
                    "connection": "gateway",
                    "model": "chat-a",
                    "fallbacks": ["not-a-target"],
                },
            )
        with pytest.raises(ValueError, match="Unknown field"):
            parse_task_assignment(
                "chat",
                {
                    "mode": "model",
                    "connection": "gateway",
                    "model": "chat-a",
                    "reasoning_effort": "medium",
                },
            )
        with pytest.raises(ValueError, match="Unknown field"):
            parse_task_assignment(
                "chat",
                {
                    "mode": "model",
                    "connection": "gateway",
                    "model": "chat-a",
                    "fallbacks": [
                        {
                            "connection": "desktop",
                            "model": "fallback",
                            "enabled": True,
                        }
                    ],
                },
            )
        with pytest.raises(ValueError, match="must not define"):
            parse_task_assignment(
                "chat",
                {"mode": "unset", "connection": "gateway", "model": "chat-a"},
            )

        config = canonical_config()
        config["connections"] = []  # type: ignore[assignment]
        with pytest.raises(ValueError, match="connections"):
            validate_model_settings(config)

        config = canonical_config()
        config["models"] = []  # type: ignore[assignment]
        with pytest.raises(ValueError, match="models"):
            validate_model_settings(config)

        config = canonical_config()
        config["models"]["chat"] = ["model"]  # type: ignore[assignment]
        with pytest.raises(ValueError, match="must be a mapping"):
            validate_model_settings(config)

        config = canonical_config()
        del config["connections"]
        with pytest.raises(ValueError, match="connections"):
            validate_model_settings(config)

    def test_empty_model_and_connection_are_rejected(self) -> None:
        for connection in ("", "   "):
            with pytest.raises(ValueError, match="connection"):
                parse_task_assignment(
                    "chat",
                    {"mode": "model", "connection": connection, "model": "chat-a"},
                )
        for model in ("", "   "):
            with pytest.raises(ValueError, match="model"):
                parse_task_assignment(
                    "chat",
                    {"mode": "model", "connection": "gateway", "model": model},
                )

        config = canonical_config()
        config["models"]["chat"]["model"] = ""  # type: ignore[typeddict-item]
        with pytest.raises(ValueError, match="model"):
            validate_model_settings(config)

        config = canonical_config()
        config["connections"][""] = {"type": "ollama"}  # type: ignore[index]
        with pytest.raises(ValueError, match="Connection names"):
            validate_model_settings(config)

    def test_unknown_connection_references_are_rejected(self) -> None:
        config = canonical_config()
        config["models"]["chat"]["connection"] = "missing"  # type: ignore[typeddict-item]
        with pytest.raises(ValueError, match="Unknown connection"):
            validate_model_settings(config)

        config = canonical_config()
        config["models"]["chat"]["fallbacks"] = [  # type: ignore[typeddict-item]
            {"connection": "missing", "model": "fallback"}
        ]
        with pytest.raises(ValueError, match="Unknown connection"):
            validate_model_settings(config)

    def test_unknown_reference_save_is_rejected_without_writing(
        self, isolated_config_dir: Path
    ) -> None:
        path = write_config(isolated_config_dir, canonical_config())
        before = path.read_bytes()

        with pytest.raises(ValueError, match="Unknown connection"):
            Config.save_model_assignment(
                "chat",
                TaskAssignment(
                    mode="model",
                    target=ModelTarget("missing", "chat-a", {}),
                    enabled=True,
                    fallbacks=(),
                ),
            )

        assert path.read_bytes() == before

    def test_duplicate_primary_and_fallback_pairs_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="Duplicate"):
            parse_task_assignment(
                "chat",
                {
                    "mode": "model",
                    "connection": "gateway",
                    "model": "chat-a",
                    "fallbacks": [
                        {"connection": "gateway", "model": "chat-a"},
                    ],
                },
            )

        with pytest.raises(ValueError, match="Duplicate"):
            parse_task_assignment(
                "chat",
                {
                    "mode": "model",
                    "connection": "gateway",
                    "model": "chat-a",
                    "fallbacks": [
                        {"connection": "desktop", "model": "fallback"},
                        {"connection": "desktop", "model": "fallback"},
                    ],
                },
            )

        # The same model on two different connections is a distinct pair.
        assignment = parse_task_assignment(
            "chat",
            {
                "mode": "model",
                "connection": "gateway",
                "model": "chat-a",
                "fallbacks": [{"connection": "desktop", "model": "chat-a"}],
            },
        )
        assert len(assignment.fallbacks) == 1

    def test_fallbacks_are_chat_only(self) -> None:
        with pytest.raises(ValueError, match="Only Chat supports fallbacks"):
            parse_task_assignment(
                "embedding",
                {
                    "mode": "model",
                    "connection": "desktop",
                    "model": "embed-a",
                    "fallbacks": [{"connection": "gateway", "model": "other"}],
                },
            )

    def test_rerank_enabled_is_a_separate_flag(self) -> None:
        assert default_model_settings()["rerank"]["enabled"] is False

        unset = parse_task_assignment("rerank", {"mode": "unset"})
        assert unset.enabled is False

        model = parse_task_assignment(
            "rerank",
            {"mode": "model", "connection": "gateway", "model": "rerank-a"},
        )
        assert model.enabled is False

        enabled = parse_task_assignment(
            "rerank",
            {
                "mode": "model",
                "connection": "gateway",
                "model": "rerank-a",
                "enabled": True,
            },
        )
        assert enabled.enabled is True

        with pytest.raises(ValueError, match="cannot be enabled"):
            parse_task_assignment("rerank", {"mode": "unset", "enabled": True})

        with pytest.raises(ValueError, match="Only Reranking"):
            parse_task_assignment(
                "chat",
                {
                    "mode": "model",
                    "connection": "gateway",
                    "model": "chat-a",
                    "enabled": False,
                },
            )

    def test_wrong_provider_type_for_a_role_is_rejected(self) -> None:
        config = canonical_config()
        config["connections"]["anthropic"] = {"type": "anthropic"}
        config["connections"]["openai"] = {"type": "openai"}

        config["models"]["rerank"] = {
            "mode": "model",
            "connection": "desktop",
            "model": "rerank-a",
            "enabled": False,
        }
        with pytest.raises(ValueError, match="cannot serve Reranking"):
            validate_model_settings(config)

        config = canonical_config()
        config["connections"]["anthropic"] = {"type": "anthropic"}
        config["models"]["embedding"] = {
            "mode": "model",
            "connection": "anthropic",
            "model": "embed-a",
        }
        with pytest.raises(ValueError, match="cannot serve Embeddings"):
            validate_model_settings(config)

        config = canonical_config()
        config["connections"]["anthropic"] = {"type": "anthropic"}
        config["models"]["ocr"] = {
            "mode": "model",
            "connection": "anthropic",
            "model": "ocr-a",
        }
        with pytest.raises(ValueError, match="cannot serve OCR"):
            validate_model_settings(config)

        config = canonical_config()
        config["connections"]["openai"] = {"type": "openai"}
        config["models"]["embedding"] = {
            "mode": "model",
            "connection": "openai",
            "model": "embed-a",
        }
        with pytest.raises(ValueError, match="cannot serve Embeddings"):
            validate_model_settings(config)

        config = canonical_config()
        config["connections"]["anthropic"] = {"type": "anthropic"}
        config["models"]["chat"] = {
            "mode": "model",
            "connection": "anthropic",
            "model": "chat-a",
        }
        assert validate_model_settings(config) is None

        config = canonical_config()
        config["connections"]["weird"] = {"type": "not_a_provider"}
        with pytest.raises(ValueError, match="unknown or missing provider type"):
            validate_model_settings(config)

    def test_connection_types_match_the_provider_registry(self) -> None:
        from esdc.providers import list_provider_types

        assert frozenset(list_provider_types()) == KNOWN_CONNECTION_TYPES

    def test_unknown_non_model_fields_survive_saves(
        self, isolated_config_dir: Path
    ) -> None:
        config = {
            "api_url": "https://example.invalid/",
            "logging": {"level": "DEBUG"},
            "custom_section": {"x": 1},
            "corpus": {"chunk_size": 1234, "custom_flag": True},
        }
        path = write_config(isolated_config_dir, config)

        Config.save_model_assignment(
            "chat",
            TaskAssignment(
                mode="model",
                target=ModelTarget("gateway", "chat-a", {}),
                enabled=True,
                fallbacks=(),
            ),
            new_connections={"gateway": {"type": "openai_compatible"}},
        )

        raw = read_config(path)
        assert raw["api_url"] == "https://example.invalid/"
        assert raw["logging"] == {"level": "DEBUG"}
        assert raw["custom_section"] == {"x": 1}
        assert raw["corpus"] == {"chunk_size": 1234, "custom_flag": True}
        assert raw["model_settings_version"] == 1
        assert raw["models"]["chat"] == {
            "mode": "model",
            "connection": "gateway",
            "model": "chat-a",
        }
        assert_no_legacy_model_keys(raw)

    def test_new_connection_name_cannot_replace_an_existing_one(
        self, isolated_config_dir: Path
    ) -> None:
        path = write_config(isolated_config_dir, canonical_config())
        before = path.read_bytes()

        with pytest.raises(ValueError, match="already exists"):
            Config.save_model_assignment(
                "chat",
                Config.get_model_assignment("chat"),
                new_connections={"gateway": {"type": "ollama"}},
            )

        assert path.read_bytes() == before


class TestPurity:
    def test_parsing_and_loading_do_not_create_clients_or_use_the_network(
        self, isolated_config_dir: Path
    ) -> None:
        write_config(isolated_config_dir, canonical_config())

        def fail(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("model parsing must not construct a client")

        with (
            patch("esdc.providers.create_provider", side_effect=fail),
            patch("esdc.providers._create_single_llm_from_config", side_effect=fail),
            patch("socket.create_connection", side_effect=fail),
        ):
            assert validate_model_settings(canonical_config()) is None
            assert parse_task_assignment(
                "chat", {"mode": "unset"}
            ) == TaskAssignment(mode="unset", target=None, enabled=True, fallbacks=())
            assert Config.get_connections()["gateway"]["type"] == "openai_compatible"
            assert Config.get_model_assignment("chat").mode == "model"


class TestChatPrimaryPromotion:
    """Changing Chat primary removes that exact target from fallbacks."""

    def test_promote_chat_fallback_preserves_options_and_remaining_order(self):
        assignment = TaskAssignment(
            mode="model",
            target=ModelTarget("gateway", "a", {}),
            fallbacks=(
                ModelTarget(
                    "backup",
                    "b",
                    {"reasoning_effort": "high", "nested": {"score": 1}},
                ),
                ModelTarget("gateway", "c", {"reasoning_effort": "low"}),
                ModelTarget("other", "d", {}),
            ),
        )

        promoted = with_chat_primary(
            assignment, ModelTarget("backup", "b", {})
        )

        assert promoted.mode == "model"
        assert promoted.target == ModelTarget(
            "backup",
            "b",
            {"reasoning_effort": "high", "nested": {"score": 1}},
        )
        assert [(pair.connection, pair.model) for pair in promoted.fallbacks] == [
            ("gateway", "c"),
            ("other", "d"),
        ]
        assert promoted.fallbacks[0].options == {"reasoning_effort": "low"}
        assert promoted.fallbacks[1].options == {}

    def test_chat_primary_promotion_uses_connection_and_model_pair(self):
        assignment = TaskAssignment(
            mode="model",
            target=ModelTarget("main", "primary", {}),
            fallbacks=(
                ModelTarget("one", "shared", {"owner": "one"}),
                ModelTarget("two", "shared", {"owner": "two"}),
            ),
        )

        promoted = with_chat_primary(
            assignment, ModelTarget("one", "shared", {})
        )

        assert promoted.target == ModelTarget("one", "shared", {"owner": "one"})
        assert promoted.fallbacks == (
            ModelTarget("two", "shared", {"owner": "two"}),
        )

    def test_chat_primary_reselection_and_new_target_preserve_other_state(self):
        reselected = with_chat_primary(
            TaskAssignment(
                mode="model",
                target=ModelTarget("gateway", "a", {"reasoning_effort": "medium"}),
                fallbacks=(ModelTarget("backup", "b", {}),),
            ),
            ModelTarget("gateway", "a", {}),
        )
        assert reselected.target == ModelTarget(
            "gateway", "a", {"reasoning_effort": "medium"}
        )
        assert reselected.fallbacks == (ModelTarget("backup", "b", {}),)

        new_target = with_chat_primary(
            TaskAssignment(
                mode="model",
                target=ModelTarget("gateway", "a", {}),
                fallbacks=(
                    ModelTarget("backup", "b", {}),
                    ModelTarget("other", "d", {}),
                ),
            ),
            ModelTarget("fresh", "new", {"reasoning_effort": "high"}),
        )
        assert new_target.target == ModelTarget(
            "fresh", "new", {"reasoning_effort": "high"}
        )
        assert [(pair.connection, pair.model) for pair in new_target.fallbacks] == [
            ("backup", "b"),
            ("other", "d"),
        ]
        assert all(pair.model != "a" for pair in new_target.fallbacks)

        first_model = with_chat_primary(
            TaskAssignment(mode="unset"),
            ModelTarget("gateway", "first", {}),
        )
        assert first_model.mode == "model"
        assert first_model.target == ModelTarget("gateway", "first", {})
        assert first_model.fallbacks == ()

    def test_chat_primary_promotion_detaches_nested_options(self):
        original_nested = {"items": [1]}
        assignment = TaskAssignment(
            mode="model",
            target=ModelTarget("gateway", "a", {}),
            fallbacks=(
                ModelTarget(
                    "backup",
                    "b",
                    {"nested": original_nested, "reasoning_effort": "high"},
                ),
            ),
        )

        promoted = with_chat_primary(
            assignment, ModelTarget("backup", "b", {})
        )

        assert promoted.target is not None
        promoted_target_options = cast(dict[str, Any], promoted.target.options)
        promoted_target_options["nested"]["items"].append(2)
        assert original_nested == {"items": [1]}
        assert assignment.fallbacks[0].options["nested"] == {"items": [1]}

        original_nested["items"].append(3)
        assert promoted.target.options["nested"] == {"items": [1, 2]}
