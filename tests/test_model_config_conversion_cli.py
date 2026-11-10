"""CLI and wizard tests for explicit model-settings conversion.

Every source file is temporary and synthetic. Nothing here reads or writes the
user's live `~/.esdc/config.yaml`, uses a real credential or opens a network.
"""

from __future__ import annotations

import stat
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml
from typer.testing import CliRunner

from esdc.configs import Config
from esdc.model_settings import default_model_settings, has_legacy_model_settings

API_KEY = "sk-synthetic-preview-key"
ACCESS_TOKEN = "synthetic-access-token"
REFRESH_TOKEN = "synthetic-refresh-token"
URL_USER = "synthetic-user"
URL_PASSWORD = "synthetic-url-password"
URL_QUERY = "synthetic-query-secret"

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch) -> Iterator[Path]:
    monkeypatch.setenv("ESDC_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("ESDC_DB_DIR", str(tmp_path))
    monkeypatch.setenv("ESDC_DB_FILE", str(tmp_path / "esdc.duckdb"))
    Config._config_cache = None
    yield tmp_path
    Config._config_cache = None


def legacy_yaml() -> str:
    return f"""\
api_url: https://legacy.example/
database_path: /tmp/synthetic-legacy.duckdb
custom_root:
  kept: true
providers:
  engine:
    provider_type: openai_compatible
    base_url: http://{URL_USER}:{URL_PASSWORD}@gateway.example/v1?api_key={URL_QUERY}
    model: chat-a
    api_key: {API_KEY}
    oauth:
      access_token: {ACCESS_TOKEN}
      refresh_token: {REFRESH_TOKEN}
default_provider: engine
provider_order: [engine]
embedding_backend: ollama
embedding_host: http://embed.example:11434
embedding_model: qwen3-embedding:0.6b
embedding_api_key: ""
embedding_provider: ""
corpus:
  chunk_size: 3000
  rerank: false
  rerank_backend: provider
  rerank_provider: ""
  rerank_model: ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF
  ocr_model: glm-ocr
  ocr_provider: ""
  metadata_model: main
  metadata_provider: ""
  cleanup_model: main
  cleanup_provider: ""
  ollama_host: ""
  n_gpu_layers: -1
"""


def canonical_config() -> dict[str, Any]:
    return {
        "model_settings_version": 1,
        "connections": {
            "gateway": {
                "type": "openai_compatible",
                "base_url": (
                    f"http://{URL_USER}:{URL_PASSWORD}@gateway.example/v1"
                    f"?api_key={URL_QUERY}"
                ),
                "api_key": API_KEY,
                "oauth": {
                    "access_token": ACCESS_TOKEN,
                    "refresh_token": REFRESH_TOKEN,
                },
            },
            "backup": {
                "type": "openai_compatible",
                "base_url": "http://backup.example/v1",
            },
        },
        "models": {
            **default_model_settings(),
            "chat": {
                "mode": "model",
                "connection": "gateway",
                "model": "chat-a",
                "fallbacks": [{"connection": "backup", "model": "fallback-a"}],
            },
        },
        "unknown_root": {"kept": True},
    }


def write_yaml(path: Path, payload: Any) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def invoke_configs(args: list[str]):
    from esdc.commands.configs import configs_app

    return runner.invoke(configs_app, args)


class TestConversionPreview:
    def test_preview_prints_a_masked_summary_and_writes_nothing(
        self, isolated_config: Path
    ):
        source = isolated_config / "legacy.yaml"
        source.write_text(legacy_yaml())
        before = source.read_text()
        Config._config_cache = {"sentinel": True}

        result = invoke_configs(
            ["convert-model-settings", "--input", str(source)]
        )

        assert result.exit_code == 0, result.output
        assert Config._config_cache == {"sentinel": True}
        output = result.output
        assert "Preview only" in output
        assert "Chat" in output and "gateway" in output
        for secret in (API_KEY, ACCESS_TOKEN, REFRESH_TOKEN, URL_PASSWORD, URL_QUERY):
            assert secret not in output
        assert URL_USER not in output
        assert source.read_text() == before
        assert not (isolated_config / "config.yaml").exists()
        assert sorted(p.name for p in isolated_config.iterdir()) == ["legacy.yaml"]

    def test_preview_does_not_initialize_a_missing_live_config(
        self, isolated_config: Path
    ):
        source = isolated_config / "legacy.yaml"
        source.write_text(legacy_yaml())

        result = invoke_configs(
            ["convert-model-settings", "--input", str(source)]
        )

        assert result.exit_code == 0, result.output
        assert not (isolated_config / "config.yaml").exists()

    def test_canonical_input_is_a_no_op(self, isolated_config: Path):
        source = isolated_config / "canonical.yaml"
        write_yaml(source, canonical_config())
        before = source.read_text()

        result = invoke_configs(
            ["convert-model-settings", "--input", str(source)]
        )

        assert result.exit_code == 0, result.output
        assert "already canonical" in result.output.lower()
        assert source.read_text() == before

    def test_a_blocked_conversion_is_reported_without_writing(
        self, isolated_config: Path
    ):
        mixed = canonical_config()
        mixed["providers"] = {"old": {"provider_type": "openai"}}
        source = isolated_config / "mixed.yaml"
        write_yaml(source, mixed)
        before = source.read_text()

        result = invoke_configs(
            ["convert-model-settings", "--input", str(source)]
        )

        assert result.exit_code == 1
        assert "mix" in result.output.lower() or "conflict" in result.output.lower()
        assert source.read_text() == before


class TestConversionApply:
    def test_apply_requires_a_named_backup(self, isolated_config: Path):
        source = isolated_config / "legacy.yaml"
        source.write_text(legacy_yaml())
        before = source.read_text()

        result = invoke_configs(
            ["convert-model-settings", "--input", str(source), "--apply"]
        )

        assert result.exit_code != 0
        assert "--backup" in result.output
        assert source.read_text() == before

    def test_apply_rejects_the_same_source_and_backup_path(
        self, isolated_config: Path
    ):
        source = isolated_config / "legacy.yaml"
        source.write_text(legacy_yaml())
        before = source.read_text()

        result = invoke_configs(
            [
                "convert-model-settings",
                "--input",
                str(source),
                "--apply",
                "--backup",
                str(source),
            ]
        )

        assert result.exit_code != 0
        assert "same" in result.output.lower()
        assert source.read_text() == before

    def test_apply_never_overwrites_an_existing_backup(
        self, isolated_config: Path
    ):
        source = isolated_config / "legacy.yaml"
        source.write_text(legacy_yaml())
        backup = isolated_config / "backup.yaml"
        backup.write_text("do not overwrite")
        before = source.read_text()

        result = invoke_configs(
            [
                "convert-model-settings",
                "--input",
                str(source),
                "--apply",
                "--backup",
                str(backup),
            ]
        )

        assert result.exit_code != 0
        assert backup.read_text() == "do not overwrite"
        assert source.read_text() == before

    def test_apply_creates_a_private_complete_backup_and_rewrites(
        self, isolated_config: Path
    ):
        source = isolated_config / "legacy.yaml"
        source.write_text(legacy_yaml())
        original = source.read_bytes()
        backup = isolated_config / "legacy.backup.yaml"

        result = invoke_configs(
            [
                "convert-model-settings",
                "--input",
                str(source),
                "--apply",
                "--backup",
                str(backup),
            ]
        )

        assert result.exit_code == 0, result.output
        assert "converted" in result.output.lower()
        assert backup.read_bytes() == original
        assert stat.S_IMODE(backup.stat().st_mode) == 0o600
        converted = yaml.safe_load(source.read_text())
        assert isinstance(converted, dict)
        assert not has_legacy_model_settings(converted)
        assert converted["model_settings_version"] == 1
        assert "providers" not in converted
        assert converted["custom_root"] == {"kept": True}

    def test_reapplying_a_canonical_file_does_not_create_another_backup(
        self, isolated_config: Path
    ):
        source = isolated_config / "legacy.yaml"
        source.write_text(legacy_yaml())
        first_backup = isolated_config / "first.backup.yaml"
        assert (
            invoke_configs(
                [
                    "convert-model-settings",
                    "--input",
                    str(source),
                    "--apply",
                    "--backup",
                    str(first_backup),
                ]
            ).exit_code
            == 0
        )
        converted = source.read_text()
        second_backup = isolated_config / "second.backup.yaml"

        result = invoke_configs(
            [
                "convert-model-settings",
                "--input",
                str(source),
                "--apply",
                "--backup",
                str(second_backup),
            ]
        )

        assert result.exit_code == 0, result.output
        assert "already canonical" in result.output.lower()
        assert not second_backup.exists()
        assert source.read_text() == converted

    def test_a_blocked_apply_writes_nothing(self, isolated_config: Path):
        mixed = canonical_config()
        mixed["providers"] = {"old": {"provider_type": "openai"}}
        source = isolated_config / "mixed.yaml"
        write_yaml(source, mixed)
        before = source.read_text()
        backup = isolated_config / "backup.yaml"

        result = invoke_configs(
            [
                "convert-model-settings",
                "--input",
                str(source),
                "--apply",
                "--backup",
                str(backup),
            ]
        )

        assert result.exit_code == 1
        assert not backup.exists()
        assert source.read_text() == before

    def test_a_source_changed_during_conversion_is_refused(
        self, isolated_config: Path, monkeypatch
    ):
        import esdc.model_config_apply as apply_mod

        source = isolated_config / "legacy.yaml"
        source.write_text(legacy_yaml())
        backup = isolated_config / "backup.yaml"
        real_convert = apply_mod.convert_model_settings

        def convert_and_touch(payload):
            conversion = real_convert(payload)
            source.write_text(legacy_yaml().replace("chunk_size: 3000", "chunk_size: 4000"))
            return conversion

        monkeypatch.setattr(apply_mod, "convert_model_settings", convert_and_touch)

        with pytest.raises(apply_mod.ModelSettingsApplyError, match="changed"):
            apply_mod.apply_model_settings(source, backup)

        assert "chunk_size: 4000" in source.read_text()
        assert backup.exists()


    def test_a_failed_replace_leaves_the_source_and_cache_intact(
        self, isolated_config: Path, monkeypatch
    ):
        import esdc.model_config_apply as apply_mod

        source = isolated_config / "legacy.yaml"
        source.write_text(legacy_yaml())
        before = source.read_bytes()
        backup = isolated_config / "backup.yaml"
        Config._config_cache = {"sentinel": True}
        monkeypatch.setattr(
            apply_mod.os, "replace", MagicMock(side_effect=OSError("disk full"))
        )

        with pytest.raises(apply_mod.ModelSettingsApplyError, match="original is intact"):
            apply_mod.apply_model_settings(source, backup)

        assert source.read_bytes() == before
        assert Config._config_cache == {"sentinel": True}
        assert backup.exists()


class TestCanonicalFlags:
    def test_old_default_provider_flag_explains_the_replacement(
        self, isolated_config: Path
    ):
        source = isolated_config / "config.yaml"
        write_yaml(source, canonical_config())
        before = source.read_text()

        result = invoke_configs(["--set-default-provider", "gateway"])

        assert result.exit_code != 0
        assert "--set-chat-primary" in result.output
        assert source.read_text() == before

    def test_old_provider_order_flag_explains_the_replacement(
        self, isolated_config: Path
    ):
        source = isolated_config / "config.yaml"
        write_yaml(source, canonical_config())
        before = source.read_text()

        result = invoke_configs(["--set-provider-order", "gateway,backup"])

        assert result.exit_code != 0
        assert "--set-chat-fallbacks" in result.output
        assert source.read_text() == before

    def test_set_chat_primary_accepts_a_connection_model_pair(
        self, isolated_config: Path
    ):
        source = isolated_config / "config.yaml"
        write_yaml(source, canonical_config())

        result = invoke_configs(["--set-chat-primary", "backup:new-chat"])

        assert result.exit_code == 0, result.output
        assignment = Config.get_model_assignment("chat")
        assert assignment.target is not None
        assert (assignment.target.connection, assignment.target.model) == (
            "backup",
            "new-chat",
        )
        # A primary change does not clear the distinct fallbacks.
        assert assignment.fallbacks[0].model == "fallback-a"

    @pytest.mark.parametrize("value", ["gateway", "gateway:", ":model", ""])
    def test_set_chat_primary_rejects_a_model_less_or_ambiguous_pair(
        self, isolated_config: Path, value: str
    ):
        source = isolated_config / "config.yaml"
        write_yaml(source, canonical_config())
        before = source.read_text()

        result = invoke_configs(["--set-chat-primary", value])

        assert result.exit_code != 0
        assert source.read_text() == before

    def test_set_chat_fallbacks_accepts_pairs_and_preserves_the_primary(
        self, isolated_config: Path
    ):
        source = isolated_config / "config.yaml"
        write_yaml(source, canonical_config())

        result = invoke_configs(
            ["--set-chat-fallbacks", "backup:first,gateway:second"]
        )

        assert result.exit_code == 0, result.output
        assignment = Config.get_model_assignment("chat")
        assert assignment.target is not None
        assert assignment.target.connection == "gateway"
        assert [pair.model for pair in assignment.fallbacks] == ["first", "second"]

    def test_primary_and_fallbacks_can_be_set_together(
        self, isolated_config: Path
    ):
        source = isolated_config / "config.yaml"
        write_yaml(source, canonical_config())

        result = invoke_configs(
            [
                "--set-chat-primary",
                "gateway:new-primary",
                "--set-chat-fallbacks",
                "backup:new-fallback",
            ]
        )

        assert result.exit_code == 0, result.output
        assignment = Config.get_model_assignment("chat")
        assert assignment.target is not None
        assert assignment.target.model == "new-primary"
        assert [pair.model for pair in assignment.fallbacks] == ["new-fallback"]

    def test_set_chat_fallbacks_requires_an_existing_primary(
        self, isolated_config: Path
    ):
        config = canonical_config()
        config["models"]["chat"] = {"mode": "unset"}
        source = isolated_config / "config.yaml"
        write_yaml(source, config)
        before = source.read_text()

        result = invoke_configs(["--set-chat-fallbacks", "backup:first"])

        assert result.exit_code != 0
        assert source.read_text() == before


class TestViewAndReset:
    def test_show_masks_canonical_connection_secrets_and_url_credentials(
        self, isolated_config: Path
    ):
        source = isolated_config / "config.yaml"
        write_yaml(source, canonical_config())
        Config._config_cache = None

        result = invoke_configs(["--show"])

        assert result.exit_code == 0, result.output
        output = result.output
        for secret in (API_KEY, ACCESS_TOKEN, REFRESH_TOKEN, URL_PASSWORD, URL_QUERY):
            assert secret not in output
        assert URL_USER not in output
        assert "connections.gateway.base_url" in output
        assert "gateway.example/v1" in output
        assert "connections.gateway.oauth.access_token" in output

    def test_reset_key_models_chat_resets_only_that_task(
        self, isolated_config: Path
    ):
        source = isolated_config / "config.yaml"
        write_yaml(source, canonical_config())
        before_connections = Config.get_connections()
        before_embedding = Config.get_model_assignment("embedding")

        result = invoke_configs(["--reset-key", "models.chat"])

        assert result.exit_code == 0, result.output
        assert Config.get_model_assignment("chat").mode == "unset"
        assert Config.get_connections() == before_connections
        assert Config.get_model_assignment("embedding") == before_embedding
        assert yaml.safe_load(source.read_text())["unknown_root"] == {"kept": True}

    def test_reset_key_rejects_an_incomplete_model_subkey(
        self, isolated_config: Path
    ):
        source = isolated_config / "config.yaml"
        write_yaml(source, canonical_config())
        before = source.read_text()

        result = invoke_configs(["--reset-key", "models.chat.mode"])

        assert result.exit_code != 0
        assert "models.chat" in result.output
        assert source.read_text() == before


class TestWizardConversionAction:
    def test_home_shows_the_update_action_only_for_legacy_settings(
        self, isolated_config: Path, monkeypatch
    ):
        import esdc.config_wizard as wizard

        legacy = isolated_config / "config.yaml"
        legacy.write_text(legacy_yaml())
        Config._config_cache = None
        select = MagicMock()
        question = MagicMock()
        question.unsafe_ask.return_value = "Exit"
        select.return_value = question
        monkeypatch.setattr(wizard.questionary, "select", select)

        wizard.run_wizard()

        labels = [
            str(choice.title) for choice in select.call_args.kwargs["choices"]
        ]
        assert "Update model settings…" in labels

        write_yaml(legacy, canonical_config())
        Config._config_cache = None
        select.reset_mock()
        wizard.run_wizard()
        labels = [
            str(choice.title) for choice in select.call_args.kwargs["choices"]
        ]
        assert "Update model settings…" not in labels

    def test_wizard_preview_cancel_writes_nothing(self, isolated_config, monkeypatch):
        import esdc.config_wizard as wizard

        legacy = isolated_config / "config.yaml"
        legacy.write_text(legacy_yaml())
        before = legacy.read_text()
        Config._config_cache = None
        select = MagicMock()
        select.return_value.unsafe_ask.side_effect = ["Update model settings…"]
        text = MagicMock()
        text.return_value.unsafe_ask.return_value = str(isolated_config / "backup.yaml")
        confirm = MagicMock()
        confirm.return_value.unsafe_ask.return_value = False
        monkeypatch.setattr(wizard.questionary, "select", select)
        monkeypatch.setattr(wizard.questionary, "text", text)
        monkeypatch.setattr(wizard.questionary, "confirm", confirm)

        wizard._update_model_settings_flow()

        assert legacy.read_text() == before
        assert not (isolated_config / "backup.yaml").exists()

    def test_wizard_apply_uses_the_shared_service(self, isolated_config, monkeypatch):
        import esdc.config_wizard as wizard

        legacy = isolated_config / "config.yaml"
        legacy.write_text(legacy_yaml())
        original = legacy.read_bytes()
        backup = isolated_config / "backup.yaml"
        Config._config_cache = None
        text = MagicMock()
        text.return_value.unsafe_ask.return_value = str(backup)
        confirm = MagicMock()
        confirm.return_value.unsafe_ask.return_value = True
        monkeypatch.setattr(wizard.questionary, "text", text)
        monkeypatch.setattr(wizard.questionary, "confirm", confirm)

        wizard._update_model_settings_flow()

        assert backup.read_bytes() == original
        converted = yaml.safe_load(legacy.read_text())
        assert isinstance(converted, dict)
        assert converted["model_settings_version"] == 1
        assert not has_legacy_model_settings(converted)
