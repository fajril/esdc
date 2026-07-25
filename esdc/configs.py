import contextlib
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Sensitive key suffixes to mask in the config UI.
SENSITIVE_KEYS = frozenset({"api_key"})

ENUM_CHOICES: dict[str, list[str]] = {
    "tool_format": ["native", "markdown", "auto"],
    "logging.level": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    "logging.server.level": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    "logging.agent.level": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    "logging.chat.level": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
}

KEY_DESCRIPTIONS: dict[str, str] = {
    "api_url": "Base URL for the ESDC API",
    "api.verify_ssl": "Verify SSL certificates for API requests",
    "database_path": "Path to the DuckDB database file",
    "tool_format": "Format for tool results (native, markdown, or auto)",
    "default_provider": "Default LLM provider name",
    "provider_order": "Ordered LLM provider failover list",
    "cache.sql_ttl": "SQL cache time-to-live in seconds",
    "logging.level": "Global logging level",
    "logging.file.enabled": "Enable logging to file",
    "logging.file.path": "Log file path",
    "logging.file.max_size": "Maximum log file size before rotation",
    "logging.file.backup_count": "Number of rotated log files to keep",
    "logging.server.level": "Log level for the server component",
    "logging.agent.level": "Log level for the agent component",
    "logging.chat.level": "Log level for the chat component",
    "semantic_search.embedding_batch_size": ("Number of embeddings per batch (10-500)"),
    "embedding_host": "Ollama host URL for embeddings (default: localhost)",
    "corpus.ocr_model": "Ollama vision model used for OCR of scanned pages",
    "corpus.metadata_model": (
        "Text LLM for metadata pre-fill at extract time ('main' = default "
        "chat provider [default], 'provider:<name>' = a configured provider, "
        "an Ollama model name to override, or '' = image-based prefill via "
        "ocr_model)"
    ),
    "corpus.cleanup_model": (
        "LLM for formatting cleanup of native-extracted pages at extract "
        "time ('main' = default chat provider [default], 'provider:<name>' "
        "= a configured provider, an Ollama model name to override, or "
        "'' = off); guarded — original text kept if the model invents "
        "numbers or changes length grossly"
    ),
    "corpus.ollama_host": (
        "Ollama server URL for corpus OCR and Ollama-named corpus models "
        "('' = local daemon, e.g. http://gpu-box:11434 for a remote server)"
    ),
    "corpus.chunk_size": "Max characters per corpus chunk",
    "corpus.chunk_overlap": "Characters carried over between corpus chunks",
    "corpus.rerank": (
        "Enable local cross-encoder rerank stage over RRF search results "
        "(off by default; first use downloads the model, ~1 GB)"
    ),
    "corpus.rerank_pool": "Number of RRF candidates scored when rerank is on",
    "corpus.rerank_model": (
        "Reranker GGUF id run in-process via llama.cpp "
        "(default ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF)"
    ),
    "corpus.ocr_dpi": "Page render resolution (DPI) for OCR",
    "corpus.num_ctx": "Ollama context window size for corpus OCR/metadata models",
    "corpus.n_gpu_layers": (
        "llama.cpp GPU offload for corpus embed/rerank "
        "(-1=auto: all layers if a GPU is present else CPU; 0=force CPU)"
    ),
    "corpus.min_chars_per_page": (
        "Text-layer character threshold below which a page counts as scanned"
    ),
    "corpus.min_image_area": (
        "Minimum embedded-image size (fraction of page area) that gets OCR'd "
        "on native-text pages; smaller images (logos, signatures) are ignored"
    ),
    "phoenix.enabled": "Enable Phoenix/OpenInference tracing",
    "phoenix.collector_endpoint": "Phoenix OTLP collector endpoint URL",
    "phoenix.project_name": "Phoenix project name for trace grouping",
}


# Model/endpoint-role groups shown under "Models & endpoints" in the wizard.
# "Chat providers" is intentionally absent here — it is served by the
# provider CRUD flows, not by flat key editing.
MODEL_SECTIONS: dict[str, list[str]] = {
    "Embeddings": [
        "embedding_host",
        "semantic_search.embedding_batch_size",
    ],
    "Corpus models": [
        "corpus.ocr_model",
        "corpus.metadata_model",
        "corpus.cleanup_model",
        "corpus.ollama_host",
        "corpus.num_ctx",
    ],
}

# Non-model settings shown under "Settings" in the wizard.
SETTINGS_SECTIONS: dict[str, list[str]] = {
    "Connection": ["api_url", "api.verify_ssl", "database_path"],
    "Chat behavior": ["tool_format"],
    "Corpus processing": [
        "corpus.chunk_size",
        "corpus.chunk_overlap",
        "corpus.rerank",
        "corpus.rerank_pool",
        "corpus.rerank_model",
        "corpus.ocr_dpi",
        "corpus.min_chars_per_page",
        "corpus.min_image_area",
        "corpus.n_gpu_layers",
    ],
    "Logging": [
        "logging.level",
        "logging.file.enabled",
        "logging.file.path",
        "logging.file.max_size",
        "logging.file.backup_count",
        "logging.server.level",
        "logging.agent.level",
        "logging.chat.level",
    ],
    "Cache": ["cache.sql_ttl"],
    "Observability": [
        "phoenix.enabled",
        "phoenix.collector_endpoint",
        "phoenix.project_name",
    ],
}


# Note: frozen=True was removed to allow method-based configuration updates
# (e.g., update_provider_config). The Config class uses classmethods for
# configuration management, so immutability is handled at the application level.
@dataclass
class Config:
    """ESDC application configuration manager."""

    APP_NAME: str = "esdc"
    DB_FILENAME: str = "esdc.duckdb"
    LEGACY_DB_FILENAME: str = "esdc.db"
    BASE_API_URL_V2: str = "https://esdc.skkmigas.go.id/"
    _config_cache: dict[str, Any] | None = None

    @classmethod
    def get_config_dir(cls) -> Path:
        """Return the config directory path (~/.esdc).

        Priority:
        1. ESDC_CONFIG_DIR environment variable
        2. ~/.esdc (default)
        """
        custom_path = os.environ.get("ESDC_CONFIG_DIR")
        if custom_path:
            return Path(custom_path)
        return Path.home() / f".{cls.APP_NAME}"

    @classmethod
    def get_config_file(cls) -> Path:
        """Return the config file path (~/.esdc/config.yaml)."""
        return cls.get_config_dir() / "config.yaml"

    @classmethod
    def get_corpus_queries_path(cls) -> Path:
        """Return the eval query set path (~/.esdc/corpus_queries.jsonl)."""
        return cls.get_config_dir() / "corpus_queries.jsonl"

    @classmethod
    def _default_db_file(cls) -> Path:
        """Return the default DuckDB file path."""
        return cls.get_config_dir() / cls.DB_FILENAME

    @classmethod
    def _legacy_default_db_file(cls) -> Path:
        """Return the pre-rename default database file path."""
        return cls.get_config_dir() / cls.LEGACY_DB_FILENAME

    @classmethod
    def _migrate_legacy_default_db_file(cls, *, update_config: bool) -> Path:
        """Rename the old default database file to the new default path once."""
        default_db_file = cls._default_db_file()
        legacy_db_file = cls._legacy_default_db_file()

        if not default_db_file.exists() and legacy_db_file.exists():
            legacy_db_file.rename(default_db_file)

        if update_config:
            config = cls._load_config() or {}
            configured_path = config.get("database_path")
            if (
                configured_path
                and Path(configured_path).expanduser() == legacy_db_file
            ):
                config["database_path"] = str(default_db_file)
                cls._save_config(config)

        return default_db_file

    @classmethod
    def _load_config(cls) -> dict[str, Any] | None:
        """Load config from YAML file (cached)."""
        if cls._config_cache is not None:
            return cls._config_cache

        config_file = cls.get_config_file()
        if config_file.exists():
            with open(config_file) as f:
                cls._config_cache = yaml.safe_load(f) or {}
                cls._normalize_database_config()
                return cls._config_cache
        cls._config_cache = {}
        return None

    @classmethod
    def _normalize_database_config(cls) -> None:
        """Keep legacy chat database.path compatible with database_path."""
        config = cls._config_cache
        if not config or "database_path" in config:
            return

        database_config = config.get("database")
        if not isinstance(database_config, dict):
            return

        db_path = database_config.get("path")
        if not db_path:
            return

        config["database_path"] = db_path
        cls._save_config(config)

    @classmethod
    def init_config(cls) -> None:
        """Create default config directory and file if not exists."""
        config_dir = cls.get_config_dir()
        config_file = cls.get_config_file()

        if not config_dir.exists():
            config_dir.mkdir(parents=True, exist_ok=True)

        if not config_file.exists():
            default_config = {
                "api_url": cls.BASE_API_URL_V2,
                "api": {"verify_ssl": True},
                "database_path": str(cls._default_db_file()),
                "tool_format": "native",  # native, markdown, or auto
                "cache": {"sql_ttl": 604800},
                "logging": {
                    "level": "INFO",
                    "file": {
                        "enabled": True,
                        "path": "logs/esdc.log",
                        "max_size": "10MB",
                        "backup_count": 5,
                    },
                    "server": {"level": "INFO"},
                    "agent": {"level": "DEBUG"},
                    "chat": {"level": "WARNING"},
                },
                "semantic_search": {
                    "embedding_batch_size": 100,  # Number of embeddings per batch (10-500)  # noqa: E501
                },
                "corpus": dict(cls.CORPUS_DEFAULTS),
                "phoenix": {
                    "enabled": False,
                    "collector_endpoint": "http://localhost:4317",
                    "project_name": "iris",
                },
            }
            with open(config_file, "w") as f:
                yaml.dump(default_config, f, default_flow_style=False)

    @classmethod
    def get_api_url(cls) -> str:
        """Get API URL with priority: env var > config.yaml > default."""
        env_url = os.environ.get("ESDC_URL")
        if env_url:
            return env_url

        config = cls._load_config()
        if config and "api_url" in config:
            return config["api_url"]

        return cls.BASE_API_URL_V2

    @classmethod
    def get_tool_format(cls) -> str:
        """Get tool format with priority: env var > config.yaml > default.

        Environment variable ESDC_TOOL_FORMAT overrides config file.
        Valid values: "native", "markdown", "auto"

        Returns:
            Tool format string (native, markdown, or auto)
        """
        # Priority 1: Environment variable
        env_format = os.environ.get("ESDC_TOOL_FORMAT", "").lower()
        if env_format in ("native", "markdown", "auto"):
            return env_format

        # Priority 2: Config file
        config = cls._load_config()
        if config and "tool_format" in config:
            config_format = config["tool_format"].lower()
            if config_format in ("native", "markdown", "auto"):
                return config_format

        # Priority 3: Default
        return "native"

    @classmethod
    def get_credentials(cls) -> tuple[str, str]:
        """Get credentials with priority: env vars > interactive prompt.

        Returns:
            tuple: (username, password)

        Note:
            This method will prompt for username and password if not
            available via environment variables.
        """
        username = os.environ.get("ESDC_USER")
        password = os.environ.get("ESDC_PASS")

        if username and password:
            return username, password

        try:
            from rich.prompt import Prompt
        except ImportError as err:
            raise RuntimeError(
                "Rich library required for interactive prompts. "
                "Install with: pip install rich"
            ) from err

        if not username:
            username = Prompt.ask("Username")
        if not password:
            password = Prompt.ask("Password", password=True)

        return username, password

    @classmethod
    def get_db_dir(cls) -> Path:
        """Return the database directory path.

        Priority:
        1. ESDC_DB_DIR environment variable
        2. config.yaml database_path (directory)
        3. ~/.esdc (default)
        """
        custom_path = os.environ.get("ESDC_DB_DIR")
        if custom_path:
            return Path(custom_path)

        config = cls._load_config()
        if config and "database_path" in config:
            db_path = Path(config["database_path"]).expanduser()
            return db_path.parent

        return cls.get_config_dir()

    @classmethod
    def get_db_file(cls) -> Path:
        """Return the database file path.

        Priority:
        1. ESDC_DB_FILE environment variable (full file path)
        2. config.yaml database_path
        3. ~/.esdc/esdc.duckdb (default)
        """
        env_file = os.environ.get("ESDC_DB_FILE")
        if env_file:
            return Path(env_file)

        config = cls._load_config()
        if config and "database_path" in config:
            db_path = Path(config["database_path"]).expanduser()
            if db_path == cls._legacy_default_db_file():
                return cls._migrate_legacy_default_db_file(update_config=True)
            return db_path

        return cls._migrate_legacy_default_db_file(update_config=False)

    @classmethod
    def get_db_path(cls) -> Path:
        """Return the database directory (backwards compatibility)."""
        return cls.get_db_dir()

    @classmethod
    def _save_config(cls, config: dict[str, Any]) -> None:
        """Save config to YAML file and update cache."""
        cls._config_cache = config
        config_file = cls.get_config_file()
        config_dir = cls.get_config_dir()
        if not config_dir.exists():
            config_dir.mkdir(parents=True, exist_ok=True)
        with open(config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    @classmethod
    def get_providers(cls) -> dict:
        """Get all provider configurations."""
        config = cls._load_config()
        return config.get("providers", {}) if config else {}

    @classmethod
    def save_provider(cls, name: str, provider_config: dict) -> None:
        """Save a provider configuration."""
        config = cls._load_config() or {}
        providers = config.get("providers", {})
        providers[name] = provider_config
        config["providers"] = providers
        cls._save_config(config)

    @classmethod
    def remove_provider(cls, name: str) -> bool:
        """Remove a provider configuration. Returns True if removed."""
        config = cls._load_config() or {}
        providers = config.get("providers", {})
        if name in providers:
            del providers[name]
            config["providers"] = providers
            cls._save_config(config)
            return True
        return False

    @classmethod
    def set_default_provider(cls, name: str) -> None:
        """Set the default provider."""
        config = cls._load_config() or {}
        config["default_provider"] = name
        provider_order = config.get("provider_order")
        if isinstance(provider_order, list):
            remaining = [p for p in provider_order if p != name]
            config["provider_order"] = [name] + remaining
        cls._save_config(config)

    @classmethod
    def get_provider_config(cls) -> dict[str, Any] | None:
        """Get provider configuration from config file.

        Returns the config for the default provider. If ``provider_order`` is
        configured, fallback provider configs are attached under
        ``fallback_configs`` for downstream LLM creation.
        """
        configs = cls.get_provider_configs_by_priority()
        if not configs:
            return None

        primary = dict(configs[0])
        fallbacks = configs[1:]
        if fallbacks:
            primary["fallback_configs"] = fallbacks
        return primary

    @classmethod
    def get_provider_order(cls) -> list[str]:
        """Return provider names in failover priority order."""
        config = cls._load_config()
        if not config:
            return []

        default_provider = config.get("default_provider")
        configured_order = config.get("provider_order", [])
        providers = config.get("providers", {})

        ordered: list[str] = []
        if default_provider and default_provider in providers:
            ordered.append(default_provider)

        if isinstance(configured_order, list):
            for provider_name in configured_order:
                if (
                    isinstance(provider_name, str)
                    and provider_name in providers
                    and provider_name not in ordered
                ):
                    ordered.append(provider_name)

        return ordered

    @classmethod
    def set_provider_order(cls, provider_names: list[str]) -> None:
        """Set ordered provider failover list."""
        config = cls._load_config() or {}
        providers = config.get("providers", {})
        unknown = [name for name in provider_names if name not in providers]
        if unknown:
            raise ValueError(f"Unknown provider(s): {', '.join(unknown)}")

        deduped = list(dict.fromkeys(provider_names))
        config["provider_order"] = deduped
        if deduped:
            config["default_provider"] = deduped[0]
        cls._save_config(config)

    @classmethod
    def get_provider_configs_by_priority(cls) -> list[dict[str, Any]]:
        """Return provider configs ordered for failover."""
        config = cls._load_config()
        if not config:
            return []

        providers = config.get("providers", {})
        ordered_names = cls.get_provider_order()
        configs: list[dict[str, Any]] = []
        for name in ordered_names:
            provider_config = providers.get(name)
            if not isinstance(provider_config, dict):
                continue
            cfg = dict(provider_config)
            cfg.setdefault("name", name)
            cfg.setdefault("provider_type", cfg.get("type") or name)
            configs.append(cfg)

        return configs

    @classmethod
    def get_default_provider(cls) -> str:
        """Get default provider name."""
        config = cls._load_config() or {}
        return config.get("default_provider", "ollama")

    @classmethod
    def get_provider_api_key(cls) -> str:
        """Get provider API key from env var or config."""
        env_key = os.environ.get("OPENAI_API_KEY")
        if env_key:
            return env_key

        provider_config = cls.get_provider_config()
        if provider_config and "api_key" in provider_config:
            return provider_config["api_key"]

        return ""

    @classmethod
    def get_provider_model(cls) -> str:
        """Get provider model from config."""
        provider_config = cls.get_provider_config()
        if provider_config and "model" in provider_config:
            return provider_config["model"]
        return "gpt-4o"

    @classmethod
    def get_log_level(cls) -> str:
        """Get log level from config.

        Priority:
        1. ESDC_LOG_LEVEL environment variable (0 = disable)
        2. config.yaml: chat.log_level

        Returns:
            Log level string: DEBUG, INFO, WARNING, ERROR, or 0 (disable)
        """
        env_level = os.environ.get("ESDC_LOG_LEVEL")
        if env_level:
            return env_level.upper()

        config = cls._load_config() or {}
        chat_config = config.get("chat", {})
        return chat_config.get("log_level", "INFO")

    @classmethod
    def get_logging_config(cls) -> dict[str, Any]:
        """Get logging configuration from config file.

        Priority:
        1. ESDC_LOG_LEVEL environment variable (applies to all components)
        2. config.yaml: logging.* structure
        3. config.yaml: chat.log_level (backward compatibility)
        4. Default values

        Returns:
            Logging configuration dict with structure:
            {
                "level": "INFO",
                "file": {"enabled": True, "path": "logs/esdc.log",
                "max_size": "10MB", "backup_count": 5},
                "server": {"level": "INFO"},
                "agent": {"level": "DEBUG"},
                "chat": {"level": "WARNING"}
            }
        """
        config = cls._load_config() or {}

        # Check environment variable first (overrides everything)
        env_level = os.environ.get("ESDC_LOG_LEVEL")
        if env_level:
            env_level = env_level.upper()
            if env_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "0"):
                env_level = "INFO"

        # Try new logging structure
        logging_config = config.get("logging", {})

        # Backward compatibility: fall back to old chat.log_level for base level
        if not logging_config:
            old_level = config.get("chat", {}).get("log_level", "INFO")
            # When migrating from old config, use old_level for all components
            logging_config = {
                "level": old_level,
                "server": {"level": old_level},
                "agent": {"level": old_level},
                "chat": {"level": old_level},
            }

        # Apply environment variable override (applies to all components)
        if env_level:
            logging_config["level"] = env_level

        # Get global level from config
        global_level = logging_config.get("level", "INFO")

        # Get component levels with defaults
        # Priority: component level > env override > global level > component default
        server_level = logging_config.get("server", {}).get("level")
        agent_level = logging_config.get("agent", {}).get("level")
        chat_level = logging_config.get("chat", {}).get("level")

        # Apply env override to components if set
        if env_level:
            server_level = env_level
            agent_level = env_level
            chat_level = env_level

        result = {
            "level": global_level,
            "file": {
                "enabled": logging_config.get("file", {}).get("enabled", True),
                "path": logging_config.get("file", {}).get("path", "logs/esdc.log"),
                "max_size": logging_config.get("file", {}).get("max_size", "10MB"),
                "backup_count": logging_config.get("file", {}).get("backup_count", 5),
            },
            "server": {"level": server_level if server_level else "INFO"},
            "agent": {"level": agent_level if agent_level else "DEBUG"},
            "chat": {"level": chat_level if chat_level else "WARNING"},
        }

        return result

    @classmethod
    def get_provider_base_url(cls) -> str:
        """Get provider base URL from config."""
        provider_config = cls.get_provider_config()
        if provider_config and "base_url" in provider_config:
            return provider_config["base_url"]
        return ""

    @classmethod
    def get_chat_db_path(cls) -> Path:
        """Get database path for chat from config."""
        config = cls._load_config() or {}
        db_config = config.get("database", {})
        if db_path := db_config.get("path"):
            return Path(db_path).expanduser().resolve()
        return cls.get_db_file().resolve()

    @classmethod
    def set_chat_db_path(cls, path: Path) -> None:
        """Set database path for chat in config."""
        config = cls._load_config() or {}
        if "database" not in config:
            config["database"] = {}
        config["database"]["path"] = str(path)
        cls._save_config(config)

    @classmethod
    def get_sql_cache_ttl(cls) -> int:
        """Get SQL cache TTL in seconds.

        Priority:
        1. ESDC_SQL_CACHE_TTL environment variable
        2. config.yaml: cache.sql_ttl
        3. 604800 (1 week default)
        """
        env_ttl = os.environ.get("ESDC_SQL_CACHE_TTL")
        if env_ttl:
            try:
                return int(env_ttl)
            except ValueError:
                pass

        config = cls._load_config() or {}
        cache_config = config.get("cache", {})
        return cache_config.get("sql_ttl", 604800)

    @classmethod
    def get_cache_dir(cls) -> Path:
        """Return the cache directory path.

        Priority:
        1. ESDC_CACHE_DIR environment variable
        2. config.yaml: cache.path
        3. ~/.esdc/cache (default)
        """
        custom_path = os.environ.get("ESDC_CACHE_DIR")
        if custom_path:
            return Path(custom_path)

        config = cls._load_config() or {}
        cache_config = config.get("cache", {})
        if cache_path := cache_config.get("path"):
            return Path(cache_path).expanduser()

        return cls.get_config_dir() / "cache"

    @classmethod
    def get_provider_config_by_name(cls, name: str) -> dict[str, Any] | None:
        """Get provider configuration by name."""
        providers = cls.get_providers()
        return providers.get(name)

    @classmethod
    def update_provider_config(cls, name: str, config_data: dict[str, Any]) -> None:
        """Update a provider configuration."""
        config = cls._load_config() or {}
        if "providers" not in config:
            config["providers"] = {}
        config["providers"][name] = config_data
        cls._save_config(config)

    @classmethod
    def persist_provider_oauth(cls, provider_name: str, oauth: dict[str, Any]) -> None:
        """Persist refreshed OAuth tokens for one provider back to disk.

        OAuth providers may rotate the refresh_token on every refresh. The
        in-memory ``ProviderConfig.oauth`` dict is updated by the caller
        immediately after a refresh, but that update is lost on process
        restart unless it is also written to the config file. This method
        performs that write.

        Only the named provider's ``oauth`` section is modified; every other
        key (including other providers) is left untouched. The config file
        is read fresh from disk (bypassing the in-memory cache) and written
        back atomically via a temp file + ``os.replace``, with permissions
        restricted to the owner (0o600) since it may contain access and
        refresh tokens.

        If the provider is not present in the on-disk config (e.g. it was
        configured purely via environment variables) or the config file does
        not exist, this logs a warning and returns without error. Callers
        should treat persistence failures as non-fatal: refreshing the token
        in memory must still succeed even if the write to disk fails.

        Args:
            provider_name: Name of the provider to update (matches
                ``ProviderConfig.name``).
            oauth: The refreshed OAuth token dict to store for the provider.
        """
        config_file = cls.get_config_file()
        if not config_file.exists():
            logger.warning(
                "Skipping OAuth token persistence for provider '%s': "
                "config file %s does not exist",
                provider_name,
                config_file,
            )
            return

        with open(config_file) as f:
            config = yaml.safe_load(f) or {}

        providers = config.get("providers")
        if not isinstance(providers, dict) or provider_name not in providers:
            logger.warning(
                "Skipping OAuth token persistence: provider '%s' not found "
                "in config file %s",
                provider_name,
                config_file,
            )
            return

        provider_entry = providers[provider_name]
        if not isinstance(provider_entry, dict):
            provider_entry = {}
            providers[provider_name] = provider_entry
        provider_entry["oauth"] = oauth

        config_dir = cls.get_config_dir()
        fd, tmp_name = tempfile.mkstemp(
            dir=config_dir, prefix=".config-", suffix=".yaml.tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            os.chmod(tmp_name, 0o600)
            os.replace(tmp_name, config_file)
        except BaseException:
            with contextlib.suppress(OSError):
                os.remove(tmp_name)
            raise

        # Invalidate rather than assign: _load_config applies
        # _normalize_database_config() on load, which raw file content
        # would bypass.
        cls._config_cache = None

    @classmethod
    def get_verify_ssl(cls) -> bool:
        """Get SSL certificate verification setting.

        Priority:
        1. ESDC_VERIFY_SSL environment variable ('true'/'1' = True, 'false'/'0' = False)
        2. config.yaml: api.verify_ssl
        3. True (verify SSL by default)

        Returns:
            True to verify SSL certificates, False to skip verification
        """
        env_val = os.environ.get("ESDC_VERIFY_SSL", "").lower()
        if env_val in ("true", "1", "yes"):
            return True
        if env_val in ("false", "0", "no"):
            return False

        config = cls._load_config() or {}
        api_config = config.get("api", {})
        return api_config.get("verify_ssl", True)

    @classmethod
    def has_chat_config(cls) -> bool:
        """Check if chat configuration exists."""
        config = cls._load_config() or {}
        return "providers" in config and len(config.get("providers", {})) > 0

    @classmethod
    def get_embedding_batch_size(cls) -> int:
        """Get embedding batch size from config.

        Priority:
        1. ESDC_EMBEDDING_BATCH_SIZE environment variable
        2. config.yaml: semantic_search.embedding_batch_size
        3. 100 (default)

        Returns:
            Batch size for embedding generation (number of documents per batch)
        """
        env_batch_size = os.environ.get("ESDC_EMBEDDING_BATCH_SIZE")
        if env_batch_size:
            try:
                size = int(env_batch_size)
                if size > 0:
                    return size
            except ValueError:
                pass

        config = cls._load_config() or {}
        semantic_config = config.get("semantic_search", {})
        return semantic_config.get("embedding_batch_size", 100)

    @classmethod
    def get_embedding_host(cls) -> str | None:
        """Get embedding service host URL from config.

        Priority:
        1. ESDC_EMBEDDING_HOST environment variable
        2. config.yaml: embedding_host
        3. None (uses localhost Ollama default)

        Returns:
            Host URL string or None for localhost default
        """
        env_host = os.environ.get("ESDC_EMBEDDING_HOST")
        if env_host:
            return env_host

        config = cls._load_config() or {}
        return config.get("embedding_host") or None

    CORPUS_DEFAULTS = {
        "chunk_size": 3000,  # max chars per chunk (~750 tokens)
        "chunk_overlap": 300,  # chars carried over between chunks
        # rerank: second-stage cross-encoder over the RRF top pool.
        # Off by default until `esdc corpus eval` justifies it; first use
        # downloads the model (~1 GB, cached).
        "rerank": False,
        "rerank_pool": 30,  # candidates scored per query when rerank is on
        # rerank_model: reranker GGUF id (llama.cpp). Runtime-only (output
        # not stored), so safe to change without reembedding.
        "rerank_model": "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF",
        "ocr_model": "glm-ocr",  # Ollama OCR model (zai-org/GLM-OCR, 0.9B)
        # metadata_model: text LLM for metadata extraction; "main" = default
        # chat provider, "" = use ocr_model on the rendered first page
        "metadata_model": "main",
        # cleanup_model: reformats native-extracted pages before review;
        # "main" = default chat provider, "" = off
        "cleanup_model": "main",
        "ocr_dpi": 200,  # page render resolution; raise to 300 if OCR quality poor
        "num_ctx": 16384,  # Ollama context window; glm-ocr crashes on images below this
        "min_chars_per_page": 50,  # text-layer chars below which a page counts as scanned  # noqa: E501
        "min_image_area": 0.05,  # embedded-image area (fraction of page) below which images are ignored  # noqa: E501
        # ollama_host: Ollama server for corpus OCR + Ollama-named text
        # models; "" = local daemon (http://127.0.0.1:11434)
        "ollama_host": "",
        # n_gpu_layers: llama.cpp GPU offload for corpus embed/rerank.
        # -1 (default) = offload all layers when a GPU backend is present
        # (Metal on the mac wheel, CUDA on a cuXXX wheel), and fall back to
        # CPU otherwise — inert/no-op on the CPU-only wheel (a GPU-less VPS
        # just runs on CPU). Set 0 to force CPU. Runtime-only, never stored.
        "n_gpu_layers": -1,
    }

    @classmethod
    def get_corpus_config(cls) -> dict[str, Any]:
        """Get corpus ingestion settings merged over defaults.

        Priority:
        1. config.yaml: corpus.* section
        2. CORPUS_DEFAULTS

        No environment variable layer and no value validation by design;
        consumers validate the values they use.
        """
        merged = dict(cls.CORPUS_DEFAULTS)
        config = cls._load_config()
        merged.update((config or {}).get("corpus", {}))
        return merged

    @classmethod
    def get_phoenix_config(cls) -> dict[str, Any]:
        """Get Phoenix observability configuration.

        Priority:
        1. Environment variables (PHOENIX_ENABLED, PHOENIX_COLLECTOR_ENDPOINT,
           PHOENIX_PROJECT_NAME)
        2. config.yaml: phoenix.enabled, phoenix.collector_endpoint,
           phoenix.project_name
        3. Defaults: enabled=False, collector_endpoint="http://localhost:4317",
           project_name="iris"
        """
        config = cls._load_config() or {}
        phoenix_config = config.get("phoenix", {})

        enabled_env = os.environ.get("PHOENIX_ENABLED")
        endpoint_env = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")
        project_env = os.environ.get("PHOENIX_PROJECT_NAME")

        if enabled_env is not None:
            enabled = enabled_env.lower() in ("true", "1", "yes")
        else:
            enabled = phoenix_config.get("enabled", False)

        return {
            "enabled": enabled,
            "collector_endpoint": endpoint_env
            or phoenix_config.get("collector_endpoint", "http://localhost:4317"),
            "project_name": project_env or phoenix_config.get("project_name", "iris"),
        }

    # Default packages available in OpenTerminal
    OPENTERM_DEFAULT_PACKAGES = (
        "matplotlib, seaborn, pandas, numpy, scipy, statsmodels, scikit-learn, plotly"
    )

    @classmethod
    def get_openwebui_config(cls) -> dict[str, Any] | None:
        """Get OpenWebUI configuration for inline file rendering.

        Returns None if OpenWebUI is not configured.
        When configured, returns dict with keys:
        url, proxy_url, terminal_server_id, api_key.

        Priority:
        1. Environment variables (OPENWEBUI_URL, OPENWEBUI_PROXY_URL,
           OPENWEBUI_TERMINAL_SERVER_ID, OPENWEBUI_API_KEY)
        2. config.yaml: openwebui section
        3. None (not configured)
        """
        env_url = os.environ.get("OPENWEBUI_URL")
        env_proxy_url = os.environ.get("OPENWEBUI_PROXY_URL")
        env_server_id = os.environ.get("OPENWEBUI_TERMINAL_SERVER_ID")
        env_api_key = os.environ.get("OPENWEBUI_API_KEY")

        config = cls._load_config() or {}
        ow_config = config.get("openwebui", {})

        url = env_url or ow_config.get("url")
        if not url:
            return None

        server_id = env_server_id or ow_config.get("terminal_server_id", "openterminal")

        # proxy_url: the URL the browser uses to reach OpenWebUI
        # Falls back to url if not explicitly set (localhost scenario)
        proxy_url = env_proxy_url or ow_config.get("proxy_url") or url.rstrip("/")

        api_key = env_api_key or ow_config.get("api_key")

        return {
            "url": url.rstrip("/"),
            "proxy_url": proxy_url.rstrip("/"),
            "terminal_server_id": server_id,
            "api_key": api_key,
        }

    @classmethod
    def get_openterminal_config(cls) -> dict[str, Any] | None:
        """Get OpenTerminal configuration.

        Returns None if OpenTerminal is not configured (tools will not be registered).
        When configured, returns dict with keys: url, api_key, packages, timeout.

        Priority:
        1. Environment variables (OPEN_TERMINAL_URL, OPEN_TERMINAL_API_KEY)
        2. config.yaml: openterminal section
        3. None (not configured)
        """
        env_url = os.environ.get("OPEN_TERMINAL_URL")
        env_api_key = os.environ.get("OPEN_TERMINAL_API_KEY")

        config = cls._load_config() or {}
        ot_config = config.get("openterminal", {})

        url = env_url or ot_config.get("url")
        if not url:
            return None

        api_key = env_api_key or ot_config.get("api_key", "")
        packages = ot_config.get("packages", cls.OPENTERM_DEFAULT_PACKAGES)
        timeout = int(os.environ.get("OPEN_TERMINAL_TIMEOUT", "0")) or ot_config.get(
            "timeout", 120
        )
        write_timeout = int(
            os.environ.get("OPEN_TERMINAL_WRITE_TIMEOUT", "0")
        ) or ot_config.get("write_timeout", 30)

        return {
            "url": url.rstrip("/"),
            "api_key": api_key,
            "packages": packages,
            "timeout": timeout,
            "write_timeout": write_timeout,
        }

    @classmethod
    def get_defaults(cls) -> dict[str, Any]:
        """Return the default configuration dict."""
        config_dir = cls.get_config_dir()
        return {
            "api_url": cls.BASE_API_URL_V2,
            "api": {"verify_ssl": True},
            "database_path": str(config_dir / cls.DB_FILENAME),
            "tool_format": "native",
            "cache": {"sql_ttl": 604800},
            "logging": {
                "level": "INFO",
                "file": {
                    "enabled": True,
                    "path": "logs/esdc.log",
                    "max_size": "10MB",
                    "backup_count": 5,
                },
                "server": {"level": "INFO"},
                "agent": {"level": "DEBUG"},
                "chat": {"level": "WARNING"},
            },
            "semantic_search": {
                "embedding_batch_size": 100,
            },
            "embedding_host": None,
            "corpus": dict(cls.CORPUS_DEFAULTS),
            "phoenix": {
                "enabled": False,
                "collector_endpoint": "http://localhost:4317",
                "project_name": "iris",
            },
        }

    @classmethod
    def get_all_config_flat(cls) -> dict[str, Any]:
        """Return all config values as a flat dict with dot-notation keys.

        Merges current config with defaults so all keys are present.
        """
        config = cls._load_config() or {}
        defaults = cls.get_defaults()
        merged = cls._deep_merge(defaults, config)
        return cls._flatten(merged)

    @classmethod
    def set_config_value(cls, key: str, value: Any) -> None:
        """Set a config value by dot-notation key and save.

        Args:
            key: Dot-notation key (e.g. 'logging.level', 'api.verify_ssl')
            value: New value. Strings 'true'/'false' are converted to bool
                   for known boolean keys. Numeric strings are converted to int.
        """
        config = cls._load_config() or {}
        value = cls._coerce_value(key, value)
        keys = key.split(".")
        d = config
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value
        cls._save_config(config)

    @classmethod
    def reset_config(cls, key: str | None = None) -> None:
        """Reset config to defaults, or reset a specific key.

        Args:
            key: Dot-notation key to reset, or None to reset all.
        """
        if key is None:
            defaults = cls.get_defaults()
            cls._save_config(defaults)
            return

        defaults = cls.get_defaults()
        default_flat = cls._flatten(defaults)
        if key not in default_flat:
            raise KeyError(f"Unknown config key: {key}")

        config = cls._load_config() or {}
        keys = key.split(".")
        d = config
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d = d.setdefault(k, {})
            else:
                d = d[k]

        default_value = default_flat[key]
        d[keys[-1]] = default_value
        cls._save_config(config)

    @classmethod
    def _flatten(cls, d: dict[str, Any], parent_key: str = "") -> dict[str, Any]:
        """Flatten a nested dict into dot-notation keys."""
        items: dict[str, Any] = {}
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(cls._flatten(v, new_key))
            else:
                items[new_key] = v
        return items

    @classmethod
    def _deep_merge(
        cls, base: dict[str, Any], override: dict[str, Any]
    ) -> dict[str, Any]:
        """Deep merge override into base dict."""
        result = base.copy()
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = cls._deep_merge(result[k], v)
            else:
                result[k] = v
        return result

    BOOLEAN_KEYS = frozenset(
        {
            "api.verify_ssl",
            "logging.file.enabled",
            "corpus.rerank",
            "phoenix.enabled",
        }
    )

    INT_KEYS = frozenset(
        {
            "cache.sql_ttl",
            "logging.file.backup_count",
            "semantic_search.embedding_batch_size",
            "corpus.rerank_pool",
            "corpus.chunk_size",
            "corpus.chunk_overlap",
            "corpus.ocr_dpi",
            "corpus.num_ctx",
            "corpus.min_chars_per_page",
            "corpus.n_gpu_layers",
        }
    )

    @classmethod
    def _coerce_value(cls, key: str, value: Any) -> Any:
        """Coerce string values to appropriate types based on key."""
        if not isinstance(value, str):
            return value
        if key in cls.BOOLEAN_KEYS:
            if value.lower() in ("true", "1", "yes"):
                return True
            if value.lower() in ("false", "0", "no"):
                return False
        if key in cls.INT_KEYS:
            try:
                return int(value)
            except ValueError:
                pass
        return value
