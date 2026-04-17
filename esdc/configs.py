import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


# Note: frozen=True was removed to allow method-based configuration updates
# (e.g., update_provider_config). The Config class uses classmethods for
# configuration management, so immutability is handled at the application level.
@dataclass
class Config:
    APP_NAME: str = "esdc"
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
    def _load_config(cls) -> dict[str, Any] | None:
        """Load config from YAML file (cached)."""
        if cls._config_cache is not None:
            return cls._config_cache

        config_file = cls.get_config_file()
        if config_file.exists():
            with open(config_file) as f:
                cls._config_cache = yaml.safe_load(f) or {}
                return cls._config_cache
        cls._config_cache = {}
        return None

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
                "database_path": str(config_dir / f"{cls.APP_NAME}.db"),
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
        3. ~/.esdc/esdc.db (default)
        """
        env_file = os.environ.get("ESDC_DB_FILE")
        if env_file:
            return Path(env_file)

        config = cls._load_config()
        if config and "database_path" in config:
            return Path(config["database_path"]).expanduser()

        return cls.get_config_dir() / f"{cls.APP_NAME}.db"

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
        cls._save_config(config)

    @classmethod
    def get_provider_config(cls) -> dict[str, Any] | None:
        """Get provider configuration from config file.

        Returns the config for the default provider.
        """
        config = cls._load_config()
        if not config:
            return None

        default_provider = config.get("default_provider")
        if not default_provider:
            return None

        providers = config.get("providers", {})
        return providers.get(default_provider)

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
        return (cls.get_db_dir() / f"{cls.APP_NAME}.db").resolve()

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
