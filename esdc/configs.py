import os
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
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
            with open(config_file, "r") as f:
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
                "database_path": str(config_dir / f"{cls.APP_NAME}.db"),
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
        except ImportError:
            raise RuntimeError(
                "Rich library required for interactive prompts. "
                "Install with: pip install rich"
            )

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
        """Get provider configuration from config file."""
        config = cls._load_config()
        if config and "provider" in config:
            return config["provider"]
        return None

    @classmethod
    def get_default_provider(cls) -> str:
        """Get default provider name."""
        provider_config = cls.get_provider_config()
        if provider_config and "name" in provider_config:
            return provider_config["name"]
        return "openai"

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
    def get_provider_base_url(cls) -> str:
        """Get provider base URL from config."""
        provider_config = cls.get_provider_config()
        if provider_config and "base_url" in provider_config:
            return provider_config["base_url"]
        return ""
