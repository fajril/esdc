"""Provider and application configuration loading."""

import contextlib
import copy
import logging
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from esdc.model_runtime import ResolvedTask, resolve_task
from esdc.model_settings import (
    LEGACY_MODEL_CORPUS_KEYS,
    MODEL_SETTINGS_REQUIRED_MESSAGE,
    MODEL_SETTINGS_VERSION,
    TASK_ROLES,
    ModelConfigurationRequired,
    TaskAssignment,
    TaskRole,
    default_model_settings,
    has_legacy_model_settings,
    parse_task_assignment,
    serialize_task_assignment,
    validate_model_settings,
)

logger = logging.getLogger(__name__)

# Sensitive key suffixes to mask in the config UI.
SENSITIVE_KEYS = frozenset({"api_key"})

ENUM_CHOICES: dict[str, list[str]] = {
    "embedding_backend": ["ollama", "openai"],
    "corpus.rerank_backend": ["local", "provider"],
    "tool_format": ["native", "markdown", "auto"],
    "logging.level": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    "logging.server.level": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    "logging.agent.level": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    "logging.chat.level": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
}

# Documented relative endpoint paths for the five canonical POD input tables.
# Only the monitoring and forecast slugs are configurable; the masters use the
# paths the POD contract declares, and no server availability is asserted.
POD_ENDPOINT_DEFAULTS: dict[str, str] = {
    "m_project": "master-data-project",
    "m_pod": "master-data-pod",
    "pod_project": "pod-project",
    "pod_monitoring": "pod-monitoring",
    "pod_forecast": "pod-forecast",
}
# Config field names under the ``api`` section, matching KEY_DESCRIPTIONS.
POD_CONFIGURABLE_PATH_KEYS: dict[str, str] = {
    "pod_monitoring": "pod_monitoring_path",
    "pod_forecast": "pod_forecast_path",
}

KEY_DESCRIPTIONS: dict[str, str] = {
    "api_url": "Base URL for the ESDC API",
    "api.verify_ssl": "Verify SSL certificates for API requests",
    "api.pod_monitoring_path": (
        "Relative path of the POD monitoring endpoint under the API version "
        "(default: pod-monitoring)"
    ),
    "api.pod_forecast_path": (
        "Relative path of the approved POD forecast endpoint under the API "
        "version (default: pod-forecast)"
    ),
    "database_path": "Path to the DuckDB database file",
    "tool_format": "Format for tool results (native, markdown, or auto)",
    "default_provider": "Default LLM provider name",
    "provider_order": "Ordered LLM provider failover list",
    "model_settings_version": "Canonical model settings schema version",
    "models.chat.mode": "Chat task mode",
    "models.embedding.mode": "Embedding task mode",
    "models.rerank.mode": "Reranking task mode",
    "models.rerank.enabled": "Whether the reranking task is enabled",
    "models.ocr.mode": "OCR task mode",
    "models.metadata.mode": "Metadata extraction task mode",
    "models.cleanup.mode": "Text cleanup task mode",
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
    "embedding_backend": (
        "HTTP embedding service used for every vector space ('ollama' [default] "
        "= Ollama service at embedding_host, 'openai' = any OpenAI-compatible "
        "/v1/embeddings server, localhost included). Query-time similarity and "
        "bulk generation both use it. The in-process backend was removed: a "
        "stored 'local' value is reported when a client is asked for, and is "
        "never rewritten."
    ),
    "embedding_model": (
        "Wire model id for the embedding service, as that server names it "
        "(e.g. Qwen3-Embedding-0.6B-8bit); for the 'ollama' backend an empty "
        "value falls back to the tag the current index was built against. The "
        "vector-space identity is MODEL_ID, not this name — the probe is what "
        "proves a server serves the model the stored vectors came from"
    ),
    "embedding_api_key": (
        "Bearer token for the 'openai' embedding backend "
        "('' = send no Authorization header, which is what LM Studio and a "
        "bare llama-server expect)"
    ),
    "embedding_provider": (
        "Saved provider whose address and credential are reused for bulk "
        "embedding generation ('' = use embedding_host/embedding_api_key). "
        "Only 'openai_compatible' and 'ollama' provider types can be reused; "
        "the provider's chat model is never used as the embedding model"
    ),
    "corpus.ocr_model": "Ollama vision model used for OCR of scanned pages",
    "corpus.ocr_provider": (
        "Saved provider reused for OCR while corpus.ocr_model stays the "
        "selected model id ('' = the Ollama daemon at corpus.ollama_host)"
    ),
    "corpus.metadata_model": (
        "Text LLM for metadata pre-fill at extract time ('main' = default "
        "chat provider [default], 'provider:<name>' = a configured provider, "
        "an Ollama model name to override, or '' = image-based prefill via "
        "ocr_model)"
    ),
    "corpus.metadata_provider": (
        "Saved provider whose connection is reused for metadata extraction "
        "while corpus.metadata_model stays the wire model id ('' = the "
        "current metadata_model string, main / provider:<name> / Ollama name)"
    ),
    "corpus.cleanup_model": (
        "LLM for formatting cleanup of native-extracted pages at extract "
        "time ('main' = default chat provider [default], 'provider:<name>' "
        "= a configured provider, an Ollama model name to override, or "
        "'' = off); guarded — original text kept if the model invents "
        "numbers or changes length grossly"
    ),
    "corpus.cleanup_provider": (
        "Saved provider whose connection is reused for formatting cleanup "
        "while corpus.cleanup_model stays the wire model id ('' = the current "
        "cleanup_model string, main / provider:<name> / Ollama name)"
    ),
    "corpus.ollama_host": (
        "Ollama server URL for corpus OCR and Ollama-named corpus models "
        "('' = local daemon, e.g. http://gpu-box:11434 for a remote server)"
    ),
    "corpus.chunk_size": "Max characters per corpus chunk",
    "corpus.chunk_overlap": "Characters carried over between corpus chunks",
    "corpus.rerank": (
        "Enable the cross-encoder rerank stage over RRF search results "
        "(off by default; scores come from the configured rerank service)"
    ),
    "corpus.rerank_pool": "Number of RRF candidates scored when rerank is on",
    "corpus.rerank_backend": (
        "Where the rerank stage runs ('provider' [default] = an "
        "OpenAI-compatible /v1/rerank server reached through "
        "corpus.rerank_provider; a stored 'local' value is retired and falls "
        "back to RRF order)"
    ),
    "corpus.rerank_provider": (
        "Saved OpenAI-compatible connection used when corpus.rerank_backend "
        "is 'provider' ('' = none; a 'local' backend keeps the value stored "
        "but unused)"
    ),
    "corpus.max_chunks_per_doc": (
        "Max chunks of the same document kept in search() results "
        "(0 = no cap, the setting used by the A/B eval run)"
    ),
    "corpus.query_instruct": (
        "Prefix search/aggregate queries with Qwen3-Embedding's documented "
        "instruction template before embedding (query side only -- chunks "
        "stay raw, so toggling this never needs a re-embed)"
    ),
    "corpus.rerank_model": (
        "Rerank model id sent to the service; a stored legacy GGUF id is "
        "passed through unchanged"
    ),
    "corpus.ocr_dpi": "Page render resolution (DPI) for OCR",
    "corpus.num_ctx": "Ollama context window size for corpus OCR/metadata models",
    "corpus.n_gpu_layers": (
        "Retired llama.cpp GPU offload key: a stored value is still read but "
        "nothing consumes it, since embedding and rerank run on a service"
    ),
    "corpus.extract_max_tokens": (
        "Max output tokens for each corpus-knowledge extraction LLM call "
        "(0 = no cap); dossiers and guideline bootstrap stay unbounded"
    ),
    "corpus.extract_timeout_seconds": (
        "Timeout in seconds for each corpus-knowledge extraction LLM call "
        "(0 = no timeout); dossiers and guideline bootstrap stay unbounded"
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


# One menu location per key: the task screens own the model/reference fields,
# Application settings owns the basic app keys, Advanced owns the remaining
# processing, detailed logging, cache and tracing controls.
TASK_SECTIONS: dict[str, list[str]] = {
    "Embeddings": [
        "embedding_backend",
        "embedding_provider",
        "embedding_host",
        "embedding_model",
        "embedding_api_key",
    ],
    "Reranking": [
        "corpus.rerank",
        "corpus.rerank_backend",
        "corpus.rerank_provider",
        "corpus.rerank_model",
    ],
    "OCR": ["corpus.ocr_model", "corpus.ocr_provider"],
    "Metadata extraction": ["corpus.metadata_model", "corpus.metadata_provider"],
    "Text cleanup": ["corpus.cleanup_model", "corpus.cleanup_provider"],
}

APPLICATION_SECTIONS: dict[str, list[str]] = {
    "Application settings": [
        "api_url",
        "api.verify_ssl",
        "database_path",
        "logging.level",
        "logging.file.enabled",
    ],
}

ADVANCED_SECTIONS: dict[str, list[str]] = {
    "Corpus processing": [
        "tool_format",
        "semantic_search.embedding_batch_size",
        "corpus.chunk_size",
        "corpus.chunk_overlap",
        "corpus.rerank_pool",
        "corpus.max_chunks_per_doc",
        "corpus.query_instruct",
        "corpus.ocr_dpi",
        "corpus.num_ctx",
        "corpus.extract_max_tokens",
        "corpus.extract_timeout_seconds",
        "corpus.min_chars_per_page",
        "corpus.min_image_area",
        "corpus.ollama_host",
    ],
    "Logging": [
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

# Aliases for the pre-task-first layout: the wizard menus are now driven by
# TASK_SECTIONS / APPLICATION_SECTIONS / ADVANCED_SECTIONS, and these names
# keep the same key coverage the section-coverage tests guard.
MODEL_SECTIONS: dict[str, list[str]] = TASK_SECTIONS
SETTINGS_SECTIONS: dict[str, list[str]] = {**APPLICATION_SECTIONS, **ADVANCED_SECTIONS}


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
            if configured_path and Path(configured_path).expanduser() == legacy_db_file:
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
                **cls._model_settings_defaults(),
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
                "corpus": {
                    key: value
                    for key, value in cls.CORPUS_DEFAULTS.items()
                    if key not in LEGACY_MODEL_CORPUS_KEYS
                },
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
    def get_pod_endpoint(cls, table: str) -> str:
        """Return the relative endpoint path for one canonical POD input table.

        Priority: ``config.yaml`` ``api.pod_monitoring_path`` or
        ``api.pod_forecast_path`` override > documented default. Only the
        monitoring and forecast slugs are configurable; the masters use the paths
        the POD contract declares. Nothing here asserts server availability.
        """
        default = POD_ENDPOINT_DEFAULTS[table]
        field = POD_CONFIGURABLE_PATH_KEYS.get(table)
        if field is None:
            return default

        config = cls._load_config()
        if not config:
            return default
        api_config = config.get("api")
        if not isinstance(api_config, dict):
            return default
        value = api_config.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip().lstrip("/")
        return default

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
        """Save config atomically, then update the cache.

        The YAML is written to a temporary file in the same directory and moved
        into place with ``os.replace``, so a reader never sees a half-written
        config. The cache is updated only after the replace succeeded: a failed
        write leaves the old file and the pre-save cache intact, and only this
        call's own temporary file is removed. An existing file's mode is
        preserved; the YAML format and key ordering behavior are unchanged.
        """
        config_dir = cls.get_config_dir()
        config_file = cls.get_config_file()
        if not config_dir.exists():
            config_dir.mkdir(parents=True, exist_ok=True)

        existing_mode = (
            config_file.stat().st_mode & 0o777 if config_file.exists() else None
        )

        fd, tmp_name = tempfile.mkstemp(
            dir=config_dir, prefix=".config-", suffix=".yaml.tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            if existing_mode is not None:
                os.chmod(tmp_name, existing_mode)
            os.replace(tmp_name, config_file)
        except BaseException:
            with contextlib.suppress(OSError):
                os.remove(tmp_name)
            raise

        cls._config_cache = config

    @classmethod
    def _model_settings_defaults(cls) -> dict[str, Any]:
        """Return fresh canonical model settings for a new configuration."""
        return {
            "model_settings_version": MODEL_SETTINGS_VERSION,
            "connections": {},
            "models": default_model_settings(),
        }

    @classmethod
    def _model_config_for_read(cls) -> dict[str, Any]:
        """Return canonical model settings, or fresh defaults for a new file.

        A file with recognized pre-canonical model keys is never converted or
        interpreted here: asking for model configuration raises
        ``ModelConfigurationRequired`` while unrelated application settings stay
        readable. A file with a canonical fragment but no version marker is
        malformed and is rejected rather than silently replaced.
        """
        config = cls._load_config()
        if not config:
            return cls._model_settings_defaults()
        if "model_settings_version" not in config:
            if has_legacy_model_settings(config):
                raise ModelConfigurationRequired(MODEL_SETTINGS_REQUIRED_MESSAGE)
            if "connections" in config or "models" in config:
                raise ValueError(
                    "`model_settings_version` is missing from model settings."
                )
            return cls._model_settings_defaults()
        validate_model_settings(config)
        return config

    @classmethod
    def _require_task_role(cls, role: str) -> TaskRole:
        if role not in TASK_ROLES:
            expected = ", ".join(TASK_ROLES)
            raise ValueError(
                f"Unknown model task: {role!r}. Expected one of: {expected}."
            )
        return cast(TaskRole, role)

    @classmethod
    def get_connections(cls) -> dict[str, dict[str, Any]]:
        """Return a detached copy of the canonical connection registry."""
        return copy.deepcopy(dict(cls._model_config_for_read()["connections"]))

    @classmethod
    def get_connection(cls, name: str) -> dict[str, Any]:
        """Return a detached copy of one canonical connection.

        Raises:
            ValueError: the name is empty or unknown.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Connection name must be a non-empty string.")
        connections = cls.get_connections()
        if name not in connections:
            raise ValueError(f"Unknown connection: {name!r}")
        return connections[name]

    @classmethod
    def get_model_assignment(cls, role: TaskRole) -> TaskAssignment:
        """Return one task's canonical assignment, detached from the cache."""
        task = cls._require_task_role(role)
        config = cls._model_config_for_read()
        return parse_task_assignment(task, config["models"][task])

    @classmethod
    def resolve_model_task(
        cls, role: TaskRole, *, apply_overrides: bool = True
    ) -> ResolvedTask:
        """Resolve one task's effective connection, model and options.

        ``mode: chat`` follows the Chat assignment dynamically; ``mode: ocr``
        and ``mode: off`` remain distinct relationships with no primary target.
        Supported per-run environment overrides are applied only when
        ``apply_overrides`` is true.
        """
        task = cls._require_task_role(role)
        return resolve_task(
            task,
            load_assignment=cls.get_model_assignment,
            load_connection=cls.get_connection,
            apply_overrides=apply_overrides,
        )

    @classmethod
    def save_model_assignment(
        cls,
        role: TaskRole,
        assignment: TaskAssignment,
        *,
        new_connections: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        """Save one task assignment and optional new connections in one write.

        The candidate is built on a deep copy, so a rejected write leaves the
        file and the pre-save cache intact. Incoming connection mappings and
        model options are copied, so a caller can keep editing its draft after
        the call without reaching the saved structure.

        Raises:
            ModelConfigurationRequired: the file predates canonical settings.
            ValueError: the role, assignment, connection or complete candidate
                is invalid, or a new connection name already exists.
        """
        task = cls._require_task_role(role)
        serialized = serialize_task_assignment(task, assignment)

        supplied_connections: Mapping[str, Mapping[str, Any]] = (
            new_connections if new_connections is not None else {}
        )
        if not isinstance(supplied_connections, Mapping):
            raise ValueError("`new_connections` must be a mapping.")

        raw_config = cls._load_config() or {}
        if "model_settings_version" not in raw_config:
            if has_legacy_model_settings(raw_config):
                raise ModelConfigurationRequired(MODEL_SETTINGS_REQUIRED_MESSAGE)
            if "connections" in raw_config or "models" in raw_config:
                raise ValueError(
                    "`model_settings_version` is missing from model settings."
                )
            candidate = copy.deepcopy(raw_config)
            candidate.update(cls._model_settings_defaults())
        else:
            candidate = copy.deepcopy(raw_config)

        connections = candidate.get("connections")
        if not isinstance(connections, dict):
            raise ValueError("`connections` must be a mapping of named connections.")
        models = candidate.get("models")
        if not isinstance(models, dict):
            raise ValueError("`models` must be a mapping of task assignments.")

        for name, entry in supplied_connections.items():
            if not isinstance(name, str) or not name.strip() or name != name.strip():
                raise ValueError(
                    "Connection names must be non-empty strings without "
                    "surrounding whitespace."
                )
            if name in connections:
                raise ValueError(f"Connection name already exists: {name!r}")
            if not isinstance(entry, Mapping):
                raise ValueError(f"Connection {name!r} must be a mapping.")
            connections[name] = copy.deepcopy(dict(entry))

        models[task] = serialized
        candidate["model_settings_version"] = MODEL_SETTINGS_VERSION
        candidate["connections"] = connections
        candidate["models"] = models
        validate_model_settings(candidate)
        cls._save_config(candidate)

    @classmethod
    def _model_config_for_write(cls) -> dict[str, Any]:
        """Return a detached canonical settings candidate for one write.

        A new/empty file starts from fresh canonical defaults; an existing
        canonical file is deep-copied; a pre-canonical file is refused rather
        than converted implicitly. Nothing is written here.
        """
        raw_config = cls._load_config() or {}
        if "model_settings_version" not in raw_config:
            if has_legacy_model_settings(raw_config):
                raise ModelConfigurationRequired(MODEL_SETTINGS_REQUIRED_MESSAGE)
            if "connections" in raw_config or "models" in raw_config:
                raise ValueError(
                    "`model_settings_version` is missing from model settings."
                )
            candidate = copy.deepcopy(raw_config)
            candidate.update(cls._model_settings_defaults())
            return candidate
        return copy.deepcopy(raw_config)

    @staticmethod
    def _connection_name_or_raise(name: object, *, label: str) -> str:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{label} must be a non-empty string.")
        if name != name.strip():
            raise ValueError(
                f"{label} must not have leading or trailing whitespace."
            )
        return name

    @staticmethod
    def _connection_references_in_config(
        config: Mapping[str, Any], name: str
    ) -> list[str]:
        """Reference paths for one canonical settings mapping."""
        models = config.get("models")
        if not isinstance(models, Mapping):
            return []
        references: list[str] = []
        for role in TASK_ROLES:
            value = models.get(role)
            if not isinstance(value, Mapping):
                continue
            assignment = parse_task_assignment(role, value)
            if (
                assignment.target is not None
                and assignment.target.connection == name
            ):
                references.append(f"models.{role}")
            for index, fallback in enumerate(assignment.fallbacks):
                if fallback.connection == name:
                    references.append(f"models.{role}.fallbacks[{index}]")
        return references

    @classmethod
    def get_connection_references(cls, name: str) -> list[str]:
        """Reference paths that assign one connection to a task.

        The order is the canonical task order (Chat, Embeddings, Reranking,
        OCR, Metadata extraction, Text cleanup), with each Chat fallback after
        its primary. ``mode: chat`` and ``mode: ocr`` relationships are not
        references by themselves: they follow the Chat/OCR task. Dotted names
        are literal mapping keys.

        Returns:
            Paths such as ``models.chat`` or ``models.chat.fallbacks[1]``;
            ``[]`` for an empty or unknown name.
        """
        if not isinstance(name, str) or not name:
            return []
        return cls._connection_references_in_config(
            cls._model_config_for_read(), name
        )

    @classmethod
    def save_connection(cls, name: str, values: Mapping[str, Any]) -> None:
        """Create or edit one canonical connection in a single atomic write.

        A connection owns transport/authentication only. The candidate is built
        on a deep copy and validated against every task reference before the
        file is replaced, so an edit that would make a dependent task invalid
        leaves the file and cache unchanged.

        Raises:
            ModelConfigurationRequired: the file predates canonical settings.
            ValueError: the name, entry, or resulting task references are
                invalid.
        """
        connection_name = cls._connection_name_or_raise(name, label="Connection name")
        if not isinstance(values, Mapping):
            raise ValueError(f"Connection {connection_name!r} must be a mapping.")

        candidate = cls._model_config_for_write()
        connections = candidate.get("connections")
        if not isinstance(connections, dict):
            raise ValueError("`connections` must be a mapping of named connections.")
        connections[connection_name] = copy.deepcopy(dict(values))
        candidate["model_settings_version"] = MODEL_SETTINGS_VERSION
        candidate["connections"] = connections
        validate_model_settings(candidate)
        cls._save_config(candidate)

    @classmethod
    def rename_connection(
        cls,
        old_name: str,
        new_name: str,
        values: Mapping[str, Any] | None = None,
    ) -> None:
        """Rename one connection and move every task reference in one write.

        The saved entry keeps all its fields, including OAuth, API version and
        unknown extension keys. ``values`` optionally replaces the moved entry
        with an edited draft in the same write. Only connection references
        change: task models, options and non-model application settings stay as
        stored. A dotted connection name is a literal mapping key.

        The candidate is validated before the file is replaced. A conflicting or
        empty name, an unknown old name, or a resulting invalid task assignment
        leaves the file and cache unchanged.

        Raises:
            ModelConfigurationRequired: the file predates canonical settings.
            ValueError: the old name is unknown, a name is invalid, the new name
                already exists, or the resulting settings are invalid.
        """
        old = cls._connection_name_or_raise(old_name, label="Connection name")
        new = cls._connection_name_or_raise(new_name, label="Connection name")
        if values is not None and not isinstance(values, Mapping):
            raise ValueError(f"Connection {new!r} must be a mapping.")

        candidate = cls._model_config_for_write()
        connections = candidate.get("connections")
        if not isinstance(connections, dict):
            raise ValueError("`connections` must be a mapping of named connections.")
        if old not in connections:
            raise ValueError(f"Unknown connection: {old!r}")
        if new != old and new in connections:
            raise ValueError(f"Connection name already exists: {new!r}")
        if new == old and values is None:
            return

        renamed: dict[str, Any] = {}
        for key, entry in connections.items():
            renamed[new if key == old else key] = entry
        if values is not None:
            renamed[new] = copy.deepcopy(dict(values))
        candidate["connections"] = renamed

        models = candidate.get("models")
        if not isinstance(models, dict):
            raise ValueError("`models` must be a mapping of task assignments.")
        for assignment in models.values():
            if not isinstance(assignment, dict) or assignment.get("mode") != "model":
                continue
            if assignment.get("connection") == old:
                assignment["connection"] = new
            fallbacks = assignment.get("fallbacks")
            if isinstance(fallbacks, list):
                for fallback in fallbacks:
                    if (
                        isinstance(fallback, dict)
                        and fallback.get("connection") == old
                    ):
                        fallback["connection"] = new

        candidate["model_settings_version"] = MODEL_SETTINGS_VERSION
        validate_model_settings(candidate)
        cls._save_config(candidate)

    @classmethod
    def remove_connection(cls, name: str) -> bool:
        """Remove one unreferenced connection; never reassigns a task.

        Returns ``False`` for an unknown name. A connection still used by any
        task target or Chat fallback raises ``ValueError`` naming every use, and
        nothing is written, so the user can reassign the tasks first.
        """
        connection_name = cls._connection_name_or_raise(name, label="Connection name")
        candidate = cls._model_config_for_write()
        connections = candidate.get("connections")
        if not isinstance(connections, dict) or connection_name not in connections:
            return False

        references = cls._connection_references_in_config(candidate, connection_name)
        if references:
            raise ValueError(
                f"Connection {connection_name!r} is still used by: "
                f"{', '.join(references)}. Reassign those tasks first; nothing "
                "was removed."
            )

        del connections[connection_name]
        candidate["model_settings_version"] = MODEL_SETTINGS_VERSION
        candidate["connections"] = connections
        validate_model_settings(candidate)
        cls._save_config(candidate)
        return True

    @classmethod
    def get_providers(cls) -> dict:
        """Return canonical connections shaped as runtime provider entries.

        This is an internal transport adapter for callers that still speak the
        former provider-entry shape. It reads ``connections`` only; a
        connection's model belongs to a task assignment and is never returned
        here.
        """
        providers: dict[str, dict[str, Any]] = {}
        for name, connection in cls.get_connections().items():
            entry = copy.deepcopy(connection)
            entry["name"] = name
            entry["provider_type"] = str(
                entry.get("type") or entry.get("provider_type") or ""
            ).strip()
            entry.pop("model", None)
            entry.pop("reasoning_effort", None)
            providers[name] = entry
        return providers

    @classmethod
    def _provider_entry(cls, config: dict[str, Any], name: str) -> dict[str, Any]:
        """Return a saved legacy provider entry, or raise.

        Used only by the legacy provider write helpers. Runtime reads use the
        canonical connection adapters instead.
        """
        providers = config.get("providers")
        entry = providers.get(name) if isinstance(providers, dict) else None
        if not isinstance(entry, dict):
            raise ValueError(f"Unknown provider reference: {name!r}")
        return entry

    @classmethod
    def get_referenced_provider(cls, name: str) -> dict[str, Any]:
        """Return a detached runtime adapter for one canonical connection.

        The saved connection supplies transport and authentication only. The
        caller sets the task's explicit wire model, so no connection-level
        model is returned or consulted.

        Raises:
            ValueError: the connection is unknown.
        """
        connections = cls.get_connections()
        if name not in connections:
            raise ValueError(f"Unknown provider reference: {name!r}")
        provider = copy.deepcopy(connections[name])
        provider["name"] = name
        provider["provider_type"] = str(
            provider.get("type") or provider.get("provider_type") or ""
        ).strip()
        provider.pop("model", None)
        provider.pop("reasoning_effort", None)
        return provider

    @classmethod
    def save_provider(cls, name: str, provider_config: dict) -> None:
        """Save a provider configuration.

        The cached mapping is never mutated: the write works on a deep copy, so a
        failed ``os.replace`` leaves the pre-save cache as it was and an unsaved
        draft cannot be picked up by a later write. The incoming mapping is copied
        too, so the caller's draft is never the object the saved structure holds.
        """
        config = copy.deepcopy(cls._load_config() or {})
        providers = config.get("providers", {})
        providers[name] = copy.deepcopy(dict(provider_config))
        config["providers"] = providers
        cls._save_config(config)

    @classmethod
    def remove_provider(cls, name: str) -> bool:
        """Remove a provider configuration. Returns True if removed.

        The cached mapping is never mutated, as in `save_provider`: the delete happens
        on a deep copy, so a failed ``os.replace`` leaves the file, the cache object,
        and its contents as they were, and an unconfirmed removal cannot be picked up
        by a later write. The cache is updated only after the move succeeded.
        """
        config = copy.deepcopy(cls._load_config() or {})
        providers = config.get("providers", {})
        if name in providers:
            del providers[name]
            config["providers"] = providers
            cls._save_config(config)
            return True
        return False

    # Config keys that can name a provider, in the order get_provider_references
    # reports them. The linked-string keys hold a legacy ``provider:<name>`` value
    # rather than a bare provider name.
    PROVIDER_REFERENCE_KEYS = (
        "default_provider",
        "provider_order",
        "embedding_provider",
        "corpus.metadata_provider",
        "corpus.cleanup_provider",
        "corpus.ocr_provider",
        "corpus.rerank_provider",
        "corpus.metadata_model",
        "corpus.cleanup_model",
    )

    PROVIDER_LINKED_MODEL_KEYS = frozenset(
        {"corpus.metadata_model", "corpus.cleanup_model"}
    )

    # Each linked model key belongs to one task, and that task's provider key decides
    # what the model string means: while the role provider is empty the string is the
    # legacy link, and once the role names a provider the same text is a literal wire
    # model that must never be rewritten.
    PROVIDER_LINKED_MODEL_ROLE_KEYS = {
        "corpus.metadata_model": "corpus.metadata_provider",
        "corpus.cleanup_model": "corpus.cleanup_provider",
    }

    @classmethod
    def get_provider_references(cls, name: str) -> list[str]:
        """Config keys that currently depend on one provider.

        Reported in this stable order: ``default_provider``, ``provider_order``,
        ``embedding_provider``, the four corpus references, then the legacy
        ``provider:<name>`` metadata and cleanup strings. Inactive references are
        included, because a name stored in the file is still a reference even when
        its task uses another source; nothing here repairs or removes one. An order
        that mentions the name is reported once.

        A metadata or cleanup model key is a reference only while its role provider is
        empty. When that provider names a provider, ``provider:<name>`` is the model
        that task calls, so the key is not reported even when its text matches: it is a
        literal wire model, not a link to ``name``.

        Only key names are returned, never a value, so no secret and no display
        label can reach the caller.

        Returns:
            The matching keys in the documented order, or ``[]`` when nothing
            refers to ``name``.
        """
        target = str(name or "").strip()
        if not target:
            return []

        flat = cls._flatten(cls._load_config() or {})
        references: list[str] = []
        for key in cls.PROVIDER_REFERENCE_KEYS:
            value = flat.get(key)
            if key == "provider_order":
                if isinstance(value, list) and target in [str(item) for item in value]:
                    references.append(key)
                continue
            if key in cls.PROVIDER_LINKED_MODEL_KEYS:
                # The role provider is what makes this string a link or a model id.
                if str(flat.get(cls.PROVIDER_LINKED_MODEL_ROLE_KEYS[key]) or ""):
                    continue
                if str(value or "") == f"provider:{target}":
                    references.append(key)
                continue
            if str(value or "") == target:
                references.append(key)
        return references

    @classmethod
    def rename_provider(
        cls,
        old_name: str,
        new_name: str,
        provider_config: Mapping[str, Any] | None = None,
    ) -> None:
        """Rename one provider and move every reference to it in one write.

        The provider entry keeps its fields, including OAuth and unknown keys.
        ``default_provider``, ``provider_order`` (list order preserved, and only the
        duplicate this rename itself creates is removed), ``embedding_provider``, the
        four corpus references and the legacy ``provider:<old_name>`` metadata and
        cleanup strings are updated — the model keys only when they are genuine links,
        which is when their role provider is empty. A model string that a named
        provider owns is a literal wire model and is left as stored. Every other
        provider, every task's explicit model and every unknown key is left as it is,
        and an inactive reference keeps its stored value.

        ``provider_config`` is the optional draft for the renamed key. When it is
        given, the entry under ``new_name`` becomes a deep copy of that mapping
        instead of the moved saved entry, so one draft's name change and its field
        changes are the same single write. Without it the moved entry keeps the
        saved fields, which is what a rename on its own must do.

        Every name is validated before the first change, so a rejected rename
        leaves the file completely unchanged. An empty ``new_name`` is rejected, an
        unknown or non-mapping ``old_name`` is rejected, and only then a
        ``new_name`` equal to ``old_name`` writes nothing. An existing different
        ``new_name`` is rejected too.

        Raises:
            ValueError: ``new_name`` is empty, ``old_name`` is unknown or is not a
                mapping, or ``new_name`` is already another provider.
        """
        target = str(new_name or "").strip()
        if not target:
            raise ValueError("Provider name cannot be empty.")

        # Read and validate on a copy: nothing is written until every name is valid.
        config = copy.deepcopy(cls._load_config() or {})
        # Validation uses the established provider-entry helper, so a missing
        # entry and an entry that is not a mapping raise the same way.
        cls._provider_entry(config, old_name)
        if target == old_name:
            return

        providers = config["providers"]
        if target in providers:
            raise ValueError(f"Provider name already exists: {target!r}")

        references = cls.get_provider_references(old_name)

        # The key moves in place: only its name changes, every field is the same
        # entry object. `provider_order` keeps its own list order.
        renamed: dict[str, Any] = {}
        for key, value in providers.items():
            renamed[target if key == old_name else key] = value
        config["providers"] = renamed

        linked = cls.PROVIDER_LINKED_MODEL_KEYS
        for key in references:
            if key == "provider_order":
                configured = config.get("provider_order")
                updated = [
                    target if str(item) == old_name else item
                    for item in configured or []
                ]
                # Only the duplicate this rename itself creates is removed; a
                # duplicate already in the file is left as the user stored it.
                # The key is reported only when the order held `old_name`, so
                # after the replacement it holds `target` at least once.
                first = updated.index(target)
                updated = [
                    item
                    for index, item in enumerate(updated)
                    if index == first or item != target
                ]
                config["provider_order"] = updated
                continue
            # Only keys that get_provider_references reported reach this loop, so a
            # linked model key here is a genuine link and keeps its legacy shape: only
            # the name inside ``provider:<old_name>`` changes. A literal wire model was
            # never reported, so it is never rewritten.
            cls._set_flat_value(
                config, key, f"provider:{target}" if key in linked else target
            )

        if provider_config is not None:
            # The draft replaces the moved entry in this same write, so the name
            # change and the draft's field changes can never be split by a failure
            # between two writes. The copy keeps the caller's mapping out of the
            # saved structure.
            config["providers"][target] = copy.deepcopy(dict(provider_config))

        cls._save_config(config)

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
        """Return the canonical Chat runtime provider adapter, or None.

        The primary entry carries the Chat target's model and options. Chat
        fallback targets are attached under ``fallback_configs`` in order.
        This reads ``connections``/``models`` only; it is an internal transport
        adapter, not a second persisted configuration system.
        """
        configs = cls.get_provider_configs_by_priority()
        if not configs:
            return None

        primary = configs[0]
        fallbacks = configs[1:]
        if fallbacks:
            primary["fallback_configs"] = fallbacks
        return primary

    @classmethod
    def get_provider_order(cls) -> list[str]:
        """Return canonical Chat connection names in effective failover order."""
        resolved = cls.resolve_model_task("chat")
        if resolved.mode != "model" or resolved.primary is None:
            return []

        ordered: list[str] = []
        for target in (resolved.primary, *resolved.fallbacks):
            if target.connection not in ordered:
                ordered.append(target.connection)
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
    def set_chat_selection(cls, provider_name: str, model_id: str) -> None:
        """Save one Chat provider/model choice as a single atomic write.

        Chat owns the provider's chat model: ``default_provider``, the front of
        ``provider_order`` and ``providers[provider_name].model`` change
        together. Every other provider field, every unknown user key and the
        remaining failover order are preserved. ``provider_order`` is written
        even when the file had none, and provider names may contain dots, so the
        provider entry is merged directly instead of being addressed through a
        dotted-key path.

        Raises:
            ValueError: the provider is unknown, or its entry is not a mapping.
        """
        config = copy.deepcopy(cls._load_config() or {})
        entry = cls._provider_entry(config, provider_name)

        config["default_provider"] = provider_name
        configured_order = config.get("provider_order")
        rest = (
            [name for name in configured_order if name != provider_name]
            if isinstance(configured_order, list)
            else []
        )
        config["provider_order"] = [provider_name] + list(dict.fromkeys(rest))

        entry["model"] = model_id
        cls._save_config(config)

    @classmethod
    def get_provider_configs_by_priority(cls) -> list[dict[str, Any]]:
        """Return canonical Chat runtime adapters in failover order."""
        resolved = cls.resolve_model_task("chat")
        if resolved.mode != "model" or resolved.primary is None:
            return []
        targets = (resolved.primary, *resolved.fallbacks)
        return [copy.deepcopy(dict(target.provider)) for target in targets]

    @classmethod
    def get_default_provider(cls) -> str:
        """Return the canonical Chat primary connection name, or ``""``."""
        resolved = cls.resolve_model_task("chat")
        if resolved.mode != "model" or resolved.primary is None:
            return ""
        return resolved.primary.connection

    @classmethod
    def get_provider_api_key(cls) -> str:
        """Get the active Chat provider API key or explicit env override."""
        env_key = os.environ.get("OPENAI_API_KEY")
        if env_key:
            return env_key

        resolved = cls.resolve_model_task("chat")
        if resolved.mode != "model" or resolved.primary is None:
            return ""
        return str(resolved.primary.provider.get("api_key") or "")

    @classmethod
    def get_provider_model(cls) -> str:
        """Return the canonical Chat primary wire model, or ``""``."""
        resolved = cls.resolve_model_task("chat")
        if resolved.mode != "model" or resolved.primary is None:
            return ""
        return resolved.primary.model

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
        """Return the canonical Chat primary connection base URL, or ``""``."""
        resolved = cls.resolve_model_task("chat")
        if resolved.mode != "model" or resolved.primary is None:
            return ""
        return str(resolved.primary.provider.get("base_url") or "")

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
        """Return one canonical connection runtime adapter, or ``None``."""
        try:
            return cls.get_referenced_provider(name)
        except ModelConfigurationRequired:
            raise
        except ValueError:
            return None

    @classmethod
    def update_provider_config(cls, name: str, config_data: dict[str, Any]) -> None:
        """Update a provider configuration.

        The cached mapping is never mutated, as in `save_provider`: the write works
        on a deep copy of the loaded config and of the incoming mapping, so a failed
        ``os.replace`` leaves the pre-save cache as it was and an unsaved change
        cannot be picked up by a later write. The cache is updated only after the
        move succeeded, and the saved structure never holds the caller's mapping.
        """
        config = copy.deepcopy(cls._load_config() or {})
        if "providers" not in config:
            config["providers"] = {}
        config["providers"][name] = copy.deepcopy(dict(config_data))
        cls._save_config(config)

    @classmethod
    def persist_provider_oauth(cls, provider_name: str, oauth: dict[str, Any]) -> None:
        """Persist refreshed OAuth tokens for one canonical connection.

        OAuth providers may rotate the refresh_token on every refresh. The
        in-memory ``ProviderConfig.oauth`` dict is updated by the caller
        immediately after a refresh, but that update is lost on process
        restart unless it is also written to the config file. This method
        performs that write against ``connections[provider_name].oauth``.

        The config file is read fresh from disk (bypassing the in-memory
        cache), so an unrelated concurrent edit is preserved. Only the named
        connection's ``oauth`` section changes; a ``providers`` section is
        never created or restored. The file is written atomically via a temp
        file + ``os.replace`` with owner-only permissions because it may hold
        access and refresh tokens.

        If the connection is absent from the on-disk config, or the config
        file does not exist, this logs a warning and returns without error.
        Callers treat persistence failures as non-fatal: refreshing the token
        in memory must still succeed even when the write to disk fails.

        Args:
            provider_name: Canonical connection name.
            oauth: The refreshed OAuth token dict to store.
        """
        config_file = cls.get_config_file()
        if not config_file.exists():
            logger.warning(
                "Skipping OAuth token persistence for connection '%s': "
                "config file %s does not exist",
                provider_name,
                config_file,
            )
            return

        with open(config_file) as f:
            config = yaml.safe_load(f) or {}

        connections = config.get("connections")
        if not isinstance(connections, dict) or provider_name not in connections:
            logger.warning(
                "Skipping OAuth token persistence: connection '%s' not found "
                "in config file %s",
                provider_name,
                config_file,
            )
            return

        connection = connections[provider_name]
        if not isinstance(connection, dict):
            logger.warning(
                "Skipping OAuth token persistence: connection '%s' in %s is "
                "not a mapping",
                provider_name,
                config_file,
            )
            return
        connection["oauth"] = copy.deepcopy(oauth)

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
        """Whether canonical Chat is configured with a model target."""
        try:
            assignment = cls.get_model_assignment("chat")
        except ModelConfigurationRequired:
            return False
        return assignment.mode == "model" and assignment.target is not None

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

    # Provider types whose saved connection can be reused for bulk embedding
    # generation in this phase. A reference decides which backend talks to it:
    # an openai_compatible provider is reached through the 'openai' backend, an
    # ollama provider through the 'ollama' backend, and the 'local' backend
    # ignores references entirely. Written without a type annotation so the
    # dataclass does not turn it into a field.
    EMBEDDING_PROVIDER_BACKENDS = {
        "openai_compatible": "openai",
        "ollama": "ollama",
    }

    @classmethod
    def _resolve_embedding_source(cls) -> tuple[str, dict[str, Any] | None]:
        """Return ``(effective backend, active provider reference copy or None)``.

        An empty ``embedding_provider`` keeps the current behavior. A ``local``
        backend ignores the stored reference, which is not repaired or cleared.
        An unknown reference, a provider type that cannot serve embeddings, or a
        provider type that disagrees with ``embedding_backend`` raises: the
        resolution never switches source silently.

        Raises:
            ValueError: the active reference is unknown or mismatched.
        """
        config = cls._load_config() or {}
        env_backend = os.environ.get("ESDC_EMBEDDING_BACKEND")
        raw = env_backend or config.get("embedding_backend")
        backend = str(raw or "ollama").strip().lower()
        name = str(config.get("embedding_provider") or "").strip()
        if not name:
            return backend, None
        if backend == "local":
            # A local backend ignores an inactive reference; its stored value
            # is neither validated nor cleared.
            return backend, None

        try:
            provider = cls.get_referenced_provider(name)
        except ValueError as err:
            raise ValueError(f"`embedding_provider`: {err}") from err

        provider_type = (
            str(provider.get("provider_type") or provider.get("type") or "")
            .strip()
            .lower()
        )
        expected = cls.EMBEDDING_PROVIDER_BACKENDS.get(provider_type)
        if expected is None:
            raise ValueError(
                f"`embedding_provider` refers to provider {name!r} (type "
                f"{provider_type or 'unknown'!r}); only 'openai_compatible' and "
                "'ollama' providers can be reused for embeddings."
            )
        if backend != expected:
            raise ValueError(
                f"`embedding_backend` '{backend}' does not match "
                f"`embedding_provider` {name!r} (type {provider_type}), which "
                f"needs backend '{expected}'."
            )
        return backend, provider

    @classmethod
    def get_embedding_host(cls) -> str | None:
        """Get embedding service host URL from config.

        Priority with no provider reference:
        1. ESDC_EMBEDDING_HOST environment variable
        2. config.yaml: embedding_host
        3. None (uses localhost Ollama default)

        With an active reference the referenced provider's address comes first:
        env > provider base_url > embedding_host. An active reference with no
        address at all raises rather than falling back to a default source.

        Returns:
            Host URL string or None for localhost default

        Raises:
            ValueError: the active reference is unknown, mismatched, or has no
                address and no configured host.
        """
        env_host = os.environ.get("ESDC_EMBEDDING_HOST")
        if env_host:
            return env_host

        config = cls._load_config() or {}
        _, provider = cls._resolve_embedding_source()
        if provider is None:
            return config.get("embedding_host") or None

        base_url = str(provider.get("base_url") or "").strip()
        if base_url:
            return base_url
        if config.get("embedding_host"):
            return str(config["embedding_host"])
        raise ValueError(
            f"`embedding_provider`: provider {provider['name']!r} has no "
            "base_url and `embedding_host` is empty; set one of them."
        )

    @classmethod
    def get_embedding_backend(cls) -> str:
        """Get the embedding service backend.

        Priority:
        1. ESDC_EMBEDDING_BACKEND environment variable
        2. config.yaml: embedding_backend
        3. "ollama" (default)

        An active `embedding_provider` reference must agree with it:
        an openai_compatible provider means "openai", an ollama provider means
        "ollama", and a stored "local" value ignores the reference.

        Reading config is not validation: a stored "local" value is returned
        as it is stored, and `esdc.embedders.get_embedder()` reports it as a
        retired value when a client is asked for. Nothing here writes config.

        Returns:
            One of "ollama", "openai", or a stored retired value such as
            "local" (lowercased, stripped).

        Raises:
            ValueError: the active reference is unknown or mismatched.
        """
        return cls._resolve_embedding_source()[0]

    @classmethod
    def get_embedding_model(cls) -> str:
        """Get the wire model id for the embedding service.

        Priority:
        1. ESDC_EMBEDDING_MODEL environment variable
        2. config.yaml: embedding_model
        3. "" (empty — the openai backend rejects this at construction, and the
        ollama backend falls back to the tag the current index was built against)

        This is the name the service is asked for, not the vector-space
        identity: `esdc.embedders.MODEL_ID` stays the pin written into
        corpus_meta/semantic_meta. A referenced provider never supplies the
        embedding model: its `model` field is that provider's chat model, owned
        by Chat.

        Raises:
            ValueError: the active reference is unknown or mismatched.
        """
        env_model = os.environ.get("ESDC_EMBEDDING_MODEL")
        if env_model:
            return env_model
        config = cls._load_config() or {}
        # Validate the reference so a mismatched one is reported rather than
        # silently used; the model itself still comes from `embedding_model`.
        cls._resolve_embedding_source()
        return str(config.get("embedding_model") or "")

    @classmethod
    def get_embedding_api_key(cls) -> str:
        """Get the bearer token for the 'openai' embedding backend.

        Priority with no provider reference:
        1. ESDC_EMBEDDING_API_KEY environment variable
        2. config.yaml: embedding_api_key
        3. "" (no Authorization header sent)

        With an active reference the credential is the referenced provider's
        own key, and an empty one means send no authentication — a key saved for
        another source is never substituted.

        Raises:
            ValueError: the active reference is unknown or mismatched.
        """
        env_key = os.environ.get("ESDC_EMBEDDING_API_KEY")
        if env_key:
            return env_key

        _, provider = cls._resolve_embedding_source()
        if provider is None:
            config = cls._load_config() or {}
            return str(config.get("embedding_api_key") or "")
        return str(provider.get("api_key") or "")

    CORPUS_DEFAULTS = {
        "chunk_size": 3000,  # max chars per chunk (~750 tokens)
        "chunk_overlap": 300,  # chars carried over between chunks
        # rerank_pool: candidates scored per query when the Reranking task is
        # enabled. The task's own model/enabled switch lives in models.rerank.
        "rerank_pool": 30,
        # max_chunks_per_doc: cap on chunks of the same document kept in
        # search() results, applied after rerank so the surviving chunk per
        # document is the best-scoring one. Measured 2026-08-05: top-10 held
        # a mean of 5.05 distinct documents (min 1) without this, and
        # lookup Pass@5/Pass@10 were identical because ranks 6-10 were
        # repeats of documents already seen. 0 disables the cap — that is
        # what the A/B eval run uses to reproduce the uncapped baseline.
        "max_chunks_per_doc": 2,
        # query_instruct: prefix search()/aggregate() queries with Qwen3's
        # documented instruction template before embedding them (query
        # side only; chunks stay raw, so this needs no re-embed).
        # On by default since 2026-08-06: Qwen3-Embedding is instruction-
        # tuned and asymmetric, so raw queries were using the model off the
        # format it was trained for. Measured over 189 queries with the cap
        # on: lookup Pass@1 78.3% -> 81.7%, cross_reference Recall@5
        # 81.7% -> 83.3%, Recall@10 85.0% -> 90.0%; the cost is roughly one
        # query at lookup Pass@5/Pass@10. Set False to A/B it back.
        "query_instruct": True,
        "ocr_dpi": 200,  # page render resolution; raise to 300 if OCR quality poor
        "num_ctx": 16384,  # Ollama context window; glm-ocr crashes on images below this
        # extract_max_tokens / extract_timeout_seconds: bounds applied ONLY to
        # corpus-knowledge extraction LLM calls (`esdc corpus learn`). Dossiers
        # and --init-guideline bootstrap keep their separate unbounded contract.
        # 0 disables each bound.
        "extract_max_tokens": 8192,
        "extract_timeout_seconds": 300,
        "min_chars_per_page": 50,  # text-layer chars below which a page counts as scanned  # noqa: E501
        "min_image_area": 0.05,  # embedded-image area (fraction of page) below which images are ignored  # noqa: E501
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
        corpus_config = (config or {}).get("corpus", {})
        if isinstance(corpus_config, Mapping):
            # Recognized pre-canonical model keys are conversion-only data.
            # They never become runtime corpus settings, even before a user
            # replaces the old file.
            merged.update(
                {
                    key: value
                    for key, value in corpus_config.items()
                    if key not in LEGACY_MODEL_CORPUS_KEYS
                }
            )
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
            **cls._model_settings_defaults(),
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
            "embedding_backend": "ollama",
            "embedding_provider": "",
            "embedding_model": "",
            "embedding_api_key": "",
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
    def _set_flat_value(cls, config: dict[str, Any], key: str, value: Any) -> None:
        """Write one already-coerced dot-notation key into ``config``.

        Provider entries are not addressed through dotted keys: a provider name
        may itself contain a dot, so provider fields are saved through the
        provider-aware methods (`save_provider`, `update_provider_config`,
        `set_chat_selection`).

        Raises:
            ValueError: the key targets a provider entry.
        """
        parts = key.split(".")
        if parts[0] == "providers":
            raise ValueError(
                f"Provider fields are not saved through dotted keys: {key!r}. "
                "Use save_provider/update_provider_config, or "
                "set_chat_selection for a chat model change."
            )
        d = config
        for part in parts[:-1]:
            if part not in d or not isinstance(d[part], dict):
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value

    @classmethod
    def set_config_values(cls, values: dict[str, Any]) -> None:
        """Set several dot-notation keys and save them in one write.

        Every value is coerced with the same rules as `set_config_value`, all
        keys are applied to one config dict, and the file is written once. An
        empty mapping writes nothing. Unrelated keys, unknown user keys, saved
        provider fields and the existing key order are preserved.

        Raises:
            ValueError: a key targets a provider entry.
        """
        if not values:
            return

        config = copy.deepcopy(cls._load_config() or {})
        for key, value in values.items():
            cls._set_flat_value(config, key, cls._coerce_value(key, value))
        cls._save_config(config)

    @classmethod
    def set_config_value(cls, key: str, value: Any) -> None:
        """Set a single config value by dot-notation key and save.

        Args:
            key: Dot-notation key (e.g. 'logging.level', 'api.verify_ssl')
            value: New value. Strings 'true'/'false' are converted to bool
                   for known boolean keys. Numeric strings are converted to int.
        """
        cls.set_config_values({key: value})

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
            "models.rerank.enabled",
            "corpus.rerank",
            "corpus.query_instruct",
            "phoenix.enabled",
        }
    )

    INT_KEYS = frozenset(
        {
            "cache.sql_ttl",
            "logging.file.backup_count",
            "model_settings_version",
            "semantic_search.embedding_batch_size",
            "corpus.rerank_pool",
            "corpus.max_chunks_per_doc",
            "corpus.chunk_size",
            "corpus.chunk_overlap",
            "corpus.ocr_dpi",
            "corpus.num_ctx",
            "corpus.min_chars_per_page",
            "corpus.n_gpu_layers",
            "corpus.extract_max_tokens",
            "corpus.extract_timeout_seconds",
        }
    )

    FLOAT_KEYS = frozenset(
        {
            # Decimal fractions of a page area; stored as float, not int.
            "corpus.min_image_area",
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
        if key in cls.FLOAT_KEYS:
            try:
                return float(value)
            except ValueError:
                pass
        return value
