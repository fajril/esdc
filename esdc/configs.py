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
    "corpus.chunk_size": "Max characters per corpus chunk",
    "corpus.chunk_overlap": "Characters carried over between corpus chunks",
    "corpus.rerank_pool": "Number of RRF candidates scored when rerank is on",
    "corpus.max_chunks_per_doc": (
        "Max chunks of the same document kept in search() results "
        "(0 = no cap, the setting used by the A/B eval run)"
    ),
    "corpus.query_instruct": (
        "Prefix search/aggregate queries with Qwen3-Embedding's documented "
        "instruction template before embedding (query side only -- chunks "
        "stay raw, so toggling this never needs a re-embed)"
    ),
    "corpus.ocr_dpi": "Page render resolution (DPI) for OCR",
    "corpus.num_ctx": "Ollama context window size for corpus OCR/metadata models",
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


# Model assignments are edited only by the canonical task picker. The generic
# key editor therefore has no task/model sections left.
TASK_SECTIONS: dict[str, list[str]] = {}

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
# (e.g. save_connection). The Config class uses classmethods for
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
    def get_provider_configs_by_priority(cls) -> list[dict[str, Any]]:
        """Return canonical Chat runtime adapters in failover order."""
        resolved = cls.resolve_model_task("chat")
        if resolved.mode != "model" or resolved.primary is None:
            return []
        targets = (resolved.primary, *resolved.fallbacks)
        return [copy.deepcopy(dict(target.provider)) for target in targets]

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

        Connections are not addressed through dotted keys: a connection name
        may itself contain a dot, so connection fields are saved through
        `save_connection`/`rename_connection`.

        Raises:
            ValueError: the key targets a connection entry.
        """
        parts = key.split(".")
        if parts[0] == "connections":
            raise ValueError(
                f"Connection fields are not saved through dotted keys: {key!r}. "
                "Use save_connection/rename_connection."
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

        A canonical task assignment is reset as a whole with
        ``models.<role>``: only that task's mode/target/fallbacks/enabled change,
        never a connection or another task. Resetting a subkey of a task is
        rejected because it could leave an invalid assignment behind.

        Args:
            key: Dot-notation key to reset, or None to reset all.
        """
        if key is None:
            defaults = cls.get_defaults()
            cls._save_config(defaults)
            return

        if key.startswith("models."):
            parts = key.split(".")
            if len(parts) != 2 or parts[1] not in TASK_ROLES:
                raise KeyError(
                    "Reset a whole task assignment with "
                    "'models.<role>' (for example models.chat)."
                )
            role = parts[1]
            config = cls._model_config_for_write()
            models = config.get("models")
            if not isinstance(models, dict):
                raise ValueError("`models` must be a mapping of task assignments.")
            models[role] = default_model_settings()[role]
            config["models"] = models
            validate_model_settings(config)
            cls._save_config(config)
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
