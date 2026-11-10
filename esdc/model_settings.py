"""Canonical connection and per-task model settings.

The public interfaces are defined here once: task roles and modes, the
immutable assignment data model, and schema parsing/validation. This module is
pure: it never reads a file, constructs a provider client, or uses the network.
Persistence belongs to :class:`esdc.configs.Config`.
"""

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

TaskRole = Literal["chat", "embedding", "rerank", "ocr", "metadata", "cleanup"]
TaskMode = Literal["model", "chat", "ocr", "off", "unset"]

MODEL_SETTINGS_VERSION = 1
MODEL_SETTINGS_REQUIRED_MESSAGE = (
    "Model settings are not configured; convert the existing model "
    "configuration first."
)

TASK_ROLES: tuple[TaskRole, ...] = (
    "chat",
    "embedding",
    "rerank",
    "ocr",
    "metadata",
    "cleanup",
)
TASK_MODES: tuple[TaskMode, ...] = ("model", "chat", "ocr", "off", "unset")

ROLE_LABELS: dict[TaskRole, str] = {
    "chat": "Chat",
    "embedding": "Embeddings",
    "rerank": "Reranking",
    "ocr": "OCR",
    "metadata": "Metadata extraction",
    "cleanup": "Text cleanup",
}

# Kept in sync with the existing provider registry. A test pins this set against
# `esdc.providers.list_provider_types()` so a provider added there cannot be
# silently omitted from canonical validation.
KNOWN_CONNECTION_TYPES: frozenset[str] = frozenset(
    {
        "ollama",
        "openai",
        "openai_compatible",
        "anthropic",
        "google",
        "azure_openai",
        "groq",
        "deepseek",
        "ollama_cloud",
        "opencode",
    }
)

# Existing provider-type eligibility. Chat, metadata and cleanup support every
# text provider; embeddings and OCR use the HTTP Ollama/OpenAI-compatible
# adapters, and reranking uses the existing OpenAI-compatible rerank adapter.
TEXT_CONNECTION_TYPES = KNOWN_CONNECTION_TYPES
EMBEDDING_CONNECTION_TYPES = frozenset({"ollama", "openai_compatible"})
RERANK_CONNECTION_TYPES = frozenset({"openai_compatible"})
OCR_CONNECTION_TYPES = frozenset({"ollama", "openai_compatible"})

ROLE_CONNECTION_TYPES: dict[TaskRole, frozenset[str]] = {
    "chat": TEXT_CONNECTION_TYPES,
    "embedding": EMBEDDING_CONNECTION_TYPES,
    "rerank": RERANK_CONNECTION_TYPES,
    "ocr": OCR_CONNECTION_TYPES,
    "metadata": TEXT_CONNECTION_TYPES,
    "cleanup": TEXT_CONNECTION_TYPES,
}

SUPPORTED_MODES: dict[TaskRole, frozenset[TaskMode]] = {
    "chat": frozenset({"model", "unset"}),
    "embedding": frozenset({"model", "unset"}),
    "rerank": frozenset({"model", "unset"}),
    "ocr": frozenset({"model", "unset"}),
    "metadata": frozenset({"model", "chat", "ocr", "unset"}),
    "cleanup": frozenset({"model", "chat", "off", "unset"}),
}

LEGACY_MODEL_ROOT_KEYS = frozenset(
    {
        "providers",
        "default_provider",
        "provider_order",
        "embedding_backend",
        "embedding_host",
        "embedding_model",
        "embedding_api_key",
        "embedding_provider",
    }
)

LEGACY_MODEL_CORPUS_KEYS = frozenset(
    {
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
)

_ASSIGNMENT_FIELDS = frozenset(
    {"mode", "connection", "model", "options", "enabled", "fallbacks"}
)
_TARGET_FIELDS = frozenset({"connection", "model", "options"})


class ModelConfigurationRequired(ValueError):  # noqa: N818
    """Raised when canonical model settings are requested from an old config."""


@dataclass(frozen=True, slots=True)
class ModelTarget:
    """One connection/model pair with its task-specific model options."""

    connection: str
    model: str
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TaskAssignment:
    """One task's canonical mode, target, enabled switch and fallbacks."""

    mode: TaskMode
    target: ModelTarget | None = None
    enabled: bool = True
    fallbacks: tuple[ModelTarget, ...] = ()


def default_model_settings() -> dict[str, Any]:
    """Return fresh canonical ``models`` defaults.

    Chat, Embeddings and OCR are unset; Reranking is unset and disabled;
    Metadata and Text cleanup follow Chat.
    """
    return {
        "chat": {"mode": "unset"},
        "embedding": {"mode": "unset"},
        "rerank": {"mode": "unset", "enabled": False},
        "ocr": {"mode": "unset"},
        "metadata": {"mode": "chat"},
        "cleanup": {"mode": "chat"},
    }


def has_legacy_model_settings(config: Mapping[str, Any]) -> bool:
    """Whether a mapping still carries any recognized pre-canonical model key."""
    if not isinstance(config, Mapping):
        return False
    if any(key in config for key in LEGACY_MODEL_ROOT_KEYS):
        return True
    corpus = config.get("corpus")
    return isinstance(corpus, Mapping) and any(
        key in corpus for key in LEGACY_MODEL_CORPUS_KEYS
    )


def _role_or_raise(role: object) -> TaskRole:
    if role not in TASK_ROLES:
        expected = ", ".join(TASK_ROLES)
        raise ValueError(f"Unknown model task: {role!r}. Expected one of: {expected}.")
    return cast(TaskRole, role)


def _supported_mode_list(role: TaskRole) -> str:
    return ", ".join(mode for mode in TASK_MODES if mode in SUPPORTED_MODES[role])


def _non_empty_name(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{label} must not have leading or trailing whitespace.")
    return value


def _options_or_raise(value: object, label: str) -> dict[str, Any]:
    """Return a deep copy of a model-options mapping; ``None`` means empty."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} options must be a mapping.")
    copied: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{label} option keys must be strings.")
        copied[key] = copy.deepcopy(item)
    return copied


def _target_from_mapping(value: Mapping[str, Any], label: str) -> ModelTarget:
    unknown = set(value) - _TARGET_FIELDS
    if unknown:
        field_name = sorted(str(name) for name in unknown)[0]
        raise ValueError(f"Unknown field {field_name!r} in {label}.")
    connection = _non_empty_name(value.get("connection"), f"{label} connection")
    model = _non_empty_name(value.get("model"), f"{label} model")
    options = _options_or_raise(value.get("options"), label)
    return ModelTarget(connection=connection, model=model, options=options)


def _duplicate_pair_message(role: TaskRole, target: ModelTarget) -> str:
    return (
        f"Duplicate connection/model pair in {ROLE_LABELS[role]} assignment: "
        f"{target.connection!r} / {target.model!r}."
    )


def parse_task_assignment(role: TaskRole, value: Mapping[str, Any]) -> TaskAssignment:
    """Parse and validate one canonical task assignment mapping.

    The returned options mappings are deep copies, detaching them from the
    incoming mapping. Role/provider eligibility and connection existence are
    validated by :func:`validate_model_settings`, which has the registry.
    """
    task = _role_or_raise(role)
    label = ROLE_LABELS[task]

    if not isinstance(value, Mapping):
        raise ValueError(f"{label} assignment must be a mapping.")

    unknown = set(value) - _ASSIGNMENT_FIELDS
    if unknown:
        field_name = sorted(str(name) for name in unknown)[0]
        raise ValueError(f"Unknown field {field_name!r} in {label} assignment.")

    mode_value = value.get("mode")
    if not isinstance(mode_value, str) or mode_value not in TASK_MODES:
        raise ValueError(f"Unknown model task mode {mode_value!r} for {label}.")
    mode = cast(TaskMode, mode_value)
    if mode not in SUPPORTED_MODES[task]:
        raise ValueError(
            f"Mode {mode!r} is not supported for {label}. Supported modes: "
            f"{_supported_mode_list(task)}."
        )

    target_fields_present = bool(set(value) & _TARGET_FIELDS)
    fallbacks_present = "fallbacks" in value
    target: ModelTarget | None
    fallbacks: tuple[ModelTarget, ...]

    if mode != "model":
        if target_fields_present or fallbacks_present:
            raise ValueError(
                f"Mode {mode!r} for {label} must not define connection, model, "
                "options, or fallbacks."
            )
        target = None
        fallbacks = ()
    else:
        if "connection" not in value or "model" not in value:
            raise ValueError(f"{label} model mode requires a connection and a model.")
        target = _target_from_mapping(
            {key: value[key] for key in _TARGET_FIELDS if key in value},
            f"{label} target",
        )
        if fallbacks_present:
            if task != "chat":
                raise ValueError("Only Chat supports fallbacks.")
            raw_fallbacks = value["fallbacks"]
            if raw_fallbacks is None:
                raw_fallbacks = []
            if not isinstance(raw_fallbacks, (list, tuple)):
                raise ValueError(
                    "Chat fallbacks must be a list of connection/model targets."
                )
            parsed_fallbacks: list[ModelTarget] = []
            for index, item in enumerate(raw_fallbacks, start=1):
                if not isinstance(item, Mapping):
                    raise ValueError(f"Chat fallback {index} must be a mapping.")
                parsed_fallbacks.append(
                    _target_from_mapping(item, f"Chat fallback {index}")
                )
            fallbacks = tuple(parsed_fallbacks)
        else:
            fallbacks = ()

        seen = {(target.connection, target.model)}
        for fallback in fallbacks:
            pair = (fallback.connection, fallback.model)
            if pair in seen:
                raise ValueError(_duplicate_pair_message(task, fallback))
            seen.add(pair)

    if task == "rerank":
        enabled_value = value.get("enabled", False)
        if not isinstance(enabled_value, bool):
            raise ValueError("Reranking `enabled` must be a boolean.")
        if enabled_value and mode != "model":
            raise ValueError("Reranking cannot be enabled without a model.")
        enabled = enabled_value
    else:
        enabled = True
        if "enabled" in value and value["enabled"] is not True:
            raise ValueError("Only Reranking has an enabled switch.")

    return TaskAssignment(
        mode=mode, target=target, enabled=enabled, fallbacks=fallbacks
    )


def _serialize_target(target: ModelTarget, label: str) -> dict[str, Any]:
    if not isinstance(target, ModelTarget):
        raise ValueError(f"{label} must be a ModelTarget.")
    connection = _non_empty_name(target.connection, f"{label} connection")
    model = _non_empty_name(target.model, f"{label} model")
    options = _options_or_raise(target.options, label)
    serialized: dict[str, Any] = {"connection": connection, "model": model}
    if options:
        serialized["options"] = options
    return serialized


def serialize_task_assignment(
    role: TaskRole, assignment: TaskAssignment
) -> dict[str, Any]:
    """Serialize an assignment to the design's plain YAML fields."""
    task = _role_or_raise(role)
    label = ROLE_LABELS[task]

    if not isinstance(assignment, TaskAssignment):
        raise ValueError("Expected a TaskAssignment.")
    if assignment.mode not in SUPPORTED_MODES[task]:
        raise ValueError(
            f"Mode {assignment.mode!r} is not supported for {label}. Supported "
            f"modes: {_supported_mode_list(task)}."
        )

    if task == "rerank":
        if not isinstance(assignment.enabled, bool):
            raise ValueError("Reranking `enabled` must be a boolean.")
        if assignment.enabled and assignment.mode != "model":
            raise ValueError("Reranking cannot be enabled without a model.")
    elif assignment.enabled is not True:
        raise ValueError("Only Reranking has an enabled switch.")

    if assignment.mode != "model":
        if assignment.target is not None:
            raise ValueError(
                f"Mode {assignment.mode!r} for {label} must not define a target."
            )
        if assignment.fallbacks:
            raise ValueError("Only a Chat model assignment can have fallbacks.")
        serialized: dict[str, Any] = {"mode": assignment.mode}
    else:
        target = assignment.target
        if target is None:
            raise ValueError(f"{label} model mode requires a target.")
        serialized = {"mode": "model", **_serialize_target(target, f"{label} target")}

        if assignment.fallbacks and task != "chat":
            raise ValueError("Only Chat supports fallbacks.")
        if assignment.fallbacks:
            serialized["fallbacks"] = [
                _serialize_target(fallback, f"Chat fallback {index}")
                for index, fallback in enumerate(assignment.fallbacks, start=1)
            ]
            seen = {(target.connection, target.model)}
            for fallback in assignment.fallbacks:
                pair = (fallback.connection, fallback.model)
                if pair in seen:
                    raise ValueError(_duplicate_pair_message(task, fallback))
                seen.add(pair)

    if task == "rerank":
        serialized["enabled"] = assignment.enabled
    return serialized


def _validate_connections(connections: Mapping[Any, Any]) -> None:
    for name, entry in connections.items():
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise ValueError(
                "Connection names must be non-empty strings without surrounding "
                "whitespace."
            )
        if not isinstance(entry, Mapping):
            raise ValueError(f"Connection {name!r} must be a mapping.")
        provider_type = entry.get("type")
        if (
            not isinstance(provider_type, str)
            or provider_type not in KNOWN_CONNECTION_TYPES
        ):
            raise ValueError(
                f"Connection {name!r} has an unknown or missing provider type: "
                f"{provider_type!r}."
            )


def _validate_target(
    role: TaskRole, target: ModelTarget, connections: Mapping[str, Any]
) -> None:
    connection = connections.get(target.connection)
    if connection is None:
        raise ValueError(f"Unknown connection reference: {target.connection!r}.")
    provider_type = connection["type"]
    allowed = ROLE_CONNECTION_TYPES[role]
    if provider_type not in allowed:
        raise ValueError(
            f"Connection {target.connection!r} (type {provider_type!r}) cannot "
            f"serve {ROLE_LABELS[role]}; allowed types: "
            f"{', '.join(sorted(allowed))}."
        )


def validate_model_settings(config: Mapping[str, Any]) -> None:
    """Validate a complete canonical model-settings mapping.

    Raises:
        ModelConfigurationRequired: the settings predate the canonical marker.
        ValueError: the version, registry, role, mode, reference or provider
            type is invalid.
    """
    if not isinstance(config, Mapping):
        raise ValueError("Model settings must be a mapping.")

    version = config.get("model_settings_version")
    if version is None:
        raise ModelConfigurationRequired(MODEL_SETTINGS_REQUIRED_MESSAGE)
    if type(version) is not int or version != MODEL_SETTINGS_VERSION:
        raise ValueError(
            f"Unsupported model settings version: {version!r}. "
            f"Expected {MODEL_SETTINGS_VERSION}."
        )

    connections = config.get("connections")
    if not isinstance(connections, Mapping):
        raise ValueError("`connections` must be a mapping of named connections.")
    _validate_connections(connections)

    models = config.get("models")
    if not isinstance(models, Mapping):
        raise ValueError("`models` must be a mapping of task assignments.")

    unknown_roles = [
        str(role)
        for role in models
        if not isinstance(role, str) or role not in TASK_ROLES
    ]
    if unknown_roles:
        raise ValueError(f"Unknown model task: {', '.join(sorted(unknown_roles))}.")

    missing_roles = [role for role in TASK_ROLES if role not in models]
    if missing_roles:
        raise ValueError(f"Missing model task: {', '.join(missing_roles)}.")

    for role in TASK_ROLES:
        assignment = parse_task_assignment(role, models[role])
        if assignment.mode != "model":
            continue
        assert assignment.target is not None
        _validate_target(role, assignment.target, connections)
        for fallback in assignment.fallbacks:
            _validate_target(role, fallback, connections)
