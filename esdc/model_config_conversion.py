"""One-time conversion of pre-canonical model settings.

This module owns all interpretation of the former provider/direct/corpus model
fields. It is a pure leaf tool: no network, file I/O, source mutation, provider
construction or environment reads. The caller decides whether to apply a
candidate.

Design rules implemented here:

* Saved providers become canonical connections. Their ``model`` and
  ``reasoning_effort`` move to task assignments; transport/authentication
  fields, OAuth, API version and unknown extensions remain on the connection.
* Connections are deduplicated only when their complete normalized
  transport/authentication mapping matches. The same endpoint with a different
  key, OAuth or API version is a distinct connection.
* Chat uses the effective old failover order only. Providers outside that order
  are not activated as fallbacks merely because they have a stored model.
* Old corpus/embedding defaults are applied to absent keys because the former
  runtime merged them before use. A saved config with no model keys at all is
  treated as fresh and receives canonical defaults instead.
* Explicit provider/model pairs are direct. ``main`` and ``provider:foo`` remain
  literal model IDs when an explicit provider is present.
* Retired local embedding/rerank settings become unset/disabled with an
  explanation; no replacement service or invented server model is created.
* A legacy candidate that validates is returned even when task-level blockers
  exist, so unset bindings are visible and the caller can refuse to apply it.
  Format-level blockers (mixed formats, unsupported versions, invalid canonical
  settings) leave ``config`` as an unchanged source copy.
"""

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from esdc.model_settings import (
    KNOWN_CONNECTION_TYPES,
    LEGACY_MODEL_CORPUS_KEYS,
    LEGACY_MODEL_ROOT_KEYS,
    MODEL_SETTINGS_VERSION,
    ROLE_CONNECTION_TYPES,
    TaskRole,
    default_model_settings,
    has_legacy_model_settings,
    validate_model_settings,
)

# Providers whose old runtime default model is a static class constant. Ollama's
# default is dynamic (it lists installed models), and openai_compatible has no
# default, so neither may be guessed here. A test pins this map to the provider
# registry.
STATIC_PROVIDER_DEFAULTS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "azure_openai": "gpt-4o",
    "deepseek": "deepseek-v4-flash",
    "google": "gemini-2.5-flash",
    "groq": "llama-3.3-70b-versatile",
    "ollama_cloud": "llama3.2",
    "openai": "gpt-4o-mini",
    "opencode": "deepseek-v4-flash",
}

OLLAMA_EMBEDDING_DEFAULT_MODEL = "qwen3-embedding:0.6b"
OPENAI_EMBEDDING_DEFAULT_HOST = "http://localhost:1234"

# Historical defaults used only by this converter. Keeping them here means the
# converter remains the sole owner of old-field interpretation.
LEGACY_EMBEDDING_DEFAULTS: dict[str, Any] = {
    "embedding_backend": "ollama",
    "embedding_host": None,
    "embedding_model": "",
    "embedding_api_key": "",
    "embedding_provider": "",
}

LEGACY_CORPUS_DEFAULTS: dict[str, Any] = {
    "rerank": False,
    "rerank_backend": "provider",
    "rerank_provider": "",
    "rerank_model": "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF",
    "ocr_model": "glm-ocr",
    "ocr_provider": "",
    "metadata_model": "main",
    "metadata_provider": "",
    "cleanup_model": "main",
    "cleanup_provider": "",
    "ollama_host": "",
    "n_gpu_layers": -1,
}

_EMBEDDING_BACKEND_TYPES = {
    "ollama": "ollama",
    "openai": "openai_compatible",
}
_EMBEDDING_PROVIDER_BACKENDS = {
    provider_type: backend
    for backend, provider_type in _EMBEDDING_BACKEND_TYPES.items()
}


@dataclass(frozen=True, slots=True)
class ModelSettingsConversion:
    """A proposed canonical config plus visible warnings and blockers."""

    config: dict[str, Any]
    warnings: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()

    def __repr__(self) -> str:
        """Redact the candidate so credentials never reach logs or reports."""
        return (
            f"ModelSettingsConversion(config=<redacted>, "
            f"warnings={self.warnings!r}, blockers={self.blockers!r})"
        )


@dataclass
class _ProviderInfo:
    name: str
    entry: Mapping[str, Any]
    connection: str
    provider_type: str
    model: str
    reasoning_options: dict[str, Any]
    model_consumed: bool = False
    reasoning_consumed: bool = False


@dataclass
class _Context:
    source: dict[str, Any]
    corpus: Mapping[str, Any]
    raw_providers: Mapping[str, Any]
    registry: "_ConnectionRegistry"
    warnings: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    providers: dict[str, _ProviderInfo] = field(default_factory=dict)
    local_corpus_used: bool = False


class _ConnectionRegistry:
    """Deterministic connection names with exact-transport deduplication."""

    def __init__(self) -> None:
        self.connections: dict[str, dict[str, Any]] = {}
        self._names_by_signature: dict[Any, str] = {}

    def add(
        self,
        connection: Mapping[str, Any],
        preferred_name: str,
        warnings: list[str],
    ) -> tuple[str, bool]:
        """Add or reuse a connection; return ``(name, reused)``."""
        signature = _freeze(connection)
        existing = self._names_by_signature.get(signature)
        if existing is not None:
            return existing, True

        name = preferred_name
        if name in self.connections:
            name = self._unique_name(preferred_name)
            warnings.append(
                f"Connection name {preferred_name!r} is already in use; "
                f"generated {name!r}."
            )
        self.connections[name] = copy.deepcopy(dict(connection))
        self._names_by_signature[signature] = name
        return name, False

    def _unique_name(self, base: str) -> str:
        index = 2
        while f"{base}-{index}" in self.connections:
            index += 1
        return f"{base}-{index}"


def _freeze(value: Any) -> Any:
    """Return a hashable, order-independent signature for a YAML value."""
    if isinstance(value, Mapping):
        pairs = sorted(
            ((str(key), _freeze(item)) for key, item in value.items()),
            key=lambda pair: pair[0],
        )
        return tuple(pairs)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _source_value(ctx: _Context, key: str, default: Any) -> Any:
    if key in ctx.source:
        return ctx.source[key]
    return default


def _corpus_value(ctx: _Context, key: str) -> Any:
    if key in ctx.corpus:
        return ctx.corpus[key]
    return LEGACY_CORPUS_DEFAULTS[key]


def _bool_value(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return bool(value)


def _reasoning_options(entry: Mapping[str, Any]) -> dict[str, Any]:
    if "reasoning_effort" not in entry or entry["reasoning_effort"] is None:
        return {}
    return {"reasoning_effort": copy.deepcopy(entry["reasoning_effort"])}


def _canonical_connection(
    entry: Mapping[str, Any],
    name: str,
    blockers: list[str],
) -> tuple[dict[str, Any] | None, str]:
    raw_type = entry.get("provider_type") or entry.get("type") or name
    provider_type = str(raw_type).strip().lower()
    if provider_type not in KNOWN_CONNECTION_TYPES:
        blockers.append(
            f"Provider {name!r} has unknown or missing type {provider_type!r}."
        )
        return None, provider_type

    connection = {
        key: copy.deepcopy(value)
        for key, value in entry.items()
        if key not in {"model", "reasoning_effort", "provider_type"}
    }
    connection["type"] = provider_type
    for key in ("api_key", "base_url", "oauth"):
        if key in connection and not connection[key]:
            connection.pop(key)
    return connection, provider_type


def _convert_providers(ctx: _Context) -> None:
    for name, entry in ctx.raw_providers.items():
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            ctx.blockers.append(
                "Provider names must be non-empty strings without surrounding "
                "whitespace."
            )
            continue
        if not isinstance(entry, Mapping):
            ctx.blockers.append(f"Provider {name!r} must be a mapping.")
            continue

        connection, provider_type = _canonical_connection(entry, name, ctx.blockers)
        if connection is None:
            continue
        connection_name, reused = ctx.registry.add(connection, name, ctx.warnings)
        if reused and connection_name != name:
            ctx.warnings.append(
                f"Provider {name!r} shares its transport with connection "
                f"{connection_name!r}; reusing that connection."
            )
        ctx.providers[name] = _ProviderInfo(
            name=name,
            entry=entry,
            connection=connection_name,
            provider_type=provider_type,
            model=_text(entry.get("model")),
            reasoning_options=_reasoning_options(entry),
        )


def _lookup_provider(ctx: _Context, name: str, label: str) -> _ProviderInfo | None:
    if not name:
        ctx.blockers.append(f"{label} references an empty provider name.")
        return None
    if not isinstance(ctx.raw_providers, Mapping) or name not in ctx.raw_providers:
        ctx.blockers.append(f"{label} references missing provider {name!r}.")
        return None
    info = ctx.providers.get(name)
    if info is None:
        ctx.blockers.append(
            f"{label} references provider {name!r}, which could not be converted."
        )
        return None
    return info


def _effective_provider_model(
    ctx: _Context,
    info: _ProviderInfo,
    label: str,
) -> str | None:
    if info.model.strip():
        info.model_consumed = True
        return info.model
    static = STATIC_PROVIDER_DEFAULTS.get(info.provider_type)
    if static is not None:
        ctx.warnings.append(
            f"{label} provider {info.name!r} has no stored model; using its "
            f"documented default {static!r}."
        )
        return static
    ctx.blockers.append(
        f"{label} provider {info.name!r} has no model and no exact static "
        "default; cannot convert it."
    )
    ctx.warnings.append(
        f"{label} provider {info.name!r} has no model; it is left unset."
    )
    return None


def _effective_chat_order(ctx: _Context) -> list[str]:
    order: list[str] = []

    default = ctx.source.get("default_provider")
    if default is not None:
        if isinstance(default, str) and default:
            if default in ctx.raw_providers:
                order.append(default)
            else:
                ctx.warnings.append(
                    f"Ignored default_provider {default!r}: no such saved provider."
                )
        else:
            ctx.warnings.append("Ignored default_provider: it is not a name.")

    configured = ctx.source.get("provider_order")
    if isinstance(configured, list):
        for name in configured:
            if not isinstance(name, str) or not name:
                ctx.warnings.append("Ignored a non-name entry in provider_order.")
                continue
            if name not in ctx.raw_providers:
                ctx.warnings.append(
                    f"Ignored provider_order entry {name!r}: no such saved provider."
                )
                continue
            if name not in order:
                order.append(name)
    elif configured is not None:
        ctx.warnings.append("Ignored provider_order: it is not a list.")

    return order


def _convert_chat(ctx: _Context) -> dict[str, Any]:
    order = _effective_chat_order(ctx)
    infos = [ctx.providers[name] for name in order if name in ctx.providers]
    if not infos:
        return {"mode": "unset"}

    primary = infos[0]
    model = _effective_provider_model(ctx, primary, "Chat")
    if model is None:
        return {"mode": "unset"}

    assignment: dict[str, Any] = {
        "mode": "model",
        "connection": primary.connection,
        "model": model,
    }
    if primary.reasoning_options:
        assignment["options"] = copy.deepcopy(primary.reasoning_options)
        primary.reasoning_consumed = True

    seen = {(primary.connection, model)}
    fallbacks: list[dict[str, Any]] = []
    for info in infos[1:]:
        fallback_model = _effective_provider_model(ctx, info, "Chat fallback")
        if fallback_model is None:
            continue
        pair = (info.connection, fallback_model)
        if pair in seen:
            ctx.warnings.append(
                f"Skipped duplicate Chat fallback {info.connection!r} / "
                f"{fallback_model!r}."
            )
            continue
        seen.add(pair)
        fallback: dict[str, Any] = {
            "connection": info.connection,
            "model": fallback_model,
        }
        if info.reasoning_options:
            fallback["options"] = copy.deepcopy(info.reasoning_options)
            info.reasoning_consumed = True
        fallbacks.append(fallback)

    if fallbacks:
        assignment["fallbacks"] = fallbacks
    return assignment


def _convert_embedding(ctx: _Context) -> dict[str, Any]:
    backend = _text(
        _source_value(
            ctx,
            "embedding_backend",
            LEGACY_EMBEDDING_DEFAULTS["embedding_backend"],
        )
    ).strip().lower() or "ollama"
    provider_name = _text(
        _source_value(
            ctx,
            "embedding_provider",
            LEGACY_EMBEDDING_DEFAULTS["embedding_provider"],
        )
    ).strip()
    host = _text(
        _source_value(
            ctx,
            "embedding_host",
            LEGACY_EMBEDDING_DEFAULTS["embedding_host"],
        )
    ).strip()
    model = _text(
        _source_value(
            ctx, "embedding_model", LEGACY_EMBEDDING_DEFAULTS["embedding_model"]
        )
    ).strip()
    api_key = _text(
        _source_value(
            ctx,
            "embedding_api_key",
            LEGACY_EMBEDDING_DEFAULTS["embedding_api_key"],
        )
    )

    if backend == "local":
        ctx.warnings.append(
            "Embedding backend 'local' is retired; Embeddings is left unset."
        )
        return {"mode": "unset"}
    if backend not in _EMBEDDING_BACKEND_TYPES:
        ctx.warnings.append(
            f"Embedding backend {backend!r} is not supported; Embeddings is "
            "left unset."
        )
        return {"mode": "unset"}

    if provider_name:
        info = _lookup_provider(ctx, provider_name, "Embedding")
        if info is None:
            return {"mode": "unset"}
        expected_backend = _EMBEDDING_PROVIDER_BACKENDS.get(info.provider_type)
        if expected_backend is None:
            ctx.blockers.append(
                f"Embedding provider {provider_name!r} has type "
                f"{info.provider_type!r}, which cannot serve Embeddings."
            )
            return {"mode": "unset"}
        if backend != expected_backend:
            ctx.blockers.append(
                f"Embedding backend {backend!r} does not match provider "
                f"{provider_name!r} (type {info.provider_type!r})."
            )
            return {"mode": "unset"}

        connection_name = info.connection
        connection_map = ctx.registry.connections[connection_name]
        if not _text(connection_map.get("base_url")).strip():
            if not host:
                ctx.blockers.append(
                    f"Embedding provider {provider_name!r} has no address and "
                    "embedding_host is empty."
                )
                return {"mode": "unset"}
            derived = copy.deepcopy(connection_map)
            derived["base_url"] = host
            connection_name, _ = ctx.registry.add(derived, "embedding", ctx.warnings)
            ctx.warnings.append(
                f"Embedding provider {provider_name!r} has no address; using "
                f"embedding_host for connection {connection_name!r}."
            )

        if not model:
            if backend == "ollama":
                model = OLLAMA_EMBEDDING_DEFAULT_MODEL
                ctx.warnings.append(
                    "Embedding model was empty; using Ollama's documented "
                    f"default {OLLAMA_EMBEDDING_DEFAULT_MODEL!r}."
                )
            else:
                ctx.warnings.append(
                    "Embedding model is empty for the openai backend; "
                    "Embeddings is left unset."
                )
                return {"mode": "unset"}
        if info.model and model == info.model:
            info.model_consumed = True
        return {
            "mode": "model",
            "connection": connection_name,
            "model": model,
        }

    if backend == "ollama":
        connection: dict[str, Any] = {"type": "ollama"}
        if host:
            connection["base_url"] = host
        connection_name, _ = ctx.registry.add(connection, "embedding", ctx.warnings)
        if not model:
            model = OLLAMA_EMBEDDING_DEFAULT_MODEL
            ctx.warnings.append(
                "Embedding model was empty; using Ollama's documented default "
                f"{OLLAMA_EMBEDDING_DEFAULT_MODEL!r}."
            )
        return {
            "mode": "model",
            "connection": connection_name,
            "model": model,
        }

    connection = {
        "type": "openai_compatible",
        "base_url": host or OPENAI_EMBEDDING_DEFAULT_HOST,
    }
    if api_key.strip():
        connection["api_key"] = api_key
    connection_name, _ = ctx.registry.add(connection, "embedding", ctx.warnings)
    if not model:
        ctx.warnings.append(
            "Embedding model is empty for the openai backend; Embeddings is "
            "left unset."
        )
        return {"mode": "unset"}
    return {
        "mode": "model",
        "connection": connection_name,
        "model": model,
    }


def _convert_rerank(ctx: _Context) -> dict[str, Any]:
    enabled = _bool_value(_corpus_value(ctx, "rerank"), False)
    backend = _text(_corpus_value(ctx, "rerank_backend")).strip().lower() or "provider"
    provider_name = _text(_corpus_value(ctx, "rerank_provider")).strip()
    model = _text(_corpus_value(ctx, "rerank_model")).strip()

    unset = {"mode": "unset", "enabled": False}
    if backend == "local":
        ctx.warnings.append(
            "Rerank backend 'local' is retired; Reranking is left unset and "
            "disabled."
        )
        return unset
    if backend != "provider":
        ctx.warnings.append(
            f"Rerank backend {backend!r} is not supported; Reranking is left "
            "unset and disabled."
        )
        return unset
    if not provider_name:
        if enabled:
            ctx.warnings.append(
                "Reranking is enabled but no rerank provider is configured; it "
                "is left unset and disabled."
            )
        return unset

    info = _lookup_provider(ctx, provider_name, "Reranking")
    if info is None:
        return unset
    if info.provider_type != "openai_compatible":
        ctx.warnings.append(
            f"Provider {provider_name!r} has type {info.provider_type!r} and "
            "cannot serve Reranking; Reranking is left unset and disabled."
        )
        return unset
    if not model:
        ctx.warnings.append(
            f"Rerank provider {provider_name!r} has no model; Reranking is left "
            "unset and disabled."
        )
        return unset
    if info.model and model == info.model:
        info.model_consumed = True
    return {
        "mode": "model",
        "connection": info.connection,
        "model": model,
        "enabled": enabled,
    }


def _corpus_connection(ctx: _Context, *, used: bool) -> str:
    host = _text(_corpus_value(ctx, "ollama_host")).strip()
    connection: dict[str, Any] = {"type": "ollama"}
    if host:
        connection["base_url"] = host
    name, _ = ctx.registry.add(connection, "corpus-ollama", ctx.warnings)
    if used:
        ctx.local_corpus_used = True
    return name


def _convert_ocr(ctx: _Context) -> dict[str, Any]:
    model = _text(_corpus_value(ctx, "ocr_model")).strip()
    provider_name = _text(_corpus_value(ctx, "ocr_provider")).strip()

    if provider_name:
        info = _lookup_provider(ctx, provider_name, "OCR")
        if info is None:
            return {"mode": "unset"}
        if not model:
            ctx.warnings.append("OCR model is empty; OCR is left unset.")
            return {"mode": "unset"}
        if info.provider_type not in ROLE_CONNECTION_TYPES["ocr"]:
            ctx.warnings.append(
                f"Provider {provider_name!r} has type {info.provider_type!r} "
                "and cannot serve OCR; OCR is left unset."
            )
            return {"mode": "unset"}
        if info.model and model == info.model:
            info.model_consumed = True
        return {
            "mode": "model",
            "connection": info.connection,
            "model": model,
        }

    if not model:
        ctx.warnings.append("OCR model is empty; OCR is left unset.")
        return {"mode": "unset"}

    connection_name = _corpus_connection(ctx, used=True)
    return {
        "mode": "model",
        "connection": connection_name,
        "model": model,
    }


def _convert_text_task(ctx: _Context, role: TaskRole) -> dict[str, Any]:
    label = "Metadata extraction" if role == "metadata" else "Text cleanup"
    model_raw = _text(_corpus_value(ctx, f"{role}_model"))
    model = model_raw.strip()
    provider_name = _text(_corpus_value(ctx, f"{role}_provider")).strip()

    if provider_name:
        info = _lookup_provider(ctx, provider_name, label)
        if info is None:
            return {"mode": "unset"}
        if not model:
            ctx.warnings.append(
                f"{label} provider {provider_name!r} has no model; {label} is "
                "left unset."
            )
            return {"mode": "unset"}
        if info.provider_type not in ROLE_CONNECTION_TYPES[role]:
            ctx.warnings.append(
                f"Provider {provider_name!r} has type {info.provider_type!r} "
                f"and cannot serve {label}; {label} is left unset."
            )
            return {"mode": "unset"}
        if info.model and model == info.model:
            info.model_consumed = True
        assignment: dict[str, Any] = {
            "mode": "model",
            "connection": info.connection,
            "model": model,
        }
        if info.reasoning_options:
            assignment["options"] = copy.deepcopy(info.reasoning_options)
            info.reasoning_consumed = True
        return assignment

    if model == "main":
        return {"mode": "chat"}
    if not model:
        return {"mode": "ocr" if role == "metadata" else "off"}

    if model_raw.startswith("provider:"):
        linked_name = model_raw[len("provider:") :]
        info = _lookup_provider(ctx, linked_name, label)
        if info is None:
            return {"mode": "unset"}
        if info.provider_type not in ROLE_CONNECTION_TYPES[role]:
            ctx.warnings.append(
                f"Provider {linked_name!r} has type {info.provider_type!r} and "
                f"cannot serve {label}; {label} is left unset."
            )
            return {"mode": "unset"}
        effective = _effective_provider_model(ctx, info, f"{label} link")
        if effective is None:
            return {"mode": "unset"}
        ctx.warnings.append(
            f"{role}_model {model_raw!r} was a provider link; it is now an "
            f"explicit assignment to connection {info.connection!r} model "
            f"{effective!r}."
        )
        assignment = {
            "mode": "model",
            "connection": info.connection,
            "model": effective,
        }
        if info.reasoning_options:
            assignment["options"] = copy.deepcopy(info.reasoning_options)
            info.reasoning_consumed = True
        return assignment

    connection_name = _corpus_connection(ctx, used=True)
    return {
        "mode": "model",
        "connection": connection_name,
        "model": model,
    }


def _warn_unused_provider_models(ctx: _Context) -> None:
    for info in ctx.providers.values():
        if info.model.strip() and not info.model_consumed:
            ctx.warnings.append(
                f"Provider {info.name!r} stored model {info.model!r} is not "
                "assigned to any task; it remains in the source backup."
            )
        if info.reasoning_options and not info.reasoning_consumed:
            ctx.warnings.append(
                f"Provider {info.name!r} stored reasoning_effort is not assigned "
                "to any task; it remains in the source backup."
            )


def _ensure_corpus_host_connection(ctx: _Context) -> None:
    host = _text(_corpus_value(ctx, "ollama_host")).strip()
    if not host or ctx.local_corpus_used:
        return
    name = _corpus_connection(ctx, used=False)
    provider_connections = {info.connection for info in ctx.providers.values()}
    if name not in provider_connections:
        ctx.warnings.append(
            f"corpus.ollama_host is not used by any task; it is kept as "
            f"connection {name!r}."
        )


def _strip_legacy_keys(candidate: dict[str, Any]) -> None:
    for key in LEGACY_MODEL_ROOT_KEYS:
        candidate.pop(key, None)
    corpus = candidate.get("corpus")
    if isinstance(corpus, dict):
        for key in LEGACY_MODEL_CORPUS_KEYS:
            corpus.pop(key, None)


def _blocked(
    source: Mapping[str, Any],
    message: str,
    warnings: list[str] | None = None,
) -> ModelSettingsConversion:
    return ModelSettingsConversion(
        config=copy.deepcopy(dict(source)),
        warnings=tuple(warnings or ()),
        blockers=(message,),
    )


def _convert_canonical(source: dict[str, Any]) -> ModelSettingsConversion:
    if has_legacy_model_settings(source):
        return _blocked(
            source,
            "Model settings mix canonical fields with legacy model keys; "
            "resolve the conflict explicitly before converting.",
        )

    version = source.get("model_settings_version")
    if type(version) is not int or version != MODEL_SETTINGS_VERSION:
        return _blocked(
            source,
            f"Unsupported model settings version: {version!r}.",
        )

    try:
        validate_model_settings(source)
    except ValueError as exc:
        return _blocked(source, f"Canonical model settings are invalid: {exc}")

    return ModelSettingsConversion(config=copy.deepcopy(source))


def _fresh_canonical(source: dict[str, Any]) -> ModelSettingsConversion:
    candidate = copy.deepcopy(source)
    candidate["model_settings_version"] = MODEL_SETTINGS_VERSION
    candidate["connections"] = {}
    candidate["models"] = default_model_settings()
    try:
        validate_model_settings(candidate)
    except ValueError as exc:
        return _blocked(
            source,
            f"Canonical model defaults are invalid: {exc}",
        )
    return ModelSettingsConversion(config=candidate)


def _convert_legacy(source: dict[str, Any]) -> ModelSettingsConversion:
    warnings: list[str] = []
    blockers: list[str] = []

    raw_providers: Mapping[str, Any] = {}
    if "providers" in source:
        supplied = source["providers"]
        if isinstance(supplied, Mapping):
            raw_providers = supplied
        else:
            blockers.append("`providers` must be a mapping.")

    corpus: Mapping[str, Any] = {}
    if "corpus" in source:
        supplied_corpus = source["corpus"]
        if isinstance(supplied_corpus, Mapping):
            corpus = supplied_corpus
        else:
            blockers.append("`corpus` must be a mapping.")

    ctx = _Context(
        source=source,
        corpus=corpus,
        raw_providers=raw_providers,
        registry=_ConnectionRegistry(),
        warnings=warnings,
        blockers=blockers,
    )
    _convert_providers(ctx)

    models = default_model_settings()
    models["chat"] = _convert_chat(ctx)
    models["embedding"] = _convert_embedding(ctx)
    models["rerank"] = _convert_rerank(ctx)
    models["ocr"] = _convert_ocr(ctx)
    models["metadata"] = _convert_text_task(ctx, "metadata")
    models["cleanup"] = _convert_text_task(ctx, "cleanup")
    _warn_unused_provider_models(ctx)
    _ensure_corpus_host_connection(ctx)

    candidate = copy.deepcopy(source)
    _strip_legacy_keys(candidate)
    candidate["model_settings_version"] = MODEL_SETTINGS_VERSION
    candidate["connections"] = ctx.registry.connections
    candidate["models"] = models

    try:
        validate_model_settings(candidate)
    except ValueError as exc:
        ctx.blockers.append(f"Converted model settings are invalid: {exc}")
        return ModelSettingsConversion(
            config=copy.deepcopy(source),
            warnings=tuple(ctx.warnings),
            blockers=tuple(ctx.blockers),
        )

    return ModelSettingsConversion(
        config=candidate,
        warnings=tuple(ctx.warnings),
        blockers=tuple(ctx.blockers),
    )


def convert_model_settings(source: Mapping[str, Any]) -> ModelSettingsConversion:
    """Convert the current saved model settings into a canonical candidate.

    The source is never mutated. A candidate with blockers leaves ``config`` as
    an unchanged source copy, so a caller that ignores the blockers cannot
    destroy the old model settings. A successful candidate removes only the
    recognized superseded keys and is validated with ``model_settings``.
    """
    if not isinstance(source, Mapping):
        return ModelSettingsConversion(
            config={},
            blockers=("Model settings source must be a mapping.",),
        )

    source_copy = copy.deepcopy(dict(source))

    if "model_settings_version" in source_copy:
        return _convert_canonical(source_copy)
    if "connections" in source_copy or "models" in source_copy:
        return _blocked(
            source_copy,
            "Model settings contain canonical fields without "
            "`model_settings_version`; resolve the incomplete or mixed "
            "configuration first.",
        )
    if not has_legacy_model_settings(source_copy):
        return _fresh_canonical(source_copy)
    return _convert_legacy(source_copy)
