# esdc/model_catalog.py
"""Read-only model discovery for saved provider connections.

Discovery is best effort. Nothing here writes config, starts a model, or
downloads weights, and no network, HTTP, JSON, or response-shape failure
raises: each one returns a short warning instead.

Unknown stays unknown. An entry with no roles and ``supports_vision=None`` is
unverified; unknown is never read as chat, never read as non-vision, and never
inferred from a model name, alias, or endpoint.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import requests

logger = logging.getLogger(__name__)

ModelRole = Literal["chat", "embedding", "rerank"]
CatalogRole = ModelRole  # D4 name for the same role set; keep one role enum

DEFAULT_TIMEOUT_SECONDS = 8.0
CONNECT_TIMEOUT_CAP = 5.0
MIN_TIMEOUT_SECONDS = 0.1

# The provider layer's own fallback hosts. No audited hostname, alias, or key
# belongs in product code.
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


@dataclass(frozen=True)
class ModelEntry:
    """One discovered model id plus whatever metadata was actually verified."""

    model_id: str
    roles: frozenset[ModelRole] = frozenset()
    supports_vision: bool | None = None


@dataclass(frozen=True)
class ModelCatalog:
    """Discovery result, in the order the model list returned it.

    ``warning`` is a short, credential-free status; ``None`` means the catalog
    is as complete as the server reported it.
    """

    entries: tuple[ModelEntry, ...] = ()
    warning: str | None = None


def api_endpoint(base_url: str, resource: str, *, root: bool = False) -> str:
    """Build one request URL from a saved provider address.

    Args:
        base_url: Saved provider address. Surrounding whitespace and a trailing
            slash are removed; any path prefix is preserved.
        resource: Path segment appended after the base, such as ``models`` or
            ``model/info``.
        root: When true, strip only a terminal ``/v1`` so the resource sits at
            the server root. When false, ensure exactly one terminal ``/v1``.

    Returns:
        The request URL, which can never contain ``/v1/v1``.

    Raises:
        ValueError: the caller passed an empty base URL.
    """
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        raise ValueError("api_endpoint requires a base URL")

    resource_path = str(resource).strip().lstrip("/")
    if root:
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
    elif not base.endswith("/v1"):
        base = f"{base}/v1"

    return f"{base}/{resource_path}"


def discover_models(
    provider: Mapping[str, Any], *, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
) -> ModelCatalog:
    """Discover the models one saved connection reports, best effort.

    Args:
        provider: Saved provider entry (a copy, never the cached config dict).
        timeout_seconds: Read timeout. The connect timeout is the smaller of
            this value and 5 seconds.

    Returns:
        Catalog with entries and, when something could not be verified, a short
        warning. Never raises.
    """
    provider_type = _provider_type(provider)
    base_url = str(provider.get("base_url") or "").strip()

    if provider_type == "ollama":
        return _discover_ollama(base_url or DEFAULT_OLLAMA_BASE_URL, timeout_seconds)

    if provider_type == "openai_compatible":
        if not base_url:
            return ModelCatalog(
                warning="Provider has no base URL; model discovery was skipped."
            )
    elif provider_type == "openai":
        base_url = base_url or DEFAULT_OPENAI_BASE_URL
    else:
        name = provider_type or "unknown"
        return ModelCatalog(
            warning=f"Model discovery is not supported for provider type {name}."
        )

    return _discover_openai_compatible(provider, base_url, timeout_seconds)


def _provider_type(provider: Mapping[str, Any]) -> str:
    """Return the saved provider type, stripped and lowercased."""
    return str(
        provider.get("provider_type") or provider.get("type") or ""
    ).strip().lower()


def _credential(provider: Mapping[str, Any]) -> str:
    """Return the saved connection credential, or an empty string.

    No placeholder key is ever substituted: an empty credential means no
    authentication header.
    """
    oauth = provider.get("oauth")
    token = oauth.get("access_token") if isinstance(oauth, Mapping) else ""
    return str(provider.get("api_key") or token or "")


def _auth_headers(provider: Mapping[str, Any]) -> dict[str, str]:
    """Bearer header, present only when a credential exists."""
    credential = _credential(provider)
    return {"Authorization": f"Bearer {credential}"} if credential else {}


def _timeout_pair(timeout_seconds: float) -> tuple[float, float]:
    """(connect, read) timeouts: connect capped at 5s, read as requested."""
    read = max(MIN_TIMEOUT_SECONDS, float(timeout_seconds))
    return (min(CONNECT_TIMEOUT_CAP, read), read)


def _failure_reason(error: Exception) -> str:
    """Short, credential-free reason for a failed request.

    Only the failure kind and, for an HTTP error, its status code are kept. The text of
    an error is never read: an Ollama response error holds the server's error body as
    its message, so a warning built from `str(error)` could repeat it. A status code is
    the only fact taken from an unknown error, otherwise its type name alone, so nothing
    here can carry a response body, a request URL, or a credential.
    """
    if isinstance(error, requests.exceptions.HTTPError):
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
        return f"HTTP {status}" if status is not None else "HTTP error"
    if isinstance(error, requests.exceptions.Timeout):
        return "timeout"
    if isinstance(error, requests.exceptions.ConnectionError):
        return "connection error"
    if isinstance(error, ValueError):
        return "invalid JSON"

    # ollama.ResponseError falls here: it is not a requests error, and its status_code
    # is read without importing ollama. Its message is the server body, so it is unused.
    status = getattr(error, "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        return f"HTTP {status}"
    return type(error).__name__


def _fetch_json(
    url: str, headers: Mapping[str, str], timeout: tuple[float, float]
) -> tuple[Any, str | None]:
    """GET one URL and parse JSON, returning ``(payload, failure_reason)``.

    ``failure_reason`` is ``None`` on success; nothing here raises to the caller.
    """
    try:
        response = requests.get(url, headers=dict(headers), timeout=timeout)
        response.raise_for_status()
        return response.json(), None
    except Exception as error:  # best effort: discovery never raises to the caller
        return None, _failure_reason(error)


def _named_items(payload: Any) -> tuple[list[Mapping[str, Any]], bool]:
    """Response items, tolerating a top-level list. ``False`` means bad shape."""
    items = payload.get("data") if isinstance(payload, Mapping) else payload
    if not isinstance(items, list):
        return [], False
    return [item for item in items if isinstance(item, Mapping)], True


def _model_ids(payload: Any) -> tuple[list[str], bool]:
    """Model ids from ``/v1/models``, deduplicated in first-seen order."""
    items, shape_ok = _named_items(payload)
    ids: list[str] = []
    for item in items:
        model_id = str(item.get("id") or "").strip()
        if model_id and model_id not in ids:
            ids.append(model_id)
    return ids, shape_ok


def _role_from_mode(mode: Any) -> ModelRole | None:
    """One verified mode; anything else stays unknown."""
    if isinstance(mode, str):
        normalized = mode.strip().lower()
        if normalized == "chat":
            return "chat"
        if normalized == "embedding":
            return "embedding"
        if normalized == "rerank":
            return "rerank"
    return None


def _roles_and_vision(
    entry: Mapping[str, Any],
) -> tuple[frozenset[ModelRole], bool | None]:
    """Role and vision status verified by one metadata entry; unknown stays unknown."""
    info = entry.get("model_info")
    if not isinstance(info, Mapping):
        return frozenset(), None

    roles: set[ModelRole] = set()
    role = _role_from_mode(info.get("mode"))
    if role is not None:
        roles.add(role)

    vision = info.get("supports_vision")
    return frozenset(roles), vision if isinstance(vision, bool) else None


def _discover_openai_compatible(
    provider: Mapping[str, Any], base_url: str, timeout_seconds: float
) -> ModelCatalog:
    """Standard ``/v1/models`` IDs, enriched best effort by root ``/model/info``."""
    headers = _auth_headers(provider)
    timeout = _timeout_pair(timeout_seconds)

    payload, reason = _fetch_json(api_endpoint(base_url, "models"), headers, timeout)
    if reason is not None:
        return ModelCatalog(warning=f"Model discovery failed ({reason}).")

    model_ids, shape_ok = _model_ids(payload)
    if not shape_ok:
        return ModelCatalog(
            warning="Model discovery returned an unexpected response shape."
        )
    if not model_ids:
        return ModelCatalog(warning="Model discovery returned no models.")

    warning = None
    metadata, meta_reason = _fetch_json(
        api_endpoint(base_url, "model/info", root=True), headers, timeout
    )
    metadata_items: list[Mapping[str, Any]] = []
    if meta_reason is not None:
        warning = f"Role metadata unavailable ({meta_reason}); models are unverified."
    else:
        metadata_items, meta_shape_ok = _named_items(metadata)
        if not meta_shape_ok:
            warning = (
                "Role metadata unavailable (unexpected response shape); "
                "models are unverified."
            )

    by_name: dict[str, Mapping[str, Any]] = {}
    by_id: dict[str, Mapping[str, Any]] = {}
    for item in metadata_items:
        name = str(item.get("model_name") or "").strip()
        identifier = str(item.get("id") or "").strip()
        if name:
            by_name.setdefault(name, item)
        if identifier:
            by_id.setdefault(identifier, item)

    entries: list[ModelEntry] = []
    for model_id in model_ids:
        # Match by model_name first, then id; unmatched entries are ignored.
        item = by_name.get(model_id) or by_id.get(model_id)
        if item is None:
            entries.append(ModelEntry(model_id=model_id))
            continue
        roles, vision = _roles_and_vision(item)
        entries.append(
            ModelEntry(model_id=model_id, roles=roles, supports_vision=vision)
        )

    # A metadata answer with nothing usable leaves the IDs unverified and adds
    # no warning of its own; a malformed answer is reported above.
    return ModelCatalog(entries=tuple(entries), warning=warning)


def _discover_ollama(base_url: str, timeout_seconds: float) -> ModelCatalog:
    """Ollama model list, with optional per-model capabilities."""
    try:
        import ollama
    except ImportError as error:
        return ModelCatalog(
            warning=f"Ollama model discovery failed ({_failure_reason(error)})."
        )

    # The Ollama client takes no per-call timeout argument, so the bound is set
    # on the client itself and applies to its list and show calls.
    read = max(MIN_TIMEOUT_SECONDS, float(timeout_seconds))
    client = ollama.Client(host=base_url, timeout=read)

    try:
        result = client.list()
    except Exception as error:  # never raises, and never retries on the local host
        return ModelCatalog(
            warning=(
                f"Ollama model list failed ({_failure_reason(error)}); "
                "models are unverified."
            )
        )

    names, shape_ok = _ollama_names(result)
    if not shape_ok:
        return ModelCatalog(
            warning="Ollama model list returned an unexpected response shape."
        )
    if not names:
        return ModelCatalog(warning="Ollama model list returned no models.")

    entries = tuple(
        ModelEntry(
            model_id=name,
            roles=frozenset(),
            supports_vision=_ollama_vision(client, name),
        )
        for name in names
    )
    return ModelCatalog(entries=entries)


def _ollama_names(result: Any) -> tuple[list[str], bool]:
    """Model names from an Ollama list response; ``False`` means unreadable shape."""
    models = getattr(result, "models", None)
    if models is None and isinstance(result, Mapping):
        models = result.get("models")
    if not isinstance(models, list):
        return [], False

    names: list[str] = []
    for model in models:
        name = getattr(model, "model", None)
        if name is None and isinstance(model, Mapping):
            name = model.get("model") or model.get("name")
        if isinstance(name, str) and name.strip() and name not in names:
            names.append(name.strip())
    return names, True


def _ollama_vision(client: Any, model: str) -> bool | None:
    """Vision capability from one show call; unknown when it cannot be read.

    A single failed or malformed call does not fail the whole catalog.
    """
    try:
        info = client.show(model)
    except Exception:
        return None

    capabilities = getattr(info, "capabilities", None)
    if capabilities is None and isinstance(info, Mapping):
        capabilities = info.get("capabilities")
    if not isinstance(capabilities, list) or not all(
        isinstance(item, str) for item in capabilities
    ):
        return None

    return "vision" in {item.strip().lower() for item in capabilities}
