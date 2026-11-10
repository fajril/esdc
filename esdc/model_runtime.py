"""Effective runtime resolution of canonical task assignments.

This module assembles runtime provider mappings from canonical connections and
task targets. It does not persist settings; :class:`esdc.configs.Config` owns
persistence and delegates here. Credentials remain inside the provider mapping,
which is excluded from dataclass reprs.
"""

import copy
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from esdc.model_settings import TaskAssignment, TaskMode, TaskRole


@dataclass(frozen=True, slots=True)
class ResolvedTarget:
    """One effective connection/model/options target for a task."""

    connection: str
    provider: dict[str, Any] = field(repr=False)
    model: str
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResolvedTask:
    """A task's effective mode, enabled state and resolved targets."""

    role: TaskRole
    mode: TaskMode
    enabled: bool
    primary: ResolvedTarget | None = None
    fallbacks: tuple[ResolvedTarget, ...] = ()
    overrides: tuple[str, ...] = ()


_EMBEDDING_BACKENDS = {
    "ollama": "ollama",
    "openai_compatible": "openai",
}


def _runtime_provider(
    connection_name: str,
    connection: Mapping[str, Any],
    model: str,
    options: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the transport adapter for one connection and target."""
    provider = copy.deepcopy(dict(connection))
    provider.pop("model", None)
    provider.pop("reasoning_effort", None)
    provider["name"] = connection_name
    provider_type = str(
        provider.get("type") or provider.get("provider_type") or ""
    ).strip()
    provider["provider_type"] = provider_type
    provider["model"] = model

    if "reasoning_effort" in options:
        provider["reasoning_effort"] = copy.deepcopy(options["reasoning_effort"])
    if "temperature" in options:
        provider["temperature"] = copy.deepcopy(options["temperature"])
    return provider


def _resolve_target(
    role: TaskRole,
    target: Any,
    load_connection: Callable[[str], dict[str, Any]],
) -> ResolvedTarget:
    if target is None:
        raise ValueError(f"Task {role!r} is in model mode without a target.")
    connection = load_connection(target.connection)
    options = dict(target.options)
    return ResolvedTarget(
        connection=target.connection,
        provider=_runtime_provider(
            target.connection, connection, target.model, options
        ),
        model=target.model,
        options=options,
    )


def _apply_embedding_overrides(
    target: ResolvedTarget | None,
    *,
    apply_overrides: bool,
) -> tuple[ResolvedTarget | None, tuple[str, ...]]:
    """Apply the supported per-run embedding environment overrides."""
    if not apply_overrides or target is None:
        return target, ()

    names: list[str] = []
    provider = copy.deepcopy(dict(target.provider))
    model = target.model

    backend_override = os.environ.get("ESDC_EMBEDDING_BACKEND")
    if backend_override:
        requested = backend_override.strip().lower()
        expected = _EMBEDDING_BACKENDS.get(
            str(provider.get("provider_type") or "").strip().lower()
        )
        if expected is None or requested != expected:
            raise ValueError(
                "`ESDC_EMBEDDING_BACKEND` "
                f"{requested!r} does not match connection "
                f"{target.connection!r} (type "
                f"{provider.get('provider_type')!r})."
            )
        names.append("ESDC_EMBEDDING_BACKEND")

    host_override = os.environ.get("ESDC_EMBEDDING_HOST")
    if host_override:
        provider["base_url"] = host_override
        names.append("ESDC_EMBEDDING_HOST")

    model_override = os.environ.get("ESDC_EMBEDDING_MODEL")
    if model_override:
        model = model_override
        names.append("ESDC_EMBEDDING_MODEL")

    key_override = os.environ.get("ESDC_EMBEDDING_API_KEY")
    if key_override:
        provider["api_key"] = key_override
        names.append("ESDC_EMBEDDING_API_KEY")

    return (
        ResolvedTarget(
            connection=target.connection,
            provider=provider,
            model=model,
            options=dict(target.options),
        ),
        tuple(names),
    )


def resolve_task(
    role: TaskRole,
    *,
    load_assignment: Callable[[TaskRole], TaskAssignment],
    load_connection: Callable[[str], dict[str, Any]],
    apply_overrides: bool = True,
) -> ResolvedTask:
    """Resolve one task through its canonical assignment and connections."""
    assignment = load_assignment(role)

    if assignment.mode == "model":
        primary = _resolve_target(role, assignment.target, load_connection)
        fallbacks = tuple(
            _resolve_target(role, target, load_connection)
            for target in assignment.fallbacks
        )
        overrides: tuple[str, ...] = ()
        if role == "embedding":
            primary, overrides = _apply_embedding_overrides(
                primary, apply_overrides=apply_overrides
            )
        return ResolvedTask(
            role=role,
            mode="model",
            enabled=assignment.enabled,
            primary=primary,
            fallbacks=fallbacks,
            overrides=overrides,
        )

    if assignment.mode == "chat" and role in {"metadata", "cleanup"}:
        chat = resolve_task(
            "chat",
            load_assignment=load_assignment,
            load_connection=load_connection,
            apply_overrides=apply_overrides,
        )
        return ResolvedTask(
            role=role,
            mode="chat",
            enabled=True,
            primary=chat.primary,
            fallbacks=chat.fallbacks,
            overrides=chat.overrides,
        )

    return ResolvedTask(
        role=role,
        mode=assignment.mode,
        enabled=assignment.enabled,
        primary=None,
        fallbacks=(),
        overrides=(),
    )
