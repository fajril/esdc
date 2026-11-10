"""Preview and apply one explicit model-settings conversion.

The pure converter remains in :mod:`esdc.model_config_conversion`. This module
owns only the file workflow: read a named source, show a masked summary, create
a private complete backup and replace the source atomically after validating
the candidate and rechecking source freshness.

No network, database, provider client or live default config is touched.
"""

from __future__ import annotations

import contextlib
import os
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from esdc.config_fields import format_config_value
from esdc.configs import Config
from esdc.model_config_conversion import (
    ModelSettingsConversion,
    convert_model_settings,
)
from esdc.model_settings import ROLE_LABELS, TASK_ROLES, TaskRole


class ModelSettingsApplyError(ValueError):
    """A source/backup path or the file workflow could not be used."""


@dataclass(frozen=True)
class ModelSettingsPreview:
    """A conversion candidate plus whether it changes the named source."""

    source: Path
    conversion: ModelSettingsConversion
    noop: bool


@dataclass(frozen=True)
class ModelSettingsApplyResult:
    """Outcome of an apply request; no source is changed unless applied."""

    applied: bool
    noop: bool
    backup: Path | None
    conversion: ModelSettingsConversion


def _regular_file(path: Path, *, label: str) -> None:
    try:
        info = path.lstat()
    except OSError as error:
        raise ModelSettingsApplyError(f"{label} cannot be read.") from error
    if stat.S_ISLNK(info.st_mode):
        raise ModelSettingsApplyError(f"{label} must not be a symbolic link.")
    if not stat.S_ISREG(info.st_mode):
        raise ModelSettingsApplyError(f"{label} must be a regular file.")


def _read_source(path: Path) -> tuple[bytes, dict[str, Any]]:
    _regular_file(path, label="Source file")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ModelSettingsApplyError("Source file cannot be read.") from error
    try:
        loaded = yaml.safe_load(raw)
    except yaml.YAMLError as error:
        raise ModelSettingsApplyError("Source file is not valid YAML.") from error
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, Mapping):
        raise ModelSettingsApplyError("Source file must contain a YAML mapping.")
    return raw, dict(loaded)


def _different_paths(source: Path, backup: Path) -> None:
    if source.resolve() == backup.resolve():
        raise ModelSettingsApplyError(
            "Source and backup must be different files."
        )


def preview_model_settings(source: Path) -> ModelSettingsPreview:
    """Read one source and return its masked conversion preview.

    Nothing is written, no cache is changed and no network is used.
    """
    source_path = Path(source).expanduser()
    _, payload = _read_source(source_path)
    conversion = convert_model_settings(payload)
    noop = not conversion.blockers and conversion.config == payload
    return ModelSettingsPreview(
        source=source_path, conversion=conversion, noop=noop
    )


def _write_private_backup(path: Path, content: bytes) -> None:
    if os.path.lexists(path):
        raise ModelSettingsApplyError(
            "Backup path already exists; choose a new file."
        )
    parent = path.parent
    if not parent.exists() or not parent.is_dir():
        raise ModelSettingsApplyError("Backup directory does not exist.")
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ModelSettingsApplyError(
            "Backup path already exists; choose a new file."
        ) from error
    except OSError as error:
        raise ModelSettingsApplyError("Backup file could not be created.") from error
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o600)
    except Exception:
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)
        raise


def _replace_source(source: Path, candidate: Mapping[str, Any], mode: int) -> None:
    directory = source.parent
    try:
        fd, temp_name = tempfile.mkstemp(
            dir=directory, prefix=".esdc-model-settings-", suffix=".yaml.tmp"
        )
    except OSError as error:
        raise ModelSettingsApplyError(
            "Source file could not be replaced."
        ) from error
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w") as handle:
            yaml.safe_dump(dict(candidate), handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_path, mode)
        os.replace(temp_path, source)
    except Exception as error:
        with contextlib.suppress(OSError):
            temp_path.unlink(missing_ok=True)
        raise ModelSettingsApplyError(
            "Source file could not be replaced; the original is intact."
        ) from error


def _same_bytes(path: Path, expected: bytes) -> bool:
    try:
        return path.read_bytes() == expected
    except OSError:
        return False


def apply_model_settings(
    source: Path, backup: Path
) -> ModelSettingsApplyResult:
    """Apply one conversion, or return a blocked/no-op result.

    Order: validate paths and candidate, create a private complete backup,
    recheck the source bytes, then atomically replace the source. A failure
    before the replacement leaves the source intact (a completed backup is kept
    as a recovery copy, never used to overwrite the source automatically).
    """
    source_path = Path(source).expanduser()
    backup_path = Path(backup).expanduser()
    _different_paths(source_path, backup_path)

    original_bytes, payload = _read_source(source_path)
    mode = stat.S_IMODE(source_path.stat().st_mode) & 0o777
    conversion = convert_model_settings(payload)
    if conversion.blockers:
        return ModelSettingsApplyResult(
            applied=False,
            noop=False,
            backup=None,
            conversion=conversion,
        )
    if conversion.config == payload:
        return ModelSettingsApplyResult(
            applied=False,
            noop=True,
            backup=None,
            conversion=conversion,
        )

    _write_private_backup(backup_path, original_bytes)
    if not _same_bytes(source_path, original_bytes):
        raise ModelSettingsApplyError(
            "Source file changed during conversion; nothing was replaced."
        )
    _replace_source(source_path, conversion.config, mode)
    Config._config_cache = None
    return ModelSettingsApplyResult(
        applied=True,
        noop=False,
        backup=backup_path,
        conversion=conversion,
    )


def _assignment_summary(role: TaskRole, value: Any) -> str:
    if not isinstance(value, Mapping):
        return f"{ROLE_LABELS[role]}: (invalid)"
    mode = value.get("mode")
    if mode == "model":
        connection = str(value.get("connection") or "")
        model = str(value.get("model") or "")
        text = f"{ROLE_LABELS[role]}: {connection} / {model}"
        fallbacks = value.get("fallbacks")
        if role == "chat" and isinstance(fallbacks, list) and fallbacks:
            pairs = ", ".join(
                f"{item.get('connection')}/{item.get('model')}"
                for item in fallbacks
                if isinstance(item, Mapping)
            )
            text += f" (fallbacks: {pairs})"
        if role == "rerank":
            text += " — enabled" if value.get("enabled") is True else " — disabled"
        return text
    if mode == "chat":
        return f"{ROLE_LABELS[role]}: use chat model"
    if mode == "ocr":
        return f"{ROLE_LABELS[role]}: use OCR result"
    if mode == "off":
        return f"{ROLE_LABELS[role]}: off"
    return f"{ROLE_LABELS[role]}: not set"


def render_preview(preview: ModelSettingsPreview) -> list[str]:
    """A masked, credential-free summary for a terminal."""
    conversion = preview.conversion
    lines = [
        f"Source: {preview.source}",
        "Preview only; nothing has been written.",
    ]
    if conversion.blockers:
        lines.append("Blockers:")
        lines.extend(f"  - {blocker}" for blocker in conversion.blockers)
    if preview.noop:
        lines.append("Already canonical; no changes.")
        return lines

    connections = conversion.config.get("connections")
    if isinstance(connections, Mapping):
        lines.append(f"Connections ({len(connections)}):")
        for name, entry in connections.items():
            if not isinstance(entry, Mapping):
                lines.append(f"  {name}: (invalid)")
                continue
            provider_type = str(entry.get("type") or "unknown")
            base_url = format_config_value("base_url", entry.get("base_url") or "")
            suffix = f" base_url={base_url}" if base_url else ""
            lines.append(f"  {name} ({provider_type}){suffix}")

    models = conversion.config.get("models")
    if isinstance(models, Mapping):
        lines.append("Assignments:")
        for role in TASK_ROLES:
            lines.append(f"  {_assignment_summary(role, models.get(role))}")

    if conversion.warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in conversion.warnings)
    return lines
