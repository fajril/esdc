"""SQLite runtime-version checks and writer connection policy."""

from __future__ import annotations

import sqlite3


def is_wal_runtime_supported(version: tuple[int, int, int]) -> bool:
    """Return whether an SQLite release meets the approved WAL safety floor."""
    major, minor, patch = version
    if major > 3:
        return True
    if major < 3:
        return False
    if minor > 51:
        return True
    if minor == 51:
        return patch >= 3
    if minor == 50:
        return patch >= 7
    if minor == 44:
        return patch >= 6
    return False


def require_wal_runtime() -> None:
    """Reject SQLite libraries below the WAL safety floor before writer setup."""
    version = sqlite3.sqlite_version_info
    if is_wal_runtime_supported(version):
        return
    version_text = ".".join(str(part) for part in version)
    raise RuntimeError(
        f"SQLite {version_text} is unsupported for WAL writers; requires "
        "SQLite >=3.51.3, or patched 3.50.7+/3.44.6+."
    )


def configure_writer(connection: sqlite3.Connection) -> None:
    """Configure and verify WAL and required pragmas on a file-backed writer."""
    require_wal_runtime()
    if connection.in_transaction:
        raise RuntimeError(
            "Cannot configure SQLite WAL while the connection has an active "
            "transaction; configure the writer before BEGIN."
        )

    try:
        journal_row = connection.execute("PRAGMA journal_mode = WAL").fetchone()
    except sqlite3.Error as exc:
        raise RuntimeError(f"Could not enable SQLite WAL journal mode: {exc}") from exc
    journal_mode = None if journal_row is None else str(journal_row[0]).lower()
    if journal_mode != "wal":
        raise RuntimeError(
            "SQLite refused WAL journal mode; "
            f"PRAGMA journal_mode returned {journal_mode!r}."
        )

    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA synchronous = FULL")
        foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()
        busy_timeout = connection.execute("PRAGMA busy_timeout").fetchone()
        synchronous = connection.execute("PRAGMA synchronous").fetchone()
    except sqlite3.Error as exc:
        raise RuntimeError(f"Could not configure SQLite writer pragmas: {exc}") from exc

    if foreign_keys is None or foreign_keys[0] != 1:
        raise RuntimeError("SQLite writer connection did not enable foreign_keys.")
    if busy_timeout is None or busy_timeout[0] != 5000:
        raise RuntimeError("SQLite writer connection did not set busy_timeout=5000.")
    if synchronous is None or synchronous[0] != 2:
        raise RuntimeError("SQLite writer connection did not set synchronous=FULL.")
