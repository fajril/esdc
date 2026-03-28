"""
KSMI ID Generation Utilities

This module produces identifiers for field, project, and zone entities in the
KSMI standard used by eSDC. It validates inputs, enforces digit uniqueness,
and calculates Luhn mod 16 checksums to keep identifiers consistent.

Key Features:
- Normalize grid and field identifiers while enforcing formatting rules.
- Generate field IDs with per-grid unique suffixes and checksums.
- Produce project IDs sequentially per field context.
- Produce zone IDs with unique three-digit suffixes and checksums.
- Expose helpers for checksum calculation and suffix allocation.

Dependencies:
- itertools.cycle: Round-robin distribution of grids when generating fields.
- typing: Type annotations for inputs and helper signatures.
"""

from __future__ import annotations

from itertools import cycle
from collections.abc import Iterable, Iterator, Sequence

_HEX_DIGITS = "0123456789ABCDEF"
_SUFFIX_RANGE = range(10_000)
_ZONE_RANGE = range(1_000)


def _iter_suffix_candidates() -> Iterator[str]:
    """Yield valid four-digit suffix candidates with unique decimal digits."""

    for value in _SUFFIX_RANGE:
        candidate = f"{value:04d}"
        if len(set(candidate)) == 4:
            yield candidate


def _iter_zone_candidates() -> Iterator[str]:
    """Yield valid three-digit zone suffix candidates with unique digits."""

    for value in _ZONE_RANGE:
        candidate = f"{value:03d}"
        if len(set(candidate)) == 3:
            yield candidate


def _normalize_grid_token(token: str) -> str:
    """Return the two-character grid code extracted from a token.

    Args:
        token: Grid code, field ID, or formatted field string.

    Returns:
        Two-character uppercase hexadecimal grid code.

    Raises:
        ValueError: If the token is empty, malformed, or contains
            non-hexadecimal characters.
    """

    raw = token.strip().upper()
    if not raw:
        raise ValueError("grid token cannot be empty")

    compact = raw.replace("-", "")
    if compact.startswith("F"):
        if len(compact) != 8:
            message = "invalid field id format in grid token"
            raise ValueError(f"{message}: {token!r}")
        grid = compact[1:3]
    elif len(compact) == 2:
        grid = compact
    elif len(compact) == 7:
        grid = compact[:2]
    else:
        raise ValueError(f"unable to derive grid from token: {token!r}")

    if any(char not in _HEX_DIGITS for char in grid):
        raise ValueError(f"grid must be hexadecimal: {token!r}")

    return grid


def _normalize_field_id(field_id: str) -> str:
    """Normalize a field ID into its compact eight-character form.

    Args:
        field_id: Field identifier string, with or without separators.

    Returns:
        Eight-character uppercase field ID without separators.

    Raises:
        ValueError: Propagated from validation failures.
    """

    raw = field_id.strip().upper().replace("-", "")
    _ = _parse_field_id(raw)
    return raw


def _parse_field_id(field_id: str) -> tuple[str, str]:
    """Validate and decompose a field ID into grid and suffix components.

    Args:
        field_id: Field identifier string, with or without separators.

    Returns:
        Tuple containing the grid code and four-digit suffix.

    Raises:
        ValueError: If the ID structure, digits, or checksum are invalid.
    """

    raw = field_id.strip().upper().replace("-", "")
    if len(raw) != 8 or not raw.startswith("F"):
        raise ValueError(f"invalid field ID format: {field_id!r}")

    grid = raw[1:3]
    suffix = raw[3:7]
    checksum = raw[7]

    if any(char not in _HEX_DIGITS for char in grid):
        raise ValueError(f"invalid grid in field ID: {field_id!r}")

    if not suffix.isdigit():
        raise ValueError(f"field suffix must be decimal digits: {field_id!r}")

    if len(set(suffix)) != 4:
        raise ValueError(f"field suffix must contain unique digits: {field_id!r}")

    expected_checksum = _luhn_mod16(grid + suffix)
    if checksum != expected_checksum:
        message = "invalid checksum for field ID"
        raise ValueError(f"{message} {field_id!r}; expected {expected_checksum}")

    return grid, suffix


def _parse_project_id(project_id: str) -> tuple[str, int]:
    """Validate and split a project ID into field payload and sequence.

    Args:
        project_id: Project identifier string with optional separators.

    Returns:
        Tuple of field payload (7 characters) and the project sequence number.

    Raises:
        ValueError: If the format, sequence digits, or range are invalid.
    """

    raw = project_id.strip().upper().replace("-", "")
    if len(raw) != 10 or not raw.startswith("P"):
        raise ValueError(f"invalid project ID format: {project_id!r}")

    field_payload = raw[1:8]
    sequence_text = raw[8:10]

    _ = _parse_field_id("F" + field_payload)

    if not sequence_text.isdigit():
        raise ValueError(f"project suffix must be decimal digits: {project_id!r}")

    sequence_number = int(sequence_text)
    if sequence_number < 1 or sequence_number > 99:
        raise ValueError(f"project sequence must be within 01-99: {project_id!r}")

    return field_payload, sequence_number


def _parse_zone_id(zone_id: str) -> tuple[str, str]:
    """Validate and split a zone ID into field payload and zone suffix.

    Args:
        zone_id: Zone identifier string with optional separators.

    Returns:
        Tuple containing the field payload (7 characters) and zone suffix.

    Raises:
        ValueError: If the format, suffix digits, or checksum are invalid.
    """

    raw = zone_id.strip().upper().replace("-", "")
    if len(raw) != 12 or not raw.startswith("Z"):
        raise ValueError(f"invalid zone ID format: {zone_id!r}")

    field_payload = raw[1:8]
    zone_suffix = raw[8:11]
    checksum = raw[11]

    _ = _parse_field_id("F" + field_payload)

    if not zone_suffix.isdigit():
        raise ValueError(f"zone suffix must be decimal digits: {zone_id!r}")

    if len(set(zone_suffix)) != 3:
        raise ValueError(f"zone suffix digits must be unique: {zone_id!r}")

    expected_checksum = _luhn_mod16(field_payload + zone_suffix)
    if checksum != expected_checksum:
        message = "invalid checksum for zone ID"
        raise ValueError(f"{message} {zone_id!r}; expected {expected_checksum}")

    return field_payload, zone_suffix


def _luhn_mod16(payload_hex: str) -> str:
    """Compute the Luhn mod 16 checksum for a hexadecimal payload.

    Args:
        payload_hex: Hexadecimal string used to calculate the checksum.

    Returns:
        Single hexadecimal character representing the checksum.

    Raises:
        ValueError: If the payload contains non-hexadecimal characters.
    """

    total = 0
    double = True

    for char in reversed(payload_hex):
        if char not in _HEX_DIGITS:
            raise ValueError(f"payload must be hexadecimal: {payload_hex!r}")

        value = int(char, 16)
        if double:
            value *= 2
            if value >= 16:
                value -= 15
        total += value
        double = not double

    remainder = total % 16
    check_digit = (16 - remainder) % 16
    return _HEX_DIGITS[check_digit]


def _next_available_suffix(used_suffixes: set[str]) -> str:
    """Return the next unused field suffix for a grid.

    Args:
        used_suffixes: Four-digit suffixes already allocated for the grid.

    Returns:
        Four-digit suffix string with unique digits.

    Raises:
        ValueError: If no suffixes remain.
    """

    for candidate in _iter_suffix_candidates():
        if candidate not in used_suffixes:
            used_suffixes.add(candidate)
            return candidate
    raise ValueError("no available suffixes remain for the specified grid")


def _next_project_sequence(used_sequences: set[int]) -> str:
    """Return the next unused project sequence number.

    Args:
        used_sequences: Sequence numbers already allocated for the field.

    Returns:
        Two-digit sequence string within 01-99.

    Raises:
        ValueError: If all sequence numbers for the field are exhausted.
    """

    for number in range(1, 100):
        if number not in used_sequences:
            used_sequences.add(number)
            return f"{number:02d}"
    raise ValueError("no project sequence numbers remain for this field")


def _next_zone_suffix(used_suffixes: set[str]) -> str:
    """Return the next unused zone suffix for a field.

    Args:
        used_suffixes: Zone suffixes already allocated for the field.

    Returns:
        Three-digit suffix string with unique digits.

    Raises:
        ValueError: If all zone suffixes are used.
    """

    for candidate in _iter_zone_candidates():
        if candidate not in used_suffixes:
            used_suffixes.add(candidate)
            return candidate
    raise ValueError("no zone suffixes remain for this field")


def _collect_existing_grids(
    existing_ids: Iterable[str],
) -> tuple[list[str], dict[str, set[str]]]:
    """Gather grid ordering and suffix usage from existing field IDs.

    Args:
        existing_ids: Iterable of field IDs to analyse.

    Returns:
        Ordered list of grids encountered and mapping of grid to used suffixes.
    """

    grid_order: list[str] = []
    grid_suffixes: dict[str, set[str]] = {}

    for field_id in existing_ids:
        grid, suffix = _parse_field_id(field_id)
        if grid not in grid_suffixes:
            grid_suffixes[grid] = {suffix}
            grid_order.append(grid)
        else:
            grid_suffixes[grid].add(suffix)

    return grid_order, grid_suffixes


def _resolve_ordered_grids(
    grid_tokens: Sequence[str] | None,
    existing_grid_order: Sequence[str],
    grid_suffixes: dict[str, set[str]],
) -> list[str]:
    """Determine the ordered grids to use when generating field IDs.

    Args:
        grid_tokens: Optional grid inputs supplied by the caller.
        existing_grid_order: Grids inferred from current field IDs.
        grid_suffixes: Mapping of grids to their used suffixes.

    Returns:
        Ordered list of grids to iterate over during generation.

    Raises:
        ValueError: If no grid information is available.
    """

    ordered: list[str] = []
    seen: set[str] = set()

    if grid_tokens:
        for token in grid_tokens:
            grid = _normalize_grid_token(token)
            if grid not in grid_suffixes:
                grid_suffixes[grid] = set()
            if grid not in seen:
                ordered.append(grid)
                seen.add(grid)
    else:
        for grid in existing_grid_order:
            if grid not in seen:
                ordered.append(grid)
                seen.add(grid)

    if not ordered:
        if not grid_suffixes:
            raise ValueError("no grid information provided")
        ordered = list(grid_suffixes)

    return ordered


def _collect_existing_projects(
    project_ids: Iterable[str],
) -> dict[str, set[int]]:
    """Map field payloads to their used project sequence numbers.

    Args:
        project_ids: Iterable of project IDs to analyse.

    Returns:
        Dictionary mapping field payloads to sets of allocated sequences.
    """

    existing: dict[str, set[int]] = {}
    for project_id in project_ids:
        field_payload, sequence_number = _parse_project_id(project_id)
        existing.setdefault(field_payload, set()).add(sequence_number)
    return existing


def _collect_existing_zones(zone_ids: Iterable[str]) -> dict[str, set[str]]:
    """Map field payloads to their used zone suffixes.

    Args:
        zone_ids: Iterable of zone IDs to analyse.

    Returns:
        Dictionary mapping field payloads to sets of allocated zone suffixes.
    """

    existing: dict[str, set[str]] = {}
    for zone_id in zone_ids:
        field_payload, zone_suffix = _parse_zone_id(zone_id)
        existing.setdefault(field_payload, set()).add(zone_suffix)
    return existing


def gen_field_id(
    current_grid: Sequence[str] | None,
    current_field_id: Sequence[str] | None,
    total_id: int = 1,
) -> list[str]:
    """Generate field IDs based on provided grids or existing field IDs.

    Args:
        current_grid: Grid identifiers to target when generating IDs.
        current_field_id: Existing field IDs used to maintain uniqueness.
        total_id: Number of field IDs requested.

    Returns:
        List of formatted field IDs (`F-XX-XXXX-X`).

    Raises:
        ValueError: If `total_id` is below one or grid data is unavailable.
    """

    if total_id < 1:
        raise ValueError("total_id must be at least 1")

    existing_grid_order, grid_suffixes = _collect_existing_grids(current_field_id or [])
    ordered_grids = _resolve_ordered_grids(
        current_grid, existing_grid_order, grid_suffixes
    )

    grid_iterator = cycle(ordered_grids)
    results: list[str] = []

    for _ in range(total_id):
        grid = next(grid_iterator)
        used_suffixes = grid_suffixes.setdefault(grid, set())
        suffix = _next_available_suffix(used_suffixes)
        results.append(f"F-{grid}-{suffix}-{_luhn_mod16(grid + suffix)}")

    return results


def gen_project_id(
    field_id: str,
    current_project_id: Sequence[str] | None = None,
    total_id: int = 1,
) -> list[str]:
    """Generate project IDs for a field using sequential numbering.

    Args:
        field_id: Field identifier that anchors the project IDs.
        current_project_id: Existing project IDs to maintain continuity.
        total_id: Number of project IDs requested.

    Returns:
        List of formatted project IDs (`P-XXXXXXX-XX`).

    Raises:
        ValueError: If `total_id` is below one or inputs are invalid.
    """

    if total_id < 1:
        raise ValueError("total_id must be at least 1")

    normalized_field = _normalize_field_id(field_id)
    field_payload = normalized_field[1:]

    existing_projects = _collect_existing_projects(current_project_id or [])
    used_sequences = existing_projects.setdefault(field_payload, set())

    results: list[str] = []

    for _ in range(total_id):
        sequence = _next_project_sequence(used_sequences)
        results.append(f"P-{field_payload}-{sequence}")

    return results


def gen_zone_id(
    field_id: str,
    current_zone_id: Sequence[str] | None = None,
    total_id: int = 1,
) -> list[str]:
    """Generate zone IDs for a field with unique suffixes and checksum.

    Args:
        field_id: Field identifier that anchors the zone IDs.
        current_zone_id: Existing zone IDs to track used suffixes.
        total_id: Number of zone IDs requested.

    Returns:
        List of formatted zone IDs (`Z-XXXXXXX-XXX-X`).

    Raises:
        ValueError: If `total_id` is below one or inputs are invalid.
    """

    if total_id < 1:
        raise ValueError("total_id must be at least 1")

    normalized_field = _normalize_field_id(field_id)
    field_payload = normalized_field[1:]

    existing_zones = _collect_existing_zones(current_zone_id or [])
    used_suffixes = existing_zones.setdefault(field_payload, set())

    results: list[str] = []

    for _ in range(total_id):
        zone_suffix = _next_zone_suffix(used_suffixes)
        checksum = _luhn_mod16(field_payload + zone_suffix)
        results.append(f"Z-{field_payload}-{zone_suffix}-{checksum}")

    return results


def verify_field_id(field_id: str) -> bool:
    """Return whether the supplied field ID passes checksum validation.

    Args:
        field_id: Field ID candidate, with or without separators.

    Returns:
        True when the field ID conforms to structure, uniqueness, and
        checksum rules; otherwise False.
    """

    try:
        _ = _parse_field_id(field_id)
    except ValueError:
        return False
    return True


def verify_zone_id(zone_id: str) -> bool:
    """Return whether the supplied zone ID passes checksum validation.

    Args:
        zone_id: Zone ID candidate, with or without separators.

    Returns:
        True when the zone ID conforms to structure, uniqueness, and
        checksum rules; otherwise False.
    """

    try:
        _ = _parse_zone_id(zone_id)
    except ValueError:
        return False
    return True
