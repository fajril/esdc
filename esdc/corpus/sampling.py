"""Stratified sampling math + corpus fingerprint for eval query generation.

Pure functions only — no LLM, no I/O. See
docs/superpowers/specs/2026-07-25-dynamic-corpus-queries-design.md.
"""

from __future__ import annotations

import hashlib
import math
import random


def compute_sample_size(
    n_docs: int, margin: float = 0.05, z: float = 1.96, p: float = 0.5
) -> int:
    """Worst-case proportion sample size with finite population correction.

    n0 = z^2 * p(1-p) / margin^2, then FPC n = n0 / (1 + (n0-1)/N),
    rounded up and clamped to [1, n_docs].
    """
    if n_docs <= 0:
        return 0
    n0 = (z * z * p * (1 - p)) / (margin * margin)
    n = n0 / (1 + (n0 - 1) / n_docs)
    return max(1, min(n_docs, math.ceil(n)))


def allocate(strata_counts: dict[str, int], n: int) -> dict[str, int]:
    """Proportional allocation of n across strata.

    Floor of 1 per non-empty stratum, largest-remainder rounding so the
    result sums to exactly n, and no stratum exceeds its own size.
    """
    strata = {k: v for k, v in strata_counts.items() if v > 0}
    if not strata:
        return {}
    total = sum(strata.values())
    n = max(1, min(n, total))

    # Start everyone at the floor of 1 (or fewer if n is tiny).
    if n <= len(strata):
        # Not enough budget for every stratum; give 1 to the n largest.
        ranked = sorted(strata, key=lambda k: strata[k], reverse=True)
        return {k: (1 if i < n else 0) for i, k in enumerate(ranked) if i < n}

    alloc = {k: 1 for k in strata}
    remaining = n - len(strata)

    # Ideal extra share (beyond the floor) per stratum.
    ideal = {k: (strata[k] / total) * n for k in strata}
    extra = {k: max(0.0, ideal[k] - 1) for k in strata}
    extra_total = sum(extra.values()) or 1.0

    quotas = {k: remaining * (extra[k] / extra_total) for k in strata}
    floors = {k: int(quotas[k]) for k in strata}
    for k, f in floors.items():
        alloc[k] += f
    assigned = sum(floors.values())

    # Largest-remainder distribution of the leftover units.
    leftover = remaining - assigned
    remainders = sorted(
        strata, key=lambda k: quotas[k] - floors[k], reverse=True
    )
    for k in remainders:
        if leftover <= 0:
            break
        alloc[k] += 1
        leftover -= 1

    # Cap at stratum size, spilling overflow to strata with headroom.
    _cap_to_size(alloc, strata)
    return alloc


def _cap_to_size(alloc: dict[str, int], strata: dict[str, int]) -> None:
    overflow = 0
    for k in alloc:
        if alloc[k] > strata[k]:
            overflow += alloc[k] - strata[k]
            alloc[k] = strata[k]
    while overflow > 0:
        # Give to any stratum with headroom, largest first.
        candidates = [k for k in alloc if alloc[k] < strata[k]]
        if not candidates:
            break
        candidates.sort(key=lambda k: strata[k] - alloc[k], reverse=True)
        alloc[candidates[0]] += 1
        overflow -= 1
