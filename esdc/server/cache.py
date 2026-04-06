"""Caching utilities for performance optimization.

This module provides configurable caching strategies optimized for
ESDC's message conversion hot paths.
"""

import hashlib
import json
from typing import Any

# JSON parsing cache
# Key: (args_str_hash, args_str)
# Value: parsed dict
_json_cache: dict[tuple[str, str], dict[str, Any]] = {}

# Cache configuration
MAX_JSON_CACHE_SIZE = 256


def _hash_messages(messages: Any) -> str:
    """Generate stable hash for messages.

    Used as cache key for message conversion memoization.

    Args:
        messages: Messages to hash (list or tuple)

    Returns:
        SHA256 hash string (first 16 chars)
    """
    try:
        if hasattr(messages, "__iter__"):
            messages_list = list(messages)
        else:
            messages_list = [messages]

        json_str = json.dumps(
            messages_list,
            sort_keys=True,
            default=str,
        )

        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
    except (TypeError, ValueError):
        return str(id(messages))


def _hash_json_args(args_str: str) -> str:
    """Hash JSON argument string for caching.

    Args:
        args_str: JSON string of arguments

    Returns:
        Hash of normalized JSON (first 8 chars)
    """
    try:
        args = json.loads(args_str) if args_str else {}
        normalized = json.dumps(args, sort_keys=True)
        return hashlib.sha256(normalized.encode()).hexdigest()[:8]
    except (json.JSONDecodeError, TypeError):
        return hashlib.sha256(args_str.encode()).hexdigest()[:8]


def get_parsed_json(args_str: str) -> dict[str, Any]:
    """Get parsed JSON with caching.

    Caches parsed JSON argument strings to avoid repeated parsing.

    Args:
        args_str: JSON string of function call arguments

    Returns:
        Parsed dictionary (empty dict on parse error)
    """
    global _json_cache

    if not args_str:
        return {}

    cache_key = (_hash_json_args(args_str), args_str)

    if cache_key in _json_cache:
        return _json_cache[cache_key]

    try:
        parsed = json.loads(args_str)
    except json.JSONDecodeError:
        parsed = {}

    if len(_json_cache) >= MAX_JSON_CACHE_SIZE:
        _json_cache.pop(next(iter(_json_cache)))

    _json_cache[cache_key] = parsed
    return parsed


def clear_all_caches():
    """Clear all caches.

    Useful for testing and memory management.
    """
    global _json_cache
    _json_cache.clear()


def get_cache_stats() -> dict[str, Any]:
    """Get cache statistics for monitoring.

    Returns:
        Dict with cache sizes and hit rates
    """
    return {
        "json_cache_size": len(_json_cache),
        "json_cache_max": MAX_JSON_CACHE_SIZE,
    }
