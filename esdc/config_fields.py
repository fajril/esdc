"""Shared display formatting and input validation for config fields.

One formatter and one parser are used by both the interactive wizard and
`configs --show`, so a secret is masked in the same way everywhere and a value
is validated before anything is written.

Nothing in this module saves config. Validation is a pure function of the key,
the input, and the flat merged config (the draft) used for fields that depend
on each other. A message never contains a secret value.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from esdc.configs import ENUM_CHOICES, Config

# Key suffixes whose value is never shown in full. Matched at the end of the
# key, so `providers.<name>.api_key` and `providers.<name>.oauth.access_token`
# are both covered.
SECRET_SUFFIXES = ("api_key", "access_token", "refresh_token")

def is_secret_key(key: str) -> bool:
    """True when a key holds a secret that must not be displayed in full."""
    return any(key.endswith(suffix) for suffix in SECRET_SUFFIXES)


_URL_SECRET_QUERY_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "token",
        "password",
        "passwd",
        "secret",
        "sig",
        "signature",
        "key",
    }
)


def mask_url_credentials(value: Any) -> str:
    """Mask userinfo and secret query values in a credential-bearing URL.

    Only URL-shaped values are changed; every other value is returned as
    ``str(value)``. The scheme, host, port, path and non-secret query names stay
    readable so a user can identify the endpoint without seeing a credential.
    """
    text = str(value)
    if not text.startswith(("http://", "https://")):
        return text
    try:
        parts = urlsplit(text)
    except ValueError:
        return "****"

    netloc = parts.netloc
    if "@" in netloc:
        host = netloc.rsplit("@", 1)[1]
        netloc = f"****@{host}"

    query_pairs = parse_qsl(parts.query, keep_blank_values=True)
    if query_pairs:
        masked_pairs = [
            (name, "****" if name.casefold() in _URL_SECRET_QUERY_NAMES else item)
            for name, item in query_pairs
        ]
        query = urlencode(masked_pairs, safe="*")
    else:
        query = parts.query

    return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))




def format_config_value(key: str, value: Any) -> str:
    """Render one value for display, masking secrets and URL credentials.

    Mask form for a secret: ``****`` for a value of four characters or fewer,
    otherwise the first two characters, ``****``, and the last two. A URL-shaped
    value also masks userinfo and secret query values while keeping the endpoint
    readable.
    """
    text = str(value)
    if is_secret_key(key):
        if len(text) <= 4:
            return "****"
        return text[:2] + "****" + text[-2:]
    return mask_url_credentials(value)


def parse_config_input(key: str, raw: Any, values: Mapping[str, Any]) -> Any:
    """Turn one field input into the value the config would store.

    `values` is the flat merged config (or draft) already held for the other
    keys, used only to check fields that depend on each other.

    Raises:
        ValueError: the input cannot be stored for this key, with a short
            message naming the field. Nothing is written.
    """
    if key in ENUM_CHOICES:
        return _match_enum(key, raw)

    value = Config._coerce_value(key, raw)

    if key in Config.BOOLEAN_KEYS and not isinstance(value, bool):
        raise ValueError(f"{key} must be true or false")
    if key in Config.INT_KEYS and not _is_int(value):
        raise ValueError(f"{key} must be a whole number")
    if key in Config.FLOAT_KEYS:
        if not _is_number(value):
            raise ValueError(f"{key} must be a decimal number")
        value = float(value)

    _check_range(key, value, values)
    return value


def _match_enum(key: str, raw: Any) -> str:
    """Return the canonical choice for an enum key, matched case-insensitively."""
    choices = ENUM_CHOICES[key]
    text = str(raw)
    for choice in choices:
        if text.casefold() == choice.casefold():
            return choice
    raise ValueError(f"{key} must be one of: {', '.join(choices)}")


def _check_range(key: str, value: Any, values: Mapping[str, Any]) -> None:
    """Check the ranges that depend on this key alone or on its related field."""
    if key == "corpus.min_image_area":
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} must be between 0 and 1")
        return

    if key == "corpus.chunk_size":
        if value <= 0:
            raise ValueError(f"{key} must be greater than 0")
        overlap = values.get("corpus.chunk_overlap")
        if _is_int(overlap) and value <= overlap:
            raise ValueError(f"{key} must be greater than corpus.chunk_overlap")
        return

    if key == "corpus.chunk_overlap":
        # Zero is the valid "no overlap" sentinel here.
        if value < 0:
            raise ValueError(f"{key} must be 0 or greater")
        size = values.get("corpus.chunk_size")
        if _is_int(size) and value >= size:
            raise ValueError(f"{key} must be smaller than corpus.chunk_size")


def _is_int(value: Any) -> bool:
    """A stored integer value, excluding bool which Python counts as int."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    """A numeric value, excluding bool which Python counts as int."""
    return _is_int(value) or isinstance(value, float)
