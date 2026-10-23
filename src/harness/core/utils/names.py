"""Agent names: unique, lowercase, addressable (phase 2, 03 §3)."""

import re
from collections.abc import Collection

from harness.core.consts import AGENT_NAME_PATTERN

_VALID = re.compile(AGENT_NAME_PATTERN)
_NOT_ALLOWED = re.compile(r"[^a-z0-9]+")
_SEPARATOR = "-"
_FALLBACK = "agent"
_NUMBERED = "{base}-{number}"


def valid_name(name: str) -> bool:
    """Return whether ``name`` may be used as an address (``agent:<name>``)."""
    return _VALID.fullmatch(name) is not None


def slug(text: str) -> str:
    """Turn any text into a valid agent name (``Code Reviewer`` -> ``code-reviewer``)."""
    return _NOT_ALLOWED.sub(_SEPARATOR, text.lower()).strip(_SEPARATOR) or _FALLBACK


def free_name(base: str, taken: Collection[str]) -> str:
    """Return ``base``, or ``base-2``, ``base-3``... - the first one nobody has."""
    if base not in taken:
        return base
    number = 2
    while _NUMBERED.format(base=base, number=number) in taken:
        number += 1
    return _NUMBERED.format(base=base, number=number)
