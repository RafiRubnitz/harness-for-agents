"""Convert internal tool names to provider wire names and back."""

import hashlib
import re

from harness.core.consts import (
    TOOL_NAME_SEPARATOR,
    TOOL_WIRE_HASH_JOINER,
    TOOL_WIRE_HASH_LENGTH,
    TOOL_WIRE_MAX_LENGTH,
    TOOL_WIRE_SEPARATOR,
    TOOL_WIRE_TRUNCATED_LENGTH,
)

_WIRE_INVALID_RE = re.compile(r"[^A-Za-z0-9_-]")


def to_wire_name(name: str) -> str:
    """Return the provider-safe name for an internal tool name.

    Dots become ``__``; other characters providers reject become ``_``. A result longer
    than the provider limit is cut and suffixed with a short hash of the internal name,
    so distinct long names stay distinct.
    """
    wire = _WIRE_INVALID_RE.sub("_", name.replace(TOOL_NAME_SEPARATOR, TOOL_WIRE_SEPARATOR))
    if len(wire) <= TOOL_WIRE_MAX_LENGTH:
        return wire
    digest = hashlib.sha1(name.encode("utf-8"), usedforsecurity=False).hexdigest()
    return (
        wire[:TOOL_WIRE_TRUNCATED_LENGTH] + TOOL_WIRE_HASH_JOINER + digest[:TOOL_WIRE_HASH_LENGTH]
    )


def build_wire_map(names: list[str]) -> dict[str, str]:
    """Return ``{wire_name: internal_name}`` for the tools of one request."""
    return {to_wire_name(name): name for name in names}


def from_wire_name(wire: str, wire_map: dict[str, str]) -> str:
    """Return the internal name for a wire name.

    Falls back to reversing the separator when the name is not in the map (a model
    calling a tool that was not offered); the tools manager then reports it as unavailable.
    """
    known = wire_map.get(wire)
    if known is not None:
        return known
    return wire.replace(TOOL_WIRE_SEPARATOR, TOOL_NAME_SEPARATOR)
