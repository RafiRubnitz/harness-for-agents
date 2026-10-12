"""The injected tool fields: ``reason`` and ``run_in_background`` (08 §3a, §6a)."""

from pydantic import JsonValue

from harness.core.consts import (
    INJECTED_COLLISION_PREFIX,
    INJECTED_REASON,
    INJECTED_RUN_IN_BACKGROUND,
)
from harness.core.types import InjectedFields
from harness.tools.consts import REASON_DESCRIPTION, RUN_IN_BACKGROUND_DESCRIPTION

_PROPERTIES = "properties"
_INJECTED = (INJECTED_REASON, INJECTED_RUN_IN_BACKGROUND)


def _schema_properties(schema: dict[str, JsonValue]) -> dict[str, JsonValue]:
    properties = schema.get(_PROPERTIES)
    return dict(properties) if isinstance(properties, dict) else {}


def injected_names(schema: dict[str, JsonValue]) -> tuple[str, str]:
    """Return the wire names of ``reason`` and ``run_in_background`` for a tool schema.

    A tool that already has a field with one of these names keeps it; the injected field
    then gets the ``_`` prefix.
    """
    own = _schema_properties(schema)
    reason, background = (
        f"{INJECTED_COLLISION_PREFIX}{name}" if name in own else name for name in _INJECTED
    )
    return reason, background


def inject_fields(schema: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Return a copy of ``schema`` with the two injected fields added (both optional)."""
    reason, background = injected_names(schema)
    properties = _schema_properties(schema)
    properties[reason] = {"type": "string", "description": REASON_DESCRIPTION}
    properties[background] = {"type": "boolean", "description": RUN_IN_BACKGROUND_DESCRIPTION}
    return {**schema, "type": schema.get("type", "object"), _PROPERTIES: properties}


def split_injected(
    raw_input: dict[str, JsonValue], schema: dict[str, JsonValue] | None = None
) -> InjectedFields:
    """Split a model-sent input into the tool's own input and the injected fields.

    Args:
        raw_input: The input exactly as the model sent it.
        schema: The tool's own schema; decides whether the injected names are prefixed.
    """
    reason_key, background_key = injected_names(schema or {})
    clean = {k: v for k, v in raw_input.items() if k not in (reason_key, background_key)}
    reason = raw_input.get(reason_key)
    return InjectedFields(
        input=clean,
        reason=reason if isinstance(reason, str) and reason.strip() else None,
        run_in_background=raw_input.get(background_key) is True,
    )
