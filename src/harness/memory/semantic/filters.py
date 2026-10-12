"""The only place that builds LanceDB filter strings (13 §4.4).

LanceDB takes filters as SQL text and has no bound parameters, so every value that
reaches a filter is validated here first: ids against a strict pattern, scopes through
their enum, times rendered from timezone-aware ``datetime`` objects.
"""

import re
from collections.abc import Sequence
from datetime import UTC, datetime

from harness.core.consts import MemoryScope
from harness.core.errors import MemoryStoreError
from harness.memory import consts

_ID_RE = re.compile(consts.SAFE_ID_PATTERN)
_MODEL_RE = re.compile(consts.SAFE_MODEL_PATTERN)


def _literal(value: str, pattern: re.Pattern[str]) -> str:
    if not pattern.fullmatch(value):
        raise MemoryStoreError(consts.MSG_UNSAFE_FILTER_VALUE.format(value=value))
    return f"'{value}'"


def _id(value: str) -> str:
    return _literal(value, _ID_RE)


def _time(value: datetime) -> str:
    if value.tzinfo is None:
        raise MemoryStoreError(consts.MSG_NAIVE_DATETIME)
    return f"timestamp '{value.astimezone(UTC).isoformat()}'"


def all_of(*parts: str) -> str:
    """Join filters with ``AND``."""
    return " AND ".join(f"({part})" for part in parts)


def valid() -> str:
    """Items that are still true."""
    return "valid_until IS NULL"


def valid_at(as_of: datetime) -> str:
    """Items that were known and true at ``as_of``."""
    moment = _time(as_of)
    return f"observed_at <= {moment} AND (valid_until IS NULL OR valid_until > {moment})"


def visible(agent_id: str, scopes: Sequence[MemoryScope]) -> str:
    """The mandatory prefilter: the agent's own items and / or the global ones."""
    parts: list[str] = []
    if MemoryScope.AGENT in scopes:
        parts.append(f"(agent_id = {_id(agent_id)} AND scope = '{MemoryScope.AGENT.value}')")
    if MemoryScope.GLOBAL in scopes:
        parts.append(f"scope = '{MemoryScope.GLOBAL.value}'")
    return " OR ".join(parts) if parts else "false"


def searchable(
    agent_id: str,
    scopes: Sequence[MemoryScope],
    *,
    include_expired: bool,
    as_of: datetime | None,
) -> str:
    """The prefilter of both searches: who may see the item, and whether it is in force."""
    who = visible(agent_id, scopes)
    if as_of is not None:
        return all_of(who, valid_at(as_of))
    return who if include_expired else all_of(who, valid())


def owned_by(agent_id: str | None) -> str:
    """Exactly one owner's items: an agent's own, or the global ones."""
    if agent_id is None:
        return f"scope = '{MemoryScope.GLOBAL.value}'"
    return f"agent_id = {_id(agent_id)} AND scope = '{MemoryScope.AGENT.value}'"


def of_agent(agent_id: str) -> str:
    """Every row that belongs to an agent (for deleting the agent)."""
    return f"agent_id = {_id(agent_id)}"


def with_id(item_id: str) -> str:
    """One item."""
    return f"id = {_id(item_id)}"


def with_ids(item_ids: Sequence[str]) -> str:
    """Several items."""
    if not item_ids:
        return "false"
    return f"id IN ({', '.join(_id(item_id) for item_id in item_ids)})"


def not_id(item_id: str) -> str:
    """Every item but one."""
    return f"id != {_id(item_id)}"


def versions_of(item_id: str) -> str:
    """The stored versions of one item."""
    return f"item_id = {_id(item_id)}"


def embedded_with(model: str) -> str:
    """Items whose vector came from ``model``."""
    return f"embed_model = {_literal(model, _MODEL_RE)}"


def not_embedded_with(model: str) -> str:
    """Items with no vector, or with a vector of another model."""
    return f"vector IS NULL OR embed_model IS NULL OR embed_model != {_literal(model, _MODEL_RE)}"


def pinned() -> str:
    """Pinned items."""
    return "pinned = true"
