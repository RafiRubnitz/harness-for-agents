# pyright: basic
# LanceDB and pyarrow ship without usable type information.
"""Arrow schemas of the two LanceDB tables and the row conversions (13 §4.1, §4.2).

The schemas are written by hand: every time column must be ``timestamp[us, UTC]`` and
the vector a nullable fixed-size list, which the Pydantic helper of LanceDB does not
guarantee.
"""

from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum

import pyarrow as pa

from harness.core.consts import MemorySource
from harness.core.records import MemoryItem, MemoryVersion
from harness.memory import consts

_TIME = pa.timestamp("us", tz="UTC")
_META = {consts.SCHEMA_VERSION_KEY: str(consts.SEMANTIC_SCHEMA_VERSION).encode()}
_ITEM_FIELDS = frozenset(MemoryItem.model_fields) - {"versions"}


def memories_schema() -> pa.Schema:
    """Schema of ``memories``."""
    return pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("scope", pa.string(), nullable=False),
            pa.field("agent_id", pa.string()),
            pa.field("kind", pa.string(), nullable=False),
            pa.field("content", pa.string(), nullable=False),
            pa.field("content_norm", pa.string(), nullable=False),
            pa.field("vector", pa.list_(pa.float32(), consts.EMBED_DIM)),
            pa.field("embed_model", pa.string()),
            pa.field("tags", pa.list_(pa.string()), nullable=False),
            pa.field("source", pa.string(), nullable=False),
            pa.field("source_msg_seq", pa.int64()),
            pa.field("pinned", pa.bool_(), nullable=False),
            pa.field("observed_at", _TIME, nullable=False),
            pa.field("created_at", _TIME, nullable=False),
            pa.field("updated_at", _TIME, nullable=False),
            pa.field("valid_until", _TIME),
            pa.field("superseded_by", pa.string()),
            pa.field("last_accessed_at", _TIME),
            pa.field("access_count", pa.int64(), nullable=False),
            pa.field("schema_version", pa.int32(), nullable=False),
        ],
        metadata=_META,
    )


def versions_schema() -> pa.Schema:
    """Schema of ``memory_versions``."""
    return pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("item_id", pa.string(), nullable=False),
            pa.field("scope", pa.string(), nullable=False),
            pa.field("agent_id", pa.string()),
            pa.field("version_no", pa.int32(), nullable=False),
            pa.field("content", pa.string(), nullable=False),
            pa.field("observed_at", _TIME, nullable=False),
            pa.field("replaced_at", _TIME, nullable=False),
            pa.field("replaced_by", pa.string(), nullable=False),
            pa.field("schema_version", pa.int32(), nullable=False),
        ],
        metadata=_META,
    )


ITEM_COLUMNS: tuple[str, ...] = tuple(
    name for name in memories_schema().names if name != consts.COL_VECTOR
)
"""Columns read for an item: everything but the vector, which no caller needs back."""


def item_to_row(item: MemoryItem) -> dict[str, object]:
    """Convert an item to a ``memories`` row."""
    row: dict[str, object] = {
        name: value.value if isinstance(value, StrEnum) else value
        for name, value in item.model_dump(exclude={"versions"}).items()
    }
    row[consts.COL_SCHEMA_VERSION] = consts.SEMANTIC_SCHEMA_VERSION
    return row


def row_to_item(row: Mapping[str, object]) -> MemoryItem:
    """Convert a ``memories`` row (with or without its vector) to an item."""
    return MemoryItem.model_validate(
        {name: value for name, value in row.items() if name in _ITEM_FIELDS}
    )


def version_to_row(
    version_id: str, item: MemoryItem, version_no: int, replaced_at: datetime, actor: MemorySource
) -> dict[str, object]:
    """Build the ``memory_versions`` row that keeps the current value of ``item``."""
    return {
        "id": version_id,
        "item_id": item.id,
        "scope": item.scope.value,
        "agent_id": item.agent_id,
        "version_no": version_no,
        "content": item.content,
        "observed_at": item.observed_at,
        "replaced_at": replaced_at,
        "replaced_by": actor.value,
        "schema_version": consts.SEMANTIC_SCHEMA_VERSION,
    }


def row_to_version(row: Mapping[str, object]) -> MemoryVersion:
    """Convert a ``memory_versions`` row to a version."""
    return MemoryVersion.model_validate({name: row[name] for name in MemoryVersion.model_fields})
