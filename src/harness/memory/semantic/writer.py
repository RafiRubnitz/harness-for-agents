"""The row-level writes shared by the memory API and the background backfill (03 §6.5)."""

import asyncio
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum

from harness.core.consts import ID_PREFIX_MEMORY_VERSION, MemorySource
from harness.core.records import MemoryItem
from harness.memory.semantic import filters as sql
from harness.memory.semantic import schema
from harness.memory.semantic.types import SemanticDeps
from harness.memory.semantic.utils import lexical_text


def merged_tags(first: Sequence[str], second: Sequence[str]) -> list[str]:
    """Return both tag lists as one, without repeats, in first-seen order."""
    return list(dict.fromkeys([*first, *second]))


class Writer:
    """Applies changes to stored items; callers hold ``lock`` around read-then-write."""

    def __init__(self, deps: SemanticDeps) -> None:
        self._deps = deps
        self.lock = asyncio.Lock()

    async def keep_version(self, item: MemoryItem, actor: MemorySource, now: datetime) -> None:
        """Move the current value of ``item`` to its versions."""
        tables = self._deps.tables
        number = len(await tables.versions(item.id)) + 1
        version_id = self._deps.ids.new(ID_PREFIX_MEMORY_VERSION)
        await tables.add_version(schema.version_to_row(version_id, item, number, now, actor))

    async def rewrite(
        self,
        item: MemoryItem,
        content: str,
        *,
        actor: MemorySource,
        observed_at: datetime,
        vector: Sequence[float] | None,
        embed_model: str | None,
        changes: dict[str, object] | None = None,
    ) -> None:
        """Give ``item`` a new content; the old one is kept as a version.

        With an unchanged content only the times and ``changes`` are written, so the
        stored vector is left alone.
        """
        now = self._deps.clock.now()
        fields: dict[str, object] = {**(changes or {}), "observed_at": observed_at}
        fields["updated_at"] = now
        if content == item.content:
            plain = {
                name: value.value if isinstance(value, StrEnum) else value
                for name, value in fields.items()
            }
            await self._deps.tables.patch(sql.with_id(item.id), plain)
            return
        await self.keep_version(item, actor, now)
        fields.update(
            content=content,
            content_norm=lexical_text(content),
            vector=list(vector) if vector is not None else None,
            embed_model=embed_model if vector is not None else None,
        )
        await self._deps.tables.upsert(item.model_copy(update=fields))

    async def set_vector(self, item: MemoryItem, vector: Sequence[float], model: str) -> None:
        """Store the vector of an item that was written without one."""
        await self._deps.tables.upsert(
            item.model_copy(update={"vector": list(vector), "embed_model": model})
        )

    async def expire(self, item_id: str, *, superseded_by: str | None = None) -> None:
        """End an item's validity now, optionally pointing to its replacement."""
        now = self._deps.clock.now()
        fields: dict[str, object] = {"valid_until": now, "updated_at": now}
        if superseded_by is not None:
            fields["superseded_by"] = superseded_by
        await self._deps.tables.patch(sql.with_id(item_id), fields)
