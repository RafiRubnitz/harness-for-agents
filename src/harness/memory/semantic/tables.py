# pyright: basic
# LanceDB and pyarrow ship without usable type information.
"""The two LanceDB tables: opening, indexes, and every read and write (13 §4).

Nothing above this module knows LanceDB. Filters arrive as strings built by
``filters``; rows leave as ``MemoryItem`` / ``MemoryVersion``.
"""

from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import lancedb
import pyarrow as pa
from lancedb.index import FTS, Bitmap, BTree
from loguru import logger

from harness.core.errors import MemoryStoreError
from harness.core.records import MemoryItem, MemoryVersion
from harness.memory import consts
from harness.memory.semantic import filters, schema

Row = Mapping[str, object]


@asynccontextmanager
async def _guard() -> AsyncGenerator[None]:
    """Turn any LanceDB failure into a ``MemoryStoreError``."""
    try:
        yield
    except MemoryStoreError:
        raise
    except Exception as error:
        raise MemoryStoreError(consts.MSG_SEMANTIC_FAILED.format(error=error)) from error


def _check_version(table_schema: pa.Schema) -> None:
    """Refuse data written by a newer app; older data would need a migration."""
    raw = (table_schema.metadata or {}).get(consts.SCHEMA_VERSION_KEY)
    found = int(raw) if raw else consts.SEMANTIC_SCHEMA_VERSION
    known = consts.SEMANTIC_SCHEMA_VERSION
    if found > known:
        raise MemoryStoreError(consts.MSG_SEMANTIC_NEWER.format(found=found, known=known))
    if found < known:
        raise MemoryStoreError(consts.MSG_SEMANTIC_NO_MIGRATION.format(found=found, known=known))


class SemanticTables:
    """``memories`` and ``memory_versions``."""

    def __init__(
        self,
        db: lancedb.AsyncConnection,
        memories: lancedb.AsyncTable,
        versions: lancedb.AsyncTable,
    ) -> None:
        self._db = db
        self._memories = memories
        self._versions = versions
        self._indexed = False
        self._writes = 0

    @classmethod
    async def open(cls, path: Path) -> "SemanticTables":
        """Open (creating when missing) the tables under ``path``.

        Raises:
            MemoryStoreError: The data was written by a newer version of the app.
        """
        async with _guard():
            db = await lancedb.connect_async(path)
            memories = await db.create_table(
                consts.TABLE_MEMORIES, schema=schema.memories_schema(), exist_ok=True
            )
            versions = await db.create_table(
                consts.TABLE_VERSIONS, schema=schema.versions_schema(), exist_ok=True
            )
            _check_version(await memories.schema())
            _check_version(await versions.schema())
            tables = cls(db, memories, versions)
            await tables.ensure_indices()
            await tables.optimize()
        return tables

    def close(self) -> None:
        """Release the tables."""
        self._memories.close()
        self._versions.close()

    async def ensure_indices(self) -> None:
        """Create the FTS and scalar indexes once the tables hold rows."""
        if self._indexed or await self._memories.count_rows() == 0:
            return
        existing = {
            column for index in await self._memories.list_indices() for column in index.columns
        }
        if consts.COL_CONTENT_NORM not in existing:
            config = FTS(
                base_tokenizer=consts.FTS_BASE_TOKENIZER,
                lower_case=True,
                stem=False,
                remove_stop_words=False,
                ascii_folding=False,
            )
            await self._memories.create_index(consts.COL_CONTENT_NORM, config=config)
        for column in consts.BTREE_COLUMNS:
            if column not in existing:
                await self._memories.create_index(column, config=BTree())
        for column in consts.BITMAP_COLUMNS:
            if column not in existing:
                await self._memories.create_index(column, config=Bitmap())
        self._indexed = True

    async def optimize(self) -> None:
        """Compact files, drop old dataset versions and bring the indexes up to date."""
        async with _guard():
            await self._memories.optimize()
            await self._versions.optimize()
        self._writes = 0

    async def _wrote(self) -> None:
        self._writes += 1
        if not self._indexed:
            await self.ensure_indices()
        if self._writes >= consts.OPTIMIZE_EVERY_WRITES:
            try:
                await self.optimize()
            except MemoryStoreError as error:
                logger.warning("semantic memory upkeep failed: {}", error)

    # --- memories: writes ---

    async def upsert(self, item: MemoryItem) -> None:
        """Insert the item, or replace the row with the same id (vector included)."""
        async with _guard():
            row = pa.Table.from_pylist([schema.item_to_row(item)], schema.memories_schema())
            await (
                self._memories.merge_insert(consts.COL_ID)
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(row, on_bad_vectors=consts.BAD_VECTOR_AS_NULL)
            )
            await self._wrote()

    async def patch(self, where: str, updates: Mapping[str, object]) -> None:
        """Set scalar columns on the rows matching ``where``."""
        async with _guard():
            await self._memories.update(updates=dict(updates), where=where)
            await self._wrote()

    async def touch(self, item_ids: Sequence[str], now: datetime) -> None:
        """Record that the items were just retrieved."""
        if not item_ids:
            return
        where = filters.with_ids(item_ids)
        async with _guard():
            await self._memories.update(updates={consts.COL_LAST_ACCESSED_AT: now}, where=where)
            await self._memories.update(
                updates_sql={consts.COL_ACCESS_COUNT: consts.SQL_ACCESS_INCREMENT}, where=where
            )
            await self._wrote()

    async def remove(self, where: str) -> None:
        """Physically delete the matching items."""
        async with _guard():
            await self._memories.delete(where)
            await self._wrote()

    # --- memories: reads ---

    async def get(self, item_id: str) -> MemoryItem | None:
        """Return one item, or ``None``."""
        found = await self.find(filters.with_id(item_id))
        return found[0] if found else None

    async def find(self, where: str, *, limit: int | None = None) -> list[MemoryItem]:
        """Return the items matching ``where`` (unordered)."""
        async with _guard():
            query = self._memories.query().where(where).select(list(schema.ITEM_COLUMNS))
            rows = await (query.limit(limit) if limit is not None else query).to_list()
        return [schema.row_to_item(row) for row in rows]

    async def count(self, where: str) -> int:
        """Return how many items match ``where``."""
        async with _guard():
            return await self._memories.count_rows(where)

    async def nearest(
        self, vector: Sequence[float], where: str, limit: int
    ) -> list[tuple[MemoryItem, float]]:
        """Return the closest items by cosine distance, nearest first, with the distance."""
        async with _guard():
            rows = await (
                self._memories.query()
                .nearest_to(list(vector))
                .distance_type(consts.DISTANCE_COSINE)
                .where(where)
                .select([*schema.ITEM_COLUMNS, consts.COL_DISTANCE])
                .limit(limit)
                .to_list()
            )
        return [(schema.row_to_item(row), float(row[consts.COL_DISTANCE])) for row in rows]

    async def matching(self, terms: Sequence[str], where: str, limit: int) -> list[MemoryItem]:
        """Return the best BM25 matches for any of ``terms``, best first."""
        if not terms:
            return []
        async with _guard():
            await self.ensure_indices()
            if not self._indexed:
                return []
            rows = await (
                self._memories.query()
                .nearest_to_text(" ".join(terms), columns=[consts.COL_CONTENT_NORM])
                .where(where)
                .select([*schema.ITEM_COLUMNS, consts.COL_SCORE])
                .limit(limit)
                .to_list()
            )
        return [schema.row_to_item(row) for row in rows]

    # --- memory_versions ---

    async def add_version(self, row: Row) -> None:
        """Store one previous value."""
        async with _guard():
            await self._versions.add(pa.Table.from_pylist([dict(row)], schema.versions_schema()))

    async def versions(self, item_id: str) -> list[MemoryVersion]:
        """Return the previous values of an item, oldest first."""
        async with _guard():
            rows = await self._versions.query().where(filters.versions_of(item_id)).to_list()
        rows.sort(key=lambda row: row[consts.COL_VERSION_NO])
        return [schema.row_to_version(row) for row in rows]

    async def remove_versions(self, where: str) -> None:
        """Physically delete the matching versions."""
        async with _guard():
            await self._versions.delete(where)
