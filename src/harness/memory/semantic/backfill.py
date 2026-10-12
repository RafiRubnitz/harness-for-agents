"""Completing missing vectors in the background, and ``reindex`` (03 §6.1, 13 §4.4).

The rows without a vector of the current model are the queue: no extra flag is kept.
"""

import asyncio
import contextlib

from loguru import logger

from harness.core.errors import MemoryStoreError
from harness.core.records import MemoryItem
from harness.memory import consts
from harness.memory.semantic import filters as sql
from harness.memory.semantic.types import SemanticDeps
from harness.memory.semantic.vectors import Vectors
from harness.memory.semantic.writer import Writer, merged_tags


class Backfill:
    """Embeds what was saved while the embeddings server was away."""

    def __init__(self, deps: SemanticDeps, vectors: Vectors, writer: Writer) -> None:
        self._deps = deps
        self._vectors = vectors
        self._writer = writer
        self._task: asyncio.Task[None] | None = None

    def _queue(self) -> str:
        return sql.not_embedded_with(self._vectors.model_id)

    async def pending(self) -> int:
        """Return how many items still wait for a vector."""
        return await self._deps.tables.count(self._queue())

    async def run_once(self, *, merge_duplicates: bool = True) -> int:
        """Embed one batch of waiting items; return how many were handled.

        A valid item that turns out to repeat an older one is merged into it: the older
        item takes the new content, the new one expires and points to it.

        Raises:
            EmbedderUnavailableError: The server is (still) down.
            MemoryStoreError: The table could not be written.
        """
        batch = await self._deps.tables.find(self._queue(), limit=consts.BACKFILL_BATCH)
        if not batch:
            return 0
        vectors = await self._vectors.many([item.content for item in batch])
        for item, vector in zip(batch, vectors, strict=True):
            async with self._writer.lock:
                current = await self._deps.tables.get(item.id)
                if current is None or current.content != item.content:
                    continue  # deleted or rewritten meanwhile; a rewrite queues it again
                await self._writer.set_vector(current, vector, self._vectors.model_id)
                if merge_duplicates and current.valid_until is None:
                    await self._merge(current, vector)
        return len(batch)

    async def _merge(self, item: MemoryItem, vector: list[float]) -> None:
        twin = await self._vectors.duplicate(item.agent_id, vector, exclude=item.id)
        if twin is None:
            return
        await self._writer.rewrite(
            twin,
            item.content,
            actor=item.source,
            observed_at=max(item.observed_at, twin.observed_at),
            vector=vector,
            embed_model=self._vectors.model_id,
            changes={consts.CHANGE_TAGS: merged_tags(twin.tags, item.tags)},
        )
        await self._writer.expire(item.id, superseded_by=twin.id)

    async def drain(self, *, merge_duplicates: bool = True) -> int:
        """Run batches until nothing waits; return how many items were handled."""
        total = 0
        while handled := await self.run_once(merge_duplicates=merge_duplicates):
            total += handled
        return total

    async def reindex(self, embed_model: str) -> None:
        """Re-embed every item whose vector is not from ``embed_model``.

        Raises:
            MemoryStoreError: The embeddings server runs another model.
            EmbedderUnavailableError: The server is down.
        """
        running = self._vectors.model_id
        if embed_model != running:
            raise MemoryStoreError(
                consts.MSG_REINDEX_MODEL_MISMATCH.format(wanted=embed_model, running=running)
            )
        await self.drain(merge_duplicates=False)

    def start(self) -> None:
        """Start the periodic background job."""
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Stop the background job."""
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _loop(self) -> None:
        while True:
            try:
                if await self.pending() and await self._deps.embedder.is_available():
                    handled = await self.drain()
                    logger.info("embedded {} memory items saved without a vector", handled)
            except MemoryStoreError as error:
                logger.debug("memory backfill will retry: {}", error)
            await asyncio.sleep(consts.BACKFILL_INTERVAL_S)
