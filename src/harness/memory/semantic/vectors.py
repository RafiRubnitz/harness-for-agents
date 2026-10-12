"""Embedding with a bounded wait, and the duplicate lookup built on it (03 §6.1, §6.4)."""

import asyncio
from collections.abc import Sequence

from loguru import logger

from harness.config import MemoryConfig
from harness.core.consts import ErrorCode
from harness.core.errors import EmbedderUnavailableError
from harness.core.records import MemoryItem
from harness.memory import consts
from harness.memory.semantic import filters as sql
from harness.memory.semantic.types import EmbedKind, SemanticDeps
from harness.memory.semantic.utils import similarity


class Vectors:
    """Talks to the embedder on behalf of the semantic memory."""

    def __init__(self, deps: SemanticDeps) -> None:
        self._deps = deps
        self._down = False

    @property
    def model_id(self) -> str:
        """The model new vectors come from."""
        return self._deps.embedder.model_id

    async def one(self, text: str, kind: EmbedKind) -> list[float] | None:
        """Embed one text, or return ``None`` when the server is down or too slow.

        The first failure of an outage is reported to the front ends; the next success
        ends the outage.
        """
        timeout = self._deps.config.section(MemoryConfig).embed_timeout_s
        try:
            async with asyncio.timeout(timeout):
                vectors = await self._deps.embedder.embed([text], kind)
        except (EmbedderUnavailableError, TimeoutError) as error:
            self._report_down(error)
            return None
        self._down = False
        return vectors[0]

    async def many(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of documents (background work: no short timeout).

        Raises:
            EmbedderUnavailableError: The server is down.
        """
        vectors = await self._deps.embedder.embed(texts, consts.EMBED_KIND_DOCUMENT)
        self._down = False
        return vectors

    async def duplicate(
        self, agent_id: str | None, vector: Sequence[float], *, exclude: str | None = None
    ) -> MemoryItem | None:
        """Return the same owner's valid item that says (nearly) the same thing, if any."""
        parts = [sql.owned_by(agent_id), sql.valid(), sql.embedded_with(self.model_id)]
        if exclude is not None:
            parts.append(sql.not_id(exclude))
        closest = await self._deps.tables.nearest(vector, sql.all_of(*parts), 1)
        if not closest:
            return None
        item, distance = closest[0]
        threshold = self._deps.config.section(MemoryConfig).dedup_similarity
        return item if similarity(distance) >= threshold else None

    def _report_down(self, error: Exception) -> None:
        if self._down:
            return
        self._down = True
        logger.warning("embeddings unavailable, falling back to lexical memory: {}", error)
        if self._deps.warn is not None:
            self._deps.warn(ErrorCode.MEMORY_EMBEDDER_UNAVAILABLE, consts.MSG_EMBEDDER_FALLBACK)
