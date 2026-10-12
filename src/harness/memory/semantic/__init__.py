"""Semantic memory (layers 4 and 5b): LanceDB tables, embeddings client, hybrid search."""

from dataclasses import dataclass

from harness.config import ConfigManager, EmbedderConfig
from harness.core.embedding import LOOPBACK_HOST
from harness.core.types import Clock, IdGenerator
from harness.core.utils.paths import HarnessHome
from harness.hooks import HooksManager
from harness.memory import consts
from harness.memory.semantic.embedder import HttpEmbedder
from harness.memory.semantic.memory import DefaultSemanticMemory
from harness.memory.semantic.server import EmbedderServer
from harness.memory.semantic.tables import SemanticTables
from harness.memory.semantic.types import (
    Embedder,
    HistoryLookup,
    SemanticDeps,
    SemanticMemory,
    Warn,
)


@dataclass(slots=True)
class Semantic:
    """The opened semantic layer and the background pieces that keep it complete."""

    memory: DefaultSemanticMemory
    tables: SemanticTables
    client: HttpEmbedder | None
    server: EmbedderServer | None

    def start(self) -> None:
        """Start the managed embeddings server (if any) and the vector backfill."""
        if self.server is not None:
            self.server.start()
        self.memory.backfill.start()

    async def close(self) -> None:
        """Stop the background work, the server and the tables."""
        await self.memory.backfill.stop()
        if self.server is not None:
            await self.server.stop()
        if self.client is not None:
            await self.client.close()
        self.tables.close()


@dataclass(frozen=True, slots=True)
class SemanticEdges:
    """What ``open_semantic`` needs from the rest of the app."""

    config: ConfigManager
    hooks: HooksManager
    history: HistoryLookup
    clock: Clock
    ids: IdGenerator
    warn: Warn | None = None


async def open_semantic(
    home: HarnessHome, edges: SemanticEdges, *, embedder: Embedder | None = None
) -> Semantic:
    """Open the tables under ``~/.harness/semantic/`` and build the memory over them.

    With ``embedder.url`` set the harness uses that server; otherwise it supervises its
    own on ``embedder.port``. A given ``embedder`` (tests) replaces both.

    Raises:
        MemoryStoreError: The tables were written by a newer version of the app.
    """
    settings = edges.config.section(EmbedderConfig)
    client: HttpEmbedder | None = None
    server: EmbedderServer | None = None
    active = embedder
    if active is None:
        url = settings.url or consts.EMBEDDER_URL.format(host=LOOPBACK_HOST, port=settings.port)
        active = client = HttpEmbedder(url, model_id=settings.model)
        if settings.url is None:
            server = EmbedderServer(
                port=settings.port,
                model=settings.model,
                log_file=home.logs_dir / consts.EMBEDDER_LOG_NAME,
                warn=edges.warn,
            )
    tables = await SemanticTables.open(home.semantic_dir)
    deps = SemanticDeps(
        tables=tables,
        embedder=active,
        history=edges.history,
        config=edges.config,
        hooks=edges.hooks,
        clock=edges.clock,
        ids=edges.ids,
        warn=edges.warn,
    )
    return Semantic(DefaultSemanticMemory(deps), tables, client, server)


__all__ = [
    "DefaultSemanticMemory",
    "Embedder",
    "EmbedderServer",
    "HistoryLookup",
    "HttpEmbedder",
    "Semantic",
    "SemanticDeps",
    "SemanticEdges",
    "SemanticMemory",
    "SemanticTables",
    "Warn",
    "open_semantic",
]
