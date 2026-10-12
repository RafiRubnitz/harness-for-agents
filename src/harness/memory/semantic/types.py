"""Protocols of the semantic layer (03 §5) and what the implementation is built from."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from pydantic import JsonValue

from harness.config import ConfigManager
from harness.core.consts import ErrorCode, MemoryKind, MemoryScope, MemorySource, SearchMode
from harness.core.embedding import EmbedKind
from harness.core.records import AgentRecord, MemoryItem, MemoryVersion, MessageRecord, ScoredItem
from harness.core.types import AgentId, Clock, IdGenerator, MemoryItemId
from harness.hooks import HooksManager
from harness.memory.semantic.tables import SemanticTables

Warn = Callable[[ErrorCode, str], None]
"""Tells the front ends that the semantic layer is degraded."""

ALL_SCOPES: tuple[MemoryScope, ...] = (MemoryScope.AGENT, MemoryScope.GLOBAL)


class Embedder(Protocol):
    """Turns text into vectors; v1 is an HTTP client of the local server (03 §6.4)."""

    @property
    def model_id(self) -> str:
        """The model whose vectors this embedder returns."""
        ...

    @property
    def dim(self) -> int:
        """Length of every vector."""
        ...

    async def embed(self, texts: Sequence[str], kind: EmbedKind) -> list[list[float]]:
        """Return one vector per text.

        Raises:
            EmbedderUnavailableError: The server is down, loading or too slow.
        """
        ...

    async def is_available(self) -> bool:
        """Return whether the server answers and its model is loaded."""
        ...


class HistoryLookup(Protocol):
    """The two things the semantic layer reads from the episodic store."""

    async def get_agent(self, agent_id: AgentId) -> AgentRecord: ...
    async def get_messages(
        self, agent_id: AgentId, *, from_seq: int = 0, limit: int | None = None
    ) -> list[MessageRecord]: ...


class SemanticMemory(Protocol):
    """Layers 4 and 5b: facts an agent or the user saved, searchable by meaning and words."""

    async def remember(
        self,
        agent_id: AgentId | None,
        content: str,
        kind: MemoryKind,
        *,
        tags: Sequence[str] = (),
        source: MemorySource = MemorySource.AGENT,
        source_msg_seq: int | None = None,
        observed_at: datetime | None = None,
    ) -> MemoryItemId:
        """Save a fact (``agent_id=None``: a global one); a near-duplicate is updated instead."""
        ...

    async def update(
        self,
        item_id: MemoryItemId,
        *,
        actor: MemorySource,
        content: str | None = None,
        **changes: JsonValue,
    ) -> None:
        """Correct an item; a replaced content moves to its versions."""
        ...

    async def supersede(
        self,
        old_id: MemoryItemId,
        new_content: str,
        *,
        actor: MemorySource,
        source_msg_seq: int | None = None,
        observed_at: datetime | None = None,
    ) -> MemoryItemId:
        """Record that a fact changed: the old item expires and points to the new one."""
        ...

    async def forget(self, item_id: MemoryItemId) -> None:
        """Soft delete: the item expires now."""
        ...

    async def delete(self, item_id: MemoryItemId) -> None:
        """Physically delete an item and its versions."""
        ...

    async def delete_agent(self, agent_id: AgentId) -> None:
        """Physically delete everything an agent owns; global items stay."""
        ...

    async def get(self, item_id: MemoryItemId) -> MemoryItem:
        """Return one item."""
        ...

    async def by_ids(self, item_ids: Sequence[MemoryItemId]) -> list[MemoryItem]:
        """Return the items with these ids, in the given order; missing ids are skipped."""
        ...

    async def search(
        self,
        agent_id: AgentId,
        query: str,
        *,
        k: int | None = None,
        mode: SearchMode = SearchMode.HYBRID,
        scopes: Sequence[MemoryScope] = ALL_SCOPES,
        filters: dict[str, JsonValue] | None = None,
        include_expired: bool = False,
        as_of: datetime | None = None,
    ) -> list[ScoredItem]:
        """Hybrid search over what the agent may see."""
        ...

    async def history(self, item_id: MemoryItemId) -> list[MemoryVersion]:
        """Return the previous values of an item, oldest first."""
        ...

    async def pinned(self, agent_id: AgentId) -> list[MemoryItem]:
        """Return the valid pinned items the agent sees, oldest first."""
        ...

    async def reindex(self, embed_model: str) -> None:
        """Re-embed every item with a new embeddings model."""
        ...

    async def list(
        self, agent_id: AgentId | None, *, include_expired: bool = False
    ) -> list[MemoryItem]:
        """Return one owner's items (``None``: the global ones), newest first."""
        ...


@dataclass(frozen=True, slots=True)
class SemanticDeps:
    """The collaborators of the semantic memory."""

    tables: SemanticTables
    embedder: Embedder
    history: HistoryLookup
    config: ConfigManager
    hooks: HooksManager
    clock: Clock
    ids: IdGenerator
    warn: Warn | None = None
