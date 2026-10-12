"""FakeRecall: the semantic memory as the context manager sees it, held in lists."""

from collections.abc import Sequence
from datetime import datetime

from harness.core.consts import MemoryKind, MemoryScope, MemorySource
from harness.core.records import MemoryItem
from harness.core.types import AgentId, MemoryItemId
from tests.fakes.clock import FAKE_CLOCK_START


def memory_item(
    item_id: str,
    content: str,
    *,
    agent_id: AgentId | None = None,
    pinned: bool = False,
    at: datetime = FAKE_CLOCK_START,
    valid_until: datetime | None = None,
) -> MemoryItem:
    """A semantic memory item; without an agent it is a global one."""
    return MemoryItem(
        id=item_id,
        scope=MemoryScope.AGENT if agent_id else MemoryScope.GLOBAL,
        agent_id=agent_id,
        kind=MemoryKind.FACT,
        content=content,
        content_norm=content.lower(),
        source=MemorySource.USER,
        pinned=pinned,
        observed_at=at,
        created_at=at,
        updated_at=at,
        valid_until=valid_until,
    )


class FakeRecall:
    """Returns the items a test stored: ``hits`` for any search, pinned ones by flag."""

    def __init__(self) -> None:
        self.items: dict[str, MemoryItem] = {}
        self.hits: list[str] = []
        self.queries: list[tuple[str, int]] = []

    def add(self, item: MemoryItem, *, hit: bool = False) -> MemoryItem:
        """Store an item; with ``hit`` every search returns it."""
        self.items[item.id] = item
        if hit:
            self.hits.append(item.id)
        return item

    async def pinned(self, agent_id: AgentId) -> list[MemoryItem]:
        """The pinned items of the agent and the global ones."""
        return [
            item
            for item in self.items.values()
            if item.pinned and item.agent_id in (None, agent_id)
        ]

    async def relevant(self, agent_id: AgentId, query: str, *, k: int) -> list[MemoryItem]:
        """The first ``k`` hits, whatever the query; the query is recorded."""
        self.queries.append((query, k))
        return [self.items[item_id] for item_id in self.hits[:k]]

    async def by_ids(self, ids: Sequence[MemoryItemId]) -> list[MemoryItem]:
        """The stored items with these ids, in the given order."""
        return [self.items[item_id] for item_id in ids if item_id in self.items]
