"""What the context manager is built from, and the edges it talks through."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from harness.config import ConfigManager, ContextConfig
from harness.core.consts import ErrorCode
from harness.core.records import MemoryItem, SummaryRecord
from harness.core.types import AgentId, Clock, IdGenerator, MemoryItemId, ModelProfile
from harness.hooks import HooksManager
from harness.memory import MemoryStore
from harness.models import ModelGateway
from harness.tools import ToolsManager


class MemoryRecall(Protocol):
    """The part of the semantic memory the context manager reads (07 §3, §6)."""

    async def pinned(self, agent_id: AgentId) -> list[MemoryItem]:
        """Return the pinned items the agent sees (its own and the global ones)."""
        ...

    async def relevant(self, agent_id: AgentId, query: str, *, k: int) -> list[MemoryItem]:
        """Return up to ``k`` valid items matching ``query`` (hybrid search)."""
        ...

    async def by_ids(self, ids: Sequence[MemoryItemId]) -> list[MemoryItem]:
        """Return the items with these ids, in the given order; missing ids are skipped."""
        ...


class ContextNotifier(Protocol):
    """How the context manager tells the front ends what it did."""

    def compacted(self, agent_id: AgentId, summary: SummaryRecord) -> None:
        """A range of messages was replaced by ``summary``."""
        ...

    def warning(self, agent_id: AgentId, code: ErrorCode, message: str) -> None:
        """Something was left out of the context."""
        ...


@dataclass(frozen=True, slots=True)
class ContextDeps:
    """The collaborators of the context manager."""

    memory: MemoryStore
    config: ConfigManager
    gateway: ModelGateway
    tools: ToolsManager
    hooks: HooksManager
    clock: Clock
    ids: IdGenerator
    recall: MemoryRecall | None = None
    notify: ContextNotifier | None = None


@dataclass(frozen=True, slots=True)
class Budgets:
    """Token budgets of one request, derived from the model's window (07 §4)."""

    window: int
    reserve: int
    system: int
    summaries: int
    memories: int
    tail_keep: int
    limit: int


def budgets_for(profile: ModelProfile, settings: ContextConfig) -> Budgets:
    """Compute the budgets of a model from ``context.*``.

    ``limit`` is where compaction starts: the configured share of the window, and never
    more than what is left once the answer's room is set aside.
    """
    window, shares = profile.context_window, settings.budget
    reserve = max(shares.reserve_min_tokens, int(window * shares.reserve_pct))
    reserve = min(reserve, profile.max_output_tokens)
    return Budgets(
        window=window,
        reserve=reserve,
        system=int(window * shares.system_pct),
        summaries=int(window * shares.summaries_pct),
        memories=int(window * shares.memories_pct),
        tail_keep=int(window * shares.tail_keep_pct),
        limit=min(int(window * settings.compaction_threshold), window - reserve),
    )
