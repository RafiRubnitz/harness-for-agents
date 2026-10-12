"""Context manager (component 7): what goes into every model request.

Other components use the ``ContextManager`` protocol.
"""

from collections.abc import Sequence
from typing import Protocol

from harness.context.manager import DefaultContextManager
from harness.context.types import ContextDeps, ContextNotifier, MemoryRecall
from harness.core.consts import CompactReason
from harness.core.records import BuiltContext, ContextReport, SummaryRecord
from harness.core.types import AgentId, Message


class ContextManager(Protocol):
    """Builds, compacts and reports an agent's context."""

    async def build(self, agent_id: AgentId, pending: Sequence[Message] = ()) -> BuiltContext:
        """Build the request for the agent's next step."""
        ...

    async def compact(
        self, agent_id: AgentId, reason: CompactReason = CompactReason.THRESHOLD
    ) -> SummaryRecord:
        """Summarise the oldest part of the history."""
        ...

    async def inspect(self, agent_id: AgentId, *, full: bool = False) -> ContextReport:
        """Return the context report (``/context``)."""
        ...


__all__ = [
    "ContextDeps",
    "ContextManager",
    "ContextNotifier",
    "DefaultContextManager",
    "MemoryRecall",
]
