"""In-memory state of one agent, the shared dependencies, and status changes."""

import asyncio
from dataclasses import dataclass, field

from harness.agents.consts import EVENT_PREVIOUS_KEY, EVENT_STATUS_KEY
from harness.agents.events import EventBus
from harness.agents.state import check_transition
from harness.config import ConfigManager
from harness.context import ContextManager
from harness.core.consts import AgentEventType, AgentStatus
from harness.core.events import AgentStatusChanged
from harness.core.records import AgentEventRecord, AgentRecord, RunRecord, ToolOutput
from harness.core.types import (
    AgentId,
    Clock,
    ContentBlock,
    IdGenerator,
    MessageId,
    MessageOrigin,
    ToolCallBlock,
    Usage,
)
from harness.core.utils.paths import HarnessHome
from harness.hooks import HooksManager
from harness.memory import MemoryStore, SemanticMemory
from harness.models import ModelGateway
from harness.tools import ToolsManager


@dataclass(frozen=True)
class AgentDeps:
    """Everything the agent manager talks to."""

    memory: MemoryStore
    config: ConfigManager
    gateway: ModelGateway
    hooks: HooksManager
    context: ContextManager
    tools: ToolsManager
    clock: Clock
    ids: IdGenerator
    home: HarnessHome
    bus: EventBus
    semantic: SemanticMemory | None = None


@dataclass(frozen=True)
class RunStart:
    """The message that opened a run (payload of ``run_start``)."""

    content: list[ContentBlock]
    origin: MessageOrigin


@dataclass(frozen=True)
class OpenCalls:
    """Tool calls of a saved assistant message that still have no tool message."""

    message_id: MessageId
    calls: list[ToolCallBlock]
    resolved: dict[str, ToolOutput]


@dataclass
class AgentRuntime:
    """One agent while the core is up."""

    record: AgentRecord
    run: RunRecord | None = None
    task: asyncio.Task[None] | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pause_requested: bool = False
    overflow_retried: bool = False
    partial_text: str = ""
    open_calls: OpenCalls | None = None
    run_usage: Usage = field(default_factory=Usage)
    cost_total: float = 0.0
    context_tokens: int | None = None
    last_seq: int = 0
    recovered: bool = False
    deleted: bool = False
    waiting_agent_messages: int = 0

    @property
    def id(self) -> AgentId:
        """The agent id."""
        return self.record.id

    @property
    def status(self) -> AgentStatus:
        """The agent's current status."""
        return self.record.status


class StatusBoard:
    """The only place that changes an agent's status: validates, persists, announces."""

    def __init__(self, deps: AgentDeps) -> None:
        self._deps = deps

    async def set(
        self,
        rt: AgentRuntime,
        target: AgentStatus,
        *,
        recovered: bool = False,
        force: bool = False,
    ) -> None:
        """Move ``rt`` to ``target``.

        Args:
            rt: The agent.
            target: The new status.
            recovered: Mark the change as a crash recovery.
            force: Skip the transition table (startup recovery only).

        Raises:
            InvalidTransitionError: The transition is not in the table.
        """
        previous = rt.status
        if not force:
            if previous is target:
                return
            check_transition(previous, target)
        now = self._deps.clock.now()
        rt.record = rt.record.model_copy(update={"status": target, "updated_at": now})
        await self._deps.memory.save_agent(rt.record)
        await self._deps.memory.append_event(
            AgentEventRecord(
                agent_id=rt.id,
                run_id=rt.run.id if rt.run else None,
                after_seq=rt.last_seq,
                type=AgentEventType.RECOVERED if recovered else AgentEventType.STATUS,
                payload={EVENT_STATUS_KEY: target.value, EVENT_PREVIOUS_KEY: previous.value},
                created_at=now,
            )
        )
        self._deps.bus.publish(
            rt.id, AgentStatusChanged(status=target, previous=previous, recovered=recovered)
        )
