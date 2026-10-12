"""``SqliteMemoryStore``: the ``MemoryStore`` implementation over SQLite (layers 2-3)."""

from datetime import datetime

from pydantic import JsonValue

from harness.core.consts import (
    AgentStatus,
    ApprovalStatus,
    BlobKind,
    PermissionAnswer,
    PermissionResolution,
    PermissionScope,
    TaskStatus,
)
from harness.core.records import (
    AgentEventRecord,
    AgentRecord,
    BackgroundTaskRecord,
    InboxItem,
    MessageRecord,
    ModelCallRecord,
    PendingApproval,
    RunRecord,
    SummaryRecord,
    ToolCallRecord,
)
from harness.core.types import AgentId, Clock, IdGenerator, PermissionRequestId, RunId
from harness.memory.blobs import FileBlobStore
from harness.memory.consts import HISTORY_SEARCH_LIMIT_DEFAULT
from harness.memory.db import convert, repo
from harness.memory.db.engine import Database
from harness.memory.db.models import PermissionRequestRow

_AGENT_OWNED_BLOBS = frozenset({BlobKind.TASK_OUTPUT, BlobKind.TOOL_OUTPUT})
_ARCHIVED = AgentStatus.ARCHIVED.value


class SqliteMemoryStore:
    """Persists agents, history, runs, approvals, tasks, events, summaries and the cost ledger."""

    def __init__(
        self, db: Database, clock: Clock, ids: IdGenerator, blobs: FileBlobStore | None = None
    ) -> None:
        self._db = db
        self._clock = clock
        self._ids = ids
        self._blobs = blobs

    # --- agents ---

    async def save_agent(self, a: AgentRecord) -> None:
        """Insert or update an agent."""
        async with self._db.write() as session:
            await repo.save_agent(session, a)

    async def get_agent(self, agent_id: AgentId) -> AgentRecord:
        """Return one agent; raise ``AgentNotFoundError`` if missing."""
        async with self._db.read() as session:
            return convert.agent_record(await repo.require_agent(session, agent_id))

    async def list_agents(self, *, include_archived: bool = False) -> list[AgentRecord]:
        """Return all agents in creation order."""
        async with self._db.read() as session:
            rows = await repo.list_agents(session, None)
        return [
            convert.agent_record(row) for row in rows if include_archived or row.status != _ARCHIVED
        ]

    async def delete_agent(self, agent_id: AgentId) -> None:
        """Delete the agent and everything it owns (cascade), then its orphaned blobs."""
        blob_ids = (
            await self._blobs.ids_for_agent(agent_id, _AGENT_OWNED_BLOBS) if self._blobs else []
        )
        async with self._db.write() as session:
            await repo.delete_agent(session, agent_id)
        if self._blobs:
            for blob_id in blob_ids:
                await self._blobs.delete(blob_id)

    async def get_config_overrides(self, agent_id: AgentId) -> dict[str, JsonValue]:
        """Return the agent's config layer."""
        async with self._db.read() as session:
            return dict((await repo.require_agent(session, agent_id)).config_overrides)

    async def set_config_overrides(
        self, agent_id: AgentId, overrides: dict[str, JsonValue]
    ) -> None:
        """Replace the agent's config layer."""
        async with self._db.write() as session:
            await repo.set_overrides(session, agent_id, overrides, self._clock.now())

    # --- layer 2 ---

    async def append_message(self, m: MessageRecord) -> MessageRecord:
        """Store a message with the agent's next seq and return it (``m.seq`` is ignored)."""
        async with self._db.write() as session:
            return await repo.insert_message(session, m)

    async def get_messages(
        self, agent_id: AgentId, *, from_seq: int = 0, limit: int | None = None
    ) -> list[MessageRecord]:
        """Return messages with ``seq >= from_seq`` in order."""
        async with self._db.read() as session:
            rows = await repo.get_messages(session, agent_id, from_seq, limit)
        return [convert.message_record(row) for row in rows]

    async def search_history(
        self, agent_id: AgentId, query: str, *, limit: int = HISTORY_SEARCH_LIMIT_DEFAULT
    ) -> list[MessageRecord]:
        """Lexical (BM25) search in one agent's own history."""
        async with self._db.read() as session:
            rows = await repo.search_history(session, agent_id, query, limit)
        return [convert.message_record(row) for row in rows]

    async def last_seq(self, agent_id: AgentId) -> int:
        """Return the last seq allocated to the agent."""
        async with self._db.read() as session:
            return await repo.last_seq(session, agent_id)

    # --- inbox ---

    async def enqueue_inbox(self, item: InboxItem) -> None:
        """Store a received message without a seq."""
        async with self._db.write() as session:
            await repo.require_agent(session, item.agent_id)
            session.add(convert.inbox_row(item))

    async def inject_inbox(
        self, agent_id: AgentId, run_id: RunId, *, step: int | None = None
    ) -> list[MessageRecord]:
        """Move the inbox into history in one transaction; return the new messages."""
        async with self._db.write() as session:
            return await repo.inject_inbox(
                session, agent_id, run_id, step=step, ids=self._ids, now=self._clock.now()
            )

    async def pending_inbox(self, agent_id: AgentId) -> list[InboxItem]:
        """Return inbox items that were not injected yet."""
        async with self._db.read() as session:
            rows = await repo.pending_inbox(session, agent_id)
        return [convert.inbox_item(row) for row in rows]

    # --- runs and recovery ---

    async def save_run(self, r: RunRecord) -> None:
        """Insert or update a run."""
        async with self._db.write() as session:
            await session.merge(convert.run_row(r))

    async def open_run(self, agent_id: AgentId) -> RunRecord | None:
        """Return the agent's open run, if any."""
        async with self._db.read() as session:
            row = await repo.open_run(session, agent_id)
        return None if row is None else convert.run_record(row)

    async def save_pending_approval(self, p: PendingApproval) -> None:
        """Persist an open approval request."""
        async with self._db.write() as session:
            tool_call_id = await repo.tool_call_id_for(session, p.agent_id, p.request.call_id)
            await session.merge(convert.approval_row(p, tool_call_id))

    async def close_pending_approval(
        self,
        request_id: PermissionRequestId,
        resolution: PermissionResolution,
        *,
        scope: PermissionScope | None = None,
        deny_message: str | None = None,
    ) -> None:
        """Close a request (answered / cancelled); the row stays as a log."""
        cancelled = resolution is PermissionResolution.CANCELLED
        async with self._db.write() as session:
            row = await session.get(PermissionRequestRow, request_id)
            if row is None:
                return
            row.status = (ApprovalStatus.CANCELLED if cancelled else ApprovalStatus.ANSWERED).value
            row.decision = None if cancelled else PermissionAnswer(resolution.value).value
            row.scope = scope.value if scope else None
            row.deny_message = deny_message
            row.answered_at = self._clock.now()

    async def pending_approvals(self, agent_id: AgentId | None = None) -> list[PendingApproval]:
        """Return open approval requests, oldest first."""
        async with self._db.read() as session:
            rows = await repo.pending_approvals(session, agent_id)
        return [convert.pending_approval(row) for row in rows]

    async def save_task(self, t: BackgroundTaskRecord) -> None:
        """Insert or update a background task."""
        async with self._db.write() as session:
            tool_call_id = await repo.require_tool_call_id(session, t.agent_id, t.call_id)
            await session.merge(convert.task_row(t, tool_call_id))

    async def list_tasks(
        self, agent_id: AgentId | None, *, status: TaskStatus | None = None
    ) -> list[BackgroundTaskRecord]:
        """Return background tasks of one agent (or of all agents)."""
        async with self._db.read() as session:
            rows = await repo.list_tasks(session, agent_id, status)
        return [convert.task_record(row) for row in rows]

    async def append_event(self, e: AgentEventRecord) -> None:
        """Persist a display event."""
        async with self._db.write() as session:
            session.add(convert.event_row(e))

    async def get_events(self, agent_id: AgentId, *, from_seq: int = 0) -> list[AgentEventRecord]:
        """Return display events positioned at or after ``from_seq``."""
        async with self._db.read() as session:
            rows = await repo.get_events(session, agent_id, from_seq)
        return [convert.event_record(row) for row in rows]

    # --- layer 3 ---

    async def save_summary(self, s: SummaryRecord) -> None:
        """Insert or update a summary."""
        async with self._db.write() as session:
            await session.merge(convert.summary_row(s))

    async def get_summaries(
        self, agent_id: AgentId, *, active_only: bool = True
    ) -> list[SummaryRecord]:
        """Return summaries; by default only those not superseded."""
        async with self._db.read() as session:
            rows = await repo.get_summaries(session, agent_id, active_only=active_only)
        return [convert.summary_record(row) for row in rows]

    # --- tool call audit ---

    async def save_tool_call(self, t: ToolCallRecord) -> None:
        """Insert or update a tool call record."""
        async with self._db.write() as session:
            await session.merge(convert.tool_call_row(t))

    async def get_tool_calls(
        self, agent_id: AgentId, *, run_id: RunId | None = None
    ) -> list[ToolCallRecord]:
        """Return an agent's tool calls in creation order."""
        async with self._db.read() as session:
            rows = await repo.get_tool_calls(session, agent_id, run_id)
        return [convert.tool_call_record(row) for row in rows]

    # --- cost ledger and meta ---

    async def record_model_call(self, call: ModelCallRecord) -> None:
        """Add one model call to the cost ledger."""
        async with self._db.write() as session:
            await session.merge(convert.model_call_row(call))

    async def cost_total(self, agent_id: AgentId) -> float:
        """Return the lifetime cost of one agent."""
        async with self._db.read() as session:
            return await repo.cost_sum(session, agent_id=agent_id)

    async def cost_since(self, since: datetime) -> float:
        """Return the cost of all agents since ``since``."""
        async with self._db.read() as session:
            return await repo.cost_sum(session, since=since)

    async def get_meta(self, key: str) -> JsonValue:
        """Return a ``harness_meta`` value, or ``None``."""
        async with self._db.read() as session:
            return await repo.get_meta(session, key)

    async def set_meta(self, key: str, value: JsonValue) -> None:
        """Insert or replace a ``harness_meta`` value."""
        async with self._db.write() as session:
            await repo.set_meta(session, key, value, self._clock.now())
