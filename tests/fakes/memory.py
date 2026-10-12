"""In-memory ``MemoryStore`` and ``BlobStore``; held to the real ones by the contract tests."""

from datetime import datetime

from pydantic import JsonValue

from harness.core.consts import (
    ID_PREFIX_BLOB,
    ID_PREFIX_MESSAGE,
    AgentStatus,
    BlobKind,
    MessageRole,
    PermissionResolution,
    PermissionScope,
    TaskStatus,
)
from harness.core.errors import AgentNotFoundError, MemoryStoreError
from harness.core.records import (
    AgentEventRecord,
    AgentRecord,
    BackgroundTaskRecord,
    DeletedFile,
    InboxItem,
    MessageRecord,
    ModelCallRecord,
    PendingApproval,
    RunRecord,
    SummaryRecord,
    ToolCallRecord,
)
from harness.core.types import (
    AgentId,
    BackupId,
    Clock,
    IdGenerator,
    PermissionRequestId,
    RunId,
)
from harness.memory import hebrew
from harness.memory.db import convert

DEFAULT_LIMIT = 20


class FakeMemoryStore:
    """Keeps everything in dictionaries; same observable behaviour as the SQLite store."""

    def __init__(self, clock: Clock, ids: IdGenerator) -> None:
        self._clock = clock
        self._ids = ids
        self.agents: dict[AgentId, AgentRecord] = {}
        self.seqs: dict[AgentId, int] = {}
        self.messages: list[MessageRecord] = []
        self.inbox: list[InboxItem] = []
        self.runs: dict[str, RunRecord] = {}
        self.approvals: dict[str, PendingApproval] = {}
        self.tasks: dict[str, BackgroundTaskRecord] = {}
        self.events: list[AgentEventRecord] = []
        self.summaries: dict[str, SummaryRecord] = {}
        self.tool_calls: dict[str, ToolCallRecord] = {}
        self.model_calls: dict[str, ModelCallRecord] = {}
        self.meta: dict[str, JsonValue] = {}

    def _require(self, agent_id: AgentId) -> AgentRecord:
        if agent_id not in self.agents:
            raise AgentNotFoundError(str(agent_id))
        return self.agents[agent_id]

    async def save_agent(self, a: AgentRecord) -> None:
        existing = self.agents.get(a.id)
        overrides = existing.config_overrides if existing else a.config_overrides
        self.agents[a.id] = a.model_copy(update={"config_overrides": dict(overrides)})
        self.seqs.setdefault(a.id, 0)

    async def get_agent(self, agent_id: AgentId) -> AgentRecord:
        return self._require(agent_id).model_copy(deep=True)

    async def list_agents(self, *, include_archived: bool = False) -> list[AgentRecord]:
        return [
            a.model_copy(deep=True)
            for a in self.agents.values()
            if include_archived or a.status is not AgentStatus.ARCHIVED
        ]

    async def delete_agent(self, agent_id: AgentId) -> None:
        self._require(agent_id)
        del self.agents[agent_id]
        del self.seqs[agent_id]
        self.messages = [m for m in self.messages if m.agent_id != agent_id]
        self.inbox = [i for i in self.inbox if i.agent_id != agent_id]
        self.events = [e for e in self.events if e.agent_id != agent_id]
        for table in (self.runs, self.approvals, self.tasks, self.summaries, self.tool_calls):
            for key in [k for k, v in table.items() if v.agent_id == agent_id]:
                del table[key]
        for key, call in self.model_calls.items():
            if call.agent_id == agent_id:
                self.model_calls[key] = call.model_copy(update={"agent_id": None})

    async def get_config_overrides(self, agent_id: AgentId) -> dict[str, JsonValue]:
        return dict(self._require(agent_id).config_overrides)

    async def set_config_overrides(
        self, agent_id: AgentId, overrides: dict[str, JsonValue]
    ) -> None:
        agent = self._require(agent_id)
        self.agents[agent_id] = agent.model_copy(
            update={"config_overrides": dict(overrides), "updated_at": self._clock.now()}
        )

    async def append_message(self, m: MessageRecord) -> MessageRecord:
        self._require(m.agent_id)
        self.seqs[m.agent_id] += 1
        stored = m.model_copy(update={"seq": self.seqs[m.agent_id]})
        self.messages.append(stored)
        return stored

    async def get_messages(
        self, agent_id: AgentId, *, from_seq: int = 0, limit: int | None = None
    ) -> list[MessageRecord]:
        found = [m for m in self.messages if m.agent_id == agent_id and m.seq >= from_seq]
        return found if limit is None else found[:limit]

    async def search_history(
        self, agent_id: AgentId, query: str, *, limit: int = DEFAULT_LIMIT
    ) -> list[MessageRecord]:
        wanted = set(hebrew.tokens(query))
        hits = [
            m
            for m in self.messages
            if m.agent_id == agent_id
            and wanted & set(hebrew.tokens(convert.search_text(m.content)))
        ]
        return hits[:limit]

    async def last_seq(self, agent_id: AgentId) -> int:
        self._require(agent_id)
        return self.seqs[agent_id]

    async def enqueue_inbox(self, item: InboxItem) -> None:
        self._require(item.agent_id)
        self.inbox.append(item)

    async def inject_inbox(
        self, agent_id: AgentId, run_id: RunId, *, step: int | None = None
    ) -> list[MessageRecord]:
        injected: list[MessageRecord] = []
        for item in await self.pending_inbox(agent_id):
            record = MessageRecord(
                id=self._ids.new(ID_PREFIX_MESSAGE),  # type: ignore[arg-type]
                agent_id=agent_id,
                seq=0,
                run_id=run_id,
                role=MessageRole.USER,
                content=item.content,
                step=step,
                origin=item.origin,
                created_at=self._clock.now(),
            )
            injected.append(await self.append_message(record))
        self.inbox = [i for i in self.inbox if i.agent_id != agent_id]
        return injected

    async def pending_inbox(self, agent_id: AgentId) -> list[InboxItem]:
        items = [i for i in self.inbox if i.agent_id == agent_id]
        return sorted(items, key=lambda i: (i.received_at, i.id))

    async def save_run(self, r: RunRecord) -> None:
        self.runs[r.id] = r.model_copy(deep=True)

    async def open_run(self, agent_id: AgentId) -> RunRecord | None:
        open_runs = [r for r in self.runs.values() if r.agent_id == agent_id and not r.ended_at]
        newest = max(open_runs, key=lambda r: (r.started_at, r.id), default=None)
        return None if newest is None else newest.model_copy(deep=True)

    async def save_pending_approval(self, p: PendingApproval) -> None:
        self.approvals[p.request_id] = p

    async def close_pending_approval(
        self,
        request_id: PermissionRequestId,
        resolution: PermissionResolution,
        *,
        scope: PermissionScope | None = None,
        deny_message: str | None = None,
    ) -> None:
        self.approvals.pop(request_id, None)

    async def pending_approvals(self, agent_id: AgentId | None = None) -> list[PendingApproval]:
        found = [p for p in self.approvals.values() if agent_id in (None, p.agent_id)]
        return sorted(found, key=lambda p: (p.created_at, p.request_id))

    async def save_task(self, t: BackgroundTaskRecord) -> None:
        known = {(c.agent_id, c.call_id) for c in self.tool_calls.values()}
        if (t.agent_id, t.call_id) not in known:
            raise MemoryStoreError(t.call_id)
        self.tasks[t.id] = t

    async def list_tasks(
        self, agent_id: AgentId | None, *, status: TaskStatus | None = None
    ) -> list[BackgroundTaskRecord]:
        return [
            t
            for t in self.tasks.values()
            if agent_id in (None, t.agent_id) and status in (None, t.status)
        ]

    async def append_event(self, e: AgentEventRecord) -> None:
        self.events.append(e)

    async def get_events(self, agent_id: AgentId, *, from_seq: int = 0) -> list[AgentEventRecord]:
        return [e for e in self.events if e.agent_id == agent_id and e.after_seq >= from_seq]

    async def save_summary(self, s: SummaryRecord) -> None:
        self.summaries[s.id] = s

    async def get_summaries(
        self, agent_id: AgentId, *, active_only: bool = True
    ) -> list[SummaryRecord]:
        found = [
            s
            for s in self.summaries.values()
            if s.agent_id == agent_id and not (active_only and s.superseded_by)
        ]
        return sorted(found, key=lambda s: (s.from_seq, s.created_at))

    async def save_tool_call(self, t: ToolCallRecord) -> None:
        self.tool_calls[t.id] = t

    async def get_tool_calls(
        self, agent_id: AgentId, *, run_id: RunId | None = None
    ) -> list[ToolCallRecord]:
        return [
            t
            for t in self.tool_calls.values()
            if t.agent_id == agent_id and run_id in (None, t.run_id)
        ]

    async def record_model_call(self, call: ModelCallRecord) -> None:
        self.model_calls[call.id] = call

    async def cost_total(self, agent_id: AgentId) -> float:
        return sum(c.cost_usd for c in self.model_calls.values() if c.agent_id == agent_id)

    async def cost_since(self, since: datetime) -> float:
        return sum(c.cost_usd for c in self.model_calls.values() if c.created_at >= since)

    async def get_meta(self, key: str) -> JsonValue:
        return self.meta.get(key)

    async def set_meta(self, key: str, value: JsonValue) -> None:
        self.meta[key] = value


class FakeBlobStore:
    """Blobs in a dictionary."""

    def __init__(self, ids: IdGenerator) -> None:
        self._ids = ids
        self.blobs: dict[str, bytes] = {}
        self.kinds: dict[str, BlobKind] = {}
        self.expiry: dict[str, datetime] = {}
        self.deleted: dict[BackupId, DeletedFile] = {}

    async def put(self, data: bytes, *, kind: BlobKind, meta: dict[str, JsonValue]) -> str:
        if kind is BlobKind.TASK_OUTPUT and "task_id" not in meta:
            raise MemoryStoreError(str(kind))  # the real store names the file after the task
        blob_id = self._ids.new(ID_PREFIX_BLOB)
        self.blobs[blob_id] = data
        self.kinds[blob_id] = kind
        return blob_id

    async def get(self, blob_id: str) -> bytes:
        if blob_id not in self.blobs:
            raise MemoryStoreError(blob_id)
        return self.blobs[blob_id]

    async def delete(self, blob_id: str) -> None:
        if blob_id not in self.blobs:
            raise MemoryStoreError(blob_id)
        del self.blobs[blob_id]

    async def save_deleted(self, record: DeletedFile) -> None:
        self.deleted[record.id] = record

    async def get_deleted(self, backup_id: BackupId) -> DeletedFile:
        if backup_id not in self.deleted:
            raise MemoryStoreError(backup_id)
        return self.deleted[backup_id]

    async def purge_expired(self, now: datetime) -> int:
        expired = [blob_id for blob_id, at in self.expiry.items() if at <= now]
        for blob_id in expired:
            self.blobs.pop(blob_id, None)
            del self.expiry[blob_id]
        return len(expired)
