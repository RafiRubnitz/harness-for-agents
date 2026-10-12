"""Memory component: history, runs and recovery state in SQLite; blobs on disk.

Other components use the ``MemoryStore`` / ``BlobStore`` protocols and ``open_memory``.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from loguru import logger
from pydantic import JsonValue

from harness.core.consts import BlobKind, PermissionResolution, PermissionScope, TaskStatus
from harness.core.errors import GatewayError, MemoryStoreError
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
    ModelRequest,
    ModelResponse,
    PermissionRequestId,
    RunId,
)
from harness.core.utils.paths import HarnessHome
from harness.memory.blobs import FileBlobStore
from harness.memory.consts import (
    HISTORY_SEARCH_LIMIT_DEFAULT,
    META_CREATED_AT,
    META_LAST_APP_VERSION,
)
from harness.memory.db.engine import Database
from harness.memory.db.migrate import migrate_to_head
from harness.memory.semantic import (
    DefaultSemanticMemory,
    Embedder,
    EmbedderServer,
    HttpEmbedder,
    Semantic,
    SemanticEdges,
    SemanticMemory,
    open_semantic,
)
from harness.memory.store import SqliteMemoryStore


class MemoryStore(Protocol):
    """Persistent agents, history (layer 2), summaries (layer 3) and recovery state."""

    async def save_agent(self, a: AgentRecord) -> None: ...
    async def get_agent(self, agent_id: AgentId) -> AgentRecord: ...
    async def list_agents(self, *, include_archived: bool = False) -> list[AgentRecord]: ...
    async def delete_agent(self, agent_id: AgentId) -> None: ...
    async def get_config_overrides(self, agent_id: AgentId) -> dict[str, JsonValue]: ...
    async def set_config_overrides(
        self, agent_id: AgentId, overrides: dict[str, JsonValue]
    ) -> None: ...

    async def append_message(self, m: MessageRecord) -> MessageRecord: ...
    async def get_messages(
        self, agent_id: AgentId, *, from_seq: int = 0, limit: int | None = None
    ) -> list[MessageRecord]: ...
    async def search_history(
        self, agent_id: AgentId, query: str, *, limit: int = HISTORY_SEARCH_LIMIT_DEFAULT
    ) -> list[MessageRecord]: ...
    async def last_seq(self, agent_id: AgentId) -> int: ...

    async def enqueue_inbox(self, item: InboxItem) -> None: ...
    async def inject_inbox(
        self,
        agent_id: AgentId,
        run_id: RunId,
        *,
        step: int | None = None,
        aging_s: float | None = None,
    ) -> list[MessageRecord]: ...
    async def pending_inbox(
        self, agent_id: AgentId, *, aging_s: float | None = None
    ) -> list[InboxItem]: ...
    async def agents_with_inbox(self) -> list[AgentId]: ...
    async def received_message(
        self, agent_id: AgentId, message_id: str
    ) -> MessageRecord | None: ...
    async def last_received(self, agent_id: AgentId) -> MessageRecord | None: ...
    async def data_version(self) -> int: ...

    async def save_run(self, r: RunRecord) -> None: ...
    async def open_run(self, agent_id: AgentId) -> RunRecord | None: ...
    async def save_pending_approval(self, p: PendingApproval) -> None: ...
    async def close_pending_approval(
        self,
        request_id: PermissionRequestId,
        resolution: PermissionResolution,
        *,
        scope: PermissionScope | None = None,
        deny_message: str | None = None,
    ) -> None: ...
    async def pending_approvals(self, agent_id: AgentId | None = None) -> list[PendingApproval]: ...
    async def save_task(self, t: BackgroundTaskRecord) -> None: ...
    async def list_tasks(
        self, agent_id: AgentId | None, *, status: TaskStatus | None = None
    ) -> list[BackgroundTaskRecord]: ...
    async def append_event(self, e: AgentEventRecord) -> None: ...
    async def get_events(
        self, agent_id: AgentId, *, from_seq: int = 0
    ) -> list[AgentEventRecord]: ...

    async def save_summary(self, s: SummaryRecord) -> None: ...
    async def get_summaries(
        self, agent_id: AgentId, *, active_only: bool = True
    ) -> list[SummaryRecord]: ...

    async def save_tool_call(self, t: ToolCallRecord) -> None: ...
    async def get_tool_calls(
        self, agent_id: AgentId, *, run_id: RunId | None = None
    ) -> list[ToolCallRecord]: ...

    async def record_model_call(self, call: ModelCallRecord) -> None: ...
    async def cost_total(self, agent_id: AgentId) -> float: ...
    async def cost_since(self, since: datetime) -> float: ...
    async def get_meta(self, key: str) -> JsonValue: ...
    async def set_meta(self, key: str, value: JsonValue) -> None: ...


class BlobStore(Protocol):
    """Binary payloads under ``blobs/``: attachments, tool output, deletion backups."""

    async def put(self, data: bytes, *, kind: BlobKind, meta: dict[str, JsonValue]) -> str: ...
    async def get(self, blob_id: str) -> bytes: ...
    async def delete(self, blob_id: str) -> None: ...
    async def purge_expired(self, now: datetime) -> int: ...
    async def save_deleted(self, record: DeletedFile) -> None: ...
    async def get_deleted(self, backup_id: BackupId) -> DeletedFile: ...


class LedgerCallSink:
    """Gateway call sink that writes every model call to the ``model_calls`` ledger."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    async def record_call(
        self,
        record: ModelCallRecord,
        request: ModelRequest,
        response: ModelResponse | None,
        error: GatewayError | None,
    ) -> None:
        """Store the call; a ledger failure is logged and never fails the model call."""
        try:
            await self._store.record_model_call(record)
        except MemoryStoreError as failure:
            logger.error("model call {} was not written to the ledger: {}", record.id, failure)


@dataclass(frozen=True, slots=True)
class Memory:
    """The opened memory component."""

    db: Database
    store: SqliteMemoryStore
    blobs: FileBlobStore

    async def close(self) -> None:
        """Release the database."""
        await self.db.close()


async def open_memory(
    home: HarnessHome, clock: Clock, ids: IdGenerator, *, app_version: str
) -> Memory:
    """Open the database the way the app does at startup (13 §5.5).

    Backup (only when a migration is pending), upgrade to head, record ``harness_meta``,
    then purge expired deletion backups.
    """
    home.ensure()
    await migrate_to_head(home.db_file, home.backups_dir, clock)
    db = Database(home.db_file)
    blobs = FileBlobStore(db, home.blobs_dir, clock, ids)
    store = SqliteMemoryStore(db, clock, ids, blobs)
    if await store.get_meta(META_CREATED_AT) is None:
        await store.set_meta(META_CREATED_AT, clock.now().isoformat())
    await store.set_meta(META_LAST_APP_VERSION, app_version)
    await blobs.purge_expired(clock.now())
    return Memory(db=db, store=store, blobs=blobs)


__all__ = [
    "BlobStore",
    "Database",
    "DefaultSemanticMemory",
    "Embedder",
    "EmbedderServer",
    "FileBlobStore",
    "HttpEmbedder",
    "LedgerCallSink",
    "Memory",
    "MemoryStore",
    "Semantic",
    "SemanticEdges",
    "SemanticMemory",
    "SqliteMemoryStore",
    "open_memory",
    "open_semantic",
]
