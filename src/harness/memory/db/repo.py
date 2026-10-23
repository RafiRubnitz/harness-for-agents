"""Queries over the SQLite schema. Every function works inside the caller's session."""

from datetime import datetime

from pydantic import JsonValue
from sqlalchemy import delete, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from harness.core.consts import ID_PREFIX_MESSAGE, ApprovalStatus, MessageRole, TaskStatus
from harness.core.errors import AgentNotFoundError, MemoryStoreError
from harness.core.records import AgentRecord, MessageRecord
from harness.core.types import AgentId, IdGenerator, RunId
from harness.core.utils.inbox import in_order
from harness.memory import hebrew
from harness.memory.consts import (
    MSG_AGENT_NOT_FOUND,
    MSG_TOOL_CALL_NOT_FOUND,
    SQL_SEARCH_HISTORY,
)
from harness.memory.db import convert
from harness.memory.db.models import (
    AgentEventRow,
    AgentRow,
    BackgroundTaskRow,
    HarnessMetaRow,
    InboxRow,
    MessageRow,
    ModelCallRow,
    PermissionRequestRow,
    RunRow,
    SummaryRow,
    ToolCallRow,
)

# --- agents ---


async def require_agent(session: AsyncSession, agent_id: AgentId) -> AgentRow:
    """Return the agent row or raise ``AgentNotFoundError``."""
    row = await session.get(AgentRow, agent_id)
    if row is None:
        raise AgentNotFoundError(MSG_AGENT_NOT_FOUND.format(agent_id=agent_id))
    return row


async def save_agent(session: AsyncSession, record: AgentRecord) -> None:
    """Insert or update an agent.

    ``config_overrides`` is written only on insert; later changes go through
    ``set_overrides`` so a stale in-memory record cannot undo them.
    """
    values = convert.agent_values(record)
    row = await session.get(AgentRow, record.id)
    if row is None:
        session.add(
            AgentRow(id=record.id, config_overrides=dict(record.config_overrides), **values)
        )
        return
    for column, value in values.items():
        setattr(row, column, value)


async def list_agents(session: AsyncSession, statuses: list[str] | None) -> list[AgentRow]:
    """Return agents in creation order, optionally only some statuses."""
    query = select(AgentRow).order_by(AgentRow.created_at, AgentRow.id)
    if statuses is not None:
        query = query.where(AgentRow.status.in_(statuses))
    return list((await session.scalars(query)).all())


async def set_overrides(
    session: AsyncSession, agent_id: AgentId, overrides: dict[str, JsonValue], now: datetime
) -> None:
    """Replace the agent's config layer."""
    row = await require_agent(session, agent_id)
    row.config_overrides = dict(overrides)
    row.updated_at = now


# --- messages ---


async def insert_message(session: AsyncSession, record: MessageRecord) -> MessageRecord:
    """Insert a message with the agent's next seq (same transaction as the counter)."""
    allocated = await session.execute(
        update(AgentRow)
        .where(AgentRow.id == record.agent_id)
        .values(last_seq=AgentRow.last_seq + 1)
        .returning(AgentRow.last_seq)
    )
    seq = allocated.scalar_one_or_none()
    if seq is None:
        raise AgentNotFoundError(MSG_AGENT_NOT_FOUND.format(agent_id=record.agent_id))
    session.add(convert.message_row(record, seq))
    await session.flush()
    return record.model_copy(update={"seq": seq})


async def get_messages(
    session: AsyncSession, agent_id: AgentId, from_seq: int, limit: int | None
) -> list[MessageRow]:
    """Return messages with ``seq >= from_seq`` in order."""
    query = (
        select(MessageRow)
        .where(MessageRow.agent_id == agent_id, MessageRow.seq >= from_seq)
        .order_by(MessageRow.seq)
    )
    if limit is not None:
        query = query.limit(limit)
    return list((await session.scalars(query)).all())


async def search_history(
    session: AsyncSession, agent_id: AgentId, query: str, limit: int
) -> list[MessageRow]:
    """BM25 search over one agent's history, best match first."""
    match = hebrew.fts_query(query)
    if not match:
        return []
    found = await session.execute(
        text(SQL_SEARCH_HISTORY), {"q": match, "agent_id": agent_id, "k": limit}
    )
    ids = [str(row.id) for row in found]
    if not ids:
        return []
    rows = (await session.scalars(select(MessageRow).where(MessageRow.id.in_(ids)))).all()
    by_id = {row.id: row for row in rows}
    return [by_id[message_id] for message_id in ids if message_id in by_id]


# --- inbox ---


async def pending_inbox(session: AsyncSession, agent_id: AgentId) -> list[InboxRow]:
    """Return the agent's inbox: higher priority first, arrival order inside a priority."""
    query = (
        select(InboxRow)
        .where(InboxRow.agent_id == agent_id)
        .order_by(InboxRow.priority.desc(), InboxRow.received_at, InboxRow.id)
    )
    return list((await session.scalars(query)).all())


async def agents_with_inbox(session: AsyncSession) -> list[str]:
    """Return the ids of the agents that have something in their inbox."""
    query = select(InboxRow.agent_id).distinct().order_by(InboxRow.agent_id)
    return list((await session.scalars(query)).all())


async def message_by_origin_id(
    session: AsyncSession, agent_id: AgentId, message_id: str
) -> MessageRow | None:
    """Return the message an agent received under the sender's ``message_id``."""
    query = (
        select(MessageRow)
        .where(MessageRow.agent_id == agent_id, MessageRow.origin_message_id == message_id)
        .order_by(MessageRow.seq.desc())
        .limit(1)
    )
    return (await session.scalars(query)).first()


async def last_received(session: AsyncSession, agent_id: AgentId) -> MessageRow | None:
    """Return the newest message of the agent that came with an origin."""
    query = (
        select(MessageRow)
        .where(MessageRow.agent_id == agent_id, MessageRow.origin_source_type.is_not(None))
        .order_by(MessageRow.seq.desc())
        .limit(1)
    )
    return (await session.scalars(query)).first()


async def inject_inbox(
    session: AsyncSession,
    agent_id: AgentId,
    run_id: RunId,
    *,
    step: int | None,
    ids: IdGenerator,
    now: datetime,
    aging_s: float | None = None,
) -> list[MessageRecord]:
    """Turn every inbox item into a ``user`` message with the next seq, and empty the inbox."""
    injected: list[MessageRecord] = []
    rows = {row.id: row for row in await pending_inbox(session, agent_id)}
    items = [convert.inbox_item(row) for row in rows.values()]
    for item in in_order(items, now, aging_s):
        row = rows[item.id]
        record = MessageRecord.model_validate(
            {
                "id": ids.new(ID_PREFIX_MESSAGE),
                "agent_id": agent_id,
                "seq": 0,
                "run_id": run_id,
                "role": MessageRole.USER,
                "content": item.content,
                "step": step,
                "origin": item.origin,
                "created_at": now,
            }
        )
        injected.append(await insert_message(session, record))
        await session.delete(row)
    return injected


# --- runs ---


async def open_run(session: AsyncSession, agent_id: AgentId) -> RunRow | None:
    """Return the agent's open run (``ended_at IS NULL``), newest first."""
    query = (
        select(RunRow)
        .where(RunRow.agent_id == agent_id, RunRow.ended_at.is_(None))
        .order_by(RunRow.started_at.desc(), RunRow.id.desc())
        .limit(1)
    )
    return (await session.scalars(query)).first()


# --- approvals ---


async def tool_call_id_for(
    session: AsyncSession, agent_id: AgentId | None, call_id: str | None
) -> str | None:
    """Return the ``tool_calls.id`` of a provider call id, if that call was recorded."""
    if agent_id is None or call_id is None:
        return None
    query = select(ToolCallRow.id).where(
        ToolCallRow.agent_id == agent_id, ToolCallRow.call_id == call_id
    )
    return (await session.scalars(query)).first()


async def require_tool_call_id(session: AsyncSession, agent_id: AgentId, call_id: str) -> str:
    """Like ``tool_call_id_for`` but the call must exist."""
    found = await tool_call_id_for(session, agent_id, call_id)
    if found is None:
        raise MemoryStoreError(MSG_TOOL_CALL_NOT_FOUND.format(call_id=call_id, agent_id=agent_id))
    return found


async def pending_approvals(
    session: AsyncSession, agent_id: AgentId | None
) -> list[PermissionRequestRow]:
    """Return open approval requests, oldest first."""
    query = (
        select(PermissionRequestRow)
        .where(PermissionRequestRow.status == ApprovalStatus.PENDING.value)
        .order_by(PermissionRequestRow.created_at, PermissionRequestRow.id)
    )
    if agent_id is not None:
        query = query.where(PermissionRequestRow.agent_id == agent_id)
    return list((await session.scalars(query)).all())


# --- tasks, events, summaries, tool calls ---


async def list_tasks(
    session: AsyncSession, agent_id: AgentId | None, status: TaskStatus | None
) -> list[BackgroundTaskRow]:
    """Return background tasks, oldest first."""
    query = select(BackgroundTaskRow).order_by(BackgroundTaskRow.started_at, BackgroundTaskRow.id)
    if agent_id is not None:
        query = query.where(BackgroundTaskRow.agent_id == agent_id)
    if status is not None:
        query = query.where(BackgroundTaskRow.status == status.value)
    return list((await session.scalars(query)).all())


async def get_events(
    session: AsyncSession, agent_id: AgentId, from_seq: int
) -> list[AgentEventRow]:
    """Return persisted display events with ``after_seq >= from_seq`` in order."""
    query = (
        select(AgentEventRow)
        .where(AgentEventRow.agent_id == agent_id, AgentEventRow.after_seq >= from_seq)
        .order_by(AgentEventRow.pk)
    )
    return list((await session.scalars(query)).all())


async def last_seq(session: AsyncSession, agent_id: AgentId) -> int:
    """Return the last seq allocated to the agent."""
    return (await require_agent(session, agent_id)).last_seq


async def get_summaries(
    session: AsyncSession, agent_id: AgentId, *, active_only: bool
) -> list[SummaryRow]:
    """Return summaries ordered by the range they cover."""
    query = (
        select(SummaryRow)
        .where(SummaryRow.agent_id == agent_id)
        .order_by(SummaryRow.from_seq, SummaryRow.created_at)
    )
    if active_only:
        query = query.where(SummaryRow.superseded_by.is_(None))
    return list((await session.scalars(query)).all())


async def get_tool_calls(
    session: AsyncSession, agent_id: AgentId, run_id: RunId | None
) -> list[ToolCallRow]:
    """Return an agent's tool calls in creation order."""
    query = (
        select(ToolCallRow)
        .where(ToolCallRow.agent_id == agent_id)
        .order_by(ToolCallRow.created_at, ToolCallRow.id)
    )
    if run_id is not None:
        query = query.where(ToolCallRow.run_id == run_id)
    return list((await session.scalars(query)).all())


# --- cost ledger and meta ---


async def cost_sum(
    session: AsyncSession, *, agent_id: AgentId | None = None, since: datetime | None = None
) -> float:
    """Sum ``model_calls.cost_usd`` for one agent and / or since a moment."""
    query = select(func.coalesce(func.sum(ModelCallRow.cost_usd), 0.0))
    if agent_id is not None:
        query = query.where(ModelCallRow.agent_id == agent_id)
    if since is not None:
        query = query.where(ModelCallRow.created_at >= since)
    return float((await session.execute(query)).scalar_one())


async def get_meta(session: AsyncSession, key: str) -> JsonValue:
    """Return a ``harness_meta`` value, or ``None``."""
    row = await session.get(HarnessMetaRow, key)
    return None if row is None else row.value


async def set_meta(session: AsyncSession, key: str, value: JsonValue, now: datetime) -> None:
    """Insert or replace a ``harness_meta`` value."""
    await session.merge(HarnessMetaRow(key=key, value=value, updated_at=now))


async def delete_agent(session: AsyncSession, agent_id: AgentId) -> None:
    """Delete the agent row; the foreign keys cascade to everything it owns."""
    await require_agent(session, agent_id)
    await session.execute(delete(AgentRow).where(AgentRow.id == agent_id))
