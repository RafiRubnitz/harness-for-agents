"""SQLAlchemy 2.0 models of the SQLite schema (13 §2).

No ``relationship()``: the repository uses explicit ``select()`` and ORM rows never leave
the memory component. Timestamps are always set by the repository from the injected clock.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from harness.memory.db.types import NAMING_CONVENTION, UTCDateTime

_CASCADE = "CASCADE"
_SET_NULL = "SET NULL"
_RESTRICT = "RESTRICT"


class Base(DeclarativeBase):
    """Declarative base with the project naming convention."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map = {  # noqa: RUF012 - SQLAlchemy reads this class attribute
        datetime: UTCDateTime,
        dict[str, Any]: JSON(none_as_null=True),
        list[Any]: JSON(none_as_null=True),
    }


class AgentRow(Base):
    """``agents``."""

    __tablename__ = "agents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(index=True, unique=True)
    description: Mapped[str] = mapped_column(Text, default="")
    model: Mapped[str]
    system_prompt: Mapped[str] = mapped_column(Text)
    tools: Mapped[list[Any]]
    template: Mapped[str | None]
    status: Mapped[str] = mapped_column(index=True)
    settings: Mapped[dict[str, Any]]
    config_overrides: Mapped[dict[str, Any]]
    pending_hook_context: Mapped[list[Any]]
    last_seq: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]


class RunRow(Base):
    """``runs``; an open run has ``ended_at IS NULL``."""

    __tablename__ = "runs"
    __table_args__ = (Index(None, "agent_id", "ended_at"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id", ondelete=_CASCADE))
    first_seq: Mapped[int]
    step_count: Mapped[int]
    stop_reason: Mapped[str | None]
    retrieved_memory_ids: Mapped[list[Any]]
    started_at: Mapped[datetime]
    ended_at: Mapped[datetime | None]


class InboxRow(Base):
    """``inbox``: received messages that have no seq yet."""

    __tablename__ = "inbox"
    __table_args__ = (Index(None, "agent_id", "priority", "received_at"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id", ondelete=_CASCADE))
    content: Mapped[list[Any]]
    origin: Mapped[dict[str, Any]]
    received_at: Mapped[datetime]
    priority: Mapped[int] = mapped_column(default=0)


class MessageRow(Base):
    """``messages`` (layer 2); ``pk`` is the stable rowid the FTS index points at."""

    __tablename__ = "messages"
    __table_args__ = (
        UniqueConstraint("agent_id", "seq"),
        Index("ix_messages_origin_source", "origin_source_type", "origin_source_id"),
        CheckConstraint(
            "(origin_source_type IS NULL) = (origin_source_id IS NULL)", name="origin_all_or_none"
        ),
    )

    pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(unique=True)
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id", ondelete=_CASCADE))
    seq: Mapped[int]
    run_id: Mapped[str | None] = mapped_column(
        ForeignKey("runs.id", ondelete=_SET_NULL), index=True
    )
    step: Mapped[int | None]
    role: Mapped[str]
    content: Mapped[list[Any]]
    search_text: Mapped[str] = mapped_column(Text)
    origin_source_type: Mapped[str | None]
    origin_source_id: Mapped[str | None]
    origin_targets: Mapped[list[Any] | None]
    trigger_id: Mapped[str | None]
    conversation_id: Mapped[str | None]
    hop: Mapped[int | None]
    reply_to: Mapped[str | None]
    origin_message_id: Mapped[str | None] = mapped_column(index=True)
    origin_source_name: Mapped[str | None]
    expects_reply: Mapped[bool | None]
    usage: Mapped[dict[str, Any] | None]
    cost_usd: Mapped[float | None] = mapped_column(Float)
    stop_reason: Mapped[str | None]
    created_at: Mapped[datetime]


class ToolCallRow(Base):
    """``tool_calls``: operational state and audit log of every tool call."""

    __tablename__ = "tool_calls"
    __table_args__ = (UniqueConstraint("agent_id", "call_id"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id", ondelete=_CASCADE))
    run_id: Mapped[str | None] = mapped_column(
        ForeignKey("runs.id", ondelete=_SET_NULL), index=True
    )
    step: Mapped[int | None]
    call_id: Mapped[str]
    request_message_id: Mapped[str] = mapped_column(ForeignKey("messages.id", ondelete=_CASCADE))
    result_message_id: Mapped[str | None] = mapped_column(
        ForeignKey("messages.id", ondelete=_SET_NULL)
    )
    tool_name: Mapped[str]
    input: Mapped[dict[str, Any]]
    reason: Mapped[str | None] = mapped_column(Text)
    run_in_background: Mapped[bool]
    status: Mapped[str] = mapped_column(index=True)
    decision: Mapped[str | None]
    decided_by: Mapped[str | None]
    asked_by: Mapped[list[Any]]
    result_content: Mapped[dict[str, Any] | None]
    is_error: Mapped[bool | None]
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata")
    created_at: Mapped[datetime]
    started_at: Mapped[datetime | None]
    finished_at: Mapped[datetime | None]
    duration_ms: Mapped[int | None]


class PermissionRequestRow(Base):
    """``permission_requests``: every approval request; closed rows stay as a log."""

    __tablename__ = "permission_requests"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    agent_id: Mapped[str | None] = mapped_column(
        ForeignKey("agents.id", ondelete=_CASCADE), index=True
    )
    run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.id", ondelete=_SET_NULL))
    subject: Mapped[str]
    tool_call_id: Mapped[str | None] = mapped_column(ForeignKey("tool_calls.id", ondelete=_CASCADE))
    call_id: Mapped[str | None]
    tool: Mapped[str | None]
    reason: Mapped[str | None] = mapped_column(Text)
    input: Mapped[dict[str, Any]]
    risk: Mapped[str | None]
    preview: Mapped[dict[str, Any] | None]
    asked_by: Mapped[list[Any]]
    hook_reasons: Mapped[list[Any]]
    summary: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(index=True)
    decision: Mapped[str | None]
    scope: Mapped[str | None]
    deny_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime]
    answered_at: Mapped[datetime | None]


class BlobRow(Base):
    """``blobs``: the index of the blob store."""

    __tablename__ = "blobs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    kind: Mapped[str]
    sha256: Mapped[str | None] = mapped_column(index=True)
    path: Mapped[str]
    size_bytes: Mapped[int]
    mime: Mapped[str | None]
    meta: Mapped[dict[str, Any]]
    agent_id: Mapped[str | None] = mapped_column(ForeignKey("agents.id", ondelete=_SET_NULL))
    created_at: Mapped[datetime]
    expires_at: Mapped[datetime | None] = mapped_column(index=True)


class BackgroundTaskRow(Base):
    """``background_tasks``."""

    __tablename__ = "background_tasks"
    __table_args__ = (Index(None, "agent_id", "status"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id", ondelete=_CASCADE))
    tool_call_id: Mapped[str] = mapped_column(
        ForeignKey("tool_calls.id", ondelete=_CASCADE), unique=True
    )
    call_id: Mapped[str]
    tool: Mapped[str]
    status: Mapped[str]
    output_blob: Mapped[str | None] = mapped_column(ForeignKey("blobs.id", ondelete=_SET_NULL))
    interrupted: Mapped[bool]
    result_summary: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    notified_at: Mapped[datetime | None]
    started_at: Mapped[datetime]
    ended_at: Mapped[datetime | None]


class ToolOutputRow(Base):
    """``tool_outputs``: full copies of truncated tool results."""

    __tablename__ = "tool_outputs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id", ondelete=_CASCADE))
    tool_call_id: Mapped[str] = mapped_column(
        ForeignKey("tool_calls.id", ondelete=_CASCADE), index=True
    )
    blob_id: Mapped[str] = mapped_column(ForeignKey("blobs.id", ondelete=_RESTRICT))
    token_count: Mapped[int | None]
    created_at: Mapped[datetime]


class DeletedFileRow(Base):
    """``deleted_files``: metadata of a ``delete_file`` backup."""

    __tablename__ = "deleted_files"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    agent_id: Mapped[str | None] = mapped_column(
        ForeignKey("agents.id", ondelete=_SET_NULL), index=True
    )
    tool_call_id: Mapped[str | None] = mapped_column(
        ForeignKey("tool_calls.id", ondelete=_SET_NULL)
    )
    blob_id: Mapped[str] = mapped_column(ForeignKey("blobs.id", ondelete=_RESTRICT))
    original_path: Mapped[str] = mapped_column(Text)
    size_bytes: Mapped[int]
    file_mtime: Mapped[datetime | None]
    reason: Mapped[str] = mapped_column(Text)
    permanent: Mapped[bool]
    trashed: Mapped[bool]
    deleted_at: Mapped[datetime]
    restored_at: Mapped[datetime | None]


class SummaryRow(Base):
    """``summaries`` (layer 3); nothing is deleted, older rows are superseded."""

    __tablename__ = "summaries"
    __table_args__ = (
        CheckConstraint("from_seq <= to_seq", name="seq_range"),
        Index(None, "agent_id", "from_seq"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id", ondelete=_CASCADE))
    from_seq: Mapped[int]
    to_seq: Mapped[int]
    text: Mapped[str] = mapped_column(Text)
    level: Mapped[int]
    superseded_by: Mapped[str | None] = mapped_column(
        ForeignKey("summaries.id", ondelete=_SET_NULL)
    )
    reason: Mapped[str]
    model_used: Mapped[str | None]
    token_count: Mapped[int | None]
    created_at: Mapped[datetime]


class ModelCallRow(Base):
    """``model_calls``: the cost ledger the budget is computed from."""

    __tablename__ = "model_calls"
    __table_args__ = (Index(None, "agent_id", "created_at"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    agent_id: Mapped[str | None] = mapped_column(ForeignKey("agents.id", ondelete=_SET_NULL))
    run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.id", ondelete=_SET_NULL))
    step: Mapped[int | None]
    message_id: Mapped[str | None] = mapped_column(ForeignKey("messages.id", ondelete=_SET_NULL))
    purpose: Mapped[str]
    model_alias: Mapped[str]
    model_used: Mapped[str]
    provider: Mapped[str]
    input_tokens: Mapped[int]
    output_tokens: Mapped[int]
    cache_read_tokens: Mapped[int]
    cache_write_tokens: Mapped[int]
    cost_usd: Mapped[float] = mapped_column(Float)
    latency_ms: Mapped[int | None]
    stop_reason: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(index=True)


class AgentEventRow(Base):
    """``agent_events``: persisted display events that are not messages."""

    __tablename__ = "agent_events"
    __table_args__ = (Index(None, "agent_id", "after_seq"),)

    pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id", ondelete=_CASCADE))
    run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.id", ondelete=_SET_NULL))
    after_seq: Mapped[int]
    type: Mapped[str]
    payload: Mapped[dict[str, Any]]
    created_at: Mapped[datetime]


class HarnessMetaRow(Base):
    """``harness_meta``: small key/value facts about this database."""

    __tablename__ = "harness_meta"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[Any] = mapped_column(JSON)
    updated_at: Mapped[datetime]
