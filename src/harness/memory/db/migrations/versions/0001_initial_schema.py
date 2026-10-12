"""initial schema

Revision ID: 0001
Revises: -
Create Date: 2026-10-11 21:38:05.026353
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from harness.memory.db.types import UTCDateTime

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_FTS_CREATE = (
    """
    CREATE VIRTUAL TABLE messages_fts USING fts5(
        search_text,
        content='messages',
        content_rowid='pk',
        tokenize='unicode61'
    )
    """,
    """
    CREATE TRIGGER messages_fts_ai AFTER INSERT ON messages BEGIN
        INSERT INTO messages_fts(rowid, search_text) VALUES (new.pk, new.search_text);
    END
    """,
    """
    CREATE TRIGGER messages_fts_ad AFTER DELETE ON messages BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, search_text)
        VALUES ('delete', old.pk, old.search_text);
    END
    """,
    """
    CREATE TRIGGER messages_fts_au AFTER UPDATE OF search_text ON messages BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, search_text)
        VALUES ('delete', old.pk, old.search_text);
        INSERT INTO messages_fts(rowid, search_text) VALUES (new.pk, new.search_text);
    END
    """,
)
_FTS_DROP = (
    "DROP TRIGGER IF EXISTS messages_fts_au",
    "DROP TRIGGER IF EXISTS messages_fts_ad",
    "DROP TRIGGER IF EXISTS messages_fts_ai",
    "DROP TABLE IF EXISTS messages_fts",
)


def upgrade() -> None:
    """Apply the change."""
    op.create_table(
        "agents",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("system_prompt", sa.Text(), nullable=False),
        sa.Column("tools", sa.JSON(), nullable=False),
        sa.Column("template", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("settings", sa.JSON(), nullable=False),
        sa.Column("config_overrides", sa.JSON(), nullable=False),
        sa.Column("pending_hook_context", sa.JSON(), nullable=False),
        sa.Column("last_seq", sa.Integer(), nullable=False),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_agents")),
    )
    with op.batch_alter_table("agents", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_agents_status"), ["status"], unique=False)

    op.create_table(
        "harness_meta",
        sa.Column("key", sa.String(), nullable=False),
        sa.Column("value", sa.JSON(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("key", name=op.f("pk_harness_meta")),
    )
    op.create_table(
        "blobs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("sha256", sa.String(), nullable=True),
        sa.Column("path", sa.String(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("mime", sa.String(), nullable=True),
        sa.Column("meta", sa.JSON(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("expires_at", UTCDateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["agent_id"], ["agents.id"], name=op.f("fk_blobs_agent_id_agents"), ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_blobs")),
    )
    with op.batch_alter_table("blobs", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_blobs_expires_at"), ["expires_at"], unique=False)
        batch_op.create_index(batch_op.f("ix_blobs_sha256"), ["sha256"], unique=False)

    op.create_table(
        "inbox",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("content", sa.JSON(), nullable=False),
        sa.Column("origin", sa.JSON(), nullable=False),
        sa.Column("received_at", UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["agent_id"], ["agents.id"], name=op.f("fk_inbox_agent_id_agents"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_inbox")),
    )
    with op.batch_alter_table("inbox", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_inbox_agent_id_received_at"), ["agent_id", "received_at"], unique=False
        )

    op.create_table(
        "runs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("first_seq", sa.Integer(), nullable=False),
        sa.Column("step_count", sa.Integer(), nullable=False),
        sa.Column("stop_reason", sa.String(), nullable=True),
        sa.Column("retrieved_memory_ids", sa.JSON(), nullable=False),
        sa.Column("started_at", UTCDateTime(), nullable=False),
        sa.Column("ended_at", UTCDateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["agent_id"], ["agents.id"], name=op.f("fk_runs_agent_id_agents"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_runs")),
    )
    with op.batch_alter_table("runs", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_runs_agent_id_ended_at"), ["agent_id", "ended_at"], unique=False
        )

    op.create_table(
        "summaries",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("from_seq", sa.Integer(), nullable=False),
        sa.Column("to_seq", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("level", sa.Integer(), nullable=False),
        sa.Column("superseded_by", sa.String(), nullable=True),
        sa.Column("reason", sa.String(), nullable=False),
        sa.Column("model_used", sa.String(), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.CheckConstraint("from_seq <= to_seq", name=op.f("ck_summaries_seq_range")),
        sa.ForeignKeyConstraint(
            ["agent_id"],
            ["agents.id"],
            name=op.f("fk_summaries_agent_id_agents"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["superseded_by"],
            ["summaries.id"],
            name=op.f("fk_summaries_superseded_by_summaries"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_summaries")),
    )
    with op.batch_alter_table("summaries", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_summaries_agent_id_from_seq"), ["agent_id", "from_seq"], unique=False
        )

    op.create_table(
        "agent_events",
        sa.Column("pk", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("run_id", sa.String(), nullable=True),
        sa.Column("after_seq", sa.Integer(), nullable=False),
        sa.Column("type", sa.String(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["agent_id"],
            ["agents.id"],
            name=op.f("fk_agent_events_agent_id_agents"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"], ["runs.id"], name=op.f("fk_agent_events_run_id_runs"), ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("pk", name=op.f("pk_agent_events")),
    )
    with op.batch_alter_table("agent_events", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_agent_events_agent_id_after_seq"),
            ["agent_id", "after_seq"],
            unique=False,
        )

    op.create_table(
        "messages",
        sa.Column("pk", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.String(), nullable=True),
        sa.Column("step", sa.Integer(), nullable=True),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("content", sa.JSON(), nullable=False),
        sa.Column("search_text", sa.Text(), nullable=False),
        sa.Column("origin_source_type", sa.String(), nullable=True),
        sa.Column("origin_source_id", sa.String(), nullable=True),
        sa.Column("origin_targets", sa.JSON(), nullable=True),
        sa.Column("trigger_id", sa.String(), nullable=True),
        sa.Column("conversation_id", sa.String(), nullable=True),
        sa.Column("hop", sa.Integer(), nullable=True),
        sa.Column("reply_to", sa.String(), nullable=True),
        sa.Column("usage", sa.JSON(), nullable=True),
        sa.Column("cost_usd", sa.Float(), nullable=True),
        sa.Column("stop_reason", sa.String(), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.CheckConstraint(
            "(origin_source_type IS NULL) = (origin_source_id IS NULL)",
            name=op.f("ck_messages_origin_all_or_none"),
        ),
        sa.ForeignKeyConstraint(
            ["agent_id"],
            ["agents.id"],
            name=op.f("fk_messages_agent_id_agents"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"], ["runs.id"], name=op.f("fk_messages_run_id_runs"), ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("pk", name=op.f("pk_messages")),
        sa.UniqueConstraint("agent_id", "seq", name=op.f("uq_messages_agent_id_seq")),
        sa.UniqueConstraint("id", name=op.f("uq_messages_id")),
    )
    with op.batch_alter_table("messages", schema=None) as batch_op:
        batch_op.create_index(
            "ix_messages_origin_source", ["origin_source_type", "origin_source_id"], unique=False
        )
        batch_op.create_index(batch_op.f("ix_messages_run_id"), ["run_id"], unique=False)

    op.create_table(
        "model_calls",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=True),
        sa.Column("run_id", sa.String(), nullable=True),
        sa.Column("step", sa.Integer(), nullable=True),
        sa.Column("message_id", sa.String(), nullable=True),
        sa.Column("purpose", sa.String(), nullable=False),
        sa.Column("model_alias", sa.String(), nullable=False),
        sa.Column("model_used", sa.String(), nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("cache_read_tokens", sa.Integer(), nullable=False),
        sa.Column("cache_write_tokens", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Float(), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("stop_reason", sa.String(), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["agent_id"],
            ["agents.id"],
            name=op.f("fk_model_calls_agent_id_agents"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["message_id"],
            ["messages.id"],
            name=op.f("fk_model_calls_message_id_messages"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"], ["runs.id"], name=op.f("fk_model_calls_run_id_runs"), ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_model_calls")),
    )
    with op.batch_alter_table("model_calls", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_model_calls_agent_id_created_at"),
            ["agent_id", "created_at"],
            unique=False,
        )
        batch_op.create_index(batch_op.f("ix_model_calls_created_at"), ["created_at"], unique=False)

    op.create_table(
        "tool_calls",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("run_id", sa.String(), nullable=True),
        sa.Column("step", sa.Integer(), nullable=True),
        sa.Column("call_id", sa.String(), nullable=False),
        sa.Column("request_message_id", sa.String(), nullable=False),
        sa.Column("result_message_id", sa.String(), nullable=True),
        sa.Column("tool_name", sa.String(), nullable=False),
        sa.Column("input", sa.JSON(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("run_in_background", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("decision", sa.String(), nullable=True),
        sa.Column("decided_by", sa.String(), nullable=True),
        sa.Column("asked_by", sa.JSON(), nullable=False),
        sa.Column("result_content", sa.JSON(), nullable=True),
        sa.Column("is_error", sa.Boolean(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("started_at", UTCDateTime(), nullable=True),
        sa.Column("finished_at", UTCDateTime(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["agent_id"],
            ["agents.id"],
            name=op.f("fk_tool_calls_agent_id_agents"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["request_message_id"],
            ["messages.id"],
            name=op.f("fk_tool_calls_request_message_id_messages"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["result_message_id"],
            ["messages.id"],
            name=op.f("fk_tool_calls_result_message_id_messages"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"], ["runs.id"], name=op.f("fk_tool_calls_run_id_runs"), ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_tool_calls")),
        sa.UniqueConstraint("agent_id", "call_id", name=op.f("uq_tool_calls_agent_id_call_id")),
    )
    with op.batch_alter_table("tool_calls", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_tool_calls_run_id"), ["run_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_tool_calls_status"), ["status"], unique=False)

    op.create_table(
        "background_tasks",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("tool_call_id", sa.String(), nullable=False),
        sa.Column("call_id", sa.String(), nullable=False),
        sa.Column("tool", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("output_blob", sa.String(), nullable=True),
        sa.Column("interrupted", sa.Boolean(), nullable=False),
        sa.Column("result_summary", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("notified_at", UTCDateTime(), nullable=True),
        sa.Column("started_at", UTCDateTime(), nullable=False),
        sa.Column("ended_at", UTCDateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["agent_id"],
            ["agents.id"],
            name=op.f("fk_background_tasks_agent_id_agents"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["output_blob"],
            ["blobs.id"],
            name=op.f("fk_background_tasks_output_blob_blobs"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["tool_call_id"],
            ["tool_calls.id"],
            name=op.f("fk_background_tasks_tool_call_id_tool_calls"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_background_tasks")),
        sa.UniqueConstraint("tool_call_id", name=op.f("uq_background_tasks_tool_call_id")),
    )
    with op.batch_alter_table("background_tasks", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_background_tasks_agent_id_status"), ["agent_id", "status"], unique=False
        )

    op.create_table(
        "deleted_files",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=True),
        sa.Column("tool_call_id", sa.String(), nullable=True),
        sa.Column("blob_id", sa.String(), nullable=False),
        sa.Column("original_path", sa.Text(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("file_mtime", UTCDateTime(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("permanent", sa.Boolean(), nullable=False),
        sa.Column("trashed", sa.Boolean(), nullable=False),
        sa.Column("deleted_at", UTCDateTime(), nullable=False),
        sa.Column("restored_at", UTCDateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["agent_id"],
            ["agents.id"],
            name=op.f("fk_deleted_files_agent_id_agents"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["blob_id"],
            ["blobs.id"],
            name=op.f("fk_deleted_files_blob_id_blobs"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["tool_call_id"],
            ["tool_calls.id"],
            name=op.f("fk_deleted_files_tool_call_id_tool_calls"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_deleted_files")),
    )
    with op.batch_alter_table("deleted_files", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_deleted_files_agent_id"), ["agent_id"], unique=False)

    op.create_table(
        "permission_requests",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=True),
        sa.Column("run_id", sa.String(), nullable=True),
        sa.Column("subject", sa.String(), nullable=False),
        sa.Column("tool_call_id", sa.String(), nullable=True),
        sa.Column("call_id", sa.String(), nullable=True),
        sa.Column("tool", sa.String(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("input", sa.JSON(), nullable=False),
        sa.Column("risk", sa.String(), nullable=True),
        sa.Column("preview", sa.JSON(), nullable=True),
        sa.Column("asked_by", sa.JSON(), nullable=False),
        sa.Column("hook_reasons", sa.JSON(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("decision", sa.String(), nullable=True),
        sa.Column("scope", sa.String(), nullable=True),
        sa.Column("deny_message", sa.Text(), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("answered_at", UTCDateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["agent_id"],
            ["agents.id"],
            name=op.f("fk_permission_requests_agent_id_agents"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["runs.id"],
            name=op.f("fk_permission_requests_run_id_runs"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["tool_call_id"],
            ["tool_calls.id"],
            name=op.f("fk_permission_requests_tool_call_id_tool_calls"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_permission_requests")),
    )
    with op.batch_alter_table("permission_requests", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_permission_requests_agent_id"), ["agent_id"], unique=False
        )
        batch_op.create_index(batch_op.f("ix_permission_requests_status"), ["status"], unique=False)

    op.create_table(
        "tool_outputs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("tool_call_id", sa.String(), nullable=False),
        sa.Column("blob_id", sa.String(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["agent_id"],
            ["agents.id"],
            name=op.f("fk_tool_outputs_agent_id_agents"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["blob_id"],
            ["blobs.id"],
            name=op.f("fk_tool_outputs_blob_id_blobs"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["tool_call_id"],
            ["tool_calls.id"],
            name=op.f("fk_tool_outputs_tool_call_id_tool_calls"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_tool_outputs")),
    )
    with op.batch_alter_table("tool_outputs", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_tool_outputs_tool_call_id"), ["tool_call_id"], unique=False
        )

    for statement in _FTS_CREATE:
        op.execute(statement)


def downgrade() -> None:
    """Revert the change."""
    for statement in _FTS_DROP:
        op.execute(statement)
    with op.batch_alter_table("tool_outputs", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_tool_outputs_tool_call_id"))

    op.drop_table("tool_outputs")
    with op.batch_alter_table("permission_requests", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_permission_requests_status"))
        batch_op.drop_index(batch_op.f("ix_permission_requests_agent_id"))

    op.drop_table("permission_requests")
    with op.batch_alter_table("deleted_files", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_deleted_files_agent_id"))

    op.drop_table("deleted_files")
    with op.batch_alter_table("background_tasks", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_background_tasks_agent_id_status"))

    op.drop_table("background_tasks")
    with op.batch_alter_table("tool_calls", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_tool_calls_status"))
        batch_op.drop_index(batch_op.f("ix_tool_calls_run_id"))

    op.drop_table("tool_calls")
    with op.batch_alter_table("model_calls", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_model_calls_created_at"))
        batch_op.drop_index(batch_op.f("ix_model_calls_agent_id_created_at"))

    op.drop_table("model_calls")
    with op.batch_alter_table("messages", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_messages_run_id"))
        batch_op.drop_index("ix_messages_origin_source")

    op.drop_table("messages")
    with op.batch_alter_table("agent_events", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_agent_events_agent_id_after_seq"))

    op.drop_table("agent_events")
    with op.batch_alter_table("summaries", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_summaries_agent_id_from_seq"))

    op.drop_table("summaries")
    with op.batch_alter_table("runs", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_runs_agent_id_ended_at"))

    op.drop_table("runs")
    with op.batch_alter_table("inbox", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_inbox_agent_id_received_at"))

    op.drop_table("inbox")
    with op.batch_alter_table("blobs", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_blobs_sha256"))
        batch_op.drop_index(batch_op.f("ix_blobs_expires_at"))

    op.drop_table("blobs")
    op.drop_table("harness_meta")
    with op.batch_alter_table("agents", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_agents_status"))

    op.drop_table("agents")
