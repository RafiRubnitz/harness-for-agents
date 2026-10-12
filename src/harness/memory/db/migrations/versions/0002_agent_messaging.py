"""agent messaging: description, unique names, inbox priority, message ids

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-12 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_SQL_AGENT_NAMES = "SELECT id, name FROM agents ORDER BY created_at, id"
_SQL_RENAME = "UPDATE agents SET name = :name WHERE id = :id"
_DUPLICATE_NAME = "{name}-{number}"


def _make_names_unique() -> None:
    """Rename agents that share a name, oldest keeps it (``coder``, ``coder-2``, ...)."""
    connection = op.get_bind()
    rows = connection.execute(sa.text(_SQL_AGENT_NAMES)).all()
    taken = {str(row.name) for row in rows}
    seen: set[str] = set()
    for row in rows:
        name = str(row.name)
        if name not in seen:
            seen.add(name)
            continue
        number = 2
        while _DUPLICATE_NAME.format(name=name, number=number) in taken:
            number += 1
        renamed = _DUPLICATE_NAME.format(name=name, number=number)
        taken.add(renamed)
        seen.add(renamed)
        connection.execute(sa.text(_SQL_RENAME), {"name": renamed, "id": row.id})


def upgrade() -> None:
    """Apply the change.

    Columns are added in place (no table rebuild), so the FTS triggers on ``messages``
    stay as they are.
    """
    op.add_column("agents", sa.Column("description", sa.Text(), nullable=False, server_default=""))
    _make_names_unique()
    op.create_index(op.f("ix_agents_name"), "agents", ["name"], unique=True)

    op.add_column("inbox", sa.Column("priority", sa.Integer(), nullable=False, server_default="0"))
    op.drop_index(op.f("ix_inbox_agent_id_received_at"), table_name="inbox")
    op.create_index(
        op.f("ix_inbox_agent_id_priority_received_at"),
        "inbox",
        ["agent_id", "priority", "received_at"],
        unique=False,
    )

    op.add_column("messages", sa.Column("origin_message_id", sa.String(), nullable=True))
    op.add_column("messages", sa.Column("origin_source_name", sa.String(), nullable=True))
    op.add_column("messages", sa.Column("expects_reply", sa.Boolean(), nullable=True))
    op.create_index(
        op.f("ix_messages_origin_message_id"), "messages", ["origin_message_id"], unique=False
    )


def downgrade() -> None:
    """Revert the change."""
    op.drop_index(op.f("ix_messages_origin_message_id"), table_name="messages")
    op.drop_column("messages", "expects_reply")
    op.drop_column("messages", "origin_source_name")
    op.drop_column("messages", "origin_message_id")

    op.drop_index(op.f("ix_inbox_agent_id_priority_received_at"), table_name="inbox")
    op.create_index(
        op.f("ix_inbox_agent_id_received_at"), "inbox", ["agent_id", "received_at"], unique=False
    )
    op.drop_column("inbox", "priority")

    op.drop_index(op.f("ix_agents_name"), table_name="agents")
    op.drop_column("agents", "description")
