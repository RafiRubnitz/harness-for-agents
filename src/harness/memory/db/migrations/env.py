"""Alembic environment: async engine, batch mode, foreign keys off while migrating (13 §5.4)."""

import asyncio
from pathlib import Path
from typing import Literal

from alembic import context
from sqlalchemy import Connection
from sqlalchemy.schema import SchemaItem

from harness.core.errors import MemoryStoreError
from harness.core.utils.paths import resolve_home
from harness.memory.consts import (
    FTS_SHADOW_SUFFIXES,
    FTS_TABLE,
    MSG_FK_VIOLATIONS,
    PRAGMA_FOREIGN_KEY_CHECK,
    PRAGMA_FOREIGN_KEYS_OFF,
)
from harness.memory.db.engine import create_engine
from harness.memory.db.models import Base

ObjectType = Literal[
    "schema",
    "table",
    "column",
    "index",
    "unique_constraint",
    "foreign_key_constraint",
    "check_constraint",
]

target_metadata = Base.metadata
_FTS_TABLES = frozenset(f"{FTS_TABLE}{suffix}" for suffix in FTS_SHADOW_SUFFIXES)


def include_object(
    _obj: SchemaItem,
    name: str | None,
    type_: ObjectType,
    _reflected: bool,
    _compare_to: SchemaItem | None,
) -> bool:
    """Hide the FTS5 table and its shadow tables from autogenerate."""
    return not (type_ == "table" and name in _FTS_TABLES)


def _database_path() -> Path:
    configured = context.config.attributes.get("db_path")
    return Path(str(configured)) if configured else resolve_home().db_file


def _run(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=True,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


async def _run_online() -> None:
    engine = create_engine(_database_path(), pragmas=False)
    try:
        async with engine.connect() as connection:
            # Outside a transaction: inside one the PRAGMA has no effect.
            await connection.exec_driver_sql(PRAGMA_FOREIGN_KEYS_OFF)
            await connection.commit()
            await connection.run_sync(_run)
            await connection.commit()
            rows = (await connection.exec_driver_sql(PRAGMA_FOREIGN_KEY_CHECK)).fetchall()
            if rows:
                raise MemoryStoreError(MSG_FK_VIOLATIONS.format(rows=rows))
    finally:
        await engine.dispose()


asyncio.run(_run_online())
