"""The async SQLite engine: PRAGMAs on every connection, and one writer at a time."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

from sqlalchemy import URL, event, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import ConnectionPoolEntry, NullPool

from harness.core.errors import MemoryStoreError
from harness.memory.consts import MSG_DB_FAILED, PRAGMA_DATA_VERSION, PRAGMAS, SQLITE_DRIVER


class _Cursor(Protocol):
    def execute(self, sql: str) -> object: ...
    def close(self) -> None: ...


class _DbapiConnection(Protocol):
    def cursor(self) -> _Cursor: ...


def sqlite_url(path: Path) -> URL:
    """Return the SQLAlchemy URL of a SQLite file."""
    return URL.create(SQLITE_DRIVER, database=str(path))


def _apply_pragmas(dbapi_connection: _DbapiConnection, _record: ConnectionPoolEntry) -> None:
    cursor = dbapi_connection.cursor()
    try:
        for pragma in PRAGMAS:
            cursor.execute(pragma)
    finally:
        cursor.close()


def create_engine(path: Path, *, pragmas: bool = True) -> AsyncEngine:
    """Create an engine for ``path``.

    ``foreign_keys`` is per connection, so the PRAGMAs are set in the ``connect`` event.
    Migrations pass ``pragmas=False`` because they must run with foreign keys off.
    """
    engine = create_async_engine(sqlite_url(path), poolclass=NullPool)
    if pragmas:
        event.listen(engine.sync_engine, "connect", _apply_pragmas)
    return engine


class Database:
    """Session factory over one SQLite file.

    Writes are serialised with a lock: the core is a single process (``core.lock``), and
    one writer at a time avoids ``SQLITE_BUSY`` on read-to-write upgrades.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.engine = create_engine(path)
        self._sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        self._write_lock = asyncio.Lock()
        self._watcher: AsyncConnection | None = None

    @asynccontextmanager
    async def read(self) -> AsyncGenerator[AsyncSession]:
        """Open a session for reading."""
        async with self._sessions() as session:
            try:
                yield session
            except SQLAlchemyError as error:
                raise MemoryStoreError(MSG_DB_FAILED.format(error=error)) from error

    @asynccontextmanager
    async def write(self) -> AsyncGenerator[AsyncSession]:
        """Open one write transaction; commits on success, rolls back on error."""
        async with self._write_lock, self._sessions() as session:
            try:
                async with session.begin():
                    yield session
            except SQLAlchemyError as error:
                raise MemoryStoreError(MSG_DB_FAILED.format(error=error)) from error

    async def data_version(self) -> int:
        """Return ``PRAGMA data_version`` as seen by one long-lived connection.

        The value changes only when a different connection (of this or another process)
        commits, so the watching connection is kept open between calls.
        """
        try:
            if self._watcher is None:
                self._watcher = await self.engine.connect()
            found = await self._watcher.execute(text(PRAGMA_DATA_VERSION))
            version = int(found.scalar_one())
            await self._watcher.rollback()
        except SQLAlchemyError as error:
            raise MemoryStoreError(MSG_DB_FAILED.format(error=error)) from error
        return version

    async def close(self) -> None:
        """Dispose the engine."""
        if self._watcher is not None:
            await self._watcher.close()
            self._watcher = None
        await self.engine.dispose()
