"""File-backed ``BlobStore`` with its index in the ``blobs`` table (13 §1, §2.14)."""

import asyncio
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import JsonValue
from sqlalchemy import delete, func, select

from harness.core.consts import ID_PREFIX_BLOB, BlobKind
from harness.core.errors import MemoryStoreError
from harness.core.records import DeletedFile
from harness.core.types import AgentId, BackupId, Clock, IdGenerator
from harness.memory.consts import (
    BLOB_DEFAULT_FILENAME,
    BLOB_DELETED_DIRNAME,
    BLOB_HASH_PREFIX_LENGTH,
    BLOB_TASK_SUFFIX,
    BLOB_TASKS_DIRNAME,
    DELETED_RETENTION_DAYS,
    META_AGENT_ID,
    META_FILENAME,
    META_MIME,
    META_TASK_ID,
    MSG_BACKUP_NOT_FOUND,
    MSG_BLOB_NOT_FOUND,
    MSG_TASK_ID_REQUIRED,
)
from harness.memory.db.engine import Database
from harness.memory.db.models import BlobRow, DeletedFileRow

_HASHED_KINDS = frozenset({BlobKind.ATTACHMENT, BlobKind.TOOL_OUTPUT})


def _text(meta: dict[str, JsonValue], key: str) -> str | None:
    value = meta.get(key)
    return value if isinstance(value, str) and value else None


class FileBlobStore:
    """Stores blob bytes under ``blobs/`` and one index row per ``put``."""

    def __init__(self, db: Database, blobs_dir: Path, clock: Clock, ids: IdGenerator) -> None:
        self._db = db
        self._dir = blobs_dir
        self._clock = clock
        self._ids = ids

    def _relative_path(
        self, kind: BlobKind, blob_id: str, digest: str, meta: dict[str, JsonValue]
    ) -> str:
        if kind in _HASHED_KINDS:
            return f"{digest[:BLOB_HASH_PREFIX_LENGTH]}/{digest}"
        if kind is BlobKind.DELETED_BACKUP:
            name = Path(_text(meta, META_FILENAME) or BLOB_DEFAULT_FILENAME).name
            return f"{BLOB_DELETED_DIRNAME}/{blob_id}/{name}"
        task_id = _text(meta, META_TASK_ID)
        if task_id is None:
            raise MemoryStoreError(MSG_TASK_ID_REQUIRED)
        return f"{BLOB_TASKS_DIRNAME}/{Path(task_id).name}{BLOB_TASK_SUFFIX}"

    async def put(self, data: bytes, *, kind: BlobKind, meta: dict[str, JsonValue]) -> str:
        """Store ``data`` and return the new blob id.

        ``meta`` may carry ``agent_id``, ``filename``, ``mime`` and (for task output) ``task_id``.
        """
        blob_id = self._ids.new(ID_PREFIX_BLOB)
        digest = hashlib.sha256(data).hexdigest()
        relative = self._relative_path(kind, blob_id, digest, meta)
        await asyncio.to_thread(self._write_file, self._dir / relative, data)
        now = self._clock.now()
        expires = (
            now + timedelta(days=DELETED_RETENTION_DAYS)
            if kind is BlobKind.DELETED_BACKUP
            else None
        )
        async with self._db.write() as session:
            session.add(
                BlobRow(
                    id=blob_id,
                    kind=kind.value,
                    sha256=digest,
                    path=relative,
                    size_bytes=len(data),
                    mime=_text(meta, META_MIME),
                    meta=dict(meta),
                    agent_id=_text(meta, META_AGENT_ID),
                    created_at=now,
                    expires_at=expires,
                )
            )
        return blob_id

    @staticmethod
    def _write_file(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    async def get(self, blob_id: str) -> bytes:
        """Return the bytes of a blob."""
        async with self._db.read() as session:
            row = await session.get(BlobRow, blob_id)
        if row is None:
            raise MemoryStoreError(MSG_BLOB_NOT_FOUND.format(blob_id=blob_id))
        try:
            return await asyncio.to_thread((self._dir / row.path).read_bytes)
        except OSError as error:
            raise MemoryStoreError(MSG_BLOB_NOT_FOUND.format(blob_id=blob_id)) from error

    async def delete(self, blob_id: str) -> None:
        """Remove a blob row, and its file once no other row shares it."""
        async with self._db.write() as session:
            row = await session.get(BlobRow, blob_id)
            if row is None:
                raise MemoryStoreError(MSG_BLOB_NOT_FOUND.format(blob_id=blob_id))
            path = row.path
            await session.delete(row)
            await session.flush()
            shared = await session.execute(
                select(func.count()).select_from(BlobRow).where(BlobRow.path == path)
            )
            orphaned = shared.scalar_one() == 0
        if orphaned:
            await asyncio.to_thread(self._remove_file, self._dir / path)

    def _remove_file(self, path: Path) -> None:
        path.unlink(missing_ok=True)
        parent = path.parent
        if parent != self._dir and parent.exists() and not any(parent.iterdir()):
            parent.rmdir()

    async def purge_expired(self, now: datetime) -> int:
        """Delete expired deletion backups (rows, ``deleted_files`` rows, files)."""
        async with self._db.write() as session:
            rows = (await session.scalars(select(BlobRow).where(BlobRow.expires_at <= now))).all()
            paths = [row.path for row in rows]
            ids = [row.id for row in rows]
            if ids:
                await session.execute(delete(DeletedFileRow).where(DeletedFileRow.blob_id.in_(ids)))
                await session.execute(delete(BlobRow).where(BlobRow.id.in_(ids)))
        for path in paths:
            await asyncio.to_thread(self._remove_file, self._dir / path)
        return len(ids)

    async def save_deleted(self, record: DeletedFile) -> None:
        """Insert or update the ``deleted_files`` row of a deletion backup."""
        values = record.model_dump()
        async with self._db.write() as session:
            row = await session.get(DeletedFileRow, record.id)
            if row is None:
                session.add(DeletedFileRow(**values))
                return
            for name, value in values.items():
                setattr(row, name, value)

    async def get_deleted(self, backup_id: BackupId) -> DeletedFile:
        """Return the record of a deletion backup.

        Raises:
            MemoryStoreError: No such backup (never made, or already purged).
        """
        async with self._db.read() as session:
            row = await session.get(DeletedFileRow, backup_id)
            if row is None:
                raise MemoryStoreError(MSG_BACKUP_NOT_FOUND.format(backup_id=backup_id))
            return DeletedFile.model_validate(row, from_attributes=True)

    async def ids_for_agent(self, agent_id: AgentId, kinds: frozenset[BlobKind]) -> list[str]:
        """Return the agent's blob ids of the given kinds (used when deleting the agent)."""
        async with self._db.read() as session:
            query = select(BlobRow.id).where(
                BlobRow.agent_id == agent_id, BlobRow.kind.in_([kind.value for kind in kinds])
            )
            return list((await session.scalars(query)).all())
