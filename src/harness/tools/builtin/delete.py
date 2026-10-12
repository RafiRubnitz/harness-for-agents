"""``delete_file`` and ``restore_deleted``: deletion with a backup (08 §4.1a)."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from send2trash import send2trash

from harness.core.consts import (
    ID_PREFIX_BACKUP,
    META_BACKUP_ID,
    BlobKind,
    Decision,
    RiskLevel,
    ToolSource,
)
from harness.core.errors import MemoryStoreError
from harness.core.records import ApprovalPreview, DeletedFile, ToolOutput
from harness.core.types import AgentId, BackupId, IdGenerator
from harness.memory import BlobStore
from harness.tools import consts
from harness.tools.builtin.utils import head
from harness.tools.function_tool import FunctionTool, error_output, text_output, tool
from harness.tools.paths import resolve_path
from harness.tools.types import ToolContext, ToolTraits

_WILDCARDS = frozenset("*?")


def _problem(path: Path, raw: str) -> str | None:
    if _WILDCARDS & set(raw):
        return consts.MSG_NO_WILDCARDS.format(path=raw)
    if not path.exists():
        return consts.MSG_NOT_FOUND.format(path=path)
    if path.is_dir():
        return consts.MSG_IS_DIRECTORY.format(path=path)
    return None


def _remove(path: Path, *, permanent: bool) -> None:
    if permanent:
        path.unlink()
    else:
        send2trash(str(path))


def _restore(path: Path, data: bytes) -> bool:
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return True


class Deleter:
    """Deletes single files, keeping a backup that can be restored until it expires."""

    def __init__(self, blobs: BlobStore, ids: IdGenerator) -> None:
        self._blobs = blobs
        self._ids = ids

    async def delete(self, raw: str, ctx: ToolContext, *, permanent: bool) -> ToolOutput:
        """Back the file up, then move it to the recycle bin (or unlink it)."""
        path = resolve_path(raw, ctx.working_dir)
        problem = await asyncio.to_thread(_problem, path, raw)
        if problem is not None:
            return error_output(problem)
        data = await asyncio.to_thread(path.read_bytes)
        stat = await asyncio.to_thread(path.stat)
        blob_id = await self._blobs.put(
            data,
            kind=BlobKind.DELETED_BACKUP,
            meta={
                consts.BACKUP_META_AGENT_ID: ctx.agent_id,
                consts.BACKUP_META_FILENAME: path.name,
                consts.BACKUP_META_ORIGINAL_PATH: str(path),
            },
        )
        record = DeletedFile(
            id=BackupId(self._ids.new(ID_PREFIX_BACKUP)),
            agent_id=ctx.agent_id,
            blob_id=blob_id,
            original_path=str(path),
            size_bytes=len(data),
            file_mtime=datetime.fromtimestamp(stat.st_mtime, UTC),
            reason=ctx.reason or "",
            permanent=permanent,
            trashed=not permanent,
            deleted_at=ctx.clock.now(),
        )
        await self._blobs.save_deleted(record)
        await asyncio.to_thread(_remove, path, permanent=permanent)
        how = consts.MSG_HOW_PERMANENT if permanent else consts.MSG_HOW_TRASHED
        return text_output(
            consts.MSG_DELETED.format(path=path, how=how, backup_id=record.id),
            metadata={
                consts.META_PATH: str(path),
                consts.META_SIZE: record.size_bytes,
                consts.META_TRASHED: record.trashed,
                META_BACKUP_ID: record.id,
            },
        )

    async def preview(self, raw: str, ctx: ToolContext, *, permanent: bool) -> ApprovalPreview:
        """Describe the file about to be deleted."""
        path = resolve_path(raw, ctx.working_dir)
        if not await asyncio.to_thread(path.is_file):
            return ApprovalPreview(path=str(path), permanent=permanent)
        data = await asyncio.to_thread(path.read_bytes)
        stat = await asyncio.to_thread(path.stat)
        return ApprovalPreview(
            path=str(path),
            size=len(data),
            mtime=datetime.fromtimestamp(stat.st_mtime, UTC),
            head=head(data),
            permanent=permanent,
        )

    async def restore(self, backup_id: BackupId, agent_id: AgentId, now: datetime) -> ToolOutput:
        """Write a backup back to its original path; refuses to overwrite a file."""
        try:
            record = await self._blobs.get_deleted(backup_id)
            data = await self._blobs.get(record.blob_id)
        except MemoryStoreError:
            return error_output(consts.MSG_BACKUP_NOT_FOUND.format(backup_id=backup_id))
        if record.agent_id not in (None, agent_id):
            return error_output(consts.MSG_BACKUP_NOT_FOUND.format(backup_id=backup_id))
        path = Path(record.original_path)
        if not await asyncio.to_thread(_restore, path, data):
            return error_output(consts.MSG_RESTORE_EXISTS.format(path=path))
        await self._blobs.save_deleted(record.model_copy(update={"restored_at": now}))
        return text_output(
            consts.MSG_RESTORED.format(path=path),
            metadata={consts.META_PATH: str(path), META_BACKUP_ID: backup_id},
        )


def delete_tools(deleter: Deleter) -> list[FunctionTool]:
    """Build ``delete_file`` and ``restore_deleted`` over one ``Deleter``."""

    async def delete_preview(
        path: str, ctx: ToolContext, permanent: bool = False
    ) -> ApprovalPreview:
        return await deleter.preview(path, ctx, permanent=permanent)

    @tool(
        group=consts.GROUP_FILES,
        exclusive=True,
        risk=RiskLevel.HIGH,
        default_permission=Decision.ASK,
        source=ToolSource.BUILTIN,
        traits=ToolTraits(path_param="path", confirm_param="permanent"),
        preview=delete_preview,
    )
    async def delete_file(path: str, ctx: ToolContext, permanent: bool = False) -> ToolOutput:
        """Delete a single file. Moves it to the Recycle Bin unless permanent=True.

        A backup is kept for a while; the result carries its ``backup_id``, which
        ``restore_deleted`` accepts. Directories and wildcards are not supported.
        """
        return await deleter.delete(path, ctx, permanent=permanent)

    @tool(
        group=consts.GROUP_FILES,
        exclusive=True,
        default_permission=Decision.ASK,
        source=ToolSource.BUILTIN,
    )
    async def restore_deleted(backup_id: str, ctx: ToolContext) -> ToolOutput:
        """Restore a file removed by ``delete_file`` to its original path.

        Fails when a file already exists there.
        """
        return await deleter.restore(BackupId(backup_id), ctx.agent_id, ctx.clock.now())

    return [delete_file, restore_deleted]
