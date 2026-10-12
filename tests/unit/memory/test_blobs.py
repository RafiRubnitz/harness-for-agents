"""The file blob store and the startup sequence."""

from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import text

from harness.core.consts import BlobKind
from harness.core.errors import MemoryStoreError
from harness.core.utils.paths import HarnessHome
from harness.memory import Memory, open_memory
from harness.memory.consts import (
    BACKUP_GLOB,
    BLOB_DELETED_DIRNAME,
    DELETED_RETENTION_DAYS,
    META_CREATED_AT,
    META_LAST_APP_VERSION,
)
from harness.memory.db.migrate import head_revision
from tests.consts import AGENT_ID
from tests.fakes.clock import FakeClock
from tests.fakes.ids import FakeIds
from tests.utils import make_agent

DATA = b"payload bytes"
FILENAME = "notes.txt"
APP_VERSION = "9.9.9"
OLD_REVISION = "0000"


def _files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*") if path.is_file()]


async def test_same_content_is_stored_once(memory: Memory, tmp_harness_home: HarnessHome) -> None:
    first = await memory.blobs.put(DATA, kind=BlobKind.ATTACHMENT, meta={})
    second = await memory.blobs.put(DATA, kind=BlobKind.TOOL_OUTPUT, meta={})

    files = _files(tmp_harness_home.blobs_dir)

    assert first != second
    assert await memory.blobs.get(first) == await memory.blobs.get(second) == DATA
    assert len(files) == 1


async def test_shared_file_is_removed_with_its_last_row(
    memory: Memory, tmp_harness_home: HarnessHome
) -> None:
    first = await memory.blobs.put(DATA, kind=BlobKind.ATTACHMENT, meta={})
    second = await memory.blobs.put(DATA, kind=BlobKind.ATTACHMENT, meta={})

    await memory.blobs.delete(first)
    still_there = await memory.blobs.get(second)
    await memory.blobs.delete(second)

    assert still_there == DATA
    assert _files(tmp_harness_home.blobs_dir) == []
    with pytest.raises(MemoryStoreError):
        await memory.blobs.get(second)
    with pytest.raises(MemoryStoreError):
        await memory.blobs.delete(second)


async def test_deleted_backup_keeps_its_name_and_expires(
    memory: Memory, tmp_harness_home: HarnessHome, clock: FakeClock
) -> None:
    meta = {"filename": f"../../{FILENAME}"}
    blob_id = await memory.blobs.put(DATA, kind=BlobKind.DELETED_BACKUP, meta=meta)
    stored = tmp_harness_home.blobs_dir / BLOB_DELETED_DIRNAME / blob_id / FILENAME

    purged_early = await memory.blobs.purge_expired(clock.now())
    clock.advance(timedelta(days=DELETED_RETENTION_DAYS).total_seconds())
    purged_late = await memory.blobs.purge_expired(clock.now())

    assert purged_early == 0
    assert purged_late == 1
    assert not stored.exists()
    assert _files(tmp_harness_home.blobs_dir) == []


async def test_task_output_needs_a_task_id(memory: Memory) -> None:
    with pytest.raises(MemoryStoreError):
        await memory.blobs.put(DATA, kind=BlobKind.TASK_OUTPUT, meta={})


async def test_deleting_an_agent_removes_its_output_blobs_only(
    memory: Memory, tmp_harness_home: HarnessHome
) -> None:
    await memory.store.save_agent(make_agent())
    owner = {"agent_id": AGENT_ID}
    await memory.blobs.put(DATA, kind=BlobKind.TOOL_OUTPUT, meta=owner)
    backup = await memory.blobs.put(DATA, kind=BlobKind.DELETED_BACKUP, meta=owner)

    await memory.store.delete_agent(AGENT_ID)

    assert await memory.blobs.get(backup) == DATA
    assert len(_files(tmp_harness_home.blobs_dir)) == 1


async def test_open_memory_creates_the_database_without_a_backup(
    tmp_harness_home: HarnessHome, clock: FakeClock, ids: FakeIds
) -> None:
    opened = await open_memory(tmp_harness_home, clock, ids, app_version=APP_VERSION)
    try:
        created = await opened.store.get_meta(META_CREATED_AT)
        version = await opened.store.get_meta(META_LAST_APP_VERSION)
    finally:
        await opened.close()

    assert created == clock.now().isoformat()
    assert version == APP_VERSION
    assert list(tmp_harness_home.backups_dir.glob(BACKUP_GLOB)) == []


async def test_open_memory_backs_up_before_a_pending_migration(
    tmp_harness_home: HarnessHome, clock: FakeClock, ids: FakeIds
) -> None:
    first = await open_memory(tmp_harness_home, clock, ids, app_version=APP_VERSION)
    created = await first.store.get_meta(META_CREATED_AT)
    async with first.db.write() as session:
        await session.execute(
            text("UPDATE alembic_version SET version_num = :rev"), {"rev": OLD_REVISION}
        )
    await first.close()
    clock.advance(1)

    with pytest.raises(Exception, match=OLD_REVISION):
        await open_memory(tmp_harness_home, clock, ids, app_version=APP_VERSION)
    backups = list(tmp_harness_home.backups_dir.glob(BACKUP_GLOB))

    assert len(backups) == 1
    assert OLD_REVISION in backups[0].name
    assert head_revision() != OLD_REVISION
    assert created is not None
