"""Global fixtures: clock, ids, tmp_harness_home, log_capture."""

import asyncio
import shutil
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from loguru import logger

from harness.core.logging import redact_record
from harness.core.utils.paths import HarnessHome
from harness.memory import Database, FileBlobStore, Memory, SqliteMemoryStore
from harness.memory.db.migrate import upgrade
from tests.fakes.clock import FakeClock
from tests.fakes.ids import FakeIds


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark tests by directory so ``-m unit`` / ``-m integration`` work without decorators."""
    for item in items:
        parts = Path(str(item.fspath)).parts
        for marker in ("unit", "integration", "live"):
            if marker in parts:
                item.add_marker(getattr(pytest.mark, marker))


@pytest.fixture
def clock() -> FakeClock:
    """A controllable clock."""
    return FakeClock()


@pytest.fixture
def ids() -> FakeIds:
    """A deterministic id generator."""
    return FakeIds()


@pytest.fixture
def tmp_harness_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> HarnessHome:
    """A temporary directory standing in for ``~/.harness``."""
    home = HarnessHome(tmp_path / "harness-home")
    home.ensure()
    monkeypatch.setenv("HARNESS_HOME", str(home.root))
    return home


@pytest.fixture
def log_capture() -> Iterator[list[str]]:
    """Collect formatted log messages (after secret redaction)."""
    records: list[str] = []
    logger.configure(patcher=redact_record)
    sink_id = logger.add(lambda message: records.append(str(message)), level="DEBUG")
    yield records
    logger.remove(sink_id)


@pytest.fixture(scope="session")
def db_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A database migrated to head once per session; tests copy it (13 §5.6)."""
    path = tmp_path_factory.mktemp("db-template") / "template.db"
    asyncio.run(upgrade(path))
    return path


@pytest.fixture
async def memory(
    tmp_harness_home: HarnessHome, db_template: Path, clock: FakeClock, ids: FakeIds
) -> AsyncIterator[Memory]:
    """The real memory component over a private copy of the migrated template."""
    shutil.copyfile(db_template, tmp_harness_home.db_file)
    db = Database(tmp_harness_home.db_file)
    blobs = FileBlobStore(db, tmp_harness_home.blobs_dir, clock, ids)
    opened = Memory(db=db, store=SqliteMemoryStore(db, clock, ids, blobs), blobs=blobs)
    yield opened
    await opened.close()
