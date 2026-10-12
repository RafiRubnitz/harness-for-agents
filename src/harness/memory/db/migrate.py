"""Run Alembic migrations from code, with a backup before any schema change (13 §5.5)."""

import asyncio
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text

from harness.core.types import Clock
from harness.memory.consts import (
    ALEMBIC_BASE,
    ALEMBIC_FILE_TEMPLATE,
    ALEMBIC_HEAD,
    ALEMBIC_VERSION_TABLE,
    BACKUP_EMPTY_REV,
    BACKUP_GLOB,
    BACKUP_KEEP,
    BACKUP_NAME,
    BACKUP_STAMP_FORMAT,
    MIGRATIONS_DIRNAME,
    SQL_CURRENT_REVISION,
    SQL_HAS_VERSION_TABLE,
    SQL_VACUUM_INTO,
)
from harness.memory.db.engine import create_engine

MIGRATIONS_DIR = Path(__file__).parent / MIGRATIONS_DIRNAME


def alembic_config(db_path: Path) -> Config:
    """Build the Alembic config for one database file (no ``alembic.ini`` needed)."""
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR).replace("%", "%%"))
    config.set_main_option("file_template", ALEMBIC_FILE_TEMPLATE)
    config.attributes["db_path"] = str(db_path)
    return config


def head_revision() -> str | None:
    """Return the newest revision id in the migrations directory."""
    return ScriptDirectory(str(MIGRATIONS_DIR)).get_current_head()


async def current_revision(db_path: Path) -> str | None:
    """Return the revision the database is at; ``None`` for a new or empty database."""
    if not db_path.exists():
        return None
    engine = create_engine(db_path, pragmas=False)
    try:
        async with engine.connect() as connection:
            found = await connection.execute(
                text(SQL_HAS_VERSION_TABLE), {"name": ALEMBIC_VERSION_TABLE}
            )
            if found.first() is None:
                return None
            row = (await connection.execute(text(SQL_CURRENT_REVISION))).first()
            return None if row is None else str(row[0])
    finally:
        await engine.dispose()


async def upgrade(db_path: Path, revision: str = ALEMBIC_HEAD) -> None:
    """Upgrade the database. Runs in a thread: Alembic's env starts its own event loop."""
    await asyncio.to_thread(command.upgrade, alembic_config(db_path), revision)


async def downgrade(db_path: Path, revision: str = ALEMBIC_BASE) -> None:
    """Downgrade the database."""
    await asyncio.to_thread(command.downgrade, alembic_config(db_path), revision)


async def backup(db_path: Path, backups_dir: Path, clock: Clock, revision: str | None) -> Path:
    """Copy the database with ``VACUUM INTO`` and keep only the newest backups."""
    backups_dir.mkdir(parents=True, exist_ok=True)
    stamp = clock.now().strftime(BACKUP_STAMP_FORMAT)
    target = backups_dir / BACKUP_NAME.format(stamp=stamp, rev=revision or BACKUP_EMPTY_REV)
    engine = create_engine(db_path, pragmas=False)
    try:
        async with engine.connect() as connection:
            autocommit = await connection.execution_options(isolation_level="AUTOCOMMIT")
            await autocommit.execute(text(SQL_VACUUM_INTO), {"path": str(target)})
    finally:
        await engine.dispose()
    for old in sorted(backups_dir.glob(BACKUP_GLOB), reverse=True)[BACKUP_KEEP:]:
        old.unlink()
    return target


async def migrate_to_head(db_path: Path, backups_dir: Path, clock: Clock) -> bool:
    """Bring the database to head; return whether a migration ran.

    A backup is taken only when an existing database is behind head.
    """
    current = await current_revision(db_path)
    if current == head_revision():
        return False
    if db_path.exists() and current is not None:
        await backup(db_path, backups_dir, clock, current)
    await upgrade(db_path)
    return True
