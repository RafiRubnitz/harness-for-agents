"""The Alembic migrations and the engine (13 §5.6)."""

from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import Connection, inspect, text

from harness.core.consts import AddressType, MessageRole
from harness.core.errors import MemoryStoreError
from harness.core.types import Address, MessageOrigin
from harness.memory import Memory
from harness.memory.consts import ALEMBIC_VERSION_TABLE, FTS_SHADOW_SUFFIXES, FTS_TABLE
from harness.memory.db.engine import create_engine
from harness.memory.db.migrate import current_revision, downgrade, head_revision, upgrade
from harness.memory.db.models import Base
from tests.consts import AGENT_ID, OTHER_AGENT_ID, USER_ID
from tests.utils import make_agent, make_message

FTS_TABLES = {f"{FTS_TABLE}{suffix}" for suffix in FTS_SHADOW_SUFFIXES}
SEARCH_WORD = "zebra"


async def _table_names(path: Path) -> set[str]:
    engine = create_engine(path)
    try:
        async with engine.connect() as connection:
            return set(await connection.run_sync(lambda c: inspect(c).get_table_names()))
    finally:
        await engine.dispose()


async def test_upgrade_empty(tmp_path: Path) -> None:
    path = tmp_path / "new.db"

    await upgrade(path)

    assert await current_revision(path) == head_revision()
    assert set(Base.metadata.tables) <= await _table_names(path)
    assert FTS_TABLE in await _table_names(path)


async def test_downgrade_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "roundtrip.db"
    await upgrade(path)

    await downgrade(path)
    after_downgrade = await _table_names(path)
    await upgrade(path)

    assert after_downgrade == {ALEMBIC_VERSION_TABLE}
    assert await current_revision(path) == head_revision()


def _diff(connection: Connection) -> list[object]:
    def include_object(_obj: object, name: str | None, type_: str, *_: object) -> bool:
        return not (type_ == "table" and name in FTS_TABLES)

    context = MigrationContext.configure(
        connection, opts={"include_object": include_object, "render_as_batch": True}
    )
    return list(compare_metadata(context, Base.metadata))


async def test_schema_matches_models(memory: Memory) -> None:
    async with memory.db.engine.connect() as connection:
        differences = await connection.run_sync(_diff)

    assert differences == []


async def test_pragmas(memory: Memory) -> None:
    async with memory.db.engine.connect() as connection:
        foreign_keys = (await connection.execute(text("PRAGMA foreign_keys"))).scalar_one()
        journal_mode = (await connection.execute(text("PRAGMA journal_mode"))).scalar_one()

    assert foreign_keys == 1
    assert journal_mode == "wal"


async def test_fts_triggers(memory: Memory) -> None:
    store = memory.store
    await store.save_agent(make_agent())
    stored = await store.append_message(make_message("msg_1", f"a {SEARCH_WORD} appeared"))

    found = await store.search_history(AGENT_ID, SEARCH_WORD)
    await store.delete_agent(AGENT_ID)
    async with memory.db.read() as session:
        left = await session.execute(
            text("SELECT count(*) FROM messages_fts WHERE messages_fts MATCH :q"),
            {"q": SEARCH_WORD},
        )

    assert [m.id for m in found] == [stored.id]
    assert left.scalar_one() == 0


async def test_origin_nullable(memory: Memory) -> None:
    store = memory.store
    await store.save_agent(make_agent())
    origin = MessageOrigin(source=Address(type=AddressType.USER, id=USER_ID))
    with_origin = make_message("msg_2", role=MessageRole.USER).model_copy(update={"origin": origin})

    assistant = await store.append_message(make_message("msg_1"))
    user = await store.append_message(with_origin)

    loaded = await store.get_messages(AGENT_ID)
    assert [m.origin for m in loaded] == [None, origin]
    assert (assistant.seq, user.seq) == (1, 2)
    with pytest.raises(MemoryStoreError):
        async with memory.db.write() as session:
            await session.execute(
                text("UPDATE messages SET origin_source_id = NULL WHERE id = :id"),
                {"id": user.id},
            )


async def test_agent_delete_cascade_keeps_other_agents(memory: Memory) -> None:
    store = memory.store
    for agent_id in (AGENT_ID, OTHER_AGENT_ID):
        await store.save_agent(make_agent(agent_id))
    await store.append_message(make_message("msg_1"))
    kept = await store.append_message(make_message("msg_2", agent_id=OTHER_AGENT_ID))

    await store.delete_agent(AGENT_ID)

    assert await store.get_messages(AGENT_ID) == []
    assert await store.get_messages(OTHER_AGENT_ID) == [kept]
    assert [a.id for a in await store.list_agents()] == [OTHER_AGENT_ID]
