"""Writing to the semantic memory: time stamps, versions, duplicates, deletion, hooks."""

from datetime import timedelta

import pytest

from harness.core.consts import (
    Decision,
    ErrorCode,
    HookEvent,
    MemoryKind,
    MemoryScope,
    MemorySource,
    SearchMode,
)
from harness.core.errors import MemoryItemNotFoundError, MemoryStoreError
from harness.core.types import MemoryItemId
from tests.consts import AGENT_ID, OTHER_AGENT_ID
from tests.fakes.clock import FAKE_CLOCK_START
from tests.fakes.embedder import FAKE_EMBED_MODEL
from tests.unit.memory.semantic.conftest import SemEnv
from tests.utils import make_message

FACT = "the deploy script lives in tools/deploy.sh"
FACT_REWORDED = "deploy.sh is kept under the tools folder"
CORRECTED = "the deploy script lives in scripts/deploy.sh"
OTHER_FACT = "the staging database is called stage-db-2"
HOUR = 3600.0
NEAR_DUPLICATE = 0.97
NOT_A_DUPLICATE = 0.5
HOOK_REASON = "contains a secret"
REDACTED = "the deploy script lives in [redacted]"


# --- time stamps ---


@pytest.mark.acceptance("03-10.1")
async def test_a_fact_learned_later_carries_the_time_it_was_said(sem: SemEnv) -> None:
    said = await sem.history.append_message(make_message("msg_1", FACT, at=FAKE_CLOCK_START))
    sem.clock.advance(HOUR)

    item_id = await sem.memory.remember(AGENT_ID, FACT, MemoryKind.FACT, source_msg_seq=said.seq)

    item = await sem.memory.get(item_id)
    assert item.observed_at == said.created_at
    assert item.created_at == sem.clock.now()
    assert item.created_at - item.observed_at == timedelta(seconds=HOUR)
    assert item.source_msg_seq == said.seq


async def test_without_a_source_message_the_fact_is_observed_now(sem: SemEnv) -> None:
    sem.clock.advance(HOUR)

    item = await sem.memory.get(await sem.save(FACT))

    assert item.observed_at == sem.clock.now()
    assert item.scope is MemoryScope.AGENT
    assert item.embed_model == FAKE_EMBED_MODEL


async def test_a_fact_without_an_agent_is_global(sem: SemEnv) -> None:
    item = await sem.memory.get(await sem.save(FACT, agent_id=None))

    assert item.scope is MemoryScope.GLOBAL
    assert item.agent_id is None


# --- update, versions ---


@pytest.mark.acceptance("03-10.2")
async def test_an_update_keeps_the_previous_value_with_its_times(sem: SemEnv) -> None:
    item_id = await sem.save(FACT)
    before = await sem.memory.get(item_id)
    sem.clock.advance(HOUR)

    await sem.memory.update(item_id, actor=MemorySource.USER, content=CORRECTED)

    after = await sem.memory.get(item_id)
    (version,) = await sem.memory.history(item_id)
    assert after.content == CORRECTED
    assert after.observed_at == after.updated_at == sem.clock.now()
    assert after.created_at == before.created_at
    assert version.content == FACT
    assert version.observed_at == before.observed_at
    assert version.replaced_at == sem.clock.now()
    assert version.replaced_by is MemorySource.USER


async def test_every_update_adds_one_version_in_order(sem: SemEnv) -> None:
    item_id = await sem.save(FACT)
    contents = [CORRECTED, FACT_REWORDED, OTHER_FACT]

    for content in contents:
        await sem.memory.update(item_id, actor=MemorySource.AGENT, content=content)

    history = await sem.memory.history(item_id)
    assert [version.content for version in history] == [FACT, *contents[:-1]]


async def test_changing_only_fields_keeps_content_vector_and_versions(sem: SemEnv) -> None:
    item_id = await sem.save(FACT)

    await sem.memory.update(
        item_id, actor=MemorySource.USER, kind=MemoryKind.DECISION.value, tags=["ops"], pinned=True
    )

    item = await sem.memory.get(item_id)
    assert (item.kind, item.tags, item.pinned) == (MemoryKind.DECISION, ["ops"], True)
    assert item.content == FACT
    assert await sem.memory.history(item_id) == []
    assert await sem.found(FACT, mode=SearchMode.VECTOR) == [FACT]


async def test_an_unknown_field_is_refused(sem: SemEnv) -> None:
    item_id = await sem.save(FACT)

    with pytest.raises(MemoryStoreError):
        await sem.memory.update(item_id, actor=MemorySource.USER, scope="global")


async def test_a_missing_item_is_reported(sem: SemEnv) -> None:
    with pytest.raises(MemoryItemNotFoundError) as raised:
        await sem.memory.update(MemoryItemId("mem_none"), actor=MemorySource.USER, content=FACT)

    assert raised.value.code is ErrorCode.MEMORY_ITEM_NOT_FOUND


# --- duplicates ---


@pytest.mark.acceptance("03-10.10")
async def test_saving_the_same_fact_twice_leaves_one_updated_item(sem: SemEnv) -> None:
    first = await sem.save(FACT, tags=("ops",))
    sem.clock.advance(HOUR)

    second = await sem.save(FACT, tags=("deploy",))

    items = await sem.memory.list(AGENT_ID)
    assert second == first
    assert [item.id for item in items] == [first]
    assert items[0].tags == ["ops", "deploy"]
    assert items[0].updated_at == sem.clock.now()


async def test_a_reworded_duplicate_updates_the_item_and_keeps_the_old_wording(
    sem: SemEnv,
) -> None:
    first = await sem.save(FACT)
    sem.embedder.close_to(FACT_REWORDED, FACT, similarity=NEAR_DUPLICATE)

    second = await sem.save(FACT_REWORDED)

    assert second == first
    assert (await sem.memory.get(first)).content == FACT_REWORDED
    assert [version.content for version in await sem.memory.history(first)] == [FACT]


async def test_a_merely_related_fact_becomes_its_own_item(sem: SemEnv) -> None:
    first = await sem.save(FACT)
    sem.embedder.close_to(OTHER_FACT, FACT, similarity=NOT_A_DUPLICATE)

    second = await sem.save(OTHER_FACT)

    assert second != first
    assert len(await sem.memory.list(AGENT_ID)) == 2


async def test_the_duplicate_threshold_comes_from_the_config(sem: SemEnv) -> None:
    await sem.set("memory.dedup_similarity", NOT_A_DUPLICATE)
    first = await sem.save(FACT)
    sem.embedder.close_to(OTHER_FACT, FACT, similarity=NOT_A_DUPLICATE)

    assert await sem.save(OTHER_FACT) == first


async def test_another_owners_identical_fact_is_not_a_duplicate(sem: SemEnv) -> None:
    mine = await sem.save(FACT)

    theirs = await sem.save(FACT, agent_id=OTHER_AGENT_ID)
    shared = await sem.save(FACT, agent_id=None)

    assert len({mine, theirs, shared}) == 3


# --- supersede, forget, delete ---


async def test_superseding_expires_the_old_item_and_points_to_the_new_one(sem: SemEnv) -> None:
    old_id = await sem.save(FACT, tags=("ops",))
    sem.clock.advance(HOUR)

    new_id = await sem.memory.supersede(old_id, CORRECTED, actor=MemorySource.AGENT)

    old, new = await sem.memory.get(old_id), await sem.memory.get(new_id)
    assert old.valid_until == sem.clock.now()
    assert old.superseded_by == new_id
    assert old.content == FACT
    assert (new.content, new.tags, new.valid_until) == (CORRECTED, ["ops"], None)
    assert [item.id for item in await sem.memory.list(AGENT_ID)] == [new_id]
    assert len(await sem.memory.list(AGENT_ID, include_expired=True)) == 2


async def test_forgetting_is_soft_and_deleting_is_physical(sem: SemEnv) -> None:
    forgotten, deleted = await sem.save(FACT), await sem.save(OTHER_FACT)
    await sem.memory.update(deleted, actor=MemorySource.USER, content=CORRECTED)

    await sem.memory.forget(forgotten)
    await sem.memory.delete(deleted)

    assert (await sem.memory.get(forgotten)).valid_until == sem.clock.now()
    assert await sem.memory.list(AGENT_ID) == []
    assert await sem.memory.history(deleted) == []
    with pytest.raises(MemoryItemNotFoundError):
        await sem.memory.get(deleted)


@pytest.mark.acceptance("03-10.13")
async def test_deleting_an_agent_removes_its_items_and_versions_only(sem: SemEnv) -> None:
    mine = await sem.save(FACT)
    await sem.memory.update(mine, actor=MemorySource.AGENT, content=CORRECTED)
    theirs = await sem.save(FACT, agent_id=OTHER_AGENT_ID)
    shared = await sem.save(FACT, agent_id=None)

    await sem.memory.delete_agent(AGENT_ID)

    assert await sem.memory.list(AGENT_ID, include_expired=True) == []
    assert await sem.memory.history(mine) == []
    assert await sem.found(CORRECTED) == [FACT]  # only the global one is left to see
    assert [item.id for item in await sem.memory.by_ids([mine, theirs, shared])] == [theirs, shared]


async def test_pinned_items_are_the_agents_own_and_the_global_valid_ones(sem: SemEnv) -> None:
    mine, shared = await sem.save(FACT), await sem.save(OTHER_FACT, agent_id=None)
    theirs, gone = await sem.save(FACT, agent_id=OTHER_AGENT_ID), await sem.save(CORRECTED)
    for item_id in (mine, shared, theirs, gone):
        await sem.memory.update(item_id, actor=MemorySource.USER, pinned=True)
    await sem.memory.forget(gone)

    pinned = await sem.memory.pinned(AGENT_ID)

    assert [item.id for item in pinned] == [mine, shared]


# --- before_memory_write ---


async def test_every_write_asks_the_hook_first(sem: SemEnv) -> None:
    item_id = await sem.save(FACT)
    await sem.memory.update(item_id, actor=MemorySource.USER, content=CORRECTED)
    await sem.memory.supersede(item_id, OTHER_FACT, actor=MemorySource.USER)

    ops = [inp.payload["op"] for inp in sem.hooks.emitted]

    assert sem.hooks.events() == [HookEvent.BEFORE_MEMORY_WRITE] * 3
    assert ops == ["remember", "update", "supersede"]
    assert sem.hooks.emitted[0].payload["content"] == FACT
    assert sem.hooks.emitted[0].agent_id == AGENT_ID


@pytest.mark.parametrize("decision", [Decision.DENY, Decision.ASK])
async def test_a_write_the_hook_does_not_allow_is_not_saved(
    sem: SemEnv, decision: Decision
) -> None:
    sem.hooks.script(HookEvent.BEFORE_MEMORY_WRITE, decision, reason=HOOK_REASON)

    with pytest.raises(MemoryStoreError) as raised:
        await sem.save(FACT)

    assert raised.value.code is ErrorCode.HOOK_DENIED
    assert HOOK_REASON in raised.value.message
    assert await sem.memory.list(AGENT_ID) == []


async def test_the_hook_may_rewrite_what_is_saved(sem: SemEnv) -> None:
    sem.hooks.script(HookEvent.BEFORE_MEMORY_WRITE, updated={"content": REDACTED})

    item = await sem.memory.get(await sem.save(FACT))

    assert item.content == REDACTED


async def test_empty_content_is_refused(sem: SemEnv) -> None:
    with pytest.raises(MemoryStoreError):
        await sem.save("   ")
