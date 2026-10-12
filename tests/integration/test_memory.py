"""Semantic memory through the real core: the user's commands, recall into context, deletion."""

from harness.core.commands import (
    AgentCreate,
    AgentDelete,
    AgentSend,
    Command,
    MemoryAdd,
    MemoryDelete,
    MemoryList,
    MemoryUpdate,
)
from harness.core.consts import AgentStatus, ErrorCode, MemoryKind, MemoryScope, MemorySource
from harness.core.events import AgentCreated, AgentDeleted, AgentStatusChanged, Error, MemoryListed
from harness.core.handle import CoreHandle
from harness.core.records import MemoryItem
from harness.core.types import AgentId, MemoryItemId, TextBlock
from tests.consts import AGENT_NAME
from tests.fakes.adapter import text_response
from tests.integration.conftest import CoreKit, next_payload

GLOBAL_FACT = "the user works in the Jerusalem time zone"
AGENT_FACT = "the release branch is called stable-7"
NEW_FACT = "the release branch is called stable-8"
QUESTION = "which branch do we release from?"
ANSWER = "from the stable branch"
MISSING = MemoryItemId("mem_missing")


async def create(core: CoreHandle, name: str = AGENT_NAME) -> AgentId:
    events = core.events()
    await core.submit(AgentCreate(agent_name=name))
    return (await next_payload(events, AgentCreated)).agent.id


async def listed(core: CoreHandle, command: Command) -> list[MemoryItem]:
    """Submit a memory command and return the list the core shows afterwards."""
    events = core.events()
    await core.submit(command)
    return (await next_payload(events, MemoryListed)).items


async def ask(core: CoreHandle, agent_id: AgentId) -> None:
    events = core.events()
    await core.submit(AgentSend(agent_id=agent_id, content=[TextBlock(text=QUESTION)]))
    async for event in events:
        payload = event.payload
        if isinstance(payload, AgentStatusChanged) and payload.status is AgentStatus.IDLE:
            return


async def test_the_user_adds_edits_and_deletes_a_global_fact(core: CoreHandle) -> None:
    (added,) = await listed(core, MemoryAdd(content=GLOBAL_FACT, kind=MemoryKind.PREFERENCE))
    (edited,) = await listed(
        core, MemoryUpdate(item_id=MemoryItemId(added.id), pinned=True, tags=["me"])
    )
    after_delete = await listed(core, MemoryDelete(item_id=MemoryItemId(added.id)))

    assert (added.scope, added.source) == (MemoryScope.GLOBAL, MemorySource.USER)
    assert (added.content, added.kind) == (GLOBAL_FACT, MemoryKind.PREFERENCE)
    assert (edited.pinned, edited.tags, edited.content) == (True, ["me"], GLOBAL_FACT)
    assert after_delete == []
    assert await listed(core, MemoryList()) == []


async def test_superseding_from_the_editor_keeps_the_old_fact_as_expired(
    core: CoreHandle,
) -> None:
    agent_id = await create(core)
    (old,) = await listed(
        core, MemoryAdd(agent_id=agent_id, content=AGENT_FACT, kind=MemoryKind.FACT)
    )

    (current,) = await listed(
        core, MemoryUpdate(item_id=MemoryItemId(old.id), content=NEW_FACT, supersede=True)
    )
    everything = await listed(core, MemoryList(agent_id=agent_id, include_expired=True))

    assert current.content == NEW_FACT
    assert current.id != old.id
    expired = next(item for item in everything if item.id == old.id)
    assert expired.superseded_by == current.id
    assert expired.valid_until is not None


async def test_a_memory_command_on_a_missing_item_is_an_error_event(core: CoreHandle) -> None:
    events = core.events()

    await core.submit(MemoryDelete(item_id=MISSING))
    error = await next_payload(events, Error)

    assert error.code is ErrorCode.MEMORY_ITEM_NOT_FOUND


async def test_pinned_and_relevant_memories_reach_the_model(core: CoreHandle, kit: CoreKit) -> None:
    agent_id = await create(core)
    (pinned,) = await listed(core, MemoryAdd(content=GLOBAL_FACT, kind=MemoryKind.FACT))
    await listed(core, MemoryUpdate(item_id=MemoryItemId(pinned.id), pinned=True))
    kit.embedder.same_meaning(AGENT_FACT, QUESTION)
    await listed(core, MemoryAdd(agent_id=agent_id, content=AGENT_FACT, kind=MemoryKind.FACT))
    kit.gateway.outcomes = [text_response(ANSWER)]

    await ask(core, agent_id)

    sent = kit.gateway.requests[-1].model_dump_json()
    assert GLOBAL_FACT in sent
    assert AGENT_FACT in sent
    assert pinned.observed_at.date().isoformat() in sent  # memories are shown with their date


async def test_memories_survive_a_restart(kit: CoreKit) -> None:
    async with kit.launch() as first:
        agent_id = await create(first)
        await listed(first, MemoryAdd(agent_id=agent_id, content=AGENT_FACT, kind=MemoryKind.FACT))

    async with kit.launch() as second:
        (kept,) = await listed(second, MemoryList(agent_id=agent_id))

    assert kept.content == AGENT_FACT


async def test_deleting_an_agent_deletes_its_memories_and_no_others(core: CoreHandle) -> None:
    doomed, kept = await create(core), await create(core, "other")
    for owner in (doomed, kept, None):
        await listed(core, MemoryAdd(agent_id=owner, content=AGENT_FACT, kind=MemoryKind.FACT))
    events = core.events()

    await core.submit(AgentDelete(agent_id=doomed))
    await next_payload(events, AgentDeleted)

    assert await listed(core, MemoryList(agent_id=doomed, include_expired=True)) == []
    assert len(await listed(core, MemoryList(agent_id=kept))) == 1
    assert len(await listed(core, MemoryList())) == 1
