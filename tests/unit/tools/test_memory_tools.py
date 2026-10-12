"""The memory tools over the real semantic memory: what an agent can save, find and change."""

import asyncio
from dataclasses import replace

import pytest
from pydantic import JsonValue

from harness.core.consts import Decision, ErrorCode, HookEvent, MemoryKind, MemorySource
from harness.core.records import ToolOutput
from harness.core.types import AgentId, MemoryItemId
from harness.tools import DefaultToolsManager
from harness.tools.builtin.memory import memory_tools
from harness.tools.consts import GROUP_MEMORY, META_ITEM_ID, META_MATCHES
from harness.tools.function_tool import FunctionTool
from harness.tools.types import ToolContext
from tests.consts import AGENT_ID, OTHER_AGENT_ID
from tests.unit.memory.semantic.conftest import SemEnv, sem  # noqa: F401 - the fixture
from tests.unit.tools.conftest import ToolsEnv, text_of
from tests.utils import make_message

FACT = "the staging server is stage-3.internal"
CORRECTED = "the staging server is stage-4.internal"
QUESTION = "where is staging?"
GLOBAL_FACT = "the user reads Hebrew and English"
SPOKEN = "we moved the nightly build to 02:00"
TOOL_NAMES = {"memory_save", "memory_search", "memory_update", "memory_forget", "history_search"}


class Memory:
    """Calls the memory tools the way the executor does, as one agent."""

    def __init__(self, env: SemEnv, agent_id: AgentId = AGENT_ID) -> None:
        self.env = env
        self.tools: dict[str, FunctionTool] = {
            tool.definition.name: tool for tool in memory_tools(env.memory, env.history)
        }
        self.ctx = ToolContext(
            agent_id=agent_id,
            run_id=None,
            call_id="call_1",
            working_dir=".",
            allowed_paths=[],
            cancel=asyncio.Event(),
            emit_progress=lambda chunk: None,
            clock=env.clock,
        )

    async def call(self, name: str, **args: JsonValue) -> ToolOutput:
        return await self.tools[name].run(args, self.ctx)

    async def save(self, content: str) -> str:
        output = await self.call("memory_save", content=content, kind=MemoryKind.FACT.value)
        return str(output.metadata[META_ITEM_ID])


@pytest.fixture
def memory(sem: SemEnv) -> Memory:  # noqa: F811 - pytest injects the fixture by name
    return Memory(sem)


def test_the_group_holds_the_five_tools_and_none_needs_approval(memory: Memory) -> None:
    definitions = [tool.definition for tool in memory.tools.values()]

    assert {definition.name for definition in definitions} == TOOL_NAMES
    assert {definition.group for definition in definitions} == {GROUP_MEMORY}
    assert {definition.default_permission for definition in definitions} == {Decision.ALLOW}
    assert {d.name for d in definitions if d.read_only} == {"memory_search", "history_search"}


@pytest.mark.acceptance("03-10.5")
async def test_what_an_agent_saved_is_found_by_it_after_a_restart(memory: Memory) -> None:
    memory.env.embedder.same_meaning(FACT, QUESTION)
    item_id = await memory.save(FACT)

    await memory.env.reopen()
    found = await Memory(memory.env).call("memory_search", query=QUESTION)

    line = text_of(found)
    assert FACT in line
    assert item_id in line
    assert memory.env.clock.now().date().isoformat() in line  # shown with its date
    assert found.metadata[META_MATCHES] == 1


async def test_saving_stores_the_agent_as_owner_and_source(memory: Memory) -> None:
    output = await memory.call(
        "memory_save", content=FACT, kind=MemoryKind.DECISION.value, tags=["infra"]
    )

    (item,) = await memory.env.memory.list(AGENT_ID)
    assert not output.is_error
    assert (item.agent_id, item.source) == (AGENT_ID, MemorySource.AGENT)
    assert (item.kind, item.tags) == (MemoryKind.DECISION, ["infra"])


async def test_an_update_corrects_and_a_supersede_replaces(memory: Memory) -> None:
    item_id = await memory.save(FACT)

    corrected = await memory.call("memory_update", id=item_id, content=CORRECTED)
    replaced = await memory.call("memory_update", id=item_id, content=FACT, supersede=True)

    new_id = str(replaced.metadata[META_ITEM_ID])
    old = await memory.env.memory.get(MemoryItemId(item_id))
    assert corrected.metadata[META_ITEM_ID] == item_id
    assert new_id != item_id
    assert (old.content, old.superseded_by) == (CORRECTED, new_id)
    assert [item.id for item in await memory.env.memory.list(AGENT_ID)] == [new_id]


async def test_forgetting_hides_the_memory_from_search(memory: Memory) -> None:
    item_id = await memory.save(FACT)

    forgotten = await memory.call("memory_forget", id=item_id)
    found = await memory.call("memory_search", query="staging server", mode="lexical")

    assert not forgotten.is_error
    assert found.metadata[META_MATCHES] == 0
    assert FACT not in text_of(found)


@pytest.mark.acceptance("03-10.9")
async def test_an_agent_can_neither_see_nor_change_another_agents_memory(memory: Memory) -> None:
    item_id = await memory.save(FACT)
    other = Memory(memory.env, OTHER_AGENT_ID)

    seen = await other.call("memory_search", query="staging server", mode="lexical")
    changed = await other.call("memory_update", id=item_id, content=CORRECTED)
    forgotten = await other.call("memory_forget", id=item_id)

    assert seen.metadata[META_MATCHES] == 0
    assert changed.is_error
    assert forgotten.is_error
    assert (await memory.env.memory.get(MemoryItemId(item_id))).content == FACT


async def test_global_facts_are_readable_but_only_the_user_changes_them(memory: Memory) -> None:
    global_id = await memory.env.save(GLOBAL_FACT, agent_id=None)

    seen = await memory.call("memory_search", query="Hebrew", mode="lexical")
    changed = await memory.call("memory_update", id=global_id, content=CORRECTED)

    assert GLOBAL_FACT in text_of(seen)
    assert changed.is_error
    assert (await memory.env.memory.get(global_id)).content == GLOBAL_FACT


async def test_errors_come_back_as_values_the_model_can_read(memory: Memory) -> None:
    memory.env.hooks.script(HookEvent.BEFORE_MEMORY_WRITE, Decision.DENY, reason="not allowed")

    denied = await memory.call("memory_save", content=FACT, kind=MemoryKind.FACT.value)
    missing = await memory.call("memory_forget", id="mem_missing")
    bad_kind = await memory.call("memory_save", content=FACT, kind="rumour")

    assert denied.is_error
    assert "not allowed" in text_of(denied)
    assert missing.is_error
    assert ErrorCode.MEMORY_ITEM_NOT_FOUND.value in missing.metadata.values()
    assert bad_kind.is_error


async def test_history_search_reads_only_the_agents_own_conversation(memory: Memory) -> None:
    history = memory.env.history
    await history.append_message(make_message("msg_1", SPOKEN))
    await history.append_message(make_message("msg_2", SPOKEN, agent_id=OTHER_AGENT_ID))

    mine = await memory.call("history_search", query="nightly build")
    nothing = await memory.call("history_search", query="unmentioned")

    assert mine.metadata[META_MATCHES] == 1
    assert SPOKEN in text_of(mine)
    assert nothing.metadata[META_MATCHES] == 0


async def test_the_tools_manager_offers_the_memory_group_only_when_it_has_a_memory(
    env: ToolsEnv,
    sem: SemEnv,  # noqa: F811 - pytest injects the fixture by name
) -> None:
    deps = replace(env.deps, semantic=sem.memory, history=sem.history)

    without = {definition.name for definition in env.manager.list()}
    offered = {d.name for d in DefaultToolsManager(deps).list() if d.group == GROUP_MEMORY}

    assert not without & TOOL_NAMES
    assert offered == TOOL_NAMES
