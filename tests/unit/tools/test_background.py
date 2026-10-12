"""Background tasks: a tool that runs while the agent goes on (08 §6a)."""

import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field, replace

import pytest
from pydantic import JsonValue

from harness.core.consts import (
    META_TASK_ID,
    ConfigLayer,
    Decision,
    ErrorCode,
    HookEvent,
    TaskStatus,
)
from harness.core.errors import ToolError
from harness.core.records import BackgroundTaskRecord, ToolCallRecord, ToolOutput
from harness.core.types import AgentId, MessageId, TaskId, ToolCallRecordId
from harness.tools import DefaultToolsManager, Tool, tool
from harness.tools.consts import GROUP_FILES, GROUP_TASKS, META_NEXT_OFFSET, META_STATUS
from harness.tools.function_tool import text_output
from harness.tools.types import ToolContext
from tests.consts import AGENT_ID, OTHER_AGENT_ID
from tests.fakes.ids import FakeIds
from tests.fakes.memory import FakeMemoryStore
from tests.unit.tools.conftest import REASON, ToolsEnv, text_of

WAITER = "waiter"
FIRST_CHUNK = "compiling module one\n"
SECOND_CHUNK = "compiling module two\n"
RESULT = "build finished without warnings"
HOOK_REASON = "results of this tool are withheld"
TASK_TOOLS = {"task_status", "task_output", "task_cancel", "task_list"}
UNKNOWN_TASK = "task_does_not_exist"
PART = 9


@dataclass
class Background:
    """A tools manager with a task store, and a tool the test releases by hand."""

    env: ToolsEnv
    manager: DefaultToolsManager
    store: FakeMemoryStore
    release: asyncio.Event = field(default_factory=asyncio.Event)
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    statuses: list[TaskStatus] = field(default_factory=list[TaskStatus])
    done: list[tuple[BackgroundTaskRecord, str]] = field(
        default_factory=list[tuple[BackgroundTaskRecord, str]]
    )
    told: asyncio.Event = field(default_factory=asyncio.Event)
    stopped: bool = False
    waiter: list[Tool] = field(default_factory=list[Tool])

    async def run(self, name: str, **args: JsonValue) -> ToolOutput:
        """Record the call the way the agent loop does, then decide and execute it."""
        call = self.env.call(name, reason=REASON, **args)
        await self.store.save_tool_call(
            ToolCallRecord(
                id=ToolCallRecordId(f"tc_{call.id}"),
                agent_id=AGENT_ID,
                call_id=call.id,
                request_message_id=MessageId("msg_1"),
                tool_name=name,
                created_at=self.env.clock.now(),
            )
        )
        decision = await self.manager.decide(AGENT_ID, call)
        return await self.manager.execute(AGENT_ID, call, decision)

    async def start(self) -> TaskId:
        """Start the waiter in the background and wait until it is really running."""
        output = await self.run(WAITER, run_in_background=True)
        await self.entered.wait()
        return TaskId(str(output.metadata[META_TASK_ID]))

    async def finish(self) -> None:
        """Let the waiter end and wait for the agent to be told."""
        self.release.set()
        await self.told.wait()


def waiter_of(bg: Background) -> Tool:
    """The waiter tool, to register it in a second manager."""
    return bg.waiter[0]


@pytest.fixture
async def bg(env: ToolsEnv, ids: FakeIds) -> AsyncGenerator[Background]:
    store = FakeMemoryStore(env.clock, ids)
    built: list[Background] = []

    def status(record: BackgroundTaskRecord) -> None:
        built[0].statuses.append(record.status)

    async def done(record: BackgroundTaskRecord, summary: str) -> None:
        built[0].done.append((record, summary))
        built[0].told.set()

    deps = replace(env.deps, history=store, task_status=status, task_done=done)
    manager = DefaultToolsManager(deps)
    built.append(Background(env, manager, store))

    @tool(name=WAITER, group=GROUP_FILES, default_permission=Decision.ALLOW)
    async def waiter(ctx: ToolContext) -> ToolOutput:
        """Print two lines with a pause the test controls between them."""
        ctx.emit_progress(FIRST_CHUNK)
        built[0].entered.set()
        try:
            await built[0].release.wait()
        except asyncio.CancelledError:
            built[0].stopped = True
            raise
        ctx.emit_progress(SECOND_CHUNK)
        return text_output(RESULT)

    manager.register(waiter)
    built[0].waiter.append(waiter)
    env.agent()
    yield built[0]
    await manager.close()


def test_every_agent_with_tools_gets_the_task_tools_without_approval(bg: Background) -> None:
    listed = {d.name: d for d in bg.manager.list(AGENT_ID) if d.group == GROUP_TASKS}

    assert set(listed) == TASK_TOOLS
    assert {d.default_permission for d in listed.values()} == {Decision.ALLOW}


@pytest.mark.acceptance("08-10.10")
async def test_a_background_call_returns_its_task_id_at_once_and_reports_the_end(
    bg: Background,
) -> None:
    output = await bg.run(WAITER, run_in_background=True)
    await bg.entered.wait()
    answer = json.loads(text_of(output))
    running = [task.status for task in bg.manager.background_tasks(AGENT_ID)]

    await bg.finish()

    ((record, summary),) = bg.done
    assert answer == {META_TASK_ID: record.id, META_STATUS: TaskStatus.RUNNING.value}
    assert running == [TaskStatus.RUNNING]
    assert (record.status, summary) == (TaskStatus.DONE, RESULT)
    assert bg.statuses == [TaskStatus.RUNNING, TaskStatus.DONE]
    assert bg.store.tasks[record.id].status is TaskStatus.DONE
    assert bg.store.tasks[record.id].output_blob is not None


@pytest.mark.acceptance("08-10.11")
async def test_task_output_returns_what_a_running_task_printed_so_far(bg: Background) -> None:
    task_id = await bg.start()

    partial = await bg.run("task_output", task_id=task_id)
    paged = await bg.run("task_output", task_id=task_id, offset=PART, limit=PART)
    await bg.finish()
    whole = await bg.run("task_output", task_id=task_id)

    assert text_of(partial) == FIRST_CHUNK
    assert partial.metadata[META_STATUS] == TaskStatus.RUNNING.value
    assert text_of(paged) == FIRST_CHUNK[PART : PART + PART]
    assert paged.metadata[META_NEXT_OFFSET] == PART + PART
    assert text_of(whole) == RESULT
    assert whole.metadata[META_STATUS] == TaskStatus.DONE.value


async def test_progress_of_a_background_task_carries_its_task_id(bg: Background) -> None:
    seen: list[tuple[TaskId | None, str]] = []

    def progress(agent_id: AgentId, call_id: str, task_id: TaskId | None, chunk: str) -> None:
        seen.append((task_id, chunk))

    deps = replace(bg.env.deps, history=bg.store, progress=progress)
    manager = DefaultToolsManager(deps)
    manager.register(waiter_of(bg))
    bg.manager = manager

    task_id = await bg.start()
    await manager.close()

    assert seen == [(task_id, FIRST_CHUNK)]


async def test_task_status_and_list_follow_the_task(bg: Background) -> None:
    task_id = await bg.start()

    before = await bg.run("task_status", task_id=task_id)
    await bg.finish()
    after = await bg.run("task_status", task_id=task_id)
    listed = await bg.run("task_list")

    assert before.metadata[META_STATUS] == TaskStatus.RUNNING.value
    assert after.metadata[META_STATUS] == TaskStatus.DONE.value
    assert task_id in text_of(listed)
    assert TaskStatus.DONE.value in text_of(listed)


async def test_task_cancel_stops_the_tool_and_tells_the_agent(bg: Background) -> None:
    task_id = await bg.start()

    output = await bg.run("task_cancel", task_id=task_id)

    ((record, _),) = bg.done
    assert bg.stopped
    assert TaskStatus.CANCELLED.value in text_of(output)
    assert record.status is TaskStatus.CANCELLED
    assert bg.statuses == [TaskStatus.RUNNING, TaskStatus.CANCELLED]
    assert bg.store.tasks[task_id].status is TaskStatus.CANCELLED


async def test_the_cancel_command_of_the_user_stops_a_task(bg: Background) -> None:
    task_id = await bg.start()

    await bg.manager.cancel_background(task_id)
    await bg.manager.cancel_background(task_id)  # a finished task is left alone

    assert bg.statuses == [TaskStatus.RUNNING, TaskStatus.CANCELLED]


async def test_a_task_that_fails_is_reported_as_failed(bg: Background) -> None:
    bg.env.hooks.script(HookEvent.AFTER_TOOL_CALL, Decision.DENY, reason=HOOK_REASON)
    await bg.start()

    await bg.finish()

    ((record, summary),) = bg.done
    assert record.status is TaskStatus.FAILED
    assert HOOK_REASON in summary
    assert record.error == summary


async def test_the_after_hook_runs_when_the_task_ends_not_when_it_starts(bg: Background) -> None:
    await bg.start()
    at_start = bg.env.hooks.events().count(HookEvent.AFTER_TOOL_CALL)

    await bg.finish()

    assert at_start == 0
    assert bg.env.hooks.events().count(HookEvent.AFTER_TOOL_CALL) == 1


async def test_a_denied_call_is_never_started_in_the_background(bg: Background) -> None:
    await bg.env.rules({"tool": WAITER, "decision": Decision.DENY.value}, layer=ConfigLayer.GLOBAL)

    output = await bg.run(WAITER, run_in_background=True)

    assert output.is_error
    assert bg.manager.background_tasks(AGENT_ID) == []
    assert bg.statuses == []


async def test_closing_leaves_a_running_task_marked_running_for_recovery(bg: Background) -> None:
    task_id = await bg.start()

    await bg.manager.close()

    assert bg.stopped
    assert bg.store.tasks[task_id].status is TaskStatus.RUNNING
    assert bg.done == []


async def test_an_agent_cannot_reach_the_task_of_another(bg: Background) -> None:
    task_id = await bg.start()

    output = await bg.run("task_status", task_id=UNKNOWN_TASK)

    assert output.is_error
    assert UNKNOWN_TASK in text_of(output)
    assert bg.manager.background_tasks(OTHER_AGENT_ID) == []
    assert [task.id for task in bg.manager.background_tasks(AGENT_ID)] == [task_id]


async def test_starting_directly_refuses_a_call_that_was_not_allowed(bg: Background) -> None:
    call = bg.env.call(WAITER)
    decision = await bg.manager.decide(AGENT_ID, call)
    denied = decision.model_copy(update={"decision": Decision.DENY})

    with pytest.raises(ToolError) as raised:
        await bg.manager.start_background(AGENT_ID, call, denied)

    assert raised.value.code is ErrorCode.TOOL_NOT_AVAILABLE
