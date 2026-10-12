"""The agent run loop: steps, tools, inbox, limits, hooks (04 §5, §6)."""

import asyncio

import pytest
from pydantic import JsonValue

from harness.core.consts import (
    AgentStatus,
    Decision,
    EventName,
    HookEvent,
    MessageRole,
    PermissionAnswer,
    RunStopReason,
    ToolCallStatus,
)
from harness.core.events import (
    AgentStatusChanged,
    PermissionRequest,
    RunLimitReached,
    TextDelta,
    ToolCall,
    ToolResult,
    UsageUpdate,
)
from harness.core.records import ToolOutput
from harness.core.types import TextBlock, ToolCallBlock, ToolResultBlock
from tests.fakes.adapter import text_response, tool_response
from tests.fakes.tools import text_output
from tests.unit.agents.conftest import AgentEnv

QUESTION = "what is in the file?"
FOLLOW_UP = "also check the other one"
ANSWER = "the file is empty"
TOOL = "lookup"
OTHER_TOOL = "store"
REASON = "need the file content"
CONTEXT_NOTE = "remember the house style"
REPLACED = "a rewritten question"
FAILURE = "disk on fire"
STEP_LIMIT = 1
UNLIMITED_STEPS = 3
PARALLEL_CALLS = 2

U, A, T = MessageRole.USER, MessageRole.ASSISTANT, MessageRole.TOOL


def call(call_id: str = "call_1", name: str = TOOL) -> ToolCallBlock:
    """A tool call as the model sends it (the reason travels inside the input)."""
    return ToolCallBlock(id=call_id, name=name, input={"reason": REASON})


@pytest.mark.acceptance("04-11.1")
async def test_message_runs_to_idle(env: AgentEnv) -> None:
    env.gateway.outcomes = [text_response(ANSWER)]
    agent_id = await env.create()

    await env.converse(agent_id, QUESTION)

    messages = await env.store.get_messages(agent_id)
    statuses = [p.status for p in env.payloads(AgentStatusChanged)]
    assert [(m.role, m.content) for m in messages] == [
        (U, [TextBlock(text=QUESTION)]),
        (A, [TextBlock(text=ANSWER)]),
    ]
    assert statuses == [AgentStatus.RUNNING, AgentStatus.IDLE]
    assert [p.text for p in env.payloads(TextDelta)] == [ANSWER]
    assert env.payloads(UsageUpdate)[0].model == env.gateway.profile.id
    assert await env.store.open_run(agent_id) is None
    assert env.stop_reasons() == [RunStopReason.END.value]


async def test_tool_call_result_goes_back_to_the_model(env: AgentEnv) -> None:
    env.tools.add(TOOL)
    env.gateway.outcomes = [tool_response(call()), text_response(ANSWER)]
    agent_id = await env.create()

    await env.converse(agent_id, QUESTION)

    messages = await env.store.get_messages(agent_id)
    result = messages[2].content[0]
    audit = await env.store.get_tool_calls(agent_id)
    assert [m.role for m in messages] == [U, A, T, A]
    assert isinstance(result, ToolResultBlock)
    assert (result.call_id, result.is_error) == (call().id, False)
    assert messages[1].content == [call().model_copy(update={"reason": REASON})]
    assert env.tools.executed == [TOOL]
    assert len(env.gateway.requests[1].messages) == len(messages) - 1
    assert [(t.status, t.reason, t.input) for t in audit] == [(ToolCallStatus.DONE, REASON, {})]
    assert [p.reason for p in env.payloads(ToolCall)] == [REASON]
    assert [p.call_id for p in env.payloads(ToolResult)] == [call().id]


@pytest.mark.acceptance("04-11.6")
@pytest.mark.parametrize("failure", ["error_value", "exception", "unknown_tool"])
async def test_tool_failure_does_not_end_the_run(env: AgentEnv, failure: str) -> None:
    async def fail(_: dict[str, JsonValue]) -> ToolOutput:
        if failure == "exception":
            raise OSError(FAILURE)
        return text_output(FAILURE, is_error=True)

    if failure != "unknown_tool":
        env.tools.add(TOOL, fail)
    env.gateway.outcomes = [tool_response(call()), text_response(ANSWER)]
    agent_id = await env.create()

    await env.converse(agent_id, QUESTION)

    messages = await env.store.get_messages(agent_id)
    result = messages[2].content[0]
    assert isinstance(result, ToolResultBlock)
    assert result.is_error
    assert messages[-1].content == [TextBlock(text=ANSWER)]
    assert env.manager.get(agent_id).status is AgentStatus.IDLE


@pytest.mark.acceptance("04-11.5")
async def test_message_sent_during_a_run_is_injected_at_the_next_step(env: AgentEnv) -> None:
    started, release = env.tools.add_blocking(TOOL)
    env.gateway.outcomes = [tool_response(call()), text_response(ANSWER)]
    agent_id = await env.create()
    await env.say(agent_id, QUESTION)
    await started.wait()

    await env.say(agent_id, FOLLOW_UP)
    waiting = await env.store.pending_inbox(agent_id)
    release.set()
    await env.manager.wait_idle(agent_id)

    messages = await env.store.get_messages(agent_id)
    assert [item.content for item in waiting] == [[TextBlock(text=FOLLOW_UP)]]
    assert [m.role for m in messages] == [U, A, T, U, A]
    assert messages[3].content == [TextBlock(text=FOLLOW_UP)]
    assert env.gateway.requests[1].messages[-1].content[-1] == TextBlock(text=FOLLOW_UP)
    assert await env.store.pending_inbox(agent_id) == []


@pytest.mark.acceptance("08-10.9")
async def test_exclusive_tools_do_not_overlap(env: AgentEnv) -> None:
    running: list[str] = []
    seen_together: list[list[str]] = []
    both_parallel_running = asyncio.Barrier(PARALLEL_CALLS)

    def tracked(name: str, *, exclusive: bool) -> None:
        async def handler(_: dict[str, JsonValue]) -> ToolOutput:
            running.append(name)
            if not exclusive:
                await both_parallel_running.wait()
            seen_together.append(list(running))
            await asyncio.sleep(0)
            running.remove(name)
            return text_output(name)

        env.tools.add(name, handler, exclusive=exclusive)

    tracked(TOOL, exclusive=False)
    tracked(OTHER_TOOL, exclusive=True)
    calls = [call("call_1"), call("call_2"), call("call_3", OTHER_TOOL)]
    response = tool_response(calls[0]).model_copy(update={"content": calls})
    env.gateway.outcomes = [response, text_response(ANSWER)]
    agent_id = await env.create()

    await env.converse(agent_id, QUESTION)

    results = (await env.store.get_messages(agent_id))[2].content
    assert [TOOL] * PARALLEL_CALLS in seen_together
    assert all(seen == [OTHER_TOOL] for seen in seen_together if OTHER_TOOL in seen)
    assert [r.call_id for r in results if isinstance(r, ToolResultBlock)] == [c.id for c in calls]


@pytest.mark.acceptance("04-11.8")
@pytest.mark.parametrize("answer", list(PermissionAnswer))
async def test_step_limit_asks_the_user(env: AgentEnv, answer: PermissionAnswer) -> None:
    env.tools.add(TOOL)
    env.gateway.outcomes = [tool_response(call()), text_response(ANSWER)]
    agent_id = await env.create(max_steps=STEP_LIMIT)
    await env.say(agent_id, QUESTION)
    await env.until_status(agent_id, AgentStatus.WAITING_APPROVAL)
    run_while_waiting = await env.store.open_run(agent_id)
    request = env.payloads(PermissionRequest)[0].request

    await env.manager.respond_permission(request.request_id, answer)
    await env.manager.wait_idle(agent_id)

    approved = answer is PermissionAnswer.APPROVE
    expected = RunStopReason.END if approved else RunStopReason.LIMIT
    assert request.subject is HookEvent.RUN_LIMIT_REACHED
    assert run_while_waiting is not None
    assert env.payloads(RunLimitReached)[0].max_steps == STEP_LIMIT
    assert env.stop_reasons() == [expected.value]
    assert env.manager.get(agent_id).status is AgentStatus.IDLE
    assert len(env.gateway.requests) == STEP_LIMIT + int(approved)


@pytest.mark.acceptance("04-11.8")
async def test_no_step_limit_never_asks(env: AgentEnv) -> None:
    env.tools.add(TOOL)
    env.gateway.outcomes = [
        *(tool_response(call(f"call_{n}")) for n in range(UNLIMITED_STEPS)),
        text_response(ANSWER),
    ]
    agent_id = await env.create(max_steps=None)

    await env.converse(agent_id, QUESTION)

    assert EventName.RUN_LIMIT_REACHED not in env.names()
    assert env.tools.executed == [TOOL] * UNLIMITED_STEPS


@pytest.mark.parametrize("decision", [Decision.ALLOW, Decision.DENY])
async def test_step_limit_hook_decides_without_the_user(env: AgentEnv, decision: Decision) -> None:
    env.tools.add(TOOL)
    env.hooks.script(HookEvent.RUN_LIMIT_REACHED, decision)
    env.gateway.outcomes = [tool_response(call()), text_response(ANSWER)]
    agent_id = await env.create(max_steps=STEP_LIMIT)

    await env.converse(agent_id, QUESTION)

    expected = RunStopReason.END if decision is Decision.ALLOW else RunStopReason.LIMIT
    assert env.payloads(PermissionRequest) == []
    assert env.stop_reasons() == [expected.value]


async def test_denied_message_never_reaches_the_inbox(env: AgentEnv) -> None:
    env.hooks.script(HookEvent.MESSAGE_RECEIVED, Decision.DENY, reason=FAILURE)
    agent_id = await env.create()

    await env.converse(agent_id, QUESTION)

    assert await env.store.pending_inbox(agent_id) == []
    assert await env.store.get_messages(agent_id) == []
    assert env.gateway.requests == []
    assert EventName.ERROR in env.names()
    assert EventName.HOOK_FIRED in env.names()


async def test_hook_can_rewrite_an_incoming_message(env: AgentEnv) -> None:
    rewritten: list[JsonValue] = [TextBlock(text=REPLACED).model_dump(mode="json")]
    env.hooks.script(HookEvent.MESSAGE_RECEIVED, updated={"content": rewritten})
    env.gateway.outcomes = [text_response(ANSWER)]
    agent_id = await env.create()

    await env.converse(agent_id, QUESTION)

    assert (await env.store.get_messages(agent_id))[0].content == [TextBlock(text=REPLACED)]


@pytest.mark.parametrize(
    ("event", "model_calls"),
    [(HookEvent.RUN_START, 0), (HookEvent.BEFORE_MODEL_CALL, 0), (HookEvent.AFTER_MODEL_CALL, 1)],
)
async def test_hook_deny_ends_the_run(env: AgentEnv, event: HookEvent, model_calls: int) -> None:
    env.tools.add(TOOL)
    env.hooks.script(event, Decision.DENY)
    env.gateway.outcomes = [tool_response(call())]
    agent_id = await env.create()

    await env.converse(agent_id, QUESTION)

    messages = await env.store.get_messages(agent_id)
    assert env.stop_reasons() == [RunStopReason.HOOK_DENIED.value]
    assert len(env.gateway.requests) == model_calls
    assert env.tools.executed == []
    assert env.manager.get(agent_id).status is AgentStatus.IDLE
    assert [m.role for m in messages if m.role is not U] == [A, T] * model_calls


async def test_hook_context_reaches_the_next_model_call_once(env: AgentEnv) -> None:
    env.hooks.script(HookEvent.RUN_START, additional_context=CONTEXT_NOTE)
    env.gateway.outcomes = [text_response(ANSWER), text_response(ANSWER)]
    agent_id = await env.create()

    await env.converse(agent_id, QUESTION)
    await env.converse(agent_id, FOLLOW_UP)

    first, second = (request.model_dump_json() for request in env.gateway.requests)
    last_message = env.gateway.requests[0].messages[-1].model_dump_json()
    assert CONTEXT_NOTE in last_message
    assert CONTEXT_NOTE not in str(env.gateway.requests[0].system)
    assert CONTEXT_NOTE in first
    assert CONTEXT_NOTE not in second
    assert (await env.store.get_agent(agent_id)).pending_hook_context == []


@pytest.mark.acceptance("04-11.1")
async def test_agents_run_side_by_side(env: AgentEnv) -> None:
    started, release = env.tools.add_blocking(TOOL)
    env.gateway.outcomes = [tool_response(call()), text_response(ANSWER), text_response(ANSWER)]
    blocked = await env.create()
    free = await env.create()
    await env.say(blocked, QUESTION)
    await started.wait()

    await env.converse(free, QUESTION)
    status_while_other_ran = env.manager.get(blocked).status
    release.set()
    await env.manager.wait_idle(blocked)

    assert status_while_other_ran is AgentStatus.RUNNING
    assert [m.role for m in await env.store.get_messages(free)] == [U, A]
    assert [m.role for m in await env.store.get_messages(blocked)] == [U, A, T, A]
