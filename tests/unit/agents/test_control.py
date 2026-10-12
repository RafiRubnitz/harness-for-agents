"""Run control: approvals, stop, pause / resume, errors and retry (04 §4, §5.2)."""

import asyncio

import pytest

from harness.agents.consts import MESSAGE_STOPPED
from harness.core.consts import (
    META_ERROR_CODE,
    AgentStatus,
    CompactReason,
    Decision,
    ErrorCode,
    ErrorKind,
    EventName,
    HookEvent,
    MessageRole,
    PermissionAnswer,
    PermissionResolution,
    PermissionScope,
    ProviderId,
    RunStopReason,
    ToolCallStatus,
)
from harness.core.errors import AgentNotFoundError, BudgetExceededError, InvalidTransitionError
from harness.core.events import Error, PermissionRequest, PermissionResolved
from harness.core.types import (
    AgentId,
    PermissionRequestId,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
)
from tests.fakes.adapter import gateway_error, text_response, tool_response
from tests.unit.agents.conftest import AgentEnv

QUESTION = "please clean the folder"
SECOND_QUESTION = "try something else"
ANSWER = "done"
SUMMARY = "SUMMARY: the folder was cleaned"
TOOL = "wipe"
REASON = "the folder is full"
DENY_NOTE = "not that folder"
UNKNOWN_AGENT = AgentId("agt_missing")
UNKNOWN_REQUEST = PermissionRequestId("perm_missing")

U, A, T = MessageRole.USER, MessageRole.ASSISTANT, MessageRole.TOOL


def call(call_id: str = "call_1") -> ToolCallBlock:
    """A tool call as the model sends it."""
    return ToolCallBlock(id=call_id, name=TOOL, input={"reason": REASON})


async def ask_for_approval(env: AgentEnv) -> tuple[AgentId, PermissionRequestId]:
    """Run an agent up to an open approval request for ``TOOL``."""
    env.tools.add(TOOL, decision=Decision.ASK)
    env.gateway.outcomes = [tool_response(call()), text_response(ANSWER)]
    agent_id = await env.create()
    await env.say(agent_id, QUESTION)
    await env.until_status(agent_id, AgentStatus.WAITING_APPROVAL)
    return agent_id, env.payloads(PermissionRequest)[0].request.request_id


async def tool_result(env: AgentEnv, agent_id: AgentId) -> ToolResultBlock:
    """The first tool result in the agent's history."""
    messages = await env.store.get_messages(agent_id)
    block = next(m for m in messages if m.role is T).content[0]
    assert isinstance(block, ToolResultBlock)
    return block


async def test_approved_tool_runs(env: AgentEnv) -> None:
    agent_id, request_id = await ask_for_approval(env)
    request = env.payloads(PermissionRequest)[0].request
    pending = await env.store.pending_approvals(agent_id)

    await env.manager.respond_permission(request_id, PermissionAnswer.APPROVE)
    await env.manager.wait_idle(agent_id)

    resolved = env.payloads(PermissionResolved)[0]
    assert (request.tool, request.call_id, request.reason) == (TOOL, call().id, REASON)
    assert request.subject is HookEvent.BEFORE_TOOL_CALL
    assert [p.request_id for p in pending] == [request_id]
    assert resolved.decision is PermissionResolution.APPROVE
    assert env.tools.executed == [TOOL]
    assert not (await tool_result(env, agent_id)).is_error
    assert await env.store.pending_approvals(agent_id) == []
    assert env.manager.get(agent_id).pending_approvals == 0


async def test_denied_tool_returns_an_error_to_the_model(env: AgentEnv) -> None:
    agent_id, request_id = await ask_for_approval(env)

    await env.manager.respond_permission(request_id, PermissionAnswer.DENY, deny_message=DENY_NOTE)
    await env.manager.wait_idle(agent_id)

    result = await tool_result(env, agent_id)
    audit = await env.store.get_tool_calls(agent_id)
    assert env.tools.executed == []
    assert result.is_error
    assert result.content == [TextBlock(text=DENY_NOTE)]
    assert [t.status for t in audit] == [ToolCallStatus.DENIED]
    assert env.stop_reasons() == [RunStopReason.END.value]


async def test_always_approve_is_remembered_for_the_agent(env: AgentEnv) -> None:
    agent_id, request_id = await ask_for_approval(env)
    env.gateway.outcomes.insert(0, tool_response(call("call_2")))

    await env.manager.respond_permission(
        request_id, PermissionAnswer.APPROVE, PermissionScope.AGENT
    )
    await env.manager.wait_idle(agent_id)

    assert env.tools.always_allowed == [TOOL]
    assert env.tools.executed == [TOOL, TOOL]
    assert len(env.payloads(PermissionRequest)) == 1


@pytest.mark.acceptance("04-11.2")
async def test_waiting_agent_does_not_block_others(env: AgentEnv) -> None:
    waiting, request_id = await ask_for_approval(env)
    env.gateway.outcomes.insert(0, text_response(ANSWER))
    other = await env.create()

    await env.converse(other, QUESTION)
    status_of_waiting = env.manager.get(waiting).status
    await env.manager.respond_permission(request_id, PermissionAnswer.APPROVE)
    await env.manager.wait_idle(waiting)

    assert status_of_waiting is AgentStatus.WAITING_APPROVAL
    assert [m.role for m in await env.store.get_messages(other)] == [U, A]
    assert env.manager.get(waiting).status is AgentStatus.IDLE


@pytest.mark.acceptance("04-11.3")
async def test_stop_keeps_what_was_streamed(env: AgentEnv) -> None:
    env.gateway.hold = asyncio.Event()
    env.gateway.outcomes = [text_response(ANSWER)]
    agent_id = await env.create()
    await env.say(agent_id, QUESTION)
    await env.gateway.streaming.wait()

    await env.manager.stop(agent_id)

    last = (await env.store.get_messages(agent_id))[-1]
    assert (last.role, last.content) == (A, [TextBlock(text=ANSWER)])
    assert last.stop_reason == MESSAGE_STOPPED
    assert env.manager.get(agent_id).status is AgentStatus.IDLE
    assert env.stop_reasons() == [RunStopReason.STOP.value]
    assert await env.store.open_run(agent_id) is None


@pytest.mark.acceptance("04-11.9")
async def test_stop_closes_an_open_approval(env: AgentEnv) -> None:
    agent_id, request_id = await ask_for_approval(env)

    await env.manager.stop(agent_id)

    result = await tool_result(env, agent_id)
    resolved = env.payloads(PermissionResolved)
    audit = await env.store.get_tool_calls(agent_id)
    assert [(p.request_id, p.decision) for p in resolved] == [
        (request_id, PermissionResolution.CANCELLED)
    ]
    assert result.is_error
    assert [t.status for t in audit] == [ToolCallStatus.CANCELLED]
    assert audit[0].metadata[META_ERROR_CODE] == ErrorCode.TOOL_CANCELLED.value
    assert env.tools.executed == []
    assert await env.store.pending_approvals(agent_id) == []
    assert env.manager.get(agent_id).status is AgentStatus.IDLE
    with pytest.raises(InvalidTransitionError):
        await env.manager.respond_permission(request_id, PermissionAnswer.APPROVE)


async def test_stop_cancels_a_running_tool(env: AgentEnv) -> None:
    started, _ = env.tools.add_blocking(TOOL)
    env.gateway.outcomes = [tool_response(call())]
    agent_id = await env.create()
    await env.say(agent_id, QUESTION)
    await started.wait()

    await env.manager.stop(agent_id)

    assert (await tool_result(env, agent_id)).is_error
    assert [m.role for m in await env.store.get_messages(agent_id)] == [U, A, T]
    assert env.manager.get(agent_id).status is AgentStatus.IDLE


@pytest.mark.acceptance("04-11.4")
async def test_pause_waits_for_the_step_and_resume_continues(env: AgentEnv) -> None:
    started, release = env.tools.add_blocking(TOOL)
    env.gateway.outcomes = [tool_response(call()), text_response(ANSWER)]
    agent_id = await env.create()
    await env.say(agent_id, QUESTION)
    await started.wait()

    await env.manager.pause(agent_id)
    status_before_step_ended = env.manager.get(agent_id).status
    release.set()
    await env.until_status(agent_id, AgentStatus.PAUSED)
    paused_run = await env.store.open_run(agent_id)
    calls_while_paused = len(env.gateway.requests)
    await env.manager.resume(agent_id)
    await env.manager.wait_idle(agent_id)

    run_starts = [e for e in env.hooks.events() if e is HookEvent.RUN_START]
    assert status_before_step_ended is AgentStatus.RUNNING
    assert paused_run is not None
    assert calls_while_paused == 1
    assert len(run_starts) == 1
    assert [m.role for m in await env.store.get_messages(agent_id)] == [U, A, T, A]
    assert env.manager.get(agent_id).status is AgentStatus.IDLE


@pytest.mark.acceptance("04-11.10")
async def test_retry_runs_the_failed_step_again_in_the_same_run(env: AgentEnv) -> None:
    env.gateway.outcomes = [gateway_error(ErrorKind.AUTH), text_response(ANSWER)]
    agent_id = await env.create()
    await env.say(agent_id, QUESTION)
    await env.until_status(agent_id, AgentStatus.ERROR)
    failed_run = await env.store.open_run(agent_id)
    shown = env.payloads(Error)[0]

    await env.manager.retry(agent_id)
    await env.manager.wait_idle(agent_id)

    run_ends = [i.payload for i in env.hooks.emitted if i.event is HookEvent.RUN_END]
    assert failed_run is not None
    assert (shown.code, shown.kind) == (ErrorCode.MODEL_AUTH, ErrorKind.AUTH)
    assert HookEvent.MODEL_ERROR in env.hooks.events()
    assert [m.role for m in await env.store.get_messages(agent_id)] == [U, A]
    assert [p["steps"] for p in run_ends] == [failed_run.step_count]
    assert env.manager.get(agent_id).status is AgentStatus.IDLE


async def test_new_message_after_an_error_opens_a_new_run(env: AgentEnv) -> None:
    env.gateway.outcomes = [gateway_error(ErrorKind.AUTH), text_response(ANSWER)]
    agent_id = await env.create()
    await env.say(agent_id, QUESTION)
    await env.until_status(agent_id, AgentStatus.ERROR)

    await env.converse(agent_id, SECOND_QUESTION)

    assert env.stop_reasons() == [RunStopReason.ERROR.value, RunStopReason.END.value]
    assert [m.role for m in await env.store.get_messages(agent_id)] == [U, U, A]
    assert env.manager.get(agent_id).status is AgentStatus.IDLE


async def test_budget_error_pauses_the_agent(env: AgentEnv) -> None:
    over_budget = BudgetExceededError(
        ErrorCode.MODEL_BUDGET_EXCEEDED.value, provider=ProviderId.ANTHROPIC, model=""
    )
    env.gateway.outcomes = [over_budget, text_response(ANSWER)]
    agent_id = await env.create()
    await env.say(agent_id, QUESTION)

    await env.until_status(agent_id, AgentStatus.PAUSED)
    await env.manager.resume(agent_id)
    await env.manager.wait_idle(agent_id)

    assert env.payloads(Error)[0].code is ErrorCode.MODEL_BUDGET_EXCEEDED
    assert env.manager.get(agent_id).status is AgentStatus.IDLE


async def test_context_overflow_is_answered_by_compacting_and_trying_again(env: AgentEnv) -> None:
    env.gateway.outcomes = [
        text_response(ANSWER),
        gateway_error(ErrorKind.CONTEXT_OVERFLOW),
        text_response(SUMMARY),
        text_response(ANSWER),
    ]
    agent_id = await env.create()
    await env.converse(agent_id, QUESTION)

    await env.converse(agent_id, SECOND_QUESTION)

    (summary,) = await env.store.get_summaries(agent_id)
    assert summary.reason is CompactReason.OVERFLOW
    assert env.gateway.outcomes == []
    assert SUMMARY in env.gateway.requests[-1].messages[0].model_dump_json()
    assert env.manager.get(agent_id).status is AgentStatus.IDLE
    assert env.payloads(Error) == []


async def test_context_overflow_without_compaction_is_an_error(env: AgentEnv) -> None:
    env.gateway.outcomes = [gateway_error(ErrorKind.CONTEXT_OVERFLOW)]
    agent_id = await env.create()

    await env.say(agent_id, QUESTION)
    await env.until_status(agent_id, AgentStatus.ERROR)

    assert env.payloads(Error)[0].kind is ErrorKind.CONTEXT_OVERFLOW


@pytest.mark.parametrize("decision", [Decision.DENY, Decision.ASK])
async def test_model_error_hook_can_give_up_or_ask(env: AgentEnv, decision: Decision) -> None:
    env.hooks.script(HookEvent.MODEL_ERROR, decision)
    env.gateway.outcomes = [gateway_error(ErrorKind.OVERLOADED), text_response(ANSWER)]
    agent_id = await env.create()
    await env.say(agent_id, QUESTION)

    if decision is Decision.ASK:
        await env.until_status(agent_id, AgentStatus.WAITING_APPROVAL)
        request_id = env.payloads(PermissionRequest)[0].request.request_id
        await env.manager.respond_permission(request_id, PermissionAnswer.APPROVE)
    await env.manager.wait_idle(agent_id)

    expected = RunStopReason.END if decision is Decision.ASK else RunStopReason.ERROR
    assert env.stop_reasons() == [expected.value]
    assert env.manager.get(agent_id).status is AgentStatus.IDLE


async def test_commands_outside_the_table_are_rejected(env: AgentEnv) -> None:
    agent_id = await env.create()

    for command in (env.manager.stop, env.manager.pause, env.manager.resume, env.manager.retry):
        with pytest.raises(InvalidTransitionError):
            await command(agent_id)
    with pytest.raises(InvalidTransitionError):
        await env.manager.unarchive(agent_id)
    with pytest.raises(InvalidTransitionError):
        await env.manager.respond_permission(UNKNOWN_REQUEST, PermissionAnswer.APPROVE)
    with pytest.raises(AgentNotFoundError):
        await env.manager.stop(UNKNOWN_AGENT)
    assert EventName.AGENT_STATUS not in env.names()
