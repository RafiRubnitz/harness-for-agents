"""Crash recovery and orderly restarts (04 §7a)."""

import asyncio

import pytest

from harness.core.consts import (
    META_ERROR_CODE,
    AddressType,
    AgentStatus,
    Decision,
    ErrorCode,
    HookEvent,
    MessageRole,
    PermissionAnswer,
    TaskStatus,
    ToolCallStatus,
)
from harness.core.events import AgentStatusChanged, PermissionRequest
from harness.core.records import BackgroundTaskRecord, ToolCallRecord
from harness.core.types import TaskId, ToolCallBlock, ToolCallRecordId, ToolResultBlock
from tests.fakes.adapter import text_response, tool_response
from tests.unit.agents.conftest import AgentEnv
from tests.utils import make_message

QUESTION = "run the long job"
ANSWER = "finished"
TOOL = "long_job"
CALL = ToolCallBlock(id="call_1", name=TOOL, input={})
TASK_ID = TaskId("task_0001")

U, A, T = MessageRole.USER, MessageRole.ASSISTANT, MessageRole.TOOL


@pytest.mark.acceptance("04-11.7")
async def test_interrupted_tool_is_reported_and_not_run_again(env: AgentEnv) -> None:
    started, _ = env.tools.add_blocking(TOOL)
    env.gateway.outcomes = [tool_response(CALL), text_response(ANSWER)]
    agent_id = await env.create()
    await env.say(agent_id, QUESTION)
    await started.wait()

    await env.crash()
    await env.restart().start()
    await env.manager.wait_idle(agent_id)

    messages = await env.store.get_messages(agent_id)
    result = messages[2].content[0]
    audit = await env.store.get_tool_calls(agent_id)
    recovered = [p.recovered for p in env.payloads(AgentStatusChanged)]
    assert [m.role for m in messages] == [U, A, T, A]
    assert isinstance(result, ToolResultBlock)
    assert result.is_error
    assert env.tools.executed == [TOOL]
    assert [t.status for t in audit] == [ToolCallStatus.INTERRUPTED]
    assert audit[0].metadata[META_ERROR_CODE] == ErrorCode.TOOL_INTERRUPTED.value
    assert recovered[0] is True
    assert env.manager.get(agent_id).status is AgentStatus.IDLE


async def test_answer_cut_mid_stream_is_asked_again(env: AgentEnv) -> None:
    env.gateway.hold = asyncio.Event()
    env.gateway.outcomes = [text_response(ANSWER), text_response(ANSWER)]
    agent_id = await env.create()
    await env.say(agent_id, QUESTION)
    await env.gateway.streaming.wait()

    await env.crash()
    env.gateway.hold = None
    await env.restart().start()
    await env.manager.wait_idle(agent_id)

    run_ends = [i.payload for i in env.hooks.emitted if i.event is HookEvent.RUN_END]
    assert [m.role for m in await env.store.get_messages(agent_id)] == [U, A]
    assert [p["steps"] for p in run_ends] == [1]
    assert env.gateway.outcomes == []


async def test_open_approval_is_asked_again_after_a_crash(env: AgentEnv) -> None:
    env.tools.add(TOOL, decision=Decision.ASK)
    env.gateway.outcomes = [tool_response(CALL), text_response(ANSWER)]
    agent_id = await env.create()
    await env.say(agent_id, QUESTION)
    await env.until_status(agent_id, AgentStatus.WAITING_APPROVAL)

    await env.crash()
    await env.restart().start()
    await env.until_status(agent_id, AgentStatus.WAITING_APPROVAL)
    request = env.payloads(PermissionRequest)[0].request
    open_requests = await env.store.pending_approvals(agent_id)
    await env.manager.respond_permission(request.request_id, PermissionAnswer.APPROVE)
    await env.manager.wait_idle(agent_id)

    assert request.call_id == CALL.id
    assert [p.request_id for p in open_requests] == [request.request_id]
    assert env.tools.executed == [TOOL]
    assert [m.role for m in await env.store.get_messages(agent_id)] == [U, A, T, A]


async def test_orderly_shutdown_resumes_without_the_recovered_mark(env: AgentEnv) -> None:
    started, _ = env.tools.add_blocking(TOOL)
    env.gateway.outcomes = [tool_response(CALL), text_response(ANSWER)]
    agent_id = await env.create()
    await env.say(agent_id, QUESTION)
    await started.wait()

    await env.manager.shutdown(stop_agents=False)
    stored_status = (await env.store.get_agent(agent_id)).status
    await env.restart().start()
    await env.manager.wait_idle(agent_id)

    assert stored_status is AgentStatus.RUNNING
    assert next(p.recovered for p in env.payloads(AgentStatusChanged)) is False
    assert env.manager.get(agent_id).recovered is False
    assert env.manager.get(agent_id).status is AgentStatus.IDLE


async def test_paused_agent_stays_paused_after_a_restart(env: AgentEnv) -> None:
    started, release = env.tools.add_blocking(TOOL)
    env.gateway.outcomes = [tool_response(CALL), text_response(ANSWER)]
    agent_id = await env.create()
    await env.say(agent_id, QUESTION)
    await started.wait()
    await env.manager.pause(agent_id)
    release.set()
    await env.until_status(agent_id, AgentStatus.PAUSED)
    await env.manager.wait_idle(agent_id)

    await env.crash()
    await env.restart().start()
    status_after_restart = env.manager.get(agent_id).status
    await env.manager.resume(agent_id)
    await env.manager.wait_idle(agent_id)

    assert status_after_restart is AgentStatus.PAUSED
    assert [m.role for m in await env.store.get_messages(agent_id)] == [U, A, T, A]


async def test_interrupted_background_task_fails_and_wakes_its_agent(env: AgentEnv) -> None:
    agent_id = await env.create()
    request = await env.store.append_message(make_message("msg_seed", agent_id=agent_id))
    now = env.deps.clock.now()
    await env.store.save_tool_call(
        ToolCallRecord(
            id=ToolCallRecordId("tc_seed"),
            agent_id=agent_id,
            call_id=CALL.id,
            request_message_id=request.id,
            tool_name=TOOL,
            created_at=now,
        )
    )
    await env.store.save_task(
        BackgroundTaskRecord(
            id=TASK_ID,
            agent_id=agent_id,
            call_id=CALL.id,
            tool=TOOL,
            status=TaskStatus.RUNNING,
            started_at=now,
        )
    )
    env.gateway.outcomes = [text_response(ANSWER)]

    await env.crash()
    await env.restart().start()
    await env.manager.wait_idle(agent_id)

    tasks = await env.store.list_tasks(agent_id)
    notice = (await env.store.get_messages(agent_id))[1]
    assert [(t.status, t.interrupted) for t in tasks] == [(TaskStatus.FAILED, True)]
    assert notice.origin is not None
    assert (notice.origin.source.type, notice.origin.source.id) == (AddressType.TASK, TASK_ID)
    assert env.manager.get(agent_id).status is AgentStatus.IDLE
    assert env.gateway.outcomes == []
