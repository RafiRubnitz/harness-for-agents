"""Background tasks through the real core: shell in the background, the agent woken at the end."""

import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from harness.app import CoreOptions, open_core
from harness.config import FileConfigManager
from harness.core.commands import AgentCreate, AgentSend, PermissionRespond
from harness.core.consts import (
    META_TASK_ID,
    AddressType,
    AgentStatus,
    MessageRole,
    PermissionAnswer,
    TaskStatus,
)
from harness.core.events import (
    AgentCreated,
    AgentStatusChanged,
    Event,
    MessageAppended,
    PermissionRequest,
    TaskStatusChanged,
)
from harness.core.handle import CoreHandle
from harness.core.records import AgentSettings
from harness.core.types import Address, AgentId, TextBlock, ToolCallBlock, ToolResultBlock
from harness.core.utils.paths import HarnessHome
from tests.consts import AGENT_NAME, USER_ID
from tests.fakes.adapter import text_response, tool_response
from tests.fakes.embedder import FakeEmbedder
from tests.fakes.gateway import FakeGateway
from tests.fakes.secrets import FakeSecretStore

QUESTION = "build the project"
WORKING = "the build runs in the background"
REPORT = "the build is done"
PARTS = ("background", "build", "output")
WORD = "-".join(PARTS)
PYTHON = sys.executable.replace("\\", "/")
SCRIPT = f"import time; time.sleep(0.5); print('-'.join({list(PARTS)!r}))"
COMMAND = f'& "{PYTHON}" -c "{SCRIPT}"'
BUILD = ToolCallBlock(
    id="call_1",
    name="shell",
    input={"command": COMMAND, "run_in_background": True, "reason": "the user asked for a build"},
)
IDLE_TURNS = 2


async def _core(home: HarnessHome, gateway: FakeGateway) -> AsyncIterator[CoreHandle]:
    config = FileConfigManager(home, secrets=FakeSecretStore(), env={}, username=USER_ID)
    options = CoreOptions(home=home, config=config, gateway=gateway, embedder=FakeEmbedder())
    async with open_core(options) as core:
        yield core


async def _agent(core: CoreHandle, work: Path) -> AgentId:
    events = core.events()
    settings = AgentSettings(working_dir=str(work))
    await core.submit(AgentCreate(agent_name=AGENT_NAME, tools=["shell"], settings=settings))
    async for event in events:
        if isinstance(event.payload, AgentCreated):
            return event.payload.agent.id
    raise AssertionError


async def _follow(core: CoreHandle, agent_id: AgentId) -> list[Event]:
    """Send the question, approve what is asked, and return once the agent rested twice."""
    events = core.events()
    seen: list[Event] = []
    idle = 0
    await core.submit(AgentSend(agent_id=agent_id, content=[TextBlock(text=QUESTION)]))
    async for event in events:
        seen.append(event)
        payload = event.payload
        if isinstance(payload, PermissionRequest):
            await core.submit(
                PermissionRespond(
                    request_id=payload.request.request_id, decision=PermissionAnswer.APPROVE
                )
            )
        if isinstance(payload, AgentStatusChanged) and payload.status is AgentStatus.IDLE:
            idle += 1
            if idle == IDLE_TURNS:
                break
    return seen


def _text(blocks: list[object]) -> str:
    return "".join(block.text for block in blocks if isinstance(block, TextBlock))


@pytest.mark.acceptance("08-10.10")
@pytest.mark.acceptance("08-10.12")
@pytest.mark.timeout(60)
async def test_a_background_shell_returns_at_once_and_wakes_the_idle_agent_when_done(
    tmp_harness_home: HarnessHome, tmp_path: Path
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    gateway = FakeGateway([tool_response(BUILD), text_response(WORKING), text_response(REPORT)])

    seen: list[Event] = []
    async for core in _core(tmp_harness_home, gateway):
        seen = await _follow(core, await _agent(core, work))

    _, second, third = gateway.requests
    (result,) = [
        block
        for message in second.messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    ]
    started = json.loads(_text(list(result.content)))
    statuses = [e.payload.status for e in seen if isinstance(e.payload, TaskStatusChanged)]
    woke = third.messages[-1]
    appended = [e.payload.message for e in seen if isinstance(e.payload, MessageAppended)]
    sources = [message.origin.source for message in appended if message.origin]
    notice = _text(list(woke.content))
    assert started["status"] == TaskStatus.RUNNING.value
    assert WORD not in second.model_dump_json()  # the agent went on before the build ended
    assert statuses == [TaskStatus.RUNNING, TaskStatus.DONE]
    assert woke.role == MessageRole.USER.value
    assert Address(type=AddressType.TASK, id=started[META_TASK_ID]) in sources
    assert started[META_TASK_ID] in notice
    assert WORD in notice
