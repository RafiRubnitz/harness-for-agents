"""User hooks through the real core, tools and hooks manager (05 §10)."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from harness.app import CoreOptions, open_core
from harness.config import FileConfigManager
from harness.core.commands import AgentCreate, AgentSend, PermissionRespond
from harness.core.consts import AgentStatus, ErrorCode, HookEvent, PermissionAnswer
from harness.core.events import (
    AgentCreated,
    AgentStatusChanged,
    Error,
    Event,
    HookFired,
    HookLoadFailed,
    PermissionRequest,
)
from harness.core.handle import CoreHandle
from harness.core.records import AgentSettings
from harness.core.types import AgentId, TextBlock, ToolCallBlock, ToolResultBlock
from harness.core.utils.paths import HarnessHome
from tests.consts import AGENT_NAME, USER_ID
from tests.fakes.adapter import text_response, tool_response
from tests.fakes.embedder import FakeEmbedder
from tests.fakes.gateway import FakeGateway
from tests.fakes.secrets import FakeSecretStore

QUESTION = "write the note"
ANSWER = "finished"
FILE = "note.txt"
CONTENT = "a line for the note\n"
HOOK_REASON = "notes are frozen this week"
HOOK_CONTEXT = "the formatter reported two warnings"
HOOK_NAME = "guard"
MODULE = "guard"
WRITE = ToolCallBlock(
    id="call_1",
    name="write_file",
    input={"path": FILE, "content": CONTENT, "reason": "the user asked for a note"},
)
READ = ToolCallBlock(id="call_2", name="read_file", input={"path": FILE})


class Harness:
    """A real core (scripted model only) whose hooks the test writes as files."""

    def __init__(self, home: HarnessHome, work: Path) -> None:
        self.home = home
        self.work = work
        self.gateway = FakeGateway()
        self.seen: list[Event] = []

    def hook(self, event: HookEvent, answer: str, *, tool: str | None = None) -> None:
        """Define one global Python hook that returns ``HookOutput(<answer>)``."""
        body = (
            "from harness.hooks import HookInput, HookOutput\n\n"
            f"async def hook(inp: HookInput) -> HookOutput:\n    return HookOutput({answer})\n"
        )
        (self.home.hooks_dir / f"{MODULE}.py").write_text(body, encoding="utf-8")
        match = f"    match: {{tool: {tool}}}\n" if tool else ""
        self.home.hooks_file.write_text(
            f"hooks:\n  - name: {HOOK_NAME}\n    event: {event.value}\n{match}"
            f"    run: python:{MODULE}:hook\n",
            encoding="utf-8",
        )

    async def open(self) -> AsyncIterator[CoreHandle]:
        config = FileConfigManager(self.home, secrets=FakeSecretStore(), env={}, username=USER_ID)
        options = CoreOptions(
            home=self.home, config=config, gateway=self.gateway, embedder=FakeEmbedder()
        )
        async with open_core(options) as core:
            yield core

    async def agent(self, core: CoreHandle) -> AgentId:
        events = core.events()
        settings = AgentSettings(working_dir=str(self.work))
        await core.submit(AgentCreate(agent_name=AGENT_NAME, tools=["files"], settings=settings))
        async for event in events:
            if isinstance(event.payload, AgentCreated):
                return event.payload.agent.id
        raise AssertionError

    async def ask(
        self,
        core: CoreHandle,
        agent_id: AgentId,
        *,
        approve: bool = True,
        then: type[object] | None = None,
    ) -> None:
        """Send the question and follow the run to idle (and to a ``then`` event)."""
        events = core.events()
        idle = False
        await core.submit(AgentSend(agent_id=agent_id, content=[TextBlock(text=QUESTION)]))
        async for event in events:
            self.seen.append(event)
            payload = event.payload
            if isinstance(payload, PermissionRequest):
                answer = PermissionAnswer.APPROVE if approve else PermissionAnswer.DENY
                await core.submit(
                    PermissionRespond(request_id=payload.request.request_id, decision=answer)
                )
            if isinstance(payload, AgentStatusChanged) and payload.status is AgentStatus.IDLE:
                idle = True
            if idle and (then is None or self.payloads(then)):
                return

    def payloads[T](self, kind: type[T]) -> list[T]:
        return [event.payload for event in self.seen if isinstance(event.payload, kind)]

    def results(self) -> list[ToolResultBlock]:
        """The tool results the model was shown in its last request."""
        blocks = [b for message in self.gateway.requests[-1].messages for b in message.content]
        return [block for block in blocks if isinstance(block, ToolResultBlock)]


@pytest.fixture
def harness(tmp_harness_home: HarnessHome, tmp_path: Path) -> Harness:
    work = tmp_path / "work"
    work.mkdir()
    return Harness(tmp_harness_home, work)


def _text(result: ToolResultBlock) -> str:
    return "".join(block.text for block in result.content if isinstance(block, TextBlock))


@pytest.mark.acceptance("05-10.2")
async def test_a_denying_hook_blocks_the_tool_and_the_model_reads_why(harness: Harness) -> None:
    harness.hook(
        HookEvent.BEFORE_TOOL_CALL, f"decision='deny', reason={HOOK_REASON!r}", tool="write_*"
    )
    harness.gateway.outcomes = [tool_response(WRITE), text_response(ANSWER)]

    async for core in harness.open():
        await harness.ask(core, await harness.agent(core))

    (result,) = harness.results()
    assert not (harness.work / FILE).exists()
    assert result.is_error
    assert HOOK_REASON in _text(result)
    assert harness.payloads(PermissionRequest) == []
    assert [fired.hook for fired in harness.payloads(HookFired)] == [HOOK_NAME]


@pytest.mark.acceptance("05-10.3")
async def test_an_asking_hook_opens_an_approval_with_its_reason(harness: Harness) -> None:
    (harness.work / FILE).write_text(CONTENT, encoding="utf-8")
    harness.hook(HookEvent.BEFORE_TOOL_CALL, f"decision='ask', reason={HOOK_REASON!r}")
    harness.gateway.outcomes = [tool_response(READ), text_response(ANSWER)]

    async for core in harness.open():
        await harness.ask(core, await harness.agent(core))

    (asked,) = harness.payloads(PermissionRequest)  # a read never asks on its own
    (result,) = harness.results()
    assert asked.request.hook_reasons == [HOOK_REASON]
    assert asked.request.tool == READ.name
    assert not result.is_error


@pytest.mark.acceptance("05-10.3")
async def test_an_allowing_hook_runs_the_tool_without_asking(harness: Harness) -> None:
    harness.hook(HookEvent.BEFORE_TOOL_CALL, "decision='allow'")
    harness.gateway.outcomes = [tool_response(WRITE), text_response(ANSWER)]

    async for core in harness.open():
        await harness.ask(core, await harness.agent(core))

    assert harness.payloads(PermissionRequest) == []  # a write asks without the hook
    assert (harness.work / FILE).read_bytes() == CONTENT.encode("utf-8")


@pytest.mark.acceptance("05-10.5")
@pytest.mark.parametrize("event", [HookEvent.BEFORE_TOOL_CALL, HookEvent.AFTER_TOOL_CALL])
async def test_what_a_hook_adds_reaches_the_model_at_the_next_step(
    harness: Harness, event: HookEvent
) -> None:
    (harness.work / FILE).write_text(CONTENT, encoding="utf-8")
    harness.hook(event, f"additional_context={HOOK_CONTEXT!r}")
    harness.gateway.outcomes = [tool_response(READ), text_response(ANSWER)]

    async for core in harness.open():
        await harness.ask(core, await harness.agent(core))

    first, second = (request.model_dump_json() for request in harness.gateway.requests)
    assert HOOK_CONTEXT not in first
    assert HOOK_CONTEXT in second


async def test_a_hook_that_denies_a_result_replaces_it_with_its_reason(harness: Harness) -> None:
    (harness.work / FILE).write_text(CONTENT, encoding="utf-8")
    harness.hook(HookEvent.AFTER_TOOL_CALL, f"decision='deny', reason={HOOK_REASON!r}")
    harness.gateway.outcomes = [tool_response(READ), text_response(ANSWER)]

    async for core in harness.open():
        await harness.ask(core, await harness.agent(core))

    (result,) = harness.results()
    assert result.is_error
    assert HOOK_REASON in _text(result)
    assert CONTENT.strip() not in _text(result)


@pytest.mark.acceptance("05-10.8")
async def test_denying_the_model_call_ends_the_run_with_a_recoverable_error(
    harness: Harness,
) -> None:
    harness.hook(HookEvent.BEFORE_MODEL_CALL, f"decision='deny', reason={HOOK_REASON!r}")

    async for core in harness.open():
        await harness.ask(core, await harness.agent(core), then=Error)

    (error,) = harness.payloads(Error)
    assert harness.gateway.requests == []
    assert (error.code, error.recoverable) == (ErrorCode.HOOK_DENIED, True)
    assert HOOK_REASON in error.message
    assert harness.payloads(AgentStatusChanged)[-1].status is AgentStatus.IDLE


async def test_a_hook_that_does_not_load_is_shown_and_the_agent_still_works(
    harness: Harness,
) -> None:
    harness.hook(HookEvent.BEFORE_TOOL_CALL, "decision='deny'")
    (harness.home.hooks_dir / f"{MODULE}.py").write_text("raise RuntimeError\n", encoding="utf-8")
    (harness.work / FILE).write_text(CONTENT, encoding="utf-8")
    harness.gateway.outcomes = [tool_response(READ), text_response(ANSWER)]

    async for core in harness.open():
        agent_id = await harness.agent(core)
        events = core.events()
        (harness.home.hooks_dir / f"{MODULE}.py").write_text("raise ValueError\n", encoding="utf-8")
        await harness.ask(core, agent_id)
        async for event in events:
            if isinstance(event.payload, HookLoadFailed):
                shown = event.payload
                break

    (result,) = harness.results()
    assert not result.is_error
    assert shown.hook == HOOK_NAME
