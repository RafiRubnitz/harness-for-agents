"""The ``messaging`` tools: list_agents, send_message, reply (phase 2, 03 §4-§5)."""

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import JsonValue

from harness.core.consts import AddressType, AgentStatus, Decision, ErrorCode, RejectReason
from harness.core.errors import AgentError
from harness.core.messaging import AgentInfo
from harness.core.records import ToolOutput
from harness.core.types import Address, TextBlock
from harness.tools.builtin.messaging import messaging_tools
from harness.tools.types import ToolContext
from tests.consts import AGENT_ID, OTHER_AGENT_ID
from tests.fakes.clock import FakeClock
from tests.fakes.messenger import FakeMessenger

COLLEAGUE = AgentInfo(
    id=OTHER_AGENT_ID,
    name="reviewer",
    address="agent:reviewer",
    description="בודק קוד",
    template=None,
    status=AgentStatus.IDLE,
    accepts_messages=True,
)
CONTENT = "please look at the diff"
MESSAGE_ID = "msg_sent_1"
BLOCKED = "agent:hermit"
SELF = Address(type=AddressType.AGENT, id=AGENT_ID)


@pytest.fixture
def messenger() -> FakeMessenger:
    return FakeMessenger([COLLEAGUE])


async def _call(
    messenger: FakeMessenger, name: str, tmp_path: Path, clock: FakeClock, **args: JsonValue
) -> ToolOutput:
    (found,) = [tool for tool in messaging_tools(messenger) if tool.definition.name == name]
    ctx = ToolContext(
        agent_id=AGENT_ID,
        run_id=None,
        call_id="call_1",
        working_dir=tmp_path,
        allowed_paths=[tmp_path],
        cancel=asyncio.Event(),
        emit_progress=lambda _text: None,
        clock=clock,
    )
    return await found.run(dict(args), ctx)


def _text(output: ToolOutput) -> str:
    return "".join(block.text for block in output.content if isinstance(block, TextBlock))


def test_no_messaging_tool_needs_approval_and_only_the_listing_is_read_only(
    messenger: FakeMessenger,
) -> None:
    definitions = [tool.definition for tool in messaging_tools(messenger)]

    assert {d.name for d in definitions} == {"list_agents", "send_message", "reply"}
    assert {d.default_permission for d in definitions} == {Decision.ALLOW}
    assert {d.name for d in definitions if d.read_only} == {"list_agents"}
    assert len({d.group for d in definitions}) == 1


async def test_the_listing_shows_the_colleagues_as_the_caller_sees_them(
    messenger: FakeMessenger, tmp_path: Path, clock: FakeClock
) -> None:
    output = await _call(messenger, "list_agents", tmp_path, clock)

    assert not output.is_error
    assert json.loads(_text(output)) == [COLLEAGUE.model_dump(mode="json")]
    assert COLLEAGUE.description in _text(output)  # Hebrew is not escaped
    assert messenger.viewers == [SELF]


async def test_an_empty_listing_says_so(
    messenger: FakeMessenger, tmp_path: Path, clock: FakeClock
) -> None:
    messenger.agents.clear()

    output = await _call(messenger, "list_agents", tmp_path, clock)

    assert not output.is_error
    assert _text(output)


async def test_a_message_is_sent_as_the_calling_agent(
    messenger: FakeMessenger, tmp_path: Path, clock: FakeClock
) -> None:
    output = await _call(
        messenger,
        "send_message",
        tmp_path,
        clock,
        to=[COLLEAGUE.address],
        content=CONTENT,
        expects_reply=True,
    )

    ((sender, message),) = messenger.sent
    shown = json.loads(_text(output))
    assert not output.is_error
    assert sender == SELF  # from the context, never from the arguments
    assert (message.to, message.content, message.expects_reply) == (
        [COLLEAGUE.address],
        CONTENT,
        True,
    )
    assert shown["delivered"] == [COLLEAGUE.address]
    assert shown["rejected"] == []


async def test_a_partly_delivered_message_reports_both_sides(
    messenger: FakeMessenger, tmp_path: Path, clock: FakeClock
) -> None:
    messenger.refusals[BLOCKED] = RejectReason.BLOCKED.value

    output = await _call(
        messenger, "send_message", tmp_path, clock, to=[COLLEAGUE.address, BLOCKED], content=CONTENT
    )

    shown = json.loads(_text(output))
    assert not output.is_error
    assert shown["delivered"] == [COLLEAGUE.address]
    assert shown["rejected"] == [[BLOCKED, RejectReason.BLOCKED.value]]


async def test_a_message_nobody_got_is_an_error_with_the_reason(
    messenger: FakeMessenger, tmp_path: Path, clock: FakeClock
) -> None:
    messenger.refusals[BLOCKED] = RejectReason.BLOCKED.value

    output = await _call(messenger, "send_message", tmp_path, clock, to=[BLOCKED], content=CONTENT)

    assert output.is_error
    assert RejectReason.BLOCKED.value in _text(output)


async def test_a_message_to_nobody_is_not_sent(
    messenger: FakeMessenger, tmp_path: Path, clock: FakeClock
) -> None:
    output = await _call(messenger, "send_message", tmp_path, clock, to=[], content=CONTENT)

    assert output.is_error
    assert messenger.sent == []


async def test_a_reply_names_the_message_it_answers(
    messenger: FakeMessenger, tmp_path: Path, clock: FakeClock
) -> None:
    output = await _call(
        messenger, "reply", tmp_path, clock, message_id=MESSAGE_ID, content=CONTENT
    )

    ((sender, message),) = messenger.sent
    assert not output.is_error
    assert sender == SELF
    assert (message.reply_to, message.content) == (MESSAGE_ID, CONTENT)


async def test_a_reply_to_an_unknown_message_is_an_error_not_a_crash(
    messenger: FakeMessenger, tmp_path: Path, clock: FakeClock
) -> None:
    problem = "no such message"
    messenger.reply_error = AgentError(problem, code=ErrorCode.TOOL_FAILED)

    output = await _call(
        messenger, "reply", tmp_path, clock, message_id=MESSAGE_ID, content=CONTENT
    )

    assert output.is_error
    assert problem in _text(output)
    assert messenger.sent == []
