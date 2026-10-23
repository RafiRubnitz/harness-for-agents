"""Agent-to-agent messaging: addresses, limits, replies and unique names (phase 2, 03)."""

import pytest

from harness.agents import MessagingService
from harness.config import MessagingConfig
from harness.core.consts import (
    AddressType,
    AgentStatus,
    ConfigLayer,
    Decision,
    ErrorCode,
    HookEvent,
    RejectReason,
)
from harness.core.errors import AgentError
from harness.core.events import AgentUpdated
from harness.core.events import Warning as WarningEvent
from harness.core.messaging import SendMessageInput, SendMessageOutput
from harness.core.records import AgentSpec, InboxItem, MessageRecord
from harness.core.types import Address, AgentId, TextBlock
from tests.fakes.adapter import text_response
from tests.fakes.clock import FakeClock
from tests.unit.agents.conftest import AgentEnv

ANSWER = "noted"
REQUEST = "please review the plan"
REPLY = "the plan looks fine"
UNKNOWN = "agent:nobody"
UNKNOWN_MESSAGE = "msg_missing"
BAD_NAMES = ["Has Space", "UPPER", "", "-starts-with-dash"]
HOUR_S = 3600.0
SMALL_LIMIT = 2
SPARE_RESPONSES = 20
API_CALLER = Address(type=AddressType.API, id="local")


def _agent(agent_id: AgentId) -> Address:
    return Address(type=AddressType.AGENT, id=agent_id)


@pytest.fixture
def service(env: AgentEnv) -> MessagingService:
    """The real service over the manager under test; every run has an answer ready."""
    env.gateway.outcomes.extend(text_response(ANSWER) for _ in range(SPARE_RESPONSES))
    deps = env.deps
    return MessagingService(
        env.manager,
        memory=env.store,
        config=deps.config,
        clock=deps.clock,
        ids=deps.ids,
        bus=deps.bus,
    )


async def _limit(env: AgentEnv, agent_id: AgentId, key: str, value: int | list[str] | bool) -> None:
    await env.deps.config.set(f"messaging.{key}", value, ConfigLayer.AGENT, agent_id)


async def _send(
    env: AgentEnv,
    service: MessagingService,
    sender: AgentId,
    target: str,
    **fields: str | bool | None,
) -> SendMessageOutput:
    """Send from one agent and wait until every agent is quiet again."""
    message = SendMessageInput.model_validate({"to": [target], "content": REQUEST, **fields})
    output = await service.send(_agent(sender), message)
    for agent in env.manager.list():
        await env.manager.wait_idle(agent.id)
    return output


async def _received(env: AgentEnv, agent_id: AgentId) -> MessageRecord:
    found = await env.store.last_received(agent_id)
    assert found is not None
    return found


async def test_a_message_reaches_the_target_by_name_and_by_id(
    env: AgentEnv, service: MessagingService
) -> None:
    sender, target = await env.create(), await env.create()
    name = env.manager.get(target).name

    by_name = await _send(env, service, sender, f"agent:{name}")
    by_id = await _send(env, service, sender, target)
    record = await _received(env, target)

    assert by_name.delivered == [f"agent:{name}"]
    assert by_id.delivered == [target]
    assert (by_name.rejected, by_id.rejected) == ([], [])
    assert record.content == [TextBlock(text=REQUEST)]
    assert record.origin is not None
    assert record.origin.source == _agent(sender)
    assert record.origin.source_name == env.manager.get(sender).name
    assert record.origin.message_id == by_id.message_id


async def test_sending_does_not_wait_for_the_target(
    env: AgentEnv, service: MessagingService
) -> None:
    sender, target = await env.create(), await env.create()
    message = SendMessageInput(to=[target], content=REQUEST, expects_reply=True)

    output = await service.send(_agent(sender), message)
    still_working = env.manager.get(target).status
    await env.manager.wait_idle(target)

    assert output.delivered == [target]
    assert still_working is not AgentStatus.IDLE  # send returned before the run ended
    origin = (await _received(env, target)).origin
    assert origin is not None
    assert origin.expects_reply
    assert origin.targets == [_agent(target)]
    assert origin.conversation_id == output.conversation_id


async def test_targets_that_cannot_get_the_message_are_reported(
    env: AgentEnv, service: MessagingService
) -> None:
    sender, target, gone = await env.create(), await env.create(), await env.create()
    await env.manager.archive(gone)
    message = SendMessageInput(to=[target, UNKNOWN, sender, gone], content=REQUEST)

    output = await service.send(_agent(sender), message)
    await env.manager.wait_idle(target)

    assert output.delivered == [target]
    assert dict(output.rejected) == {
        UNKNOWN: RejectReason.NOT_FOUND.value,
        sender: RejectReason.SELF.value,
        gone: RejectReason.ARCHIVED.value,
    }


@pytest.mark.parametrize("spelling", ["agent:{name}", "{name}", "agent:{id}", "agent:*", "*"])
async def test_a_denied_sender_is_blocked(
    env: AgentEnv, service: MessagingService, spelling: str
) -> None:
    sender, target, other = await env.create(), await env.create(), await env.create()
    pattern = spelling.format(name=env.manager.get(sender).name, id=sender)
    await _limit(env, target, "deny_from", [pattern])

    blocked = await _send(env, service, sender, target)
    free = await _send(env, service, sender, other)

    assert blocked.rejected == [(target, RejectReason.BLOCKED.value)]
    assert free.delivered == [other]
    assert await env.store.last_received(target) is None


async def test_allow_from_lets_in_only_the_listed(env: AgentEnv, service: MessagingService) -> None:
    friend, stranger, target = await env.create(), await env.create(), await env.create()
    await _limit(env, target, "allow_from", [f"agent:{env.manager.get(friend).name}"])

    refused = await _send(env, service, stranger, target)
    accepted = await _send(env, service, friend, target)
    outside = await service.send(API_CALLER, SendMessageInput(to=[target], content=REQUEST))

    assert refused.rejected == [(target, RejectReason.BLOCKED.value)]
    assert accepted.delivered == [target]
    assert outside.rejected == [(target, RejectReason.BLOCKED.value)]


async def test_a_hook_can_refuse_a_message(env: AgentEnv, service: MessagingService) -> None:
    sender, target = await env.create(), await env.create()
    env.hooks.script(HookEvent.MESSAGE_RECEIVED, Decision.DENY, reason="not now")

    refused = await _send(env, service, sender, target)
    accepted = await _send(env, service, sender, target)

    assert refused.rejected == [(target, RejectReason.HOOK.value)]
    assert accepted.delivered == [target]


async def test_the_listing_hides_the_viewer_the_archived_and_the_hidden(
    env: AgentEnv, service: MessagingService
) -> None:
    viewer, open_one, hidden, gone, deaf = [await env.create() for _ in range(5)]
    await env.manager.archive(gone)
    await _limit(env, hidden, "discoverable", False)
    await _limit(env, deaf, "deny_from", [f"agent:{viewer}"])

    listed = {info.id: info for info in service.list_agents(_agent(viewer))}

    assert set(listed) == {open_one, deaf}
    assert listed[open_one].address == f"agent:{env.manager.get(open_one).name}"
    assert listed[open_one].accepts_messages
    assert not listed[deaf].accepts_messages


async def test_the_description_is_what_others_see(env: AgentEnv, service: MessagingService) -> None:
    viewer = await env.create()
    described = await env.manager.create(AgentSpec(name="reviewer", description=REQUEST))

    (info,) = service.list_agents(_agent(viewer))

    assert (info.id, info.name, info.description) == (described, "reviewer", REQUEST)


async def test_each_forward_between_agents_is_one_more_hop(
    env: AgentEnv, service: MessagingService
) -> None:
    first, second, third = await env.create(), await env.create(), await env.create()

    await _send(env, service, first, second)
    await _send(env, service, second, third)
    hops = [(await _received(env, agent)).origin for agent in (second, third)]

    assert hops[0] is not None
    assert hops[1] is not None
    assert hops[1].hop == hops[0].hop + 1


async def test_a_chain_longer_than_the_limit_is_cut(
    env: AgentEnv, service: MessagingService
) -> None:
    first, second, third = await env.create(), await env.create(), await env.create()
    await _send(env, service, first, second)
    reached = (await _received(env, second)).origin
    assert reached is not None
    await _limit(env, third, "max_agent_hops", reached.hop)

    cut = await _send(env, service, second, third)
    await env.say(second, "the user speaks: the chain starts over")
    await env.manager.wait_idle(second)
    fresh = await _send(env, service, second, third)

    assert cut.rejected == [(third, RejectReason.HOP_LIMIT.value)]
    assert fresh.delivered == [third]


async def test_the_hourly_limit_is_per_pair_and_frees_up_with_time(
    env: AgentEnv, service: MessagingService, clock: FakeClock
) -> None:
    sender, other, target = await env.create(), await env.create(), await env.create()
    await _limit(env, target, "max_per_hour", SMALL_LIMIT)

    allowed = [await _send(env, service, sender, target) for _ in range(SMALL_LIMIT)]
    over = await _send(env, service, sender, target)
    another_sender = await _send(env, service, other, target)
    clock.advance(HOUR_S)
    later = await _send(env, service, sender, target)

    assert all(output.delivered == [target] for output in allowed)
    assert over.rejected == [(target, RejectReason.RATE_LIMIT.value)]
    assert another_sender.delivered == [target]
    assert later.delivered == [target]


async def test_a_conversation_stops_at_its_limit_and_the_user_is_told_once(
    env: AgentEnv, service: MessagingService
) -> None:
    sender, target = await env.create(), await env.create()
    await _limit(env, target, "max_messages_per_conversation", SMALL_LIMIT)
    opened = await _send(env, service, sender, target)
    conversation = opened.conversation_id
    for _ in range(SMALL_LIMIT - 1):
        await _send(env, service, sender, target, conversation_id=conversation)

    stopped = [
        await _send(env, service, sender, target, conversation_id=conversation) for _ in range(2)
    ]
    new_conversation = await _send(env, service, sender, target)

    warnings = [
        warning
        for warning in env.payloads(WarningEvent)
        if warning.code is ErrorCode.MESSAGING_CONVERSATION_LIMIT
    ]
    reasons = [output.rejected for output in stopped]
    assert reasons == [[(target, RejectReason.CONVERSATION_LIMIT.value)]] * len(stopped)
    assert len(warnings) == 1
    assert conversation in warnings[0].message
    assert new_conversation.delivered == [target]
    assert new_conversation.conversation_id != conversation


async def test_a_reply_goes_back_to_the_sender_in_the_same_conversation(
    env: AgentEnv, service: MessagingService
) -> None:
    asker, answerer = await env.create(), await env.create()
    asked = await _send(env, service, asker, answerer, expects_reply=True)

    replied = await service.reply(answerer, asked.message_id, REPLY)
    await env.manager.wait_idle(asker)
    question = (await _received(env, answerer)).origin
    answer = await _received(env, asker)

    assert replied.delivered == [str(_agent(asker))]
    assert replied.conversation_id == asked.conversation_id
    assert answer.content == [TextBlock(text=REPLY)]
    assert question is not None
    assert answer.origin is not None
    assert answer.origin.reply_to == asked.message_id
    assert answer.origin.hop == question.hop + 1
    assert not answer.origin.expects_reply  # nothing asks for an answer to an answer


async def test_only_a_message_from_an_agent_can_be_replied_to(
    env: AgentEnv, service: MessagingService
) -> None:
    agent_id = await env.create()
    outside = await service.send(API_CALLER, SendMessageInput(to=[agent_id], content=REQUEST))
    await env.manager.wait_idle(agent_id)

    with pytest.raises(AgentError):
        await service.reply(agent_id, UNKNOWN_MESSAGE, REPLY)
    with pytest.raises(AgentError):
        await service.reply(agent_id, outside.message_id, REPLY)


async def test_an_outside_caller_starts_a_chain_at_zero(
    env: AgentEnv, service: MessagingService
) -> None:
    target, next_one = await env.create(), await env.create()

    await service.send(API_CALLER, SendMessageInput(to=[target], content=REQUEST))
    await env.manager.wait_idle(target)
    await _send(env, service, target, next_one)
    from_outside = (await _received(env, target)).origin
    forwarded = (await _received(env, next_one)).origin

    assert from_outside is not None
    assert forwarded is not None
    assert from_outside.source == API_CALLER
    assert forwarded.hop == from_outside.hop + 1


# --- the manager: unique names, the waiting counter, waking from the database ---


@pytest.mark.parametrize("name", BAD_NAMES)
async def test_a_name_that_cannot_be_an_address_is_refused(env: AgentEnv, name: str) -> None:
    with pytest.raises(AgentError) as raised:
        await env.manager.create(AgentSpec(name=name))

    assert raised.value.code is ErrorCode.AGENT_NAME_INVALID
    assert env.manager.list() == []


async def test_two_agents_cannot_share_a_name(env: AgentEnv) -> None:
    first, second = await env.create(), await env.create()
    taken = env.manager.get(first).name

    with pytest.raises(AgentError) as on_create:
        await env.manager.create(AgentSpec(name=taken))
    with pytest.raises(AgentError) as on_rename:
        await env.manager.update(second, {"name": taken})
    await env.manager.update(first, {"name": taken})  # keeping your own name is fine

    assert on_create.value.code is ErrorCode.AGENT_NAME_TAKEN
    assert on_rename.value.code is ErrorCode.AGENT_NAME_TAKEN
    assert env.manager.get(second).name != taken


async def test_messages_from_agents_are_counted_until_they_are_read(
    env: AgentEnv, service: MessagingService
) -> None:
    sender, target = await env.create(), await env.create()

    await _send(env, service, sender, target)
    counts = [
        update.agent.waiting_agent_messages
        for update in env.payloads(AgentUpdated)
        if update.agent.id == target
    ]

    assert max(counts) == 1
    assert env.manager.get(target).waiting_agent_messages == 0


async def test_the_user_is_not_counted_as_an_agent_message(env: AgentEnv) -> None:
    agent_id = await env.create()
    env.gateway.outcomes.append(text_response(ANSWER))

    await env.converse(agent_id, REQUEST)

    counts = [update.agent.waiting_agent_messages for update in env.payloads(AgentUpdated)]
    assert set(counts) <= {0}


async def test_whatever_waits_in_the_database_wakes_an_idle_agent(
    env: AgentEnv, service: MessagingService
) -> None:
    sender, target = await env.create(), await env.create()
    await _send(env, service, sender, target)
    record = await _received(env, target)
    assert record.origin is not None
    late = InboxItem(
        id="inb_written_elsewhere",
        agent_id=target,
        content=[TextBlock(text=REPLY)],
        origin=record.origin,
        received_at=env.deps.clock.now(),
    )
    await env.store.enqueue_inbox(late)  # as another process would: no in-process wake

    await env.manager.wake_pending()
    await env.manager.wait_idle(target)

    assert await env.store.pending_inbox(target) == []
    assert (await _received(env, target)).content == [TextBlock(text=REPLY)]


def test_the_defaults_let_agents_talk(env: AgentEnv) -> None:
    rules = env.deps.config.section(MessagingConfig)

    assert rules.discoverable
    assert rules.deny_from == []
