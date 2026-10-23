"""Hooks can match a message by who sent it and who gets it (phase 2, 03 §7)."""

import pytest

from harness.core.consts import AddressType, HookEvent
from harness.core.hook_payloads import MessageReceivedPayload
from harness.core.hook_types import HookInput, HookSpec
from harness.core.types import Address, MessageOrigin, TextBlock
from harness.hooks.utils import matches
from tests.consts import AGENT_ID, AGENT_NAME, OTHER_AGENT_ID, USER_ID
from tests.fakes.clock import FAKE_CLOCK_START

SENDER_NAME = "planner"
FROM_AGENT = MessageOrigin(
    source=Address(type=AddressType.AGENT, id=OTHER_AGENT_ID), source_name=SENDER_NAME
)
FROM_USER = MessageOrigin(source=Address(type=AddressType.USER, id=USER_ID))


def _fits(match: dict[str, str], origin: MessageOrigin) -> bool:
    spec = HookSpec(name="guard", event=HookEvent.MESSAGE_RECEIVED, match=match, run="x")
    payload = MessageReceivedPayload(content=[TextBlock(text="hi")], origin=origin)
    inp = HookInput(
        event=HookEvent.MESSAGE_RECEIVED,
        agent_id=AGENT_ID,
        agent_name=AGENT_NAME,
        payload=payload.model_dump(mode="json"),
        timestamp=FAKE_CLOCK_START,
    )
    return matches(spec, inp, None)


@pytest.mark.parametrize(
    "pattern", [f"agent:{SENDER_NAME}", f"agent:{OTHER_AGENT_ID}", "agent:*", "*"]
)
def test_the_sender_is_matched_by_name_id_or_wildcard(pattern: str) -> None:
    assert _fits({"source": pattern}, FROM_AGENT)


@pytest.mark.parametrize("pattern", ["agent:someone-else", "user:*", f"agent:{AGENT_NAME}"])
def test_another_sender_does_not_match(pattern: str) -> None:
    assert not _fits({"source": pattern}, FROM_AGENT)


def test_agent_patterns_do_not_catch_the_user() -> None:
    assert not _fits({"source": "agent:*"}, FROM_USER)
    assert _fits({"source": "user:*"}, FROM_USER)


@pytest.mark.parametrize("pattern", [f"agent:{AGENT_NAME}", f"agent:{AGENT_ID}", "agent:*"])
def test_the_target_is_the_agent_that_gets_the_message(pattern: str) -> None:
    assert _fits({"target": pattern}, FROM_AGENT)


def test_source_and_target_must_both_fit() -> None:
    both = {"source": f"agent:{SENDER_NAME}", "target": f"agent:{AGENT_NAME}"}
    wrong_target = {"source": f"agent:{SENDER_NAME}", "target": f"agent:{SENDER_NAME}"}

    assert _fits(both, FROM_AGENT)
    assert not _fits(wrong_target, FROM_AGENT)
