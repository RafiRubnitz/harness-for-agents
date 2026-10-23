"""Agent names and inbox order (phase 2, 03 §3, §7a)."""

from datetime import timedelta

import pytest

from harness.core.consts import USER_PRIORITY, AddressType
from harness.core.records import InboxItem
from harness.core.types import Address, MessageOrigin, TextBlock
from harness.core.utils.inbox import effective_priority, in_order
from harness.core.utils.names import free_name, slug, valid_name
from tests.consts import AGENT_ID, USER_ID
from tests.fakes.clock import FAKE_CLOCK_START

ORIGIN = MessageOrigin(source=Address(type=AddressType.USER, id=USER_ID))
AGING_S = 100.0
LOW, MIDDLE = 5, 40


def _item(item_id: str, priority: int, waited_s: float = 0.0) -> InboxItem:
    return InboxItem(
        id=item_id,
        agent_id=AGENT_ID,
        content=[TextBlock(text=item_id)],
        origin=ORIGIN,
        received_at=FAKE_CLOCK_START - timedelta(seconds=waited_s),
        priority=priority,
    )


@pytest.mark.parametrize("name", ["coder", "code-reviewer", "agent-2", "a1"])
def test_names_that_can_be_addresses(name: str) -> None:
    assert valid_name(name)


@pytest.mark.parametrize(
    "name", ["", "Coder", "two words", "a:b", "-x", "x-", "a_b", "שלום", "a/b"]
)
def test_names_that_cannot_be_addresses(name: str) -> None:
    assert not valid_name(name)


@pytest.mark.parametrize("text", ["Code Reviewer", "  --My Agent!!  ", "שלום", "", "A__b"])
def test_a_slug_is_always_a_valid_name(text: str) -> None:
    assert valid_name(slug(text))


def test_a_slug_keeps_the_words() -> None:
    made = slug("Code Reviewer")

    assert "code" in made
    assert "reviewer" in made


def test_a_free_name_is_never_one_that_is_taken() -> None:
    taken: set[str] = set()
    for _ in range(4):
        name = free_name("chat", taken)
        assert name not in taken
        assert valid_name(name)
        taken.add(name)

    assert "chat" in taken  # the plain name goes first


def test_higher_priority_goes_first_and_ties_keep_arrival_order() -> None:
    items = [_item("b_low", LOW, 1), _item("a_mid", MIDDLE, 2), _item("c_mid", MIDDLE, 3)]

    ordered = in_order(items, FAKE_CLOCK_START, None)

    assert [item.id for item in ordered] == ["c_mid", "a_mid", "b_low"]


def test_waiting_raises_priority_one_step_per_period() -> None:
    fresh = _item("fresh", LOW)
    one_period = _item("one", LOW, AGING_S)
    two_periods = _item("two", LOW, 2 * AGING_S)
    now = FAKE_CLOCK_START

    gains = [
        effective_priority(item, now, AGING_S) - LOW for item in (fresh, one_period, two_periods)
    ]

    assert gains[0] == 0
    assert gains[1] > 0
    assert gains[2] == 2 * gains[1]


@pytest.mark.parametrize("aging_s", [None, 0.0])
def test_aging_can_be_turned_off(aging_s: float | None) -> None:
    old = _item("old", LOW, 1000 * AGING_S)

    assert effective_priority(old, FAKE_CLOCK_START, aging_s) == LOW


def test_no_amount_of_waiting_gets_ahead_of_the_user() -> None:
    ancient = _item("ancient", MIDDLE, 10_000 * AGING_S)
    user = _item("user", USER_PRIORITY)

    ordered = in_order([ancient, user], FAKE_CLOCK_START, AGING_S)

    assert [item.id for item in ordered] == ["user", "ancient"]
    assert effective_priority(ancient, FAKE_CLOCK_START, AGING_S) < USER_PRIORITY
