"""The transition table (04 §4)."""

import itertools

import pytest

from harness.agents.state import (
    ACTIVE,
    ARCHIVABLE,
    TRANSITIONS,
    can_transition,
    check_transition,
    require_status,
)
from harness.core.consts import AgentStatus
from harness.core.errors import InvalidTransitionError

ALL_PAIRS = list(itertools.product(AgentStatus, AgentStatus))


@pytest.mark.parametrize(("current", "target"), sorted(TRANSITIONS))
def test_listed_transitions_pass(current: AgentStatus, target: AgentStatus) -> None:
    check_transition(current, target)

    assert can_transition(current, target)


@pytest.mark.parametrize(
    ("current", "target"), [pair for pair in ALL_PAIRS if pair not in TRANSITIONS]
)
def test_unlisted_transitions_raise(current: AgentStatus, target: AgentStatus) -> None:
    with pytest.raises(InvalidTransitionError):
        check_transition(current, target)


def test_reserved_status_is_unreachable() -> None:
    reachable = {target for _, target in TRANSITIONS}

    assert AgentStatus.REFLECTING not in reachable


def test_every_archivable_status_can_be_archived() -> None:
    assert all(can_transition(status, AgentStatus.ARCHIVED) for status in ARCHIVABLE)
    assert not any(can_transition(status, AgentStatus.ARCHIVED) for status in ACTIVE)


def test_require_status_names_the_action() -> None:
    action = "stop"

    with pytest.raises(InvalidTransitionError, match=action):
        require_status(AgentStatus.IDLE, ACTIVE, action)
