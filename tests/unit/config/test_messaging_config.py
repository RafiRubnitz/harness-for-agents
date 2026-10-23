"""The ``messaging``, ``inbox`` and ``local_server`` sections (phase 2: 03 §5-§7a, 02 §5a)."""

import pytest
from pydantic import ValidationError

from harness.config import HarnessConfig, InboxConfig, requires_restart
from harness.core.consts import USER_PRIORITY, AddressType, ConfigLayer
from harness.core.errors import ConfigError
from tests.consts import AGENT_ID
from tests.unit.config.conftest import ManagerFactory

CUSTOM = 7
OUTSIDE_SOURCES = [source for source in AddressType if source is not AddressType.USER]


def test_the_user_is_above_every_other_source() -> None:
    inbox = InboxConfig()

    assert inbox.priority_of(AddressType.USER) == USER_PRIORITY
    assert all(inbox.priority_of(source) < USER_PRIORITY for source in OUTSIDE_SOURCES)


def test_a_priority_can_be_changed_per_source() -> None:
    inbox = InboxConfig(priorities={AddressType.AGENT.value: CUSTOM})

    assert inbox.priority_of(AddressType.AGENT) == CUSTOM
    assert inbox.priority_of(AddressType.USER) == USER_PRIORITY


@pytest.mark.parametrize(
    "priorities",
    [
        {AddressType.USER.value: CUSTOM},
        {AddressType.AGENT.value: USER_PRIORITY},
        {AddressType.AGENT.value: USER_PRIORITY + 1},
        {AddressType.AGENT.value: -1},
    ],
)
def test_nothing_can_be_ranked_with_or_above_the_user(priorities: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        InboxConfig(priorities=priorities)


def test_the_local_server_is_off_until_it_is_turned_on() -> None:
    assert not HarnessConfig().local_server.enabled


def test_turning_the_local_server_on_needs_a_restart() -> None:
    assert requires_restart(["local_server.enabled"])
    assert not requires_restart(["messaging.max_per_hour"])


async def test_messaging_rules_can_differ_per_agent(make_manager: ManagerFactory) -> None:
    manager = make_manager()

    await manager.set("messaging.discoverable", False, ConfigLayer.AGENT, AGENT_ID)

    assert not manager.config(AGENT_ID).messaging.discoverable
    assert manager.config().messaging.discoverable


async def test_the_local_server_cannot_be_set_per_agent(make_manager: ManagerFactory) -> None:
    manager = make_manager()

    with pytest.raises(ConfigError):
        await manager.set("local_server.enabled", True, ConfigLayer.AGENT, AGENT_ID)
