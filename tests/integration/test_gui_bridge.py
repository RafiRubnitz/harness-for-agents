"""The GUI's bridge and state over the real core (no window): threads, history, stop."""

import time
from collections.abc import Callable, Iterator

import pytest

from harness.core import commands as c
from harness.core.consts import AgentStatus
from harness.core.errors import CoreLockedError
from harness.core.types import AgentId
from harness.core.utils.slash import parse_input
from harness.gui.bridge import BridgeError, CoreBridge
from harness.gui.consts import ACTIVE_STATUSES
from harness.gui.items import AssistantItem, UserItem
from harness.gui.render import item_text
from harness.gui.state import AppState
from tests.fakes.adapter import text_response
from tests.integration.conftest import CoreKit

WAIT_S = 10.0
TICK_S = 0.005
STOP_LIMIT_S = 1.0
NAMES = ["alpha", "beta", "gamma"]
MANY_NAMES = [f"worker-{number}" for number in range(10)]
QUESTION = "מה שלומך?"
ANSWER = "הכול טוב, thanks"


class Screen:
    """What the window does each tick, without Tk: poll the bridge, fold into the state."""

    def __init__(self, bridge: CoreBridge) -> None:
        self.bridge = bridge
        self.state = AppState()
        self.slowest_poll = 0.0

    def tick(self) -> None:
        started = time.perf_counter()
        events = self.bridge.poll()
        self.slowest_poll = max(self.slowest_poll, time.perf_counter() - started)
        for event in events:
            self.state.apply(event)

    def until(self, done: Callable[[], bool]) -> None:
        deadline = time.monotonic() + WAIT_S
        while not done():
            if time.monotonic() > deadline:
                message = "the GUI state never got there"
                raise AssertionError(message)
            self.tick()
            time.sleep(TICK_S)

    def agent(self, name: str) -> AgentId:
        self.bridge.submit(c.AgentCreate(agent_name=name))
        self.until(lambda: name in self.names())
        return next(view.state.id for view in self.state.ordered() if view.state.name == name)

    def names(self) -> list[str]:
        return [view.state.name for view in self.state.ordered()]

    def status(self, agent_id: AgentId) -> AgentStatus:
        return self.state.agents[agent_id].state.status


@pytest.fixture
def screen(kit: CoreKit) -> Iterator[Screen]:
    bridge = CoreBridge(kit.launch)
    bridge.start()
    yield Screen(bridge)
    bridge.stop(stop_agents=True)


def run_side_by_side(screen: Screen, kit: CoreKit, names: list[str]) -> None:
    """Run one agent per name at the same time and check the window never waited."""
    kit.gateway.outcomes = [text_response(ANSWER) for _ in names]
    agents = [screen.agent(name) for name in names]
    seen: dict[AgentId, set[AgentStatus]] = {agent_id: set() for agent_id in agents}

    def watch() -> bool:
        for agent_id in agents:
            seen[agent_id].add(screen.status(agent_id))
        return all(
            AgentStatus.RUNNING in statuses and screen.status(agent_id) is AgentStatus.IDLE
            for agent_id, statuses in seen.items()
        )

    for agent_id in agents:
        screen.bridge.submit(parse_input(QUESTION, agent_id))
    screen.until(watch)

    for agent_id in agents:
        texts = [item_text(item) for item in screen.state.agents[agent_id].conversation.items]
        assert ANSWER in texts
    assert screen.slowest_poll < TICK_S * 10


@pytest.mark.acceptance("02-10.1")
def test_three_agents_run_at_once_and_polling_never_waits(screen: Screen, kit: CoreKit) -> None:
    run_side_by_side(screen, kit, NAMES)


@pytest.mark.acceptance("04-11.11")
@pytest.mark.timeout(60)
def test_ten_agents_running_at_once_do_not_stall_the_window(screen: Screen, kit: CoreKit) -> None:
    run_side_by_side(screen, kit, MANY_NAMES)


@pytest.mark.acceptance("02-10.4")
def test_stop_ends_the_selected_agent_within_a_second(screen: Screen, kit: CoreKit) -> None:
    kit.gateway.outcomes = [text_response(ANSWER)]
    kit.gateway.hold = kit.gateway.streaming.__class__()  # never set: the model call hangs
    agent_id = screen.agent(NAMES[0])
    screen.bridge.submit(parse_input(QUESTION, agent_id))
    screen.until(lambda: screen.status(agent_id) is AgentStatus.RUNNING)
    screen.until(kit.gateway.streaming.is_set)

    started = time.perf_counter()
    screen.bridge.submit(c.AgentStop(agent_id=agent_id))
    screen.until(lambda: screen.status(agent_id) not in ACTIVE_STATUSES)

    assert time.perf_counter() - started < STOP_LIMIT_S


@pytest.mark.acceptance("02-10.6")
def test_reopening_shows_every_agent_with_its_history(kit: CoreKit) -> None:
    kit.gateway.outcomes = [text_response(ANSWER)]
    first = Screen(CoreBridge(kit.launch))
    first.bridge.start()
    agents = [first.agent(name) for name in NAMES]
    talker = agents[0]
    first.bridge.submit(parse_input(QUESTION, talker))
    first.until(
        lambda: ANSWER in [item_text(i) for i in first.state.agents[talker].conversation.items]
    )
    first.until(lambda: first.status(talker) is AgentStatus.IDLE)
    first.bridge.stop()

    second = Screen(CoreBridge(kit.launch))
    second.bridge.start()
    try:
        second.bridge.submit(c.AgentList(include_archived=True))
        second.until(lambda: len(second.state.agents) == len(NAMES))
        second.bridge.submit(c.AgentLoadHistory(agent_id=talker))
        second.until(lambda: second.state.agents[talker].loaded)
    finally:
        second.bridge.stop()

    assert second.names() == NAMES
    items = second.state.agents[talker].conversation.items
    assert [(type(item), item_text(item)) for item in items] == [
        (UserItem, QUESTION),
        (AssistantItem, ANSWER),
    ]


def test_second_bridge_is_refused_while_the_first_holds_the_lock(
    screen: Screen, kit: CoreKit
) -> None:
    other = CoreBridge(kit.launch)

    with pytest.raises(CoreLockedError):
        other.start()

    assert screen.bridge.running
    assert not other.running


def test_a_stopped_bridge_refuses_commands(kit: CoreKit) -> None:
    bridge = CoreBridge(kit.launch)
    bridge.start()
    info = bridge.info()

    bridge.stop()

    assert info.models
    assert not bridge.running
    with pytest.raises(BridgeError):
        bridge.submit(c.AgentList(include_archived=True))
