"""Helpers for GUI tests: agent states and a feed that wraps payloads in real envelopes."""

from datetime import timedelta

import pytest

from harness.agents import EventBus
from harness.core.consts import AgentStatus, HookEvent, RiskLevel
from harness.core.events import AgentListed, EventPayload
from harness.core.records import AgentSettings, AgentState, PermissionRequestPayload
from harness.core.types import AgentId, PermissionRequestId
from harness.gui.state import AppState
from tests.consts import AGENT_ID, OTHER_AGENT_ID, TEST_MODEL_ALIAS
from tests.fakes.clock import FAKE_CLOCK_START, FakeClock

THIRD_AGENT_ID = AgentId("agt_0003")
TOOL = "write_file"
CALL_ID = "call-1"
REASON = "לשמור את הפתק שביקשת"
REQUEST_ID = PermissionRequestId("prm_0001")


def agent_state(
    agent_id: AgentId = AGENT_ID,
    *,
    name: str | None = None,
    status: AgentStatus = AgentStatus.IDLE,
    order: int = 0,
    tools: list[str] | None = None,
) -> AgentState:
    """An agent as the core reports it; ``order`` spaces the creation times."""
    at = FAKE_CLOCK_START + timedelta(seconds=order)
    return AgentState(
        id=agent_id,
        name=name or str(agent_id),
        model=TEST_MODEL_ALIAS,
        template=None,
        status=status,
        settings=AgentSettings(),
        tools=tools or [],
        run_id=None,
        step=None,
        pending_approvals=0,
        background_tasks=0,
        cost_usd_total=0.0,
        context_tokens=None,
        created_at=at,
        updated_at=at,
    )


def permission_request(reason: str | None = REASON) -> PermissionRequestPayload:
    """A request to run ``TOOL``, as the agent manager publishes it."""
    return PermissionRequestPayload(
        request_id=REQUEST_ID,
        subject=HookEvent.BEFORE_TOOL_CALL,
        call_id=CALL_ID,
        tool=TOOL,
        reason=reason,
        input={"path": "note.txt"},
        risk=RiskLevel.MEDIUM,
    )


class Feed:
    """An ``AppState`` fed through a real event bus, so envelopes are genuine."""

    def __init__(self) -> None:
        self.state = AppState()
        self._bus = EventBus(FakeClock())

    def send(self, payload: EventPayload, agent_id: AgentId | None = AGENT_ID) -> None:
        """Publish one payload and fold it into the state."""
        self.state.apply(self._bus.publish(agent_id, payload))

    def app(self, payload: EventPayload) -> None:
        """Publish a payload that belongs to no agent."""
        self.send(payload, None)


@pytest.fixture
def feed() -> Feed:
    """A state that already knows three idle agents; the first one is selected."""
    made = Feed()
    agents = [
        agent_state(agent_id, order=order)
        for order, agent_id in enumerate((AGENT_ID, OTHER_AGENT_ID, THIRD_AGENT_ID))
    ]
    made.app(AgentListed(agents=agents))
    made.state.take_changes()
    return made
