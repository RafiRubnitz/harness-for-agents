"""The cost budget: nothing when off, a warning near the limit, no call at the limit."""

from dataclasses import dataclass, field
from datetime import datetime

import pytest

from harness.config import FileConfigManager
from harness.config import consts as config_consts
from harness.core.consts import ConfigLayer, ErrorCode, ProviderId
from harness.core.errors import BudgetExceededError
from harness.core.types import AgentId
from harness.core.utils.paths import HarnessHome
from harness.models import BudgetGuard, DefaultModelGateway
from tests.consts import AGENT_ID, TEST_API_KEY, USER_ID
from tests.fakes.adapter import FakeAdapter, text_response
from tests.fakes.clock import FakeClock
from tests.fakes.secrets import FakeSecretStore
from tests.utils import make_request

MAIN = config_consts.MODEL_ALIAS_MAIN
AGENT_LIMIT = 2.0
DAY_LIMIT = 10.0
WARN_PCT = 50.0
BELOW_WARNING = AGENT_LIMIT * WARN_PCT / 100 / 2
PAST_WARNING = AGENT_LIMIT * WARN_PCT / 100 + 0.1
RAISED_LIMIT = AGENT_LIMIT * 10


@dataclass
class Ledger:
    """What was spent, as the model-call table would report it."""

    by_agent: dict[AgentId, float] = field(default_factory=dict[AgentId, float])
    today: float = 0.0
    asked_since: list[datetime] = field(default_factory=list[datetime])

    async def cost_total(self, agent_id: AgentId) -> float:
        return self.by_agent.get(agent_id, 0.0)

    async def cost_since(self, since: datetime) -> float:
        self.asked_since.append(since)
        return self.today


@dataclass
class Kit:
    gateway: DefaultModelGateway
    adapter: FakeAdapter
    ledger: Ledger
    config: FileConfigManager
    warnings: list[tuple[AgentId | None, str]]

    async def call(self) -> None:
        self.adapter.outcomes.append(text_response("ok"))
        await self.gateway.complete(make_request(model=MAIN))


@pytest.fixture
async def kit(tmp_harness_home: HarnessHome, clock: FakeClock) -> Kit:
    config = FileConfigManager(
        tmp_harness_home,
        secrets=FakeSecretStore({config_consts.SECRET_ANTHROPIC: TEST_API_KEY}),
        env={},
        username=USER_ID,
    )
    settings = {"enabled": True, "per_agent_usd": AGENT_LIMIT, "per_day_usd": DAY_LIMIT}
    for key, value in {**settings, "warn_at_pct": WARN_PCT}.items():
        await config.set(f"limits.budget.{key}", value, ConfigLayer.GLOBAL)
    adapter, ledger = FakeAdapter([]), Ledger()
    warnings: list[tuple[AgentId | None, str]] = []
    gateway = DefaultModelGateway(config, {ProviderId.ANTHROPIC: adapter}, clock)
    gateway.set_budget(
        BudgetGuard(config, ledger, clock, lambda agent, text: warnings.append((agent, text)))
    )
    return Kit(gateway, adapter, ledger, config, warnings)


async def test_under_the_warning_line_calls_pass_quietly(kit: Kit) -> None:
    kit.ledger.by_agent[AGENT_ID] = BELOW_WARNING

    await kit.call()

    assert len(kit.adapter.requests) == 1
    assert kit.warnings == []


async def test_past_the_warning_line_warns_once_and_still_calls(kit: Kit) -> None:
    kit.ledger.by_agent[AGENT_ID] = PAST_WARNING

    await kit.call()
    await kit.call()

    assert len(kit.adapter.requests) == len(["first", "second"])
    assert [agent for agent, _ in kit.warnings] == [AGENT_ID]
    assert "agent" in kit.warnings[0][1]


@pytest.mark.parametrize("scope", ["agent", "day"])
async def test_a_used_up_budget_stops_the_call_before_it_is_sent(kit: Kit, scope: str) -> None:
    if scope == "agent":
        kit.ledger.by_agent[AGENT_ID] = AGENT_LIMIT
    else:
        kit.ledger.today = DAY_LIMIT

    with pytest.raises(BudgetExceededError) as raised:
        await kit.call()

    assert kit.adapter.requests == []
    assert raised.value.code is ErrorCode.MODEL_BUDGET_EXCEEDED
    assert raised.value.recoverable
    assert not raised.value.retryable
    assert scope in raised.value.message


async def test_raising_the_budget_lets_calls_through_again(kit: Kit) -> None:
    kit.ledger.by_agent[AGENT_ID] = AGENT_LIMIT
    with pytest.raises(BudgetExceededError):
        await kit.call()

    await kit.config.set("limits.budget.per_agent_usd", RAISED_LIMIT, ConfigLayer.GLOBAL)
    await kit.call()

    assert len(kit.adapter.requests) == 1


async def test_a_disabled_budget_checks_nothing(kit: Kit) -> None:
    kit.ledger.by_agent[AGENT_ID] = AGENT_LIMIT
    await kit.config.set("limits.budget.enabled", False, ConfigLayer.GLOBAL)

    await kit.call()

    assert len(kit.adapter.requests) == 1
    assert kit.ledger.asked_since == []


async def test_the_day_starts_at_local_midnight(kit: Kit, clock: FakeClock) -> None:
    await kit.call()

    since = kit.ledger.asked_since[0]
    local = since.astimezone()
    assert since <= clock.now()
    assert (local.hour, local.minute, local.second) == (0, 0, 0)
    assert local.date() == clock.now().astimezone().date()
