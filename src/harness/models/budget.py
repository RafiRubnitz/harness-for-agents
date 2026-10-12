"""The optional cost budget: a warning near the limit, a stop at it (01 §7)."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from harness.config import BudgetConfig, ConfigManager
from harness.core.errors import BudgetExceededError
from harness.core.types import AgentId, Clock, ModelRequest
from harness.models.consts import (
    BUDGET_SCOPE_AGENT,
    BUDGET_SCOPE_DAY,
    MSG_BUDGET_EXCEEDED,
    MSG_BUDGET_WARNING,
    PERCENT,
)

type BudgetNotifier = Callable[[AgentId | None, str], None]
"""Called with the agent that made the call and a ready warning text."""


class CostLedger(Protocol):
    """Where the guard reads what was already spent (the ``model_calls`` table)."""

    async def cost_total(self, agent_id: AgentId) -> float:
        """Lifetime cost of one agent."""
        ...

    async def cost_since(self, since: datetime) -> float:
        """Cost of all agents since ``since``."""
        ...


@dataclass(frozen=True, slots=True)
class _Spend:
    """What one budget has used so far."""

    scope: str
    owner: str
    used: float
    limit: float


class BudgetGuard:
    """Checks the per-agent and per-day budgets before a model call is sent."""

    def __init__(
        self,
        config: ConfigManager,
        ledger: CostLedger,
        clock: Clock,
        notify: BudgetNotifier | None = None,
    ) -> None:
        self._config = config
        self._ledger = ledger
        self._clock = clock
        self._notify = notify
        self._warned: set[str] = set()

    def _day_start(self) -> datetime:
        """Midnight of the local calendar day, as UTC."""
        local = self._clock.now().astimezone()
        return local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(UTC)

    async def check(self, req: ModelRequest) -> None:
        """Let the call through, warn, or refuse it.

        Raises:
            BudgetExceededError: A budget is used up; the call must not be sent.
        """
        budget = self._config.section(BudgetConfig)
        if not budget.enabled:
            return
        agent_id = req.metadata.agent_id
        spent: list[_Spend] = []
        if agent_id is not None:
            total = await self._ledger.cost_total(agent_id)
            spent.append(_Spend(BUDGET_SCOPE_AGENT, str(agent_id), total, budget.per_agent_usd))
        today = await self._ledger.cost_since(self._day_start())
        spent.append(_Spend(BUDGET_SCOPE_DAY, "", today, budget.per_day_usd))
        for spend in spent:
            self._judge(req, spend, budget.warn_at_pct)

    def _judge(self, req: ModelRequest, spend: "_Spend", warn_pct: float) -> None:
        scope, used, limit = spend.scope, spend.used, spend.limit
        key = f"{scope}:{spend.owner}"
        if used >= limit:
            raise BudgetExceededError(
                MSG_BUDGET_EXCEEDED.format(scope=scope, used=used, limit=limit),
                provider="",
                model=req.model,
            )
        if used < limit * warn_pct / PERCENT:
            self._warned.discard(key)  # the budget was raised: warn again next time
            return
        if key in self._warned:
            return
        self._warned.add(key)
        if self._notify is not None:
            percent = used / limit * PERCENT
            message = MSG_BUDGET_WARNING.format(
                scope=scope, used=used, limit=limit, percent=percent
            )
            self._notify(req.metadata.agent_id, message)
