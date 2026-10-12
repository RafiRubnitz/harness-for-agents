"""``DefaultContextManager``: builds every model request and compacts when it grows (07)."""

import asyncio
from collections.abc import Sequence

from loguru import logger

from harness.context.assembly import Assembler, Assembly
from harness.context.compaction import Compactor
from harness.context.consts import EXACT_COUNT_FROM_SHARE, MAX_COMPACTIONS_PER_BUILD
from harness.context.types import ContextDeps
from harness.core.consts import CompactReason
from harness.core.errors import ContextError, GatewayError
from harness.core.records import BuiltContext, ContextReport, SummaryRecord
from harness.core.types import AgentId, Message

_FULL_TEXT_EXCLUDE = {"metadata"}
_FULL_TEXT_INDENT = 2


class DefaultContextManager:
    """Stateless between steps: everything is read from the memory layers on each build."""

    def __init__(self, deps: ContextDeps) -> None:
        self._deps = deps
        self._assembler = Assembler(deps)
        self._compactor = Compactor(deps)
        self._locks: dict[AgentId, asyncio.Lock] = {}

    async def _over_limit(self, assembly: Assembly) -> bool:
        """Whether the request passed the compaction line.

        The local estimate decides while it is far from the line; near it the provider's
        exact count is asked for (07 §4).
        """
        limit, estimate = assembly.budgets.limit, assembly.report.total_tokens
        if estimate <= limit * EXACT_COUNT_FROM_SHARE:
            return False
        try:
            return await self._deps.gateway.count_tokens(assembly.request) > limit
        except GatewayError as error:
            logger.debug("exact token count failed, using the estimate: {}", error.message)
            return estimate > limit

    async def build(self, agent_id: AgentId, pending: Sequence[Message] = ()) -> BuiltContext:
        """Build the request for the agent's next step, compacting first when needed.

        A model with a smaller window than the last one is handled the same way: the
        budget comes from the active model, so the first build after the switch compacts.
        """
        assembly = await self._assembler.assemble(agent_id, pending, retrieve=True)
        for _ in range(MAX_COMPACTIONS_PER_BUILD):
            if not await self._over_limit(assembly):
                break
            try:
                await self.compact(agent_id, CompactReason.THRESHOLD)
            except ContextError as error:
                logger.warning("context of {} is over its limit: {}", agent_id, error.message)
                break
            assembly = await self._assembler.assemble(agent_id, pending)
        return BuiltContext(request=assembly.request, report=assembly.report)

    async def compact(
        self, agent_id: AgentId, reason: CompactReason = CompactReason.THRESHOLD
    ) -> SummaryRecord:
        """Summarise the oldest part of the history.

        Raises:
            ContextError: Nothing can be compacted, a hook denied it, or no model answered.
        """
        async with self._locks.setdefault(agent_id, asyncio.Lock()):
            return await self._compactor.compact(agent_id, reason)

    async def inspect(self, agent_id: AgentId, *, full: bool = False) -> ContextReport:
        """Return the report of the request that would be built now; nothing is changed."""
        assembly = await self._assembler.assemble(agent_id)
        if not full:
            return assembly.report
        text = assembly.request.model_dump_json(
            indent=_FULL_TEXT_INDENT, exclude=_FULL_TEXT_EXCLUDE
        )
        return assembly.report.model_copy(update={"full_text": text})
