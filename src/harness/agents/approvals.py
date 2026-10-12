"""Approval requests: persist, announce, wait for the user's answer (04 §5.2, stage 3)."""

import asyncio
from dataclasses import dataclass

from harness.agents.consts import MSG_UNKNOWN_REQUEST
from harness.agents.runtime import AgentDeps, AgentRuntime, StatusBoard
from harness.core.consts import (
    ID_PREFIX_PERMISSION,
    AgentStatus,
    PermissionAnswer,
    PermissionResolution,
    PermissionScope,
)
from harness.core.errors import InvalidTransitionError
from harness.core.events import PermissionRequest, PermissionResolved
from harness.core.records import PendingApproval, PermissionRequestPayload
from harness.core.types import AgentId, PermissionRequestId, RunId


@dataclass(frozen=True)
class Answer:
    """The user's answer to one request."""

    decision: PermissionAnswer
    scope: PermissionScope = PermissionScope.ONCE
    deny_message: str | None = None

    @property
    def approved(self) -> bool:
        """Whether the user approved."""
        return self.decision is PermissionAnswer.APPROVE


@dataclass
class _OpenRequest:
    agent_id: AgentId | None
    future: asyncio.Future[Answer]


class ApprovalDesk:
    """Holds the open requests; only the asking agent waits, the others keep running."""

    def __init__(self, deps: AgentDeps, status: StatusBoard) -> None:
        self._deps = deps
        self._status = status
        self._open: dict[PermissionRequestId, _OpenRequest] = {}

    def new_id(self) -> PermissionRequestId:
        """Return an id for a new request."""
        return PermissionRequestId(self._deps.ids.new(ID_PREFIX_PERMISSION))

    def count(self, agent_id: AgentId) -> int:
        """Return how many requests of an agent are open."""
        return sum(1 for entry in self._open.values() if entry.agent_id == agent_id)

    async def ask(
        self,
        request: PermissionRequestPayload,
        *,
        rt: AgentRuntime | None,
        run_id: RunId | None = None,
    ) -> Answer:
        """Send a request and wait for the answer.

        A running agent moves to ``waiting_approval`` while it waits and back afterwards.

        Args:
            request: What is being asked.
            rt: The asking agent; ``None`` for an agent that is not stored yet.
            run_id: The run the request belongs to.
        """
        agent_id = rt.id if rt else None
        future: asyncio.Future[Answer] = asyncio.get_running_loop().create_future()
        self._open[request.request_id] = _OpenRequest(agent_id, future)
        await self._deps.memory.save_pending_approval(
            PendingApproval(
                request_id=request.request_id,
                agent_id=agent_id,
                run_id=run_id,
                request=request,
                created_at=self._deps.clock.now(),
            )
        )
        self._deps.bus.publish(agent_id, PermissionRequest(request=request))
        waiting = rt is not None and rt.status is AgentStatus.RUNNING
        if rt is not None and waiting:
            await self._status.set(rt, AgentStatus.WAITING_APPROVAL)
        answer = await future
        if rt is not None and waiting and rt.status is AgentStatus.WAITING_APPROVAL:
            await self._status.set(rt, AgentStatus.RUNNING)
        return answer

    async def respond(
        self,
        request_id: PermissionRequestId,
        decision: PermissionAnswer,
        scope: PermissionScope = PermissionScope.ONCE,
        deny_message: str | None = None,
    ) -> None:
        """Answer an open request.

        Raises:
            InvalidTransitionError: No open request has this id.
        """
        entry = self._open.pop(request_id, None)
        if entry is None:
            raise InvalidTransitionError(MSG_UNKNOWN_REQUEST.format(request_id=request_id))
        await self._deps.memory.close_pending_approval(
            request_id,
            PermissionResolution(decision.value),
            scope=scope,
            deny_message=deny_message,
        )
        self._deps.bus.publish(
            entry.agent_id,
            PermissionResolved(
                request_id=request_id, decision=PermissionResolution(decision.value), scope=scope
            ),
        )
        if not entry.future.done():
            entry.future.set_result(Answer(decision, scope, deny_message))

    async def cancel_for(self, agent_id: AgentId) -> None:
        """Close every open request of an agent as cancelled."""
        mine = [rid for rid, entry in self._open.items() if entry.agent_id == agent_id]
        for request_id in mine:
            entry = self._open.pop(request_id)
            await self._deps.memory.close_pending_approval(
                request_id, PermissionResolution.CANCELLED
            )
            self._deps.bus.publish(
                agent_id,
                PermissionResolved(request_id=request_id, decision=PermissionResolution.CANCELLED),
            )
            if not entry.future.done():
                entry.future.cancel()
