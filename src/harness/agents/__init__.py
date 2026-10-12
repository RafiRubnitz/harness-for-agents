"""Agent manager (component 4): agent lifecycle, the run loop, approvals, recovery.

Other components use the ``AgentManager`` protocol.
"""

from collections.abc import AsyncIterator
from typing import Protocol

from pydantic import JsonValue

from harness.agents.events import EventBus, Subscription
from harness.agents.manager import DefaultAgentManager
from harness.agents.messaging import MessagingService
from harness.agents.runtime import AgentDeps
from harness.core.consts import PermissionAnswer, PermissionScope
from harness.core.events import Event
from harness.core.records import AgentSpec, AgentState
from harness.core.types import AgentId, ContentBlock, MessageOrigin, PermissionRequestId


class AgentManager(Protocol):
    """Creates agents, runs them, and reports everything they do as events."""

    async def create(self, spec: AgentSpec) -> AgentId: ...
    async def create_from_template(
        self, template: str, *, overrides: dict[str, JsonValue] | None = None
    ) -> AgentId: ...
    async def update(self, agent_id: AgentId, changes: dict[str, JsonValue]) -> None: ...
    async def apply_template(self, template: str, agent_ids: list[AgentId]) -> None: ...
    async def archive(self, agent_id: AgentId) -> None: ...
    async def unarchive(self, agent_id: AgentId) -> None: ...
    async def delete(self, agent_id: AgentId) -> None: ...

    async def send(
        self,
        agent_id: AgentId,
        content: list[ContentBlock],
        origin: MessageOrigin | None = None,
    ) -> bool: ...
    async def stop(self, agent_id: AgentId) -> None: ...
    async def pause(self, agent_id: AgentId) -> None: ...
    async def resume(self, agent_id: AgentId) -> None: ...
    async def retry(self, agent_id: AgentId) -> None: ...
    async def respond_permission(
        self,
        request_id: PermissionRequestId,
        decision: PermissionAnswer,
        scope: PermissionScope = PermissionScope.ONCE,
        deny_message: str | None = None,
    ) -> None: ...
    async def shutdown(self, *, stop_agents: bool) -> None: ...

    def get(self, agent_id: AgentId) -> AgentState: ...
    def list(self, *, include_archived: bool = False) -> list[AgentState]: ...
    def events(self) -> AsyncIterator[Event]: ...


__all__ = [
    "AgentDeps",
    "AgentManager",
    "DefaultAgentManager",
    "EventBus",
    "MessagingService",
    "Subscription",
]
