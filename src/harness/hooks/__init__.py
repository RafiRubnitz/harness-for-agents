"""Hooks manager (component 5): run user hooks around agent events.

Other components use the ``HooksManager`` protocol; ``DefaultHooksManager`` is the real
one and ``NullHooksManager`` a stand-in that never fires. Hook authors import
``HookInput`` and ``HookOutput`` from here.
"""

from typing import Protocol

from harness.core.consts import HookEvent
from harness.core.hook_types import HookInput, HookOutput, HookSpec, MergedOutput
from harness.core.types import AgentId
from harness.hooks.manager import DefaultHooksManager, HooksDeps


class HooksManager(Protocol):
    """Fires hooks for an event and merges what they return."""

    async def emit(self, event: HookEvent, inp: HookInput) -> MergedOutput:
        """Run every hook matching ``event`` and return the merged output."""
        ...

    def register(self, hook: HookSpec) -> None:
        """Add a hook at runtime."""
        ...

    def unregister(self, name: str) -> None:
        """Remove a hook by name."""
        ...

    def list(
        self, event: HookEvent | None = None, agent_id: AgentId | None = None
    ) -> list[HookSpec]:
        """Return the hooks that apply to an event / agent."""
        ...

    async def reload(self) -> None:
        """Re-read hook definitions."""
        ...


class NullHooksManager:
    """A ``HooksManager`` with no hooks: every event passes through unchanged."""

    async def emit(self, event: HookEvent, inp: HookInput) -> MergedOutput:
        """Return an empty decision and the payload as it was given."""
        return MergedOutput(payload=dict(inp.payload))

    def register(self, hook: HookSpec) -> None:
        """Ignore the hook."""

    def unregister(self, name: str) -> None:
        """Nothing to remove."""

    def list(
        self, event: HookEvent | None = None, agent_id: AgentId | None = None
    ) -> list[HookSpec]:
        """No hooks."""
        return []

    async def reload(self) -> None:
        """Nothing to reload."""


__all__ = [
    "DefaultHooksManager",
    "HookInput",
    "HookOutput",
    "HooksDeps",
    "HooksManager",
    "NullHooksManager",
]
