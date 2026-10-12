"""In-memory ``AgentLayerStore``."""

from pydantic import JsonValue

from harness.core.types import AgentId


class FakeAgentLayerStore:
    """Agent config overrides kept in a dict instead of the database."""

    def __init__(self) -> None:
        self.overrides: dict[AgentId, dict[str, JsonValue]] = {}

    async def get_config_overrides(self, agent_id: AgentId) -> dict[str, JsonValue]:
        """Return the agent's stored overrides."""
        return dict(self.overrides.get(agent_id, {}))

    async def set_config_overrides(
        self, agent_id: AgentId, overrides: dict[str, JsonValue]
    ) -> None:
        """Persist the agent's overrides."""
        self.overrides[agent_id] = dict(overrides)
