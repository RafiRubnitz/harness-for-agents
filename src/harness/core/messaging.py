"""Agent messaging models: what the tools and the local server exchange (phase2-03 §4, §5).

The ``Messenger`` protocol lives here so the tools and the local server can use the
messaging service without importing the agent manager.
"""

from typing import Protocol

from pydantic import BaseModel, ConfigDict

from harness.core.consts import AgentStatus
from harness.core.types import Address, AgentId


class AgentInfo(BaseModel):
    """One agent as another agent (or an outside caller) sees it."""

    model_config = ConfigDict(frozen=True)
    id: AgentId
    name: str
    address: str
    description: str
    template: str | None
    status: AgentStatus
    accepts_messages: bool


class SendMessageInput(BaseModel):
    """A message to one or more agents; every target gets its own copy."""

    model_config = ConfigDict(extra="forbid")
    to: list[str]
    content: str
    expects_reply: bool = False
    conversation_id: str | None = None
    reply_to: str | None = None


class SendMessageOutput(BaseModel):
    """What happened to a sent message, target by target."""

    message_id: str
    conversation_id: str
    delivered: list[str]
    rejected: list[tuple[str, str]]


class Messenger(Protocol):
    """Sends messages to agents and lists who can be written to."""

    def list_agents(self, viewer: Address) -> list[AgentInfo]:
        """Return the discoverable agents, without the viewer itself."""
        ...

    async def send(self, sender: Address, message: SendMessageInput) -> SendMessageOutput:
        """Put the message in every target's inbox; returns once it is stored."""
        ...

    async def reply(self, sender: AgentId, message_id: str, content: str) -> SendMessageOutput:
        """Answer a message the sender received from another agent."""
        ...
