"""FakeMessenger: a scripted ``Messenger`` that records what was sent."""

from harness.core.errors import AgentError
from harness.core.messaging import AgentInfo, SendMessageInput, SendMessageOutput
from harness.core.types import Address, AgentId

FAKE_MESSAGE_ID = "msg_fake"
FAKE_CONVERSATION_ID = "conv_fake"


class FakeMessenger:
    """Delivers to every target unless a test put a reason in ``refusals``."""

    def __init__(self, agents: list[AgentInfo] | None = None) -> None:
        self.agents: list[AgentInfo] = list(agents or [])
        self.refusals: dict[str, str] = {}
        self.sent: list[tuple[Address, SendMessageInput]] = []
        self.viewers: list[Address] = []
        self.reply_error: AgentError | None = None

    def list_agents(self, viewer: Address) -> list[AgentInfo]:
        """Return the scripted agents and remember who asked."""
        self.viewers.append(viewer)
        return list(self.agents)

    async def send(self, sender: Address, message: SendMessageInput) -> SendMessageOutput:
        """Record the message; targets named in ``refusals`` are rejected."""
        self.sent.append((sender, message))
        return SendMessageOutput(
            message_id=FAKE_MESSAGE_ID,
            conversation_id=message.conversation_id or FAKE_CONVERSATION_ID,
            delivered=[to for to in message.to if to not in self.refusals],
            rejected=[(to, self.refusals[to]) for to in message.to if to in self.refusals],
        )

    async def reply(self, sender: AgentId, message_id: str, content: str) -> SendMessageOutput:
        """Record a reply as a message to ``message_id``; raise ``reply_error`` when set."""
        if self.reply_error is not None:
            raise self.reply_error
        message = SendMessageInput(to=[message_id], content=content, reply_to=message_id)
        return await self.send(Address.model_validate({"type": "agent", "id": sender}), message)
