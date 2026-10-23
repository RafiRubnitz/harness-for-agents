"""``MessagingService``: one agent writes to another (phase 2, 03).

Sending is always asynchronous: the message goes into the target's inbox through
``AgentManager.send`` (hooks, commit, wake) and the sender carries on. A reply is just
another message that names the one it answers.
"""

from collections import deque
from datetime import datetime
from fnmatch import fnmatchcase
from typing import Protocol

from harness.agents.consts import (
    FIRST_HOP,
    MSG_CONVERSATION_LIMIT,
    MSG_REPLY_NOT_AGENT,
    MSG_REPLY_UNKNOWN,
    SECONDS_PER_HOUR,
)
from harness.agents.events import EventBus
from harness.config import ConfigManager, MessagingConfig
from harness.core.consts import (
    ADDRESS_ANY,
    ADDRESS_SEPARATOR,
    ID_PREFIX_CONVERSATION,
    ID_PREFIX_MESSAGE,
    AddressType,
    AgentStatus,
    ErrorCode,
    RejectReason,
)
from harness.core.errors import AgentError, HarnessError
from harness.core.events import Warning as WarningEvent
from harness.core.messaging import AgentInfo, SendMessageInput, SendMessageOutput
from harness.core.records import AgentState
from harness.core.types import (
    Address,
    AgentId,
    Clock,
    ContentBlock,
    IdGenerator,
    MessageId,
    MessageOrigin,
    TextBlock,
)
from harness.memory import MemoryStore


class _Agents(Protocol):
    """What the service needs from the agent manager."""

    async def send(
        self,
        agent_id: AgentId,
        content: list[ContentBlock],
        origin: MessageOrigin | None = None,
    ) -> bool: ...
    def list(self, *, include_archived: bool = False) -> list[AgentState]: ...


def address_of(agent: AgentState) -> str:
    """The address other agents write to: ``agent:<name>``."""
    return f"{AddressType.AGENT.value}{ADDRESS_SEPARATOR}{agent.name}"


def _spellings(sender: Address, name: str | None) -> list[str]:
    """Every way a sender can be written in ``allow_from`` / ``deny_from``."""
    found = [str(sender), sender.id]
    if name:
        found += [f"{sender.type.value}{ADDRESS_SEPARATOR}{name}", name]
    return found


def _listed(patterns: list[str], spellings: list[str]) -> bool:
    return any(
        pattern == ADDRESS_ANY or fnmatchcase(spelling, pattern)
        for pattern in patterns
        for spelling in spellings
    )


class MessagingService:
    """Resolves addresses, applies the limits, and hands messages to the agent manager.

    The hourly counters and the per-conversation counters live in memory: they start
    from zero when the core restarts.
    """

    def __init__(
        self,
        agents: _Agents,
        *,
        memory: MemoryStore,
        config: ConfigManager,
        clock: Clock,
        ids: IdGenerator,
        bus: EventBus,
    ) -> None:
        self._agents = agents
        self._memory = memory
        self._config = config
        self._clock = clock
        self._ids = ids
        self._bus = bus
        self._sent: dict[tuple[str, AgentId], deque[datetime]] = {}
        self._conversations: dict[str, int] = {}
        self._stopped: set[str] = set()

    # --- addresses ---

    def resolve(self, reference: str) -> AgentState | None:
        """Find an agent by ``agent:<name>``, ``agent:<id>``, a bare name or a bare id."""
        prefix = f"{AddressType.AGENT.value}{ADDRESS_SEPARATOR}"
        wanted = reference.strip().removeprefix(prefix)
        agents = self._agents.list(include_archived=True)
        by_id = [agent for agent in agents if agent.id == wanted]
        by_name = [agent for agent in agents if agent.name == wanted]
        found = by_id or by_name
        return found[0] if found else None

    def _name_of(self, sender: Address) -> str | None:
        if sender.type is not AddressType.AGENT:
            return None
        found = self.resolve(sender.id)
        return found.name if found else None

    def _accepts(self, target: AgentState, sender: Address, name: str | None) -> bool:
        rules = self._config.section(MessagingConfig, target.id)
        spellings = _spellings(sender, name)
        return _listed(rules.allow_from, spellings) and not _listed(rules.deny_from, spellings)

    def list_agents(self, viewer: Address) -> list[AgentInfo]:
        """The agents ``viewer`` may discover: not itself, not archived, not hidden."""
        name = self._name_of(viewer)
        return [
            AgentInfo(
                id=agent.id,
                name=agent.name,
                address=address_of(agent),
                description=agent.description,
                template=agent.template,
                status=agent.status,
                accepts_messages=self._accepts(agent, viewer, name),
            )
            for agent in self._agents.list()
            if agent.id != viewer.id
            and self._config.section(MessagingConfig, agent.id).discoverable
        ]

    # --- limits ---

    def _over_rate(self, sender: Address, target: AgentState, now: datetime) -> bool:
        sent = self._sent.setdefault((str(sender), target.id), deque())
        while sent and (now - sent[0]).total_seconds() >= SECONDS_PER_HOUR:
            sent.popleft()
        return len(sent) >= self._config.section(MessagingConfig, target.id).max_per_hour

    def _conversation_over(self, conversation_id: str, sender: Address, target: AgentState) -> bool:
        limit = self._config.section(MessagingConfig, target.id).max_messages_per_conversation
        if self._conversations.get(conversation_id, 0) < limit:
            return False
        if conversation_id not in self._stopped:  # tell the user once
            self._stopped.add(conversation_id)
            message = MSG_CONVERSATION_LIMIT.format(
                conversation_id=conversation_id,
                limit=limit,
                sender=sender,
                target=address_of(target),
            )
            self._bus.publish(
                target.id,
                WarningEvent(code=ErrorCode.MESSAGING_CONVERSATION_LIMIT, message=message),
            )
        return True

    def _refusal(
        self, target: AgentState | None, sender: Address, name: str | None, hop: int
    ) -> RejectReason | None:
        """Why ``target`` cannot get this message, before the per-pair counters."""
        if target is None:
            return RejectReason.NOT_FOUND
        if sender.type is AddressType.AGENT and target.id == sender.id:
            return RejectReason.SELF
        if target.status is AgentStatus.ARCHIVED:
            return RejectReason.ARCHIVED
        if not self._accepts(target, sender, name):
            return RejectReason.BLOCKED
        if hop > self._config.section(MessagingConfig, target.id).max_agent_hops:
            return RejectReason.HOP_LIMIT
        return None

    # --- sending ---

    async def _hop(self, sender: Address, reply_to: str | None) -> int:
        """Hops since the last message that did not come from an agent."""
        if sender.type is not AddressType.AGENT:
            return 0
        agent_id = AgentId(sender.id)
        answered = await self._memory.received_message(agent_id, reply_to) if reply_to else None
        last = answered or await self._memory.last_received(agent_id)
        if last is None or last.origin is None:
            return FIRST_HOP
        return last.origin.hop + 1

    async def _conversation(self, sender: Address, message: SendMessageInput) -> str:
        if message.conversation_id:
            return message.conversation_id
        if message.reply_to and sender.type is AddressType.AGENT:
            answered = await self._memory.received_message(AgentId(sender.id), message.reply_to)
            if answered and answered.origin and answered.origin.conversation_id:
                return answered.origin.conversation_id
        return self._ids.new(ID_PREFIX_CONVERSATION)

    async def _deliver(
        self, target: AgentState, origin: MessageOrigin, content: str
    ) -> RejectReason | None:
        sender = origin.source
        conversation_id = str(origin.conversation_id)
        now = self._clock.now()
        if self._over_rate(sender, target, now):
            return RejectReason.RATE_LIMIT
        if self._conversation_over(conversation_id, sender, target):
            return RejectReason.CONVERSATION_LIMIT
        try:
            accepted = await self._agents.send(target.id, [TextBlock(text=content)], origin)
        except HarnessError:  # archived or deleted since it was resolved
            return RejectReason.NOT_FOUND
        if not accepted:
            return RejectReason.HOOK
        self._sent[(str(sender), target.id)].append(now)
        self._conversations[conversation_id] = self._conversations.get(conversation_id, 0) + 1
        return None

    async def send(self, sender: Address, message: SendMessageInput) -> SendMessageOutput:
        """Send ``message`` to every address in ``message.to``; never waits for an answer.

        A target that cannot get the message is reported in ``rejected`` with the reason;
        the others still get it.
        """
        name = self._name_of(sender)
        hop = await self._hop(sender, message.reply_to)
        message_id = self._ids.new(ID_PREFIX_MESSAGE)
        conversation_id = await self._conversation(sender, message)
        found = {reference: self.resolve(reference) for reference in message.to}
        targets = [
            Address(type=AddressType.AGENT, id=agent.id) for agent in found.values() if agent
        ]
        origin = MessageOrigin(
            source=sender,
            targets=targets,
            conversation_id=conversation_id,
            hop=hop,
            reply_to=MessageId(message.reply_to) if message.reply_to else None,
            message_id=message_id,
            expects_reply=message.expects_reply,
            source_name=name,
        )
        delivered: list[str] = []
        rejected: list[tuple[str, str]] = []
        done: set[AgentId] = set()
        for reference, target in found.items():
            reason = self._refusal(target, sender, name, hop)
            if target is not None and reason is None and target.id not in done:
                done.add(target.id)
                reason = await self._deliver(target, origin, message.content)
            if reason is None:
                delivered.append(reference)
            else:
                rejected.append((reference, reason.value))
        return SendMessageOutput(
            message_id=message_id,
            conversation_id=conversation_id,
            delivered=delivered,
            rejected=rejected,
        )

    async def reply(self, sender: AgentId, message_id: str, content: str) -> SendMessageOutput:
        """Answer a message an agent sent to ``sender``, in the same conversation.

        Raises:
            AgentError: ``sender`` got no such message, or it did not come from an agent.
        """
        answered = await self._memory.received_message(sender, message_id)
        if answered is None or answered.origin is None:
            raise AgentError(
                MSG_REPLY_UNKNOWN.format(message_id=message_id), code=ErrorCode.TOOL_FAILED
            )
        source = answered.origin.source
        if source.type is not AddressType.AGENT:
            raise AgentError(
                MSG_REPLY_NOT_AGENT.format(message_id=message_id), code=ErrorCode.TOOL_FAILED
            )
        return await self.send(
            Address(type=AddressType.AGENT, id=sender),
            SendMessageInput(
                to=[str(source)],
                content=content,
                conversation_id=answered.origin.conversation_id,
                reply_to=message_id,
            ),
        )
