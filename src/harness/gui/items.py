"""What one agent's conversation is made of on screen, and how history becomes it."""

from dataclasses import dataclass, field

from pydantic import JsonValue

from harness.core.consts import (
    INJECTED_REASON,
    INJECTED_RUN_IN_BACKGROUND,
    AddressType,
    AgentEventType,
    Decision,
    MessageRole,
)
from harness.core.records import AgentEventRecord, MessageRecord
from harness.core.types import (
    ContentBlock,
    FileBlock,
    ImageBlock,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
)
from harness.gui import consts
from harness.gui.consts import NoticeLevel, ToolStatus

_INJECTED = frozenset({INJECTED_REASON, INJECTED_RUN_IN_BACKGROUND})
_QUIET_DECISIONS = frozenset({None, Decision.ALLOW.value})


@dataclass
class UserItem:
    """A message the agent received; ``label`` names a source other than the user."""

    text: str
    label: str | None = None


@dataclass
class AssistantItem:
    """The agent's answer (Markdown); ``streaming`` while deltas still arrive."""

    text: str
    streaming: bool = False


@dataclass
class ThinkingItem:
    """Model reasoning, folded by default."""

    text: str
    streaming: bool = False


@dataclass
class ToolItem:
    """One tool call with its result, folded to a single row by default."""

    call_id: str
    tool: str
    input: dict[str, JsonValue]
    reason: str | None = None
    status: ToolStatus = ToolStatus.RUNNING
    duration_ms: int | None = None
    output: str = ""
    background: bool = False


@dataclass
class NoticeItem:
    """A line from the harness itself: status, warning, or error."""

    text: str
    level: NoticeLevel = NoticeLevel.INFO
    mono: bool = False


Item = UserItem | AssistantItem | ThinkingItem | ToolItem | NoticeItem


@dataclass
class Conversation:
    """The ordered items of one agent, with tool rows reachable by call id."""

    items: list[Item] = field(default_factory=list[Item])
    tools: dict[str, int] = field(default_factory=dict[str, int])
    dirty_from: int | None = None

    def touch(self, index: int) -> None:
        """Mark everything from ``index`` on as needing a redraw."""
        self.dirty_from = index if self.dirty_from is None else min(self.dirty_from, index)

    def add(self, item: Item) -> int:
        """Append an item and return its index."""
        self.items.append(item)
        index = len(self.items) - 1
        if isinstance(item, ToolItem):
            self.tools[item.call_id] = index
        self.touch(index)
        return index

    def tool(self, call_id: str) -> ToolItem | None:
        """The row of one tool call, if it is known."""
        index = self.tools.get(call_id)
        found = self.items[index] if index is not None else None
        if not isinstance(found, ToolItem) or index is None:
            return None
        self.touch(index)
        return found

    def drop_streaming(self) -> None:
        """Remove the trailing items that deltas built; the stored message replaces them."""
        while self.items:
            last = self.items[-1]
            if not (isinstance(last, AssistantItem | ThinkingItem) and last.streaming):
                break
            self.items.pop()
            self.touch(len(self.items))

    def take_dirty(self) -> int | None:
        """Return the first stale index and clear the mark."""
        dirty, self.dirty_from = self.dirty_from, None
        return dirty


def clip(text: str, limit: int) -> str:
    """Bound a text, marking the cut."""
    return text if len(text) <= limit else text[:limit] + consts.TRUNCATION_MARK


def blocks_text(blocks: list[ContentBlock]) -> str:
    """The readable text of content blocks; attachments become placeholders."""
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, TextBlock):
            parts.append(block.text)
        elif isinstance(block, ImageBlock):
            parts.append(consts.MSG_BLOCK_IMAGE)
        elif isinstance(block, FileBlock):
            parts.append(consts.MSG_BLOCK_FILE)
    return "\n".join(parts)


def shown_input(raw: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """A tool input without the fields the harness injected (the reason has its own line)."""
    return {key: value for key, value in raw.items() if key not in _INJECTED}


def add_user(conversation: Conversation, message: MessageRecord) -> None:
    """Show a received message; tool results inside it close their tool rows."""
    origin = message.origin
    label = None
    if origin is not None and origin.source.type is AddressType.AGENT:
        label = consts.AGENT_ORIGIN_LABEL.format(name=origin.source_name or origin.source.id)
    elif origin is not None and origin.source.type is not AddressType.USER:
        label = consts.ORIGIN_LABEL.format(source=origin.source.type.value)
    text = blocks_text(message.content)
    if text:
        conversation.add(UserItem(text, label))
    add_results(conversation, message)


def add_results(conversation: Conversation, message: MessageRecord) -> None:
    """Close the tool rows whose results a stored message carries."""
    for block in message.content:
        if not isinstance(block, ToolResultBlock):
            continue
        row = conversation.tool(block.call_id)
        if row is None or row.status in (ToolStatus.OK, ToolStatus.FAILED):
            continue
        row.status = ToolStatus.FAILED if block.is_error else ToolStatus.OK
        row.output = clip(blocks_text(block.content), consts.TOOL_OUTPUT_MAX_CHARS)


def add_assistant(conversation: Conversation, message: MessageRecord) -> None:
    """Show a stored answer: thinking, text, and a row per tool call."""
    conversation.drop_streaming()
    for block in message.content:
        if isinstance(block, ThinkingBlock):
            conversation.add(ThinkingItem(block.text))
        elif isinstance(block, TextBlock) and block.text:
            conversation.add(AssistantItem(block.text))
        elif isinstance(block, ToolCallBlock) and conversation.tool(block.id) is None:
            reason = block.reason or _reason_in(block.input)
            conversation.add(ToolItem(block.id, block.name, shown_input(block.input), reason))


def _reason_in(raw: dict[str, JsonValue]) -> str | None:
    reason = raw.get(INJECTED_REASON)
    return reason if isinstance(reason, str) else None


def add_message(conversation: Conversation, message: MessageRecord) -> None:
    """Show one stored message of any role."""
    if message.role is MessageRole.ASSISTANT:
        add_assistant(conversation, message)
    elif message.role is MessageRole.USER:
        add_user(conversation, message)
    else:
        add_results(conversation, message)


def _str(payload: dict[str, JsonValue], key: str) -> str:
    value = payload.get(key)
    return "" if value is None else str(value)


def notice_for(record: AgentEventRecord) -> NoticeItem | None:
    """The line a stored display event turns into; quiet events give ``None``."""
    payload = record.payload
    if record.type is AgentEventType.ERROR:
        text = consts.MSG_ERROR.format(message=_str(payload, "message"), code=_str(payload, "code"))
        return NoticeItem(clip(text, consts.NOTICE_MAX_CHARS), NoticeLevel.ERROR)
    if record.type is AgentEventType.RECOVERED:
        return NoticeItem(consts.MSG_RECOVERED, NoticeLevel.WARNING)
    if record.type is AgentEventType.LIMIT_REACHED:
        text = consts.MSG_LIMIT.format(
            steps=_str(payload, "steps"), max_steps=_str(payload, "max_steps")
        )
        return NoticeItem(text, NoticeLevel.WARNING)
    if record.type is AgentEventType.COMPACTED:
        text = consts.MSG_COMPACTED.format(
            from_seq=_str(payload, "from_seq"), to_seq=_str(payload, "to_seq")
        )
        return NoticeItem(text)
    if record.type is AgentEventType.HOOK_RUN and payload.get("decision") not in _QUIET_DECISIONS:
        text = consts.MSG_HOOK.format(
            hook=_str(payload, "hook"),
            decision=_str(payload, "decision"),
            reason=_str(payload, "reason"),
        )
        return NoticeItem(text.strip(), NoticeLevel.WARNING)
    return None


def from_history(messages: list[MessageRecord], events: list[AgentEventRecord]) -> Conversation:
    """Rebuild a conversation from what the database holds (02 §5)."""
    conversation = Conversation()
    pending = sorted(events, key=lambda record: (record.after_seq, record.created_at))
    for message in sorted(messages, key=lambda record: record.seq):
        while pending and pending[0].after_seq < message.seq:
            _add_notice(conversation, pending.pop(0))
        add_message(conversation, message)
    for record in pending:
        _add_notice(conversation, record)
    for item in conversation.items:
        if isinstance(item, ToolItem) and item.status is ToolStatus.RUNNING:
            item.status = ToolStatus.WAITING
    return conversation


def _add_notice(conversation: Conversation, record: AgentEventRecord) -> None:
    notice = notice_for(record)
    if notice is not None:
        conversation.add(notice)
