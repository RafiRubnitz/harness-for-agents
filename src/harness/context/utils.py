"""Pure helpers for building model requests out of history."""

from collections.abc import Sequence
from html import escape

from harness.context.consts import (
    AGENT_MESSAGE_BLOCK,
    AGENT_MESSAGE_CLOSE,
    AGENT_MESSAGE_CLOSE_SHOWN,
    BYTES_PER_TOKEN_ESTIMATE,
    LINE_SEPARATOR,
    TAG_AGENT_MESSAGE,
    TAGGED_BLOCK,
    TEXT_ENCODING,
)
from harness.core.consts import AddressType, MessageRole
from harness.core.records import MessageRecord
from harness.core.types import ContentBlock, Message, MessageOrigin, TextBlock


def agent_message(origin: MessageOrigin, body: str) -> str:
    """Wrap a message another agent sent, so the model can tell it from the user's words.

    The attributes come from the core (never from the content), and a closing tag inside
    the body is defused so the sender cannot end the block early.
    """
    return AGENT_MESSAGE_BLOCK.format(
        tag=TAG_AGENT_MESSAGE,
        source=escape(str(origin.source)),
        name=escape(origin.source_name or origin.source.id),
        message_id=escape(origin.message_id or ""),
        conversation=escape(origin.conversation_id or ""),
        expects_reply=str(origin.expects_reply).lower(),
        hop=origin.hop,
        body=body.replace(AGENT_MESSAGE_CLOSE, AGENT_MESSAGE_CLOSE_SHOWN),
    )


def _content(record: MessageRecord) -> list[ContentBlock]:
    """The blocks the model gets for one record; agent messages are wrapped."""
    origin = record.origin
    from_agent = origin is not None and origin.source.type is AddressType.AGENT
    if origin is None or not from_agent or record.role is not MessageRole.USER:
        return list(record.content)
    others = [block for block in record.content if not isinstance(block, TextBlock)]
    return [TextBlock(text=agent_message(origin, plain_text(record.content))), *others]


def to_messages(records: Sequence[MessageRecord]) -> list[Message]:
    """Turn history records into model messages.

    ``tool`` records become ``user`` messages (they hold tool results), empty records are
    dropped, and neighbours with the same role are merged so roles always alternate.
    """
    messages: list[Message] = []
    for record in records:
        if not record.content:
            continue
        role = "assistant" if record.role is MessageRole.ASSISTANT else "user"
        if messages and messages[-1].role == role:
            messages[-1] = Message(role=role, content=[*messages[-1].content, *_content(record)])
        else:
            messages.append(Message(role=role, content=_content(record)))
    return messages


def estimate_tokens(text: str) -> int:
    """Rough token count of a text.

    Counted from UTF-8 bytes, not characters: Hebrew takes about twice the tokens of
    English per character, and so does its byte length.
    """
    return len(text.encode(TEXT_ENCODING)) // BYTES_PER_TOKEN_ESTIMATE


def blocks_text(blocks: Sequence[ContentBlock]) -> str:
    """The JSON text of content blocks, for size estimates."""
    return "".join(block.model_dump_json() for block in blocks)


def blocks_tokens(blocks: Sequence[ContentBlock]) -> int:
    """Rough token count of content blocks."""
    return estimate_tokens(blocks_text(blocks))


def plain_text(blocks: Sequence[ContentBlock]) -> str:
    """The text blocks of a content list, one per line."""
    return LINE_SEPARATOR.join(block.text for block in blocks if isinstance(block, TextBlock))


def tagged(tag: str, body: str) -> str:
    """Wrap ``body`` in a marked block (``<tag>...</tag>``)."""
    return TAGGED_BLOCK.format(tag=tag, body=body)


def with_last_user_text(messages: Sequence[Message], text: str) -> list[Message]:
    """Append a text block to the last message when it is the user's, else as a new one."""
    block = TextBlock(text=text)
    result = list(messages)
    if result and result[-1].role == "user":
        result[-1] = Message(role="user", content=[*result[-1].content, block])
    else:
        result.append(Message(role="user", content=[block]))
    return result


def with_first_user_blocks(
    messages: Sequence[Message], blocks: Sequence[ContentBlock]
) -> list[Message]:
    """Put blocks at the very start of the conversation, in a user message."""
    result = list(messages)
    if result and result[0].role == "user":
        result[0] = Message(role="user", content=[*blocks, *result[0].content])
    else:
        result.insert(0, Message(role="user", content=list(blocks)))
    return result
