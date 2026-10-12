"""Pure helpers for building model requests out of history."""

from collections.abc import Sequence

from harness.context.consts import (
    BYTES_PER_TOKEN_ESTIMATE,
    LINE_SEPARATOR,
    TAGGED_BLOCK,
    TEXT_ENCODING,
)
from harness.core.consts import MessageRole
from harness.core.records import MessageRecord
from harness.core.types import ContentBlock, Message, TextBlock


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
            messages[-1] = Message(role=role, content=[*messages[-1].content, *record.content])
        else:
            messages.append(Message(role=role, content=list(record.content)))
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
