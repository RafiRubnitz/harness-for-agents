"""Pure transformations of history records on their way into a model request (07 §5, §7, §8).

Nothing here touches the stored history: every function returns changed copies.
"""

import json
from collections.abc import Sequence

from harness.context.consts import (
    MSG_OLD_IMAGE,
    MSG_OLD_RESULT,
    MSG_OLD_RESULT_STORED,
    MSG_TRUNCATED,
    MSG_TRUNCATED_UNSTORED,
    OLD_RESULT_MIN_TOKENS,
    TRANSCRIPT_BLOCK_MAX_CHARS,
    TRANSCRIPT_CUT_MARK,
    TRANSCRIPT_FILE,
    TRANSCRIPT_IMAGE,
    TRANSCRIPT_LINE,
    TRANSCRIPT_TOOL_CALL,
    TRANSCRIPT_TOOL_ERROR,
    TRANSCRIPT_TOOL_RESULT,
    TRUNCATE_HEAD_SHARE,
)
from harness.context.utils import blocks_tokens, estimate_tokens, plain_text
from harness.core.consts import MessageRole
from harness.core.records import MessageRecord
from harness.core.types import (
    ContentBlock,
    FileBlock,
    ImageBlock,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
)


def _with_content(record: MessageRecord, content: Sequence[ContentBlock]) -> MessageRecord:
    return record.model_copy(update={"content": list(content)})


# --- thinking ---


def strip_old_thinking(records: Sequence[MessageRecord]) -> list[MessageRecord]:
    """Keep thinking blocks only in the current tool-use turn (07 §8).

    The current turn is everything after the last message the user (or another sender)
    wrote; thinking in earlier turns is dropped, as the provider API expects.
    """
    last_user = max(
        (i for i, record in enumerate(records) if record.role is MessageRole.USER), default=-1
    )
    result: list[MessageRecord] = []
    for index, record in enumerate(records):
        if index < last_user and any(isinstance(b, ThinkingBlock) for b in record.content):
            kept = [block for block in record.content if not isinstance(block, ThinkingBlock)]
            result.append(_with_content(record, kept))
        else:
            result.append(record)
    return result


# --- big tool results ---


def truncate_text(text: str, limit_tokens: int, result_id: str | None) -> str:
    """Cut the middle out of a long text, leaving a note with the stored result's id."""
    total = estimate_tokens(text)
    if total <= limit_tokens or limit_tokens <= 0:
        return text
    keep_chars = len(text) * limit_tokens // total
    head_chars = int(keep_chars * TRUNCATE_HEAD_SHARE)
    tail_chars = keep_chars - head_chars
    omitted = total - limit_tokens
    note = (
        MSG_TRUNCATED.format(omitted=omitted, result_id=result_id)
        if result_id
        else MSG_TRUNCATED_UNSTORED.format(omitted=omitted)
    )
    return text[:head_chars] + note + (text[-tail_chars:] if tail_chars else "")


def _shrunk_result(block: ToolResultBlock, limit_tokens: int) -> ToolResultBlock:
    text = plain_text(block.content)
    if estimate_tokens(text) <= limit_tokens:
        return block
    others = [inner for inner in block.content if not isinstance(inner, TextBlock)]
    cut = TextBlock(text=truncate_text(text, limit_tokens, block.result_id))
    return block.model_copy(update={"content": [cut, *others]})


def shrink_results(records: Sequence[MessageRecord], limit_tokens: int) -> list[MessageRecord]:
    """Truncate every tool result that is longer than ``limit_tokens`` (07 §7)."""
    result: list[MessageRecord] = []
    for record in records:
        content: list[ContentBlock] = [
            _shrunk_result(block, limit_tokens) if isinstance(block, ToolResultBlock) else block
            for block in record.content
        ]
        result.append(_with_content(record, content))
    return result


# --- old tool results ---


def _old_note(block: ToolResultBlock) -> ToolResultBlock:
    tokens = blocks_tokens(block.content)
    has_image = any(isinstance(inner, ImageBlock) for inner in block.content)
    if tokens <= OLD_RESULT_MIN_TOKENS and not has_image:
        return block
    note = (
        MSG_OLD_RESULT_STORED.format(tokens=tokens, result_id=block.result_id)
        if block.result_id
        else MSG_OLD_RESULT.format(tokens=tokens)
    )
    return block.model_copy(update={"content": [TextBlock(text=note)]})


def _aged(block: ContentBlock) -> ContentBlock:
    if isinstance(block, ToolResultBlock):
        return _old_note(block)
    if isinstance(block, ImageBlock):
        return TextBlock(text=MSG_OLD_IMAGE)
    return block


def stub_old_results(records: Sequence[MessageRecord], keep_steps: int) -> list[MessageRecord]:
    """Replace tool results and images older than ``keep_steps`` steps by a short note.

    The line between old and recent moves only once every ``keep_steps`` steps, not on
    every step: a line that moved each step would change the request's prefix each
    time and the prompt cache would never be hit.
    """
    steps = [i for i, record in enumerate(records) if record.role is MessageRole.ASSISTANT]
    if keep_steps <= 0 or len(steps) <= keep_steps:
        return list(records)
    old_steps = (len(steps) - keep_steps) // keep_steps * keep_steps
    if old_steps == 0:
        return list(records)
    boundary = steps[old_steps]
    aged = [_with_content(r, [_aged(block) for block in r.content]) for r in records[:boundary]]
    return [*aged, *records[boundary:]]


# --- pinned blocks ---


def pin_text(
    records: Sequence[MessageRecord], first_seq: int, text: str
) -> tuple[list[MessageRecord], bool]:
    """Put a text block at the start of the message that opened a run.

    Returns:
        The records, and whether the opening message was among them.
    """
    result = list(records)
    for index, record in enumerate(result):
        if record.seq >= first_seq and record.role is MessageRole.USER:
            result[index] = _with_content(record, [TextBlock(text=text), *record.content])
            return result, True
    return result, False


# --- compaction range ---


def tail_start(records: Sequence[MessageRecord], keep_tokens: int) -> int:
    """Index of the first record of the newest part that fits in ``keep_tokens``.

    The last record always stays in the tail, however big it is.
    """
    kept = 0
    for index in range(len(records) - 1, -1, -1):
        kept += blocks_tokens(records[index].content)
        if kept > keep_tokens:
            return min(index + 1, len(records) - 1)
    return 0


def safe_cut(records: Sequence[MessageRecord], target: int) -> int | None:
    """The cut closest to ``target`` that splits no tool call from its result (07 §5).

    A cut at ``c`` compacts ``records[:c]`` and keeps ``records[c:]``. It is safe when
    every tool call before it was answered before it, and the kept part does not open
    with tool results.

    Returns:
        The cut (``1..len(records)``), preferring the nearest one below ``target``, or
        ``None`` when no safe cut exists.
    """
    safe: list[int] = []
    open_calls: set[str] = set()
    for index, record in enumerate(records):
        if index > 0 and not open_calls and record.role is not MessageRole.TOOL:
            safe.append(index)
        for block in record.content:
            if isinstance(block, ToolCallBlock):
                open_calls.add(block.id)
            elif isinstance(block, ToolResultBlock):
                open_calls.discard(block.call_id)
    if records and not open_calls:
        safe.append(len(records))
    below = [cut for cut in safe if cut <= target]
    if below:
        return below[-1]
    above = [cut for cut in safe if cut > target]
    return above[0] if above else None


# --- transcript for the summary model ---


def _clip(text: str) -> str:
    if len(text) <= TRANSCRIPT_BLOCK_MAX_CHARS:
        return text
    half = TRANSCRIPT_BLOCK_MAX_CHARS // 2
    return text[:half] + TRANSCRIPT_CUT_MARK + text[-half:]


def _block_line(block: ContentBlock) -> str | None:
    match block:
        case TextBlock():
            return block.text
        case ToolCallBlock():
            rendered = json.dumps(block.input, ensure_ascii=False)
            return TRANSCRIPT_TOOL_CALL.format(name=block.name, input=_clip(rendered))
        case ToolResultBlock():
            template = TRANSCRIPT_TOOL_ERROR if block.is_error else TRANSCRIPT_TOOL_RESULT
            return template.format(text=_clip(plain_text(block.content)))
        case ImageBlock():
            return TRANSCRIPT_IMAGE
        case FileBlock():
            return TRANSCRIPT_FILE
        case ThinkingBlock():
            return None


def transcript(records: Sequence[MessageRecord]) -> list[str]:
    """Render records as plain lines for the summary model; long tool data is clipped."""
    lines: list[str] = []
    for record in records:
        parts = [line for block in record.content if (line := _block_line(block))]
        if parts:
            text = " ".join(parts)
            lines.append(TRANSCRIPT_LINE.format(seq=record.seq, role=record.role.value, text=text))
    return lines


def chunked(lines: Sequence[str], chunk_tokens: int) -> list[list[str]]:
    """Split lines into runs of at most ``chunk_tokens`` (a single longer line stands alone)."""
    chunks: list[list[str]] = []
    current: list[str] = []
    used = 0
    for line in lines:
        size = estimate_tokens(line)
        if current and used + size > chunk_tokens:
            chunks.append(current)
            current, used = [], 0
        current.append(line)
        used += size
    if current:
        chunks.append(current)
    return chunks
