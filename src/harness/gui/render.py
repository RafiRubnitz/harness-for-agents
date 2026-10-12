"""Conversation items to rows of styled pieces, ready for a text widget. No Tk in here."""

import json
from collections.abc import Callable
from dataclasses import dataclass

from harness.gui import bidi, consts
from harness.gui.bidi import Piece
from harness.gui.consts import NoticeLevel, Tag, ToolStatus
from harness.gui.items import (
    AssistantItem,
    Item,
    NoticeItem,
    ThinkingItem,
    ToolItem,
    UserItem,
    clip,
)
from harness.gui.markdown import Line, Span, plain, render

type Style = tuple[Tag, ...]

_STATUS_MARK = {
    ToolStatus.RUNNING: (consts.RUNNING_SYMBOL, Tag.MUTED),
    ToolStatus.WAITING: (consts.WAITING_SYMBOL, Tag.WARNING),
    ToolStatus.OK: (consts.OK_SYMBOL, Tag.SUCCESS),
    ToolStatus.FAILED: (consts.ERROR_SYMBOL.strip(), Tag.ERROR),
}
_NOTICE_TAG = {
    NoticeLevel.INFO: Tag.MUTED,
    NoticeLevel.WARNING: Tag.WARNING,
    NoticeLevel.ERROR: Tag.ERROR,
}
_REASON_LINE = "  reason: {reason}"
_INDENT = "  "
_JSON_INDENT = 2


@dataclass(frozen=True, slots=True)
class Row:
    """One display line: pieces in visual order plus the tags of the whole line."""

    pieces: tuple[Piece[Style], ...]
    tags: Style = ()


@dataclass(frozen=True, slots=True)
class Layout:
    """How wide a row may be and how to measure text in a style."""

    width: int
    measure: Callable[[str, Style], int]


def tool_summary(item: ToolItem) -> str:
    """The short text after the tool name: its scalar arguments on one line."""
    values = [str(value) for value in item.input.values() if isinstance(value, str | int | float)]
    flat = " ".join(" ".join(values).split())
    return clip(flat, consts.TOOL_SUMMARY_MAX_CHARS)


def _tool_header(item: ToolItem, *, expanded: bool) -> Line:
    mark, mark_tag = _STATUS_MARK[item.status]
    failed = item.status is ToolStatus.FAILED
    base: Style = (Tag.TOOL, Tag.ERROR) if failed else (Tag.TOOL,)
    arrow = consts.TOOL_EXPANDED if expanded else consts.TOOL_COLLAPSED
    spans = [
        Span(arrow, (Tag.TOOL, Tag.MUTED)),
        Span(item.tool, (*base, Tag.BOLD)),
        Span(f"  {tool_summary(item)}", base),
        Span(f"  {mark}", (Tag.TOOL, mark_tag)),
    ]
    if item.duration_ms is not None:
        seconds = item.duration_ms / consts.MS_PER_SECOND
        spans.append(
            Span(f" {consts.TOOL_DURATION.format(seconds=seconds)}", (Tag.TOOL, Tag.MUTED))
        )
    return Line(tuple(spans), (Tag.TOOL,))


def _tool_lines(item: ToolItem, *, expanded: bool) -> list[Line]:
    lines = [_tool_header(item, expanded=expanded)]
    failed = item.status is ToolStatus.FAILED
    if failed and not expanded and item.output.strip():
        first = item.output.strip().splitlines()[0]
        lines += plain(f"{_INDENT}{consts.ERROR_SYMBOL}{first}", Tag.ERROR)
    if not expanded:
        return lines
    if item.reason:
        lines += plain(_REASON_LINE.format(reason=item.reason), Tag.MUTED)
    shown = json.dumps(item.input, ensure_ascii=False, indent=_JSON_INDENT)
    lines += plain(shown, Tag.CODE_BLOCK, bidi=False)
    if item.output:
        output_tags = (Tag.MONO, Tag.ERROR) if failed else (Tag.MONO,)
        lines += plain(item.output.rstrip("\n"), *output_tags)
    return lines


def _user_lines(item: UserItem) -> list[Line]:
    lines = plain(item.text, Tag.USER)
    lead = [Span(consts.PROMPT_SYMBOL, (Tag.PROMPT,))]
    if item.label:
        lead.append(Span(item.label, (Tag.MUTED,)))
    lines[0] = Line((*lead, *lines[0].spans), lines[0].tags)
    return lines


def _thinking_lines(item: ThinkingItem, *, expanded: bool) -> list[Line]:
    arrow = consts.TOOL_EXPANDED if expanded else consts.TOOL_COLLAPSED
    lines = plain(f"{arrow}{consts.THINKING_LABEL}", Tag.MUTED)
    if expanded:
        lines += plain(item.text, Tag.MUTED, Tag.ITALIC)
    return lines


def _notice_lines(item: NoticeItem) -> list[Line]:
    tag = _NOTICE_TAG[item.level]
    text = f"{consts.ERROR_SYMBOL}{item.text}" if item.level is NoticeLevel.ERROR else item.text
    if item.mono:
        return plain(text, Tag.MONO, tag, bidi=False)
    return plain(text, tag)


def item_lines(item: Item, *, expanded: bool = False) -> list[Line]:
    """The logical lines of one item, followed by a blank line."""
    if isinstance(item, UserItem):
        lines = _user_lines(item)
    elif isinstance(item, AssistantItem):
        lines = render(item.text)
    elif isinstance(item, ThinkingItem):
        lines = _thinking_lines(item, expanded=expanded)
    elif isinstance(item, ToolItem):
        lines = _tool_lines(item, expanded=expanded)
    else:
        lines = _notice_lines(item)
    return [*lines, Line(())]


def foldable(item: Item) -> bool:
    """Whether a click on the item opens and closes it."""
    return isinstance(item, ToolItem | ThinkingItem)


def item_text(item: Item) -> str:
    """The original text of an item, for the clipboard (never the reordered display text)."""
    if isinstance(item, ToolItem):
        shown = json.dumps(item.input, ensure_ascii=False)
        return f"{item.tool} {shown}\n{item.output}".rstrip()
    return item.text


def _styles(line: Line) -> list[Style]:
    return [span.tags for span in line.spans for _ in span.text]


def _wrap(text: str, styles: list[Style], layout: Layout | None) -> list[tuple[int, int]]:
    if layout is None:
        return [(0, len(text))]
    position = 0

    def measure(word: str) -> int:
        nonlocal position
        size = layout.measure(word, styles[position])
        position += len(word)
        return size

    return bidi.wrap(text, max(layout.width, consts.MIN_WRAP_WIDTH_PX), measure)


def line_rows(line: Line, layout: Layout | None = None) -> list[Row]:
    """Lay one logical line out as display rows (more than one when an RTL line wraps)."""
    text = line.text
    if not line.bidi or not bidi.has_rtl(text):
        pieces = tuple(Piece(span.text, False, span.tags) for span in line.spans)
        return [Row(pieces, (*line.tags, Tag.LTR))]
    rtl = bidi.is_rtl(text)
    styles = _styles(line)
    direction = Tag.RTL if rtl else Tag.LTR
    rows: list[Row] = []
    for start, end in _wrap(text, styles, layout):
        pieces = bidi.layout(text[start:end].rstrip(" ") or " ", styles[start:end], rtl=rtl)
        rows.append(Row(tuple(pieces), (*line.tags, direction)))
    return rows


def item_rows(item: Item, *, expanded: bool = False, layout: Layout | None = None) -> list[Row]:
    """Every display row of one item."""
    return [row for line in item_lines(item, expanded=expanded) for row in line_rows(line, layout)]
