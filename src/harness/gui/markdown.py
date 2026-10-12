"""Markdown to styled lines for the conversation stream (14 §3)."""

from dataclasses import dataclass, field
from typing import Final

from markdown_it import MarkdownIt
from markdown_it.token import Token

from harness.gui.consts import (
    BULLET,
    HEADING_TAGS,
    LIST_INDENT,
    QUOTE_PREFIX,
    RULE_CHAR,
    RULE_WIDTH,
    TABLE_SEPARATOR,
    Tag,
)

_PARSER: Final = MarkdownIt("commonmark").enable(["table", "strikethrough"])
_INLINE_TAGS: Final = {"strong": Tag.BOLD, "em": Tag.ITALIC, "s": Tag.STRIKE, "link": Tag.LINK}
_SPACED_BLOCKS: Final = frozenset(
    {"paragraph", "heading", "table", "blockquote", "bullet_list", "ordered_list"}
)
_OPEN: Final = "_open"
_CLOSE: Final = "_close"
_ORDERED_ITEM: Final = "{number}. "
_LINK_TARGET: Final = " ({href})"
_ATTR_HREF: Final = "href"
_ATTR_START: Final = "start"


@dataclass(frozen=True, slots=True)
class Span:
    """A run of text with its inline styles."""

    text: str
    tags: tuple[Tag, ...] = ()


@dataclass(frozen=True, slots=True)
class Line:
    """One logical line; ``bidi`` is off for code and tables, which stay left-to-right."""

    spans: tuple[Span, ...]
    tags: tuple[Tag, ...] = ()
    bidi: bool = True

    @property
    def text(self) -> str:
        """The plain text of the line."""
        return "".join(span.text for span in self.spans)


def plain(text: str, *tags: Tag, bidi: bool = True) -> list[Line]:
    """Unformatted text as lines that all carry the same tags."""
    return [Line((Span(part, tags),), tags, bidi) for part in text.split("\n")]


@dataclass
class _Lists:
    """Open lists: the next number of each (``None`` for a bullet list)."""

    counters: list[int | None] = field(default_factory=list[int | None])
    pending_marker: str | None = None

    def marker(self) -> str:
        number = self.counters[-1]
        if number is None:
            return BULLET
        self.counters[-1] = number + 1
        return _ORDERED_ITEM.format(number=number)


class _Builder:
    """Walks the token stream and collects lines."""

    def __init__(self) -> None:
        self.lines: list[Line] = []
        self._lists = _Lists()
        self._quotes = 0
        self._block: Tag | None = None
        self._table: list[list[str]] = []
        self._in_table = False

    # --- inline ---

    def _inline(self, token: Token) -> list[list[Span]]:
        """The spans of an inline token, split at line breaks."""
        rows: list[list[Span]] = [[]]
        active: list[Tag] = []
        href: str | None = None
        for child in token.children or []:
            kind = child.type
            if kind in ("softbreak", "hardbreak"):
                rows.append([])
            elif kind == "code_inline":
                rows[-1].append(Span(child.content, (*active, Tag.CODE)))
            elif kind.endswith(_OPEN):
                active.append(_INLINE_TAGS[kind.removesuffix(_OPEN)])
                href = str(child.attrGet(_ATTR_HREF) or "") or href
            elif kind.endswith(_CLOSE):
                closed = active.pop()
                if closed is Tag.LINK and href and href != _text_of(rows[-1][-1:]):
                    rows[-1].append(Span(_LINK_TARGET.format(href=href), (*active, Tag.MUTED)))
                    href = None
            else:
                rows[-1].append(Span(child.content, tuple(active)))
        return rows

    def _prefix(self) -> list[Span]:
        spans: list[Span] = []
        if self._quotes:
            spans.append(Span(QUOTE_PREFIX * self._quotes, (Tag.MUTED,)))
        depth = len(self._lists.counters)
        if depth:
            marker = self._lists.pending_marker
            self._lists.pending_marker = None
            indent = LIST_INDENT * (depth - 1)
            spans.append(Span(indent + (marker or " " * len(BULLET)), (Tag.MUTED,)))
        return spans

    def _emit_inline(self, token: Token) -> None:
        block = (self._block,) if self._block is not None else ()
        tags = (*block, Tag.QUOTE) if self._quotes else block
        for row in self._inline(token):
            styled = [Span(span.text, (*block, *span.tags)) for span in row]
            self.lines.append(Line((*self._prefix(), *styled), tags))

    # --- blocks ---

    def _code(self, token: Token) -> None:
        for row in token.content.rstrip("\n").split("\n"):
            self.lines.append(Line((Span(row, (Tag.CODE_BLOCK,)),), (Tag.CODE_BLOCK,), bidi=False))

    def _end_table(self) -> None:
        columns = max(len(row) for row in self._table)
        widths = [
            max(len(row[index]) for row in self._table if index < len(row))
            for index in range(columns)
        ]
        for number, row in enumerate(self._table):
            cells = [cell.ljust(widths[index]) for index, cell in enumerate(row)]
            tags = (Tag.MONO, Tag.BOLD) if number == 0 else (Tag.MONO,)
            self.lines.append(Line((Span(TABLE_SEPARATOR.join(cells), tags),), (Tag.MONO,), False))
        self._table, self._in_table = [], False

    def _open(self, token: Token) -> None:
        kind = token.type.removesuffix(_OPEN)
        if kind == "heading":
            level = min(int(token.tag[1:]), len(HEADING_TAGS))
            self._block = HEADING_TAGS[level - 1]
        elif kind == "blockquote":
            self._quotes += 1
        elif kind == "bullet_list":
            self._lists.counters.append(None)
        elif kind == "ordered_list":
            self._lists.counters.append(int(token.attrGet(_ATTR_START) or 1))
        elif kind == "list_item":
            self._lists.pending_marker = self._lists.marker()
        elif kind == "table":
            self._in_table = True
        elif kind == "tr":
            self._table.append([])

    def _close(self, token: Token) -> None:
        kind = token.type.removesuffix(_CLOSE)
        if kind == "heading":
            self._block = None
        elif kind == "blockquote":
            self._quotes -= 1
        elif kind in ("bullet_list", "ordered_list"):
            self._lists.counters.pop()
        elif kind == "table":
            self._end_table()
        if kind in _SPACED_BLOCKS and not self._nested():
            self.lines.append(Line(()))

    def _nested(self) -> bool:
        return bool(self._lists.counters or self._quotes or self._in_table)

    def feed(self, token: Token) -> None:
        """Handle one block-level token."""
        kind = token.type
        if kind == "inline":
            if self._in_table:
                self._table[-1].append(_text_of([s for row in self._inline(token) for s in row]))
            else:
                self._emit_inline(token)
        elif kind in ("fence", "code_block"):
            self._code(token)
            self.lines.append(Line(()))
        elif kind == "hr":
            self.lines.append(Line((Span(RULE_CHAR * RULE_WIDTH, (Tag.MUTED,)),), bidi=False))
            self.lines.append(Line(()))
        elif kind == "html_block":
            self.lines.extend(plain(token.content.rstrip("\n")))
        elif kind.endswith(_OPEN):
            self._open(token)
        elif kind.endswith(_CLOSE):
            self._close(token)


def _text_of(spans: list[Span]) -> str:
    return "".join(span.text for span in spans)


def render(text: str) -> list[Line]:
    """Parse Markdown into lines; trailing blank lines are dropped."""
    builder = _Builder()
    for token in _PARSER.parse(text):
        builder.feed(token)
    lines = builder.lines
    while lines and not lines[-1].spans:
        lines.pop()
    return lines
