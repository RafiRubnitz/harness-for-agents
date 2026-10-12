"""Hebrew / RTL layout for a Tk text widget (02 §7).

Tk on Windows draws every run of one font with its own call, and the system reorders
the letters inside that run only. So a mixed line comes out with its words in the wrong
order. The layout here computes the visual order of the whole line with the Unicode
bidi algorithm, then hands each RTL run back in logical order as a separate piece: the
widget inserts every such piece as its own chunk, and the system puts its letters right.
"""

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Protocol, cast

from bidi import algorithm as _algorithm  # pyright: ignore[reportMissingTypeStubs]

_RTL_CLASSES: Final = frozenset({"R", "AL"})
_STRONG_CLASSES: Final = frozenset({"L", "R", "AL"})
_MARK_CLASS: Final = "NSM"
_DIR_RTL: Final = "R"
_DIR_LTR: Final = "L"
_KEY_CHARS: Final = "chars"
_KEY_CHAR: Final = "ch"
_KEY_INDEX: Final = "index"
_KEY_BASE_LEVEL: Final = "base_level"
_KEY_BASE_DIR: Final = "base_dir"
_WORD: Final = re.compile(r"\S+\s*|\s+")

type _Storage = dict[str, object]


class _Algorithm(Protocol):
    """The steps of the pure-Python bidi implementation used to keep source indexes."""

    PARAGRAPH_LEVELS: dict[str, int]

    def get_empty_storage(self) -> _Storage: ...
    def get_embedding_levels(self, text: str, storage: _Storage) -> None: ...
    def explicit_embed_and_overrides(self, storage: _Storage) -> None: ...
    def resolve_weak_types(self, storage: _Storage) -> None: ...
    def resolve_neutral_types(self, storage: _Storage, *, debug: bool) -> None: ...
    def resolve_implicit_levels(self, storage: _Storage, *, debug: bool) -> None: ...
    def reorder_resolved_levels(self, storage: _Storage, *, debug: bool) -> None: ...
    def apply_mirroring(self, storage: _Storage, *, debug: bool) -> None: ...


_bidi: Final = cast("_Algorithm", _algorithm)


@dataclass(frozen=True, slots=True)
class Glyph:
    """One displayed character and the index of its source character."""

    char: str
    index: int
    rtl: bool


@dataclass(frozen=True, slots=True)
class Piece[T]:
    """A run to insert as one chunk; an RTL run holds its text in logical order."""

    text: str
    rtl: bool
    style: T


def is_rtl_char(char: str) -> bool:
    """Whether a character is a strong right-to-left letter."""
    return unicodedata.bidirectional(char) in _RTL_CLASSES


def has_rtl(text: str) -> bool:
    """Whether the text holds any right-to-left letter."""
    return any(is_rtl_char(char) for char in text)


def is_rtl(text: str) -> bool:
    """Whether the first strong character of the text is right-to-left."""
    for char in text:
        kind = unicodedata.bidirectional(char)
        if kind in _STRONG_CLASSES:
            return kind in _RTL_CLASSES
    return False


def visual_order(line: str, *, rtl: bool) -> list[Glyph]:
    """The characters of one line in display order, left to right."""
    storage = _bidi.get_empty_storage()
    direction = _DIR_RTL if rtl else _DIR_LTR
    storage[_KEY_BASE_LEVEL] = _bidi.PARAGRAPH_LEVELS[direction]
    storage[_KEY_BASE_DIR] = direction
    _bidi.get_embedding_levels(line, storage)
    chars = cast("list[dict[str, object]]", storage[_KEY_CHARS])
    for index, entry in enumerate(chars):
        entry[_KEY_INDEX] = index
    marks = [unicodedata.bidirectional(char) == _MARK_CLASS for char in line]
    _bidi.explicit_embed_and_overrides(storage)
    _bidi.resolve_weak_types(storage)
    _bidi.resolve_neutral_types(storage, debug=False)
    _bidi.resolve_implicit_levels(storage, debug=False)
    _bidi.reorder_resolved_levels(storage, debug=False)
    _bidi.apply_mirroring(storage, debug=False)
    ordered = cast("list[dict[str, object]]", storage[_KEY_CHARS])
    glyphs: list[Glyph] = []
    for entry in ordered:
        char, index = cast("str", entry[_KEY_CHAR]), cast("int", entry[_KEY_INDEX])
        glyphs.append(Glyph(char, index, is_rtl_char(line[index]) or marks[index]))
    return glyphs


def layout[T](line: str, styles: list[T], *, rtl: bool) -> list[Piece[T]]:
    """Split one line into pieces in display order.

    ``styles`` gives the style of every source character. A piece never mixes styles or
    directions; the text of an RTL piece is in logical order, ready for the system to
    reorder when the piece is drawn as a chunk of its own.
    """
    if not has_rtl(line):
        return _plain(line, styles)
    pieces: list[Piece[T]] = []
    run: list[Glyph] = []

    def flush() -> None:
        if not run:
            return
        ordered = sorted(run, key=lambda glyph: glyph.index) if run[0].rtl else run
        text = "".join(glyph.char for glyph in ordered)
        pieces.append(Piece(text, run[0].rtl, styles[run[0].index]))
        run.clear()

    for glyph in visual_order(line, rtl=rtl):
        if run and (glyph.rtl != run[0].rtl or styles[glyph.index] != styles[run[0].index]):
            flush()
        run.append(glyph)
    flush()
    return pieces


def _plain[T](line: str, styles: list[T]) -> list[Piece[T]]:
    pieces: list[Piece[T]] = []
    start = 0
    for index in range(1, len(line) + 1):
        if index == len(line) or styles[index] != styles[start]:
            pieces.append(Piece(line[start:index], False, styles[start]))
            start = index
    return pieces


def wrap(line: str, width: int, measure: Callable[[str], int]) -> list[tuple[int, int]]:
    """Break a line between words so every part fits ``width``; returns ``(start, end)`` ranges.

    Needed for RTL lines only: the widget wraps in display order, which would put the end
    of a right-to-left sentence on the first row. A single word wider than ``width`` keeps
    a row to itself.
    """
    ranges: list[tuple[int, int]] = []
    start = position = used = 0
    for word in _WORD.findall(line):
        size = measure(word)
        if used and used + size > width:
            ranges.append((start, position))
            start, used = position, 0
        used += size
        position += len(word)
    ranges.append((start, len(line)))
    return ranges
