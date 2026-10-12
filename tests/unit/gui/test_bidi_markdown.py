"""Bidi layout and markdown rendering: the pure text pipeline behind the stream."""

import pytest

from harness.gui import bidi
from harness.gui.consts import Tag
from harness.gui.markdown import Line, Span, plain, render
from harness.gui.render import Layout, line_rows
from tests.consts import HEBREW_MIXED, HEBREW_SAMPLES

MIXED_LINE = "תריץ את main.py עם Python 3.12"
LATIN_RUNS = ["Python 3.12", "main.py"]
HEBREW_RUNS = ["עם", "את", "תריץ"]
ENGLISH = "plain english line"
CHAR_WIDTH = 10


def _measure(text: str, _style: object = None) -> int:
    return CHAR_WIDTH * len(text)


def _texts(pieces: list[bidi.Piece[str]], *, rtl: bool) -> list[str]:
    return [piece.text.strip() for piece in pieces if piece.rtl is rtl and piece.text.strip()]


def test_direction_follows_the_first_strong_character() -> None:
    assert bidi.is_rtl(MIXED_LINE)
    assert not bidi.is_rtl(f"{ENGLISH} שלום")
    assert bidi.has_rtl(f"{ENGLISH} שלום")
    assert not bidi.has_rtl(ENGLISH)


@pytest.mark.acceptance("02-10.7")
def test_mixed_line_is_laid_out_in_display_order_and_aligned_right() -> None:
    styles = [""] * len(MIXED_LINE)

    pieces = bidi.layout(MIXED_LINE, styles, rtl=True)
    rows = line_rows(plain(MIXED_LINE)[0])

    # left to right on screen: the Latin runs come out reversed, each one intact
    assert _texts(pieces, rtl=False) == LATIN_RUNS
    # Hebrew words run right to left; inside a word the letters stay in typing order,
    # because the system reverses each chunk when drawing it
    assert _texts(pieces, rtl=True) == HEBREW_RUNS
    assert sorted("".join(piece.text for piece in pieces)) == sorted(MIXED_LINE)
    assert [Tag.RTL in row.tags for row in rows] == [True]


@pytest.mark.parametrize("sample", HEBREW_SAMPLES)
def test_layout_keeps_every_character(sample: str) -> None:
    pieces = bidi.layout(sample, [""] * len(sample), rtl=bidi.is_rtl(sample))

    assert sorted("".join(piece.text for piece in pieces)) == sorted(sample)


def test_latin_line_is_one_untouched_piece() -> None:
    rows = line_rows(plain(ENGLISH)[0])

    assert [piece.text for piece in rows[0].pieces] == [ENGLISH]
    assert Tag.LTR in rows[0].tags
    assert not any(piece.rtl for piece in rows[0].pieces)


def test_a_piece_never_mixes_styles() -> None:
    half = len(HEBREW_MIXED) // 2
    styles = ["a"] * half + ["b"] * (len(HEBREW_MIXED) - half)

    pieces = bidi.layout(HEBREW_MIXED, styles, rtl=True)

    assert {piece.style for piece in pieces} == {"a", "b"}
    assert sum(len(piece.text) for piece in pieces if piece.style == "a") == half


def test_wrap_breaks_between_words_and_covers_the_line() -> None:
    words = MIXED_LINE.split(" ")
    width = _measure(max(words, key=len)) + CHAR_WIDTH

    ranges = bidi.wrap(MIXED_LINE, width, _measure)

    parts = [MIXED_LINE[start:end] for start, end in ranges]
    assert len(parts) > 1
    assert "".join(parts).split() == words
    assert all(_measure(part.rstrip()) <= width for part in parts)


def test_rtl_line_wraps_into_several_right_aligned_rows() -> None:
    line = plain(" ".join([MIXED_LINE] * 3))[0]
    layout = Layout(_measure(MIXED_LINE), _measure)

    rows = line_rows(line, layout)

    assert len(rows) > 1
    assert all(Tag.RTL in row.tags for row in rows)
    shown = "".join(piece.text for row in rows for piece in row.pieces)
    assert sorted(shown.replace(" ", "")) == sorted(line.text.replace(" ", ""))


def test_lines_marked_not_bidi_are_left_alone() -> None:
    line = Line((Span(MIXED_LINE, (Tag.CODE_BLOCK,)),), (Tag.CODE_BLOCK,), bidi=False)

    rows = line_rows(line)

    assert [piece.text for piece in rows[0].pieces] == [MIXED_LINE]


def _tags_of(lines: list[Line], text: str) -> set[Tag]:
    found = [span for line in lines for span in line.spans if span.text == text]
    assert found, f"no span {text!r}"
    return set(found[0].tags)


def test_markdown_inline_styles() -> None:
    lines = render("some **bold**, *italic*, ~~gone~~ and `code`")

    assert Tag.BOLD in _tags_of(lines, "bold")
    assert Tag.ITALIC in _tags_of(lines, "italic")
    assert Tag.STRIKE in _tags_of(lines, "gone")
    assert Tag.CODE in _tags_of(lines, "code")


def test_markdown_blocks() -> None:
    source = "# Title\n\n- first\n- second\n\n```python\nx = 1\n```\n\n> quoted"

    lines = render(source)

    texts = [line.text for line in lines]
    assert Tag.HEADING_1 in _tags_of(lines, "Title")
    assert sum("first" in text or "second" in text for text in texts) == len(["first", "second"])
    code = next(line for line in lines if line.text == "x = 1")
    assert Tag.CODE_BLOCK in code.tags
    assert not code.bidi
    assert any(Tag.QUOTE in line.tags and "quoted" in line.text for line in lines)


def test_markdown_table_keeps_every_cell() -> None:
    cells = ["name", "value", "alpha", "1"]
    source = f"| {cells[0]} | {cells[1]} |\n|---|---|\n| {cells[2]} | {cells[3]} |"

    text = "\n".join(line.text for line in render(source))

    assert all(cell in text for cell in cells)


def test_plain_text_is_one_line_per_source_line() -> None:
    lines = plain("one\ntwo", Tag.MUTED)

    assert [line.text for line in lines] == ["one", "two"]
    assert all(Tag.MUTED in span.tags for line in lines for span in line.spans)
