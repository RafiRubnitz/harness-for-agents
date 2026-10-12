"""A small HTML to Markdown conversion for the plain-fetch fallback of ``web_scrape``."""

import re
from html.parser import HTMLParser

from harness.tools import consts

_BLANK_LINES = re.compile(r"\n{3,}")
_SPACES = re.compile(r"[ \t\r\f\v]+")


class _Converter(HTMLParser):
    """Keeps headings, paragraphs, lists, links, emphasis and code; drops the rest."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skipped = 0
        self._preformatted = 0
        self._links: list[str | None] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Open a tag."""
        if tag in consts.MD_SKIPPED_TAGS:
            self._skipped += 1
        elif tag in consts.MD_HEADINGS:
            self.parts.append("\n\n" + consts.MD_HEADINGS[tag] + " ")
        elif tag in consts.MD_BLOCK_TAGS:
            self.parts.append("\n\n")
        elif tag == "br":
            self.parts.append("\n")
        elif tag == "li":
            self.parts.append("\n" + consts.MD_BULLET)
        elif tag == "pre":
            self._preformatted += 1
            self.parts.append("\n\n" + consts.MD_FENCE + "\n")
        elif tag in consts.MD_INLINE:
            self.parts.append(consts.MD_INLINE[tag])
        elif tag == "a":
            self._links.append(dict(attrs).get("href"))
            self.parts.append("[")

    def handle_endtag(self, tag: str) -> None:
        """Close a tag."""
        if tag in consts.MD_SKIPPED_TAGS:
            self._skipped = max(self._skipped - 1, 0)
        elif tag in consts.MD_HEADINGS or tag in consts.MD_BLOCK_TAGS:
            self.parts.append("\n\n")
        elif tag == "pre":
            self._preformatted = max(self._preformatted - 1, 0)
            self.parts.append("\n" + consts.MD_FENCE + "\n\n")
        elif tag in consts.MD_INLINE:
            self.parts.append(consts.MD_INLINE[tag])
        elif tag == "a" and self._links:
            href = self._links.pop()
            self.parts.append(f"]({href})" if href else "]")

    def handle_data(self, data: str) -> None:
        """Keep visible text; white space collapses outside ``<pre>``."""
        if self._skipped:
            return
        self.parts.append(data if self._preformatted else _SPACES.sub(" ", data.replace("\n", " ")))


def html_to_markdown(html: str) -> str:
    """Convert an HTML page to readable Markdown (scripts, styles and markup removed)."""
    converter = _Converter()
    converter.feed(html)
    converter.close()
    lines: list[str] = []
    fenced = False
    for raw in "".join(converter.parts).splitlines():
        line = raw.rstrip()
        if line.strip() == consts.MD_FENCE:
            fenced = not fenced
            line = consts.MD_FENCE
        lines.append(line if fenced else line.lstrip())
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()
