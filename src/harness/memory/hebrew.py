"""Hebrew-aware text normalisation for lexical search (03 §6.3).

``normalize`` is idempotent and never drops a letter. ``index_text`` adds, next to every
Hebrew word, the same word without its attached prefix letters, so that a search for
"קובץ" finds "והקובץ".
"""

import re

from harness.memory.consts import (
    FTS_OR,
    HEBREW_FINALS,
    HEBREW_LETTER_FIRST,
    HEBREW_LETTER_LAST,
    HEBREW_MARKS_FIRST,
    HEBREW_MARKS_LAST,
    HEBREW_MAX_PREFIX_LETTERS,
    HEBREW_MIN_STEM_LETTERS,
    HEBREW_PREFIX_LETTERS,
    HEBREW_PUNCTUATION_KEPT,
)

_FINALS = str.maketrans(HEBREW_FINALS)
_MARKS = {
    code: None
    for code in range(HEBREW_MARKS_FIRST, HEBREW_MARKS_LAST + 1)
    if code not in HEBREW_PUNCTUATION_KEPT
}
_TOKEN_RE = re.compile(r"\w+")
_HEBREW_WORD_RE = re.compile(rf"[{HEBREW_LETTER_FIRST}-{HEBREW_LETTER_LAST}]+")


def normalize(text: str) -> str:
    """Strip niqqud and cantillation, fold final letters, and lowercase."""
    return text.translate(_MARKS).translate(_FINALS).lower()


def strip_prefixes(word: str) -> list[str]:
    """Return ``word`` without 1..n leading prefix letters, while 3+ letters remain."""
    variants: list[str] = []
    rest = word
    for _ in range(HEBREW_MAX_PREFIX_LETTERS):
        if rest[:1] not in HEBREW_PREFIX_LETTERS or len(rest) - 1 < HEBREW_MIN_STEM_LETTERS:
            break
        rest = rest[1:]
        variants.append(rest)
    return variants


def _variants(normalized: str) -> list[str]:
    seen: dict[str, None] = {}
    for word in _HEBREW_WORD_RE.findall(normalized):
        for variant in strip_prefixes(word):
            seen.setdefault(variant)
    return list(seen)


def index_text(text: str) -> str:
    """Return the text to index: normalised, followed by the prefix-stripped variants."""
    normalized = normalize(text)
    variants = _variants(normalized)
    return f"{normalized}\n{' '.join(variants)}" if variants else normalized


def tokens(text: str) -> list[str]:
    """Return the normalised search tokens of ``text``, with prefix-stripped variants."""
    normalized = normalize(text)
    found: dict[str, None] = dict.fromkeys(_TOKEN_RE.findall(normalized))
    for variant in _variants(normalized):
        found.setdefault(variant)
    return list(found)


def fts_query(text: str) -> str:
    """Build an FTS5 ``MATCH`` expression: every token quoted, joined with ``OR``."""
    return FTS_OR.join(f'"{token}"' for token in tokens(text))
