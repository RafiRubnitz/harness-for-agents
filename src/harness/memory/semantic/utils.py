"""Small pure helpers of the semantic layer."""

import re

from harness.memory import hebrew

_EDGE_PUNCTUATION_RE = re.compile(r"^\W+|\W+$")


def lexical_terms(text: str) -> list[str]:
    """Return the terms a text is indexed and searched by (03 §6.3).

    Every whitespace-separated word without its surrounding punctuation (so that
    ``config_v2.yaml`` stays one exact term), its word parts, and each Hebrew word
    without its attached prefix letters. The same function builds ``content_norm`` and
    the query, which is why the index uses a plain whitespace tokenizer.
    """
    found: dict[str, None] = {}
    for word in hebrew.normalize(text).split():
        core = _EDGE_PUNCTUATION_RE.sub("", word)
        if core:
            found.setdefault(core)
    for token in hebrew.tokens(text):
        found.setdefault(token)
    return list(found)


def lexical_text(text: str) -> str:
    """Return ``content_norm``: the terms of ``text`` joined by spaces."""
    return " ".join(lexical_terms(text))


def similarity(distance: float) -> float:
    """Convert a cosine distance to a cosine similarity."""
    return 1.0 - distance
