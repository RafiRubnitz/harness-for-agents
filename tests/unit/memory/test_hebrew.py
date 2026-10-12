"""Hebrew normalisation for lexical search (03 §6.3)."""

import unicodedata

import pytest

from harness.memory import hebrew
from harness.memory.consts import HEBREW_FINALS, HEBREW_MIN_STEM_LETTERS, HEBREW_PREFIX_LETTERS
from tests.consts import HEBREW_FINALS as FINALS_SAMPLE
from tests.consts import HEBREW_MIXED, HEBREW_NIQQUD, HEBREW_SAMPLES

STEM = "קובץ"
PREFIXED = "והקובץ"
SHORT_WORD = "של"


@pytest.mark.parametrize("sample", HEBREW_SAMPLES)
def test_normalize_is_idempotent(sample: str) -> None:
    once = hebrew.normalize(sample)

    assert hebrew.normalize(once) == once


@pytest.mark.parametrize("sample", HEBREW_SAMPLES)
def test_normalize_keeps_every_letter(sample: str) -> None:
    letters = [c for c in sample if c.isalpha()]

    normalized = hebrew.normalize(sample)

    assert len([c for c in normalized if c.isalpha()]) == len(letters)


def test_normalize_strips_niqqud() -> None:
    normalized = hebrew.normalize(HEBREW_NIQQUD)

    assert not any(unicodedata.category(c) == "Mn" for c in normalized)


def test_normalize_folds_final_letters() -> None:
    normalized = hebrew.normalize(FINALS_SAMPLE)

    assert not set(normalized) & set(HEBREW_FINALS)


def test_normalize_lowercases_latin() -> None:
    assert hebrew.normalize(HEBREW_MIXED) == hebrew.normalize(HEBREW_MIXED).lower()
    assert "modelrequest" in hebrew.normalize(HEBREW_MIXED)


@pytest.mark.acceptance("03-10.8")
def test_prefixed_word_is_indexed_with_its_stem() -> None:
    indexed = hebrew.index_text(PREFIXED).split()

    assert hebrew.normalize(STEM) in indexed
    assert hebrew.normalize(PREFIXED) in indexed


def test_prefix_is_not_stripped_below_minimum_stem() -> None:
    word = HEBREW_PREFIX_LETTERS[0] + "אב"

    assert len(word) - 1 < HEBREW_MIN_STEM_LETTERS
    assert hebrew.strip_prefixes(word) == []
    assert hebrew.strip_prefixes(SHORT_WORD) == []


def test_fts_query_quotes_every_token() -> None:
    query = hebrew.fts_query('build_context() "AND" NOT *')

    assert query
    assert all(part.startswith('"') and part.endswith('"') for part in query.split(" OR "))


def test_fts_query_of_punctuation_only_is_empty() -> None:
    assert hebrew.fts_query('"*()') == ""
