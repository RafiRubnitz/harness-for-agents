"""Pure ranking of search candidates: RRF fusion, recency boost, post-filters (03 §6.2)."""

from collections.abc import Sequence
from datetime import datetime

from pydantic import JsonValue

from harness.core.records import MemoryItem, ScoredItem
from harness.memory import consts


def _ranks(items: Sequence[MemoryItem]) -> dict[str, int]:
    return {item.id: rank for rank, item in enumerate(items, start=1)}


def recency(item: MemoryItem, now: datetime) -> float:
    """Return 1 for a fact observed now, halving every ``RECENCY_HALF_LIFE_DAYS``."""
    age_days = max((now - item.observed_at).total_seconds(), 0.0) / consts.SECONDS_PER_DAY
    return 0.5 ** (age_days / consts.RECENCY_HALF_LIFE_DAYS)


def fuse(
    by_vector: Sequence[MemoryItem],
    by_text: Sequence[MemoryItem],
    *,
    now: datetime,
    recency_weight: float,
) -> list[ScoredItem]:
    """Merge two ranked lists with Reciprocal Rank Fusion, best first.

    ``score = Σ 1 / (RRF_K + rank)`` over the lists an item appears in, then lifted by up
    to ``recency_weight`` for a fresh item. Ties keep the vector list's order.
    """
    vector_ranks, text_ranks = _ranks(by_vector), _ranks(by_text)
    items = {item.id: item for item in (*by_vector, *by_text)}
    scored: list[ScoredItem] = []
    for item_id, item in items.items():
        vector_rank, text_rank = vector_ranks.get(item_id), text_ranks.get(item_id)
        score = sum(1.0 / (consts.RRF_K + rank) for rank in (vector_rank, text_rank) if rank)
        score *= 1.0 + recency_weight * recency(item, now)
        scored.append(
            ScoredItem(item=item, score=score, vector_rank=vector_rank, bm25_rank=text_rank)
        )
    scored.sort(key=lambda hit: hit.score, reverse=True)
    return scored


def _as_set(value: JsonValue) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, list):
        return {entry for entry in value if isinstance(entry, str)}
    return set()


def passes(item: MemoryItem, wanted: dict[str, JsonValue] | None) -> bool:
    """Apply the post-fusion filters: ``kind`` (one of) and ``tags`` (all of)."""
    if not wanted:
        return True
    kinds = _as_set(wanted.get(consts.FILTER_KIND))
    if kinds and item.kind.value not in kinds:
        return False
    return _as_set(wanted.get(consts.FILTER_TAGS)) <= set(item.tags)
