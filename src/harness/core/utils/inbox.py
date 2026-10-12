"""Inbox order: higher priority first, arrival order inside a priority (phase2-03 §7a)."""

from collections.abc import Sequence
from datetime import datetime

from harness.core.consts import USER_PRIORITY
from harness.core.records import InboxItem

AGING_STEP = 10
"""How much a waiting message gains for every full ``aging_s`` it has waited."""


def effective_priority(item: InboxItem, now: datetime, aging_s: float | None) -> int:
    """The priority of an item right now.

    A message that waited longer than ``aging_s`` climbs one step per period, so it is
    never starved; it still stays below the user.
    """
    if item.priority >= USER_PRIORITY or not aging_s or aging_s <= 0:
        return item.priority
    waited = (now - item.received_at).total_seconds()
    gained = AGING_STEP * int(max(waited, 0) // aging_s)
    return min(item.priority + gained, USER_PRIORITY - 1)


def in_order(items: Sequence[InboxItem], now: datetime, aging_s: float | None) -> list[InboxItem]:
    """Sort inbox items the way they are handled."""
    return sorted(items, key=lambda i: (-effective_priority(i, now, aging_s), i.received_at, i.id))
