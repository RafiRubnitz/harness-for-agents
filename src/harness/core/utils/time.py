"""The only place that reads the real clock."""

import asyncio
from datetime import UTC, datetime

from harness.core.consts import MSG_NAIVE_DATETIME

_MS_PER_SECOND = 1000


class SystemClock:
    """Real wall clock and real sleep."""

    def now(self) -> datetime:
        """Return the current timezone-aware UTC time."""
        return datetime.now(UTC)

    async def sleep(self, seconds: float) -> None:
        """Wait for ``seconds`` without blocking the loop."""
        await asyncio.sleep(seconds)


def ensure_utc(value: datetime) -> datetime:
    """Return ``value`` converted to UTC; raise ``ValueError`` for a naive datetime."""
    if value.tzinfo is None:
        raise ValueError(MSG_NAIVE_DATETIME)
    return value.astimezone(UTC)


def elapsed_ms(start: datetime, end: datetime) -> int:
    """Return the whole milliseconds between two datetimes."""
    return int((end - start).total_seconds() * _MS_PER_SECOND)
