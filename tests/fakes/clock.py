"""FakeClock: a controllable implementation of the core Clock protocol."""

import asyncio
from datetime import UTC, datetime, timedelta

FAKE_CLOCK_START = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


class FakeClock:
    """Time moves only when a test calls ``advance`` or the code under test sleeps."""

    def __init__(self, start: datetime = FAKE_CLOCK_START) -> None:
        self._now = start
        self.sleeps: list[float] = []

    def now(self) -> datetime:
        """Return the current fake time."""
        return self._now

    async def sleep(self, seconds: float) -> None:
        """Record the sleep, advance fake time, and yield to the loop once."""
        self.sleeps.append(seconds)
        self.advance(seconds)
        await asyncio.sleep(0)

    def advance(self, seconds: float) -> None:
        """Move fake time forward."""
        self._now += timedelta(seconds=seconds)
