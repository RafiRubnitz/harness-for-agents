"""Deterministic id generator: ``agt_0001``, ``run_0001``, ..."""

from collections import defaultdict

FAKE_ID_WIDTH = 4


class FakeIds:
    """Counts up separately per prefix."""

    def __init__(self) -> None:
        self._counters: defaultdict[str, int] = defaultdict(int)

    def new(self, prefix: str) -> str:
        """Return the next id for ``prefix``."""
        self._counters[prefix] += 1
        return f"{prefix}{self._counters[prefix]:0{FAKE_ID_WIDTH}d}"
