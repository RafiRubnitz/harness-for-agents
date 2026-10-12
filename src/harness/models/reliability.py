"""Circuit breaker for a failing provider."""

from datetime import datetime

from harness.models.consts import BREAKER_COOLDOWN_S, BREAKER_FAILURE_THRESHOLD


class CircuitBreaker:
    """Opens after consecutive retryable failures and lets one call through after a cooldown."""

    def __init__(
        self,
        threshold: int = BREAKER_FAILURE_THRESHOLD,
        cooldown_s: float = BREAKER_COOLDOWN_S,
    ) -> None:
        self._threshold = threshold
        self._cooldown_s = cooldown_s
        self._failures = 0
        self._opened_at: datetime | None = None

    def is_open(self, now: datetime) -> bool:
        """Return whether calls to the provider are currently blocked."""
        if self._opened_at is None:
            return False
        return (now - self._opened_at).total_seconds() < self._cooldown_s

    def record_failure(self, now: datetime) -> None:
        """Count a retryable failure; open (or re-open) at the threshold."""
        self._failures += 1
        if self._failures >= self._threshold:
            self._opened_at = now

    def record_success(self) -> None:
        """Close the breaker."""
        self._failures = 0
        self._opened_at = None
