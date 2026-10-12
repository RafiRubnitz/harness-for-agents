"""The provider adapter protocol: implement it to add a provider."""

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Protocol

from pydantic import JsonValue

from harness.core.consts import ProviderId
from harness.core.errors import GatewayError
from harness.core.types import Capabilities, ModelRequest, ModelResponse, StreamEvent


@dataclass(frozen=True, slots=True)
class AdapterSettings:
    """Connection settings an adapter reads before every call (so config changes are hot)."""

    api_key: str
    base_url: str | None
    timeout_s: float


type SettingsProvider = Callable[[], AdapterSettings]


class ProviderAdapter(Protocol):
    """Translates the internal request / response format to one provider and back.

    Adapters receive requests whose ``model`` is already the provider's model id and
    whose ``max_tokens`` is set. They raise only ``GatewayError``.
    """

    id: ProviderId

    def capabilities(self, model_id: str) -> Capabilities:
        """Return what the model supports."""
        ...

    def to_provider(self, req: ModelRequest) -> dict[str, JsonValue]:
        """Build the provider request body."""
        ...

    def from_provider(self, raw: object) -> ModelResponse:
        """Convert a complete (non-streamed) provider response."""
        ...

    def stream(self, req: ModelRequest) -> AsyncIterator[StreamEvent]:
        """Stream one call; the last event is ``end`` and carries the whole response."""
        ...

    def classify_error(self, e: Exception) -> GatewayError:
        """Classify a provider / network exception."""
        ...

    async def count_tokens(self, req: ModelRequest) -> int | None:
        """Count input tokens with the provider API; ``None`` when it has none."""
        ...
