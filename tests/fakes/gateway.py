"""FakeGateway: a scripted ``ModelGateway`` for agent / context tests."""

import asyncio
from collections.abc import AsyncIterator, Callable

from harness.core.consts import ProviderId, StreamEventType
from harness.core.types import (
    Capabilities,
    ModelPrices,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    StreamEvent,
)
from tests.consts import TEST_MODEL_ALIAS, TEST_MODEL_ID
from tests.fakes.adapter import events_for

FAKE_CONTEXT_WINDOW = 100_000
FAKE_MAX_OUTPUT_TOKENS = 4_000
FAKE_TOKENS_PER_MESSAGE = 10

type GatewayOutcome = ModelResponse | Exception


class FakeGateway:
    """Returns scripted responses in order and records every request."""

    def __init__(self, outcomes: list[GatewayOutcome] | None = None) -> None:
        self.outcomes: list[GatewayOutcome] = list(outcomes or [])
        self.requests: list[ModelRequest] = []
        self.hold: asyncio.Event | None = None
        self.streaming = asyncio.Event()
        self.counter: Callable[[ModelRequest], int] | None = None
        self.profile = ModelProfile(
            alias=TEST_MODEL_ALIAS,
            provider=ProviderId.ANTHROPIC,
            id=TEST_MODEL_ID,
            context_window=FAKE_CONTEXT_WINDOW,
            max_output_tokens=FAKE_MAX_OUTPUT_TOKENS,
            capabilities=Capabilities(tools=True, thinking=True, prompt_cache="explicit"),
            prices=ModelPrices(input=1.0, output=1.0),
        )

    def _next(self, req: ModelRequest) -> ModelResponse:
        self.requests.append(req)
        if not self.outcomes:
            message = "FakeGateway has no scripted outcome left"
            raise AssertionError(message)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome.model_copy(update={"model_used": self.profile.id})

    async def complete(self, req: ModelRequest) -> ModelResponse:
        """Return the next scripted response."""
        return self._next(req)

    async def stream(self, req: ModelRequest) -> AsyncIterator[StreamEvent]:
        """Stream the next scripted response; with ``hold`` set, pause before the end."""
        response = self._next(req)
        for event in events_for(response):
            if event.type is StreamEventType.USAGE and self.hold is not None:
                self.streaming.set()
                await self.hold.wait()
            if event.type is StreamEventType.END:
                yield event.model_copy(update={"response": response})
            else:
                yield event

    async def count_tokens(self, req: ModelRequest) -> int:
        """A fixed number of tokens per message, unless a test set its own ``counter``."""
        if self.counter is not None:
            return self.counter(req)
        return FAKE_TOKENS_PER_MESSAGE * len(req.messages)

    def list_models(self) -> list[ModelProfile]:
        """The single fake model."""
        return [self.profile]

    def resolve(self, alias_or_id: str) -> ModelProfile:
        """Every name resolves to the fake model."""
        return self.profile
