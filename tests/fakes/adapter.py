"""FakeAdapter: a scripted ``ProviderAdapter`` that never touches the network."""

import asyncio
import json
from collections.abc import AsyncIterator

from pydantic import JsonValue

from harness.core.consts import ErrorKind, ProviderId, StopReason, StreamEventType
from harness.core.errors import GatewayError
from harness.core.types import (
    Capabilities,
    ModelRequest,
    ModelResponse,
    StreamEvent,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    Usage,
)

type Outcome = ModelResponse | Exception | list[StreamEvent]


def text_response(text: str, usage: Usage | None = None) -> ModelResponse:
    """A plain text response."""
    return ModelResponse(
        content=[TextBlock(text=text)],
        stop_reason=StopReason.END,
        usage=usage or Usage(input=len(text), output=len(text)),
        model_used="",
    )


def tool_response(call: ToolCallBlock, usage: Usage | None = None) -> ModelResponse:
    """A response that asks for one tool call."""
    return ModelResponse(
        content=[call],
        stop_reason=StopReason.TOOL_CALL,
        usage=usage or Usage(input=1, output=1),
        model_used="",
    )


def events_for(response: ModelResponse) -> list[StreamEvent]:
    """The stream a provider would produce for a complete response."""
    events = [StreamEvent(type=StreamEventType.START)]
    for block in response.content:
        if isinstance(block, TextBlock):
            events.append(StreamEvent(type=StreamEventType.TEXT_DELTA, text=block.text))
        elif isinstance(block, ThinkingBlock):
            events.append(StreamEvent(type=StreamEventType.THINKING_DELTA, text=block.text))
        elif isinstance(block, ToolCallBlock):
            events.extend(
                [
                    StreamEvent(
                        type=StreamEventType.TOOL_CALL_START,
                        call_id=block.id,
                        tool_name=block.name,
                    ),
                    StreamEvent(
                        type=StreamEventType.TOOL_CALL_DELTA,
                        call_id=block.id,
                        input_json_delta=json.dumps(block.input),
                    ),
                    StreamEvent(
                        type=StreamEventType.TOOL_CALL_END, call_id=block.id, tool_call=block
                    ),
                ]
            )
    events.append(StreamEvent(type=StreamEventType.USAGE, usage=response.usage))
    events.append(StreamEvent(type=StreamEventType.END, response=response))
    return events


def gateway_error(kind: ErrorKind, retry_after_s: float | None = None) -> GatewayError:
    """A classified provider failure."""
    return GatewayError(
        kind.value,
        kind=kind,
        provider=ProviderId.ANTHROPIC,
        model="",
        retry_after_s=retry_after_s,
    )


class FakeAdapter:
    """Replays scripted outcomes, one per call, and records every request it received.

    An outcome is a response (streamed normally), an exception (raised before any event),
    or an explicit event list (an exception inside the list is not supported; use
    ``fail_after`` for a failure in the middle of a stream).
    """

    def __init__(
        self,
        outcomes: list[Outcome] | None = None,
        provider: ProviderId = ProviderId.ANTHROPIC,
        token_count: int | None = None,
    ) -> None:
        self.id = provider
        self.outcomes: list[Outcome] = list(outcomes or [])
        self.requests: list[ModelRequest] = []
        self.token_count = token_count
        self.fail_after: tuple[int, Exception] | None = None
        self.hang = False

    def capabilities(self, model_id: str) -> Capabilities:
        """Every capability on."""
        return Capabilities(tools=True, images=True, thinking=True, prompt_cache="explicit")

    def to_provider(self, req: ModelRequest) -> dict[str, JsonValue]:
        """The request as plain JSON."""
        return req.model_dump(mode="json")

    def from_provider(self, raw: object) -> ModelResponse:
        """Validate a plain mapping as a response."""
        return ModelResponse.model_validate(raw)

    async def stream(self, req: ModelRequest) -> AsyncIterator[StreamEvent]:
        """Replay the next scripted outcome."""
        self.requests.append(req)
        if self.hang:
            await asyncio.Event().wait()
        if not self.outcomes:
            message = "FakeAdapter has no scripted outcome left"
            raise AssertionError(message)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        events = events_for(outcome) if isinstance(outcome, ModelResponse) else outcome
        for index, event in enumerate(events):
            if self.fail_after is not None and index == self.fail_after[0]:
                raise self.fail_after[1]
            yield event

    def classify_error(self, e: Exception) -> GatewayError:
        """Pass ``GatewayError`` through; anything else is an invalid request."""
        if isinstance(e, GatewayError):
            return e
        return gateway_error(ErrorKind.INVALID_REQUEST)

    async def count_tokens(self, req: ModelRequest) -> int | None:
        """Return the scripted token count."""
        return self.token_count
