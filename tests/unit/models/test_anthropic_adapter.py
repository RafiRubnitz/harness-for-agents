"""The Anthropic adapter against a mocked HTTP transport."""

import json
from collections.abc import Callable

import httpx2
import pytest

from harness.core.consts import ErrorKind, StopReason, StreamEventType
from harness.core.errors import GatewayError
from harness.core.types import StreamEvent, ToolCallBlock
from harness.models import AdapterSettings, AnthropicAdapter
from tests.consts import TEST_API_KEY, TEST_MODEL_ID
from tests.utils import make_request, read_data, tool_spec

DOTTED_TOOL = "mcp.github.create_issue"
RETRY_AFTER_S = 7.0
TOKEN_COUNT = 321
TIMEOUT_S = 5.0

type Handler = Callable[[httpx2.Request], httpx2.Response]


def _adapter(handler: Handler) -> AnthropicAdapter:
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    settings = AdapterSettings(api_key=TEST_API_KEY, base_url=None, timeout_s=TIMEOUT_S)
    return AnthropicAdapter(lambda: settings, http_client=client)


def _error(status: int, kind: str, message: str = "boom", headers: dict[str, str] | None = None):
    body = {"type": "error", "error": {"type": kind, "message": message}}

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, json=body, headers=headers)

    return handler


async def _collect(adapter: AnthropicAdapter, **kwargs: object) -> list[StreamEvent]:
    request = make_request(model=TEST_MODEL_ID, **kwargs)  # type: ignore[arg-type]
    return [event async for event in adapter.stream(request)]


@pytest.mark.acceptance("01-10.3")
async def test_stream_parses_sse_and_sends_the_expected_request() -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(
            200,
            content=read_data("providers/anthropic_tool_call.sse").encode("utf-8"),
            headers={"content-type": "text/event-stream"},
        )

    events = await _collect(_adapter(handler), tools=[tool_spec(DOTTED_TOOL)])

    sent = json.loads(seen[0].content)
    assert sent["stream"] is True
    assert sent["model"] == TEST_MODEL_ID
    assert seen[0].headers["x-api-key"] == TEST_API_KEY
    final = events[-1]
    assert final.type is StreamEventType.END
    assert final.response is not None
    assert final.response.stop_reason is StopReason.TOOL_CALL
    calls = [block for block in final.response.content if isinstance(block, ToolCallBlock)]
    assert [call.name for call in calls] == [DOTTED_TOOL]


@pytest.mark.acceptance("01-10.4")
async def test_rate_limit_is_classified_with_retry_after() -> None:
    headers = {"retry-after": str(RETRY_AFTER_S)}
    adapter = _adapter(_error(429, "rate_limit_error", headers=headers))

    with pytest.raises(GatewayError) as raised:
        await _collect(adapter)

    assert raised.value.kind is ErrorKind.RATE_LIMIT
    assert raised.value.retry_after_s == RETRY_AFTER_S
    assert raised.value.retryable
    assert raised.value.model == TEST_MODEL_ID


@pytest.mark.parametrize(
    ("status", "error_type", "message", "kind"),
    [
        (529, "overloaded_error", "overloaded", ErrorKind.OVERLOADED),
        (500, "api_error", "internal", ErrorKind.OVERLOADED),
        (401, "authentication_error", "bad key", ErrorKind.AUTH),
        (403, "permission_error", "forbidden", ErrorKind.AUTH),
        (400, "invalid_request_error", "prompt is too long: 1 > 0", ErrorKind.CONTEXT_OVERFLOW),
        (413, "request_too_large", "too large", ErrorKind.CONTEXT_OVERFLOW),
        (400, "invalid_request_error", "bad field", ErrorKind.INVALID_REQUEST),
    ],
)
async def test_http_errors_are_classified(
    status: int, error_type: str, message: str, kind: ErrorKind
) -> None:
    adapter = _adapter(_error(status, error_type, message))

    with pytest.raises(GatewayError) as raised:
        await _collect(adapter)

    assert raised.value.kind is kind


async def test_network_failure_is_a_retryable_timeout() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        message = "connection refused"
        raise httpx2.ConnectError(message, request=request)

    with pytest.raises(GatewayError) as raised:
        await _collect(_adapter(handler))

    assert raised.value.kind is ErrorKind.TIMEOUT
    assert raised.value.retryable


async def test_truncated_stream_is_a_retryable_error() -> None:
    sse = read_data("providers/anthropic_tool_call.sse")
    truncated = sse[: sse.index("event: message_delta")]

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200, content=truncated.encode("utf-8"), headers={"content-type": "text/event-stream"}
        )

    with pytest.raises(GatewayError) as raised:
        await _collect(_adapter(handler))

    assert raised.value.retryable


async def test_count_tokens_uses_the_counting_endpoint() -> None:
    paths: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        paths.append(request.url.path)
        return httpx2.Response(200, json={"input_tokens": TOKEN_COUNT})

    counted = await _adapter(handler).count_tokens(make_request(model=TEST_MODEL_ID))

    assert counted == TOKEN_COUNT
    assert paths[0].endswith("/count_tokens")


def test_from_provider_converts_a_complete_message() -> None:
    raw = {
        "model": TEST_MODEL_ID,
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "refusal",
        "usage": {"input_tokens": 3, "output_tokens": 4},
    }

    response = _adapter(_error(500, "api_error")).from_provider(raw)

    assert response.stop_reason is StopReason.REFUSAL
    assert response.usage.input == raw["usage"]["input_tokens"]  # type: ignore[index]
    assert response.model_used == TEST_MODEL_ID
