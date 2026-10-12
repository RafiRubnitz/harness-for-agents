"""The OpenAI / DeepSeek adapters: wire format and a mocked HTTP transport."""

import json
from collections.abc import Callable

import httpx2
import pytest
from pydantic import JsonValue

from harness.core.consts import ErrorKind, StopReason, StreamEventType
from harness.core.errors import GatewayError
from harness.core.types import (
    ImageBlock,
    Message,
    ModelResponse,
    Source,
    StreamEvent,
    TextBlock,
    ThinkingBlock,
    ThinkingConfig,
    ToolCallBlock,
    ToolResultBlock,
)
from harness.core.utils.tool_names import to_wire_name
from harness.models import AdapterSettings, AnthropicAdapter, DeepSeekAdapter, OpenAIAdapter
from tests.consts import TEST_API_KEY, TEST_MODEL_ID
from tests.utils import make_request, read_data, tool_spec, user_message

DOTTED_TOOL = "mcp.github.create_issue"
CALL_ID = "call_abc"
TOOL_INPUT: dict[str, JsonValue] = {"title": "שלום", "labels": ["bug"], "reason": "asked"}
ANSWER_PARTS = ["Hello", ", ", "עולם"]
REASONING_PARTS = ["first ", "then"]
PROMPT_TOKENS = 120
CACHED_TOKENS = 100
COMPLETION_TOKENS = 30
RETRY_AFTER_S = 7.0
TIMEOUT_S = 5.0
MAX_TOKENS = 512
SCHEMA: dict[str, JsonValue] = {"type": "object", "properties": {"ok": {"type": "boolean"}}}

type Handler = Callable[[httpx2.Request], httpx2.Response]
type Chunk = dict[str, JsonValue]


def _settings() -> AdapterSettings:
    return AdapterSettings(api_key=TEST_API_KEY, base_url=None, timeout_s=TIMEOUT_S)


def _adapter(handler: Handler, kind: type[OpenAIAdapter] = OpenAIAdapter) -> OpenAIAdapter:
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    return kind(_settings, http_client=client)


def _chunk(delta: Chunk, finish: str | None = None) -> Chunk:
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def _usage_chunk() -> Chunk:
    usage: Chunk = {
        "prompt_tokens": PROMPT_TOKENS,
        "completion_tokens": COMPLETION_TOKENS,
        "prompt_tokens_details": {"cached_tokens": CACHED_TOKENS},
    }
    return {"choices": [], "usage": usage}


def _tool_chunks(name: str, call_id: str, tool_input: dict[str, JsonValue]) -> list[Chunk]:
    """A tool call the way the API streams it: header first, then the arguments in pieces."""
    arguments = json.dumps(tool_input, ensure_ascii=False)
    half = len(arguments) // 2
    head: Chunk = {"index": 0, "id": call_id, "type": "function"}
    head["function"] = {"name": to_wire_name(name), "arguments": ""}
    pieces: list[Chunk] = [
        {"index": 0, "function": {"arguments": part}}
        for part in (arguments[:half], arguments[half:])
    ]
    return [
        _chunk({"role": "assistant", "tool_calls": [head]}),
        *[_chunk({"tool_calls": [piece]}) for piece in pieces],
        _chunk({}, "tool_calls"),
        _usage_chunk(),
    ]


def _text_chunks(parts: list[str], key: str = "content") -> list[Chunk]:
    return [_chunk({key: part}) for part in parts]


def _sse(chunks: list[Chunk], *, done: bool = True) -> Handler:
    lines = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
    if done:
        lines.append("data: [DONE]\n\n")

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            content="".join(lines).encode("utf-8"),
            headers={"content-type": "text/event-stream"},
        )

    return handler


def _error(status: int, code: str | None, message: str, headers: dict[str, str] | None = None):
    body = {"error": {"message": message, "type": "error", "code": code}}

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, json=body, headers=headers)

    return handler


async def _collect(adapter: OpenAIAdapter | AnthropicAdapter, **kw: object) -> list[StreamEvent]:
    request = make_request(model=TEST_MODEL_ID, **kw)  # type: ignore[arg-type]
    return [event async for event in adapter.stream(request)]


def _final(events: list[StreamEvent]) -> ModelResponse:
    assert events[-1].type is StreamEventType.END
    assert events[-1].response is not None
    return events[-1].response


# --- request body ------------------------------------------------------------------------


def test_request_body_orders_tool_results_before_user_content() -> None:
    call = ToolCallBlock(id=CALL_ID, name=DOTTED_TOOL, input=TOOL_INPUT)
    result = ToolResultBlock(call_id=CALL_ID, content=[TextBlock(text="done")], is_error=True)
    image = ImageBlock(source=Source(kind="base64", media_type="image/png", data="AAAA"))
    request = make_request(model=TEST_MODEL_ID, tools=[tool_spec(DOTTED_TOOL)]).model_copy(
        update={
            "system": "be brief",
            "max_tokens": MAX_TOKENS,
            "messages": [
                user_message("open an issue"),
                Message(role="assistant", content=[ThinkingBlock(text="hm"), call]),
                Message(role="user", content=[result, TextBlock(text="and now?"), image]),
            ],
        }
    )

    body = OpenAIAdapter(_settings).to_provider(request)

    messages = body["messages"]
    assert isinstance(messages, list)
    roles = [m["role"] for m in messages if isinstance(m, dict)]
    assert roles == ["system", "user", "assistant", "tool", "user"]
    sent = json.loads(json.dumps(messages))
    assert sent[2]["tool_calls"][0]["function"]["name"] == to_wire_name(DOTTED_TOOL)
    assert json.loads(sent[2]["tool_calls"][0]["function"]["arguments"]) == TOOL_INPUT
    assert sent[3]["tool_call_id"] == CALL_ID
    assert "done" in sent[3]["content"]
    assert [part["type"] for part in sent[4]["content"]] == ["text", "image_url"]
    assert json.loads(json.dumps(body["tools"]))[0]["function"]["name"] == to_wire_name(DOTTED_TOOL)
    assert "hm" not in json.dumps(body)


def test_providers_differ_only_where_their_apis_do() -> None:
    request = make_request(model=TEST_MODEL_ID).model_copy(
        update={
            "max_tokens": MAX_TOKENS,
            "thinking": ThinkingConfig(budget_tokens=MAX_TOKENS),
            "temperature": 0.5,
            "response_format": SCHEMA,
        }
    )

    openai_body = OpenAIAdapter(_settings).to_provider(request)
    deepseek_body = DeepSeekAdapter(_settings).to_provider(request)

    assert openai_body["max_completion_tokens"] == deepseek_body["max_tokens"] == MAX_TOKENS
    assert "reasoning_effort" in openai_body
    assert "reasoning_effort" not in deepseek_body
    assert "temperature" not in openai_body
    assert json.loads(json.dumps(openai_body["response_format"]))["json_schema"]["schema"] == SCHEMA
    assert deepseek_body["response_format"] == {"type": "json_object"}
    assert openai_body["messages"] == deepseek_body["messages"]


# --- streaming ---------------------------------------------------------------------------


@pytest.mark.acceptance("01-10.3")
async def test_streamed_tool_call_is_assembled_and_named_internally() -> None:
    seen: list[httpx2.Request] = []
    reply = _sse(_tool_chunks(DOTTED_TOOL, CALL_ID, TOOL_INPUT))

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return reply(request)

    events = await _collect(_adapter(handler), tools=[tool_spec(DOTTED_TOOL)])

    sent = json.loads(seen[0].content)
    assert sent["stream"] is True
    assert sent["stream_options"] == {"include_usage": True}
    assert seen[0].headers["authorization"] == f"Bearer {TEST_API_KEY}"
    response = _final(events)
    assert response.stop_reason is StopReason.TOOL_CALL
    assert response.content == [ToolCallBlock(id=CALL_ID, name=DOTTED_TOOL, input=TOOL_INPUT)]
    kinds = [event.type for event in events]
    assert kinds.index(StreamEventType.TOOL_CALL_START) < kinds.index(StreamEventType.TOOL_CALL_END)
    deltas = [e.input_json_delta or "" for e in events if e.type is StreamEventType.TOOL_CALL_DELTA]
    assert json.loads("".join(deltas)) == TOOL_INPUT


async def test_text_stream_reports_usage_with_cached_tokens_apart() -> None:
    chunks = [*_text_chunks(ANSWER_PARTS), _chunk({}, "stop"), _usage_chunk()]

    events = await _collect(_adapter(_sse(chunks)))

    response = _final(events)
    texts = [event.text for event in events if event.type is StreamEventType.TEXT_DELTA]
    assert texts == ANSWER_PARTS
    assert response.content == [TextBlock(text="".join(ANSWER_PARTS))]
    assert response.stop_reason is StopReason.END
    assert response.usage.cache_read == CACHED_TOKENS
    assert response.usage.input == PROMPT_TOKENS - CACHED_TOKENS
    assert response.usage.output == COMPLETION_TOKENS


async def test_deepseek_reasoning_becomes_a_thinking_block() -> None:
    chunks = [
        *_text_chunks(REASONING_PARTS, "reasoning_content"),
        *_text_chunks(ANSWER_PARTS),
        _chunk({}, "stop"),
    ]

    events = await _collect(_adapter(_sse(chunks), DeepSeekAdapter))

    thinking = [event.text for event in events if event.type is StreamEventType.THINKING_DELTA]
    assert thinking == REASONING_PARTS
    assert _final(events).content == [
        ThinkingBlock(text="".join(REASONING_PARTS)),
        TextBlock(text="".join(ANSWER_PARTS)),
    ]


@pytest.mark.acceptance("01-10.1")
@pytest.mark.parametrize("kind", [OpenAIAdapter, DeepSeekAdapter])
async def test_same_request_gives_the_same_response_shape_as_anthropic(
    kind: type[OpenAIAdapter],
) -> None:
    def anthropic_handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            content=read_data("providers/anthropic_tool_call.sse").encode("utf-8"),
            headers={"content-type": "text/event-stream"},
        )

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(anthropic_handler))
    anthropic = AnthropicAdapter(_settings, http_client=client)
    expected = _final(await _collect(anthropic, tools=[tool_spec(DOTTED_TOOL)]))
    texts = [block.text for block in expected.content if isinstance(block, TextBlock)]
    call = next(block for block in expected.content if isinstance(block, ToolCallBlock))
    chunks = [*_text_chunks(texts), *_tool_chunks(call.name, call.id, call.input)]

    events = await _collect(_adapter(_sse(chunks), kind), tools=[tool_spec(DOTTED_TOOL)])

    response = _final(events)
    shown = [block for block in expected.content if not isinstance(block, ThinkingBlock)]
    assert response.content == shown
    assert response.stop_reason is expected.stop_reason
    assert set(response.model_dump()) == set(expected.model_dump())


async def test_stream_that_ends_without_a_finish_reason_is_retryable() -> None:
    with pytest.raises(GatewayError) as raised:
        await _collect(_adapter(_sse(_text_chunks(ANSWER_PARTS))))

    assert raised.value.retryable


def test_complete_response_is_converted() -> None:
    arguments = json.dumps(TOOL_INPUT)
    function = {"name": to_wire_name("read_file"), "arguments": arguments}
    message = {
        "content": ANSWER_PARTS[0],
        "tool_calls": [{"id": CALL_ID, "type": "function", "function": function}],
    }
    raw = {
        "model": TEST_MODEL_ID,
        "choices": [{"message": message, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": PROMPT_TOKENS, "completion_tokens": COMPLETION_TOKENS},
    }

    response = OpenAIAdapter(_settings).from_provider(raw)

    assert response.model_used == TEST_MODEL_ID
    assert response.stop_reason is StopReason.TOOL_CALL
    assert response.content == [
        TextBlock(text=ANSWER_PARTS[0]),
        ToolCallBlock(id=CALL_ID, name="read_file", input=TOOL_INPUT),
    ]
    assert (response.usage.input, response.usage.output) == (PROMPT_TOKENS, COMPLETION_TOKENS)


# --- errors ------------------------------------------------------------------------------


@pytest.mark.acceptance("01-10.4")
async def test_rate_limit_is_classified_with_retry_after() -> None:
    headers = {"retry-after": str(RETRY_AFTER_S)}
    adapter = _adapter(_error(429, "rate_limit_exceeded", "slow down", headers))

    with pytest.raises(GatewayError) as raised:
        await _collect(adapter)

    assert raised.value.kind is ErrorKind.RATE_LIMIT
    assert raised.value.retry_after_s == RETRY_AFTER_S
    assert raised.value.retryable
    assert raised.value.model == TEST_MODEL_ID


@pytest.mark.parametrize(
    ("status", "code", "message", "kind"),
    [
        (429, "insufficient_quota", "you exceeded your quota", ErrorKind.AUTH),
        (500, None, "internal", ErrorKind.OVERLOADED),
        (503, None, "unavailable", ErrorKind.OVERLOADED),
        (401, "invalid_api_key", "bad key", ErrorKind.AUTH),
        (403, None, "forbidden", ErrorKind.AUTH),
        (400, "context_length_exceeded", "too long", ErrorKind.CONTEXT_OVERFLOW),
        (400, None, "This model's maximum context length is 8", ErrorKind.CONTEXT_OVERFLOW),
        (400, None, "bad field", ErrorKind.INVALID_REQUEST),
    ],
)
async def test_http_errors_are_classified(
    status: int, code: str | None, message: str, kind: ErrorKind
) -> None:
    adapter = _adapter(_error(status, code, message))

    with pytest.raises(GatewayError) as raised:
        await _collect(adapter)

    assert raised.value.kind is kind
    assert raised.value.retryable is (kind is ErrorKind.OVERLOADED)


async def test_network_failure_is_a_retryable_timeout() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        message = "connection refused"
        raise httpx2.ConnectError(message, request=request)

    with pytest.raises(GatewayError) as raised:
        await _collect(_adapter(handler, DeepSeekAdapter))

    assert raised.value.kind is ErrorKind.TIMEOUT
    assert raised.value.provider == DeepSeekAdapter.id


async def test_there_is_no_counting_endpoint() -> None:
    adapter = OpenAIAdapter(_settings)

    assert await adapter.count_tokens(make_request(model=TEST_MODEL_ID)) is None
    assert adapter.capabilities("no-such-model").tools
