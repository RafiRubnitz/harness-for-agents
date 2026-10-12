"""Anthropic request building and stream parsing (no network)."""

import json

import pytest

from harness.core.consts import StopReason, StreamEventType
from harness.core.errors import GatewayError
from harness.core.types import (
    CacheHint,
    ImageBlock,
    Message,
    Source,
    TextBlock,
    ThinkingBlock,
    ThinkingConfig,
    ToolCallBlock,
    ToolResultBlock,
)
from harness.core.utils.tool_names import build_wire_map, to_wire_name
from harness.models import consts
from harness.models.adapters.anthropic_wire import (
    INVALID_JSON_KEY,
    StreamParser,
    parse_tool_input,
    to_provider,
)
from tests.consts import HEBREW_PLAIN, TEST_MODEL_ID
from tests.utils import make_request, read_data, tool_spec

DOTTED_TOOL = "mcp.github.create_issue"
CALL_ID = "toolu_01"
SIGNATURE = "sig-abc"
MAX_TOKENS = 8_000
BUDGET = 2_000


def _sse_events(name: str) -> list[dict[str, object]]:
    lines = read_data(f"providers/{name}").splitlines()
    return [json.loads(line.removeprefix("data: ")) for line in lines if line.startswith("data: ")]


def _parse(name: str, tools: list[str]) -> tuple[list[StreamEventType], StreamParser]:
    parser = StreamParser(TEST_MODEL_ID, build_wire_map(tools))
    types = [event.type for raw in _sse_events(name) for event in parser.feed(raw)]
    return types, parser


@pytest.mark.acceptance("01-10.2")
def test_dotted_tool_name_is_sent_as_wire_name() -> None:
    request = make_request(tools=[tool_spec(DOTTED_TOOL)])

    body = to_provider(request)

    tools = body["tools"]
    assert isinstance(tools, list)
    assert isinstance(tools[0], dict)
    assert tools[0]["name"] == to_wire_name(DOTTED_TOOL)
    assert body["tool_choice"] == consts.ANTHROPIC_TOOL_CHOICE[request.tool_choice]  # type: ignore[index]


@pytest.mark.acceptance("01-10.2")
@pytest.mark.acceptance("01-10.3")
def test_streamed_tool_call_is_assembled_with_internal_name() -> None:
    types, parser = _parse("anthropic_tool_call.sse", [DOTTED_TOOL])

    response = parser.response()

    thinking, text, call = response.content
    assert isinstance(thinking, ThinkingBlock)
    assert thinking.signature == SIGNATURE
    assert isinstance(text, TextBlock)
    assert isinstance(call, ToolCallBlock)
    assert call.name == DOTTED_TOOL
    assert call.id == CALL_ID
    assert call.input == {"title": "bug", "reason": "user asked"}
    assert response.stop_reason is StopReason.TOOL_CALL
    assert types[0] is StreamEventType.START
    assert types[-1] is StreamEventType.END
    assert StreamEventType.TOOL_CALL_END in types


def test_stream_usage_combines_start_and_delta() -> None:
    events = _sse_events("anthropic_tool_call.sse")
    start_usage = events[0]["message"]["usage"]  # type: ignore[index]
    final_usage = next(e for e in events if e["type"] == "message_delta")["usage"]

    _, parser = _parse("anthropic_tool_call.sse", [DOTTED_TOOL])
    usage = parser.response().usage

    assert usage.input == start_usage["input_tokens"]
    assert usage.cache_read == start_usage["cache_read_input_tokens"]
    assert usage.cache_write == start_usage["cache_creation_input_tokens"]
    assert usage.output == final_usage["output_tokens"]  # type: ignore[index]


def test_stream_without_stop_reason_is_an_error() -> None:
    parser = StreamParser(TEST_MODEL_ID, {})
    parser.feed({"type": "message_start", "message": {"usage": {}}})

    with pytest.raises(GatewayError) as raised:
        parser.response()

    assert raised.value.retryable


def test_invalid_tool_json_is_kept_under_marker_key() -> None:
    broken = '{"title": '

    assert parse_tool_input(broken) == {INVALID_JSON_KEY: broken}
    assert parse_tool_input("") == {}


def test_history_blocks_are_converted_and_unsigned_thinking_is_dropped() -> None:
    call = ToolCallBlock(id=CALL_ID, name=DOTTED_TOOL, input={"title": HEBREW_PLAIN})
    result = ToolResultBlock(call_id=CALL_ID, content=[TextBlock(text=HEBREW_PLAIN)])
    request = make_request()
    request.messages.extend(
        [
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(text="unsigned"),
                    ThinkingBlock(text="signed", signature=SIGNATURE),
                    call,
                ],
            ),
            Message(role="user", content=[result]),
        ]
    )

    body = to_provider(request)

    messages = body["messages"]
    assert isinstance(messages, list)
    assistant = messages[1]["content"]  # type: ignore[index,call-overload]
    assert [block["type"] for block in assistant] == ["thinking", "tool_use"]  # type: ignore[index,union-attr]
    assert assistant[1]["name"] == to_wire_name(DOTTED_TOOL)  # type: ignore[index,call-overload]
    tool_result = messages[2]["content"][0]  # type: ignore[index,call-overload]
    assert tool_result["tool_use_id"] == CALL_ID  # type: ignore[index,call-overload]
    assert tool_result["content"][0]["text"] == HEBREW_PLAIN  # type: ignore[index,call-overload]


def test_cache_hints_mark_system_tools_and_last_message() -> None:
    request = make_request(tools=[tool_spec("read_file")])
    request.system = "system prompt"
    request.cache = [
        CacheHint(position="system"),
        CacheHint(position="tools"),
        CacheHint(position="messages"),
    ]

    body = to_provider(request)

    marker = consts.ANTHROPIC_CACHE_CONTROL
    assert body["system"][-1]["cache_control"] == marker  # type: ignore[index,call-overload]
    assert body["tools"][-1]["cache_control"] == marker  # type: ignore[index,call-overload]
    assert body["messages"][-1]["content"][-1]["cache_control"] == marker  # type: ignore[index,call-overload]


def test_thinking_replaces_temperature_and_fits_under_max_tokens() -> None:
    request = make_request()
    request.max_tokens = MAX_TOKENS
    request.temperature = 0.5
    request.thinking = ThinkingConfig(budget_tokens=BUDGET)

    body = to_provider(request)

    assert body["thinking"] == {"type": "enabled", "budget_tokens": BUDGET}
    assert "temperature" not in body
    assert body["max_tokens"] == MAX_TOKENS


def test_blob_source_must_be_resolved_before_the_adapter() -> None:
    request = make_request()
    blob = Source(kind="blob", media_type="image/png", blob_id="blob_1")
    request.messages[0].content.append(ImageBlock(source=blob))

    with pytest.raises(GatewayError):
        to_provider(request)
