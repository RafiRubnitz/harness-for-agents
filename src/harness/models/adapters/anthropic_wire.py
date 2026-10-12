"""Anthropic Messages wire format: request building and stream parsing (no network)."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

from pydantic import JsonValue

from harness.core.consts import ErrorKind, ProviderId, StopReason, StreamEventType
from harness.core.errors import GatewayError
from harness.core.types import (
    ContentBlock,
    FileBlock,
    ImageBlock,
    ModelRequest,
    ModelResponse,
    Source,
    StreamEvent,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
    Usage,
)
from harness.core.utils.tool_names import from_wire_name, to_wire_name
from harness.models.consts import (
    ANTHROPIC_CACHE_CONTROL,
    ANTHROPIC_DEFAULT_MAX_TOKENS,
    ANTHROPIC_MIN_THINKING_BUDGET,
    ANTHROPIC_STOP_REASONS,
    ANTHROPIC_TOOL_CHOICE,
    MSG_BLOB_SOURCE,
    MSG_STREAM_ENDED_EARLY,
)

type Wire = dict[str, JsonValue]

INVALID_JSON_KEY = "__invalid_json__"


def _cache_control() -> Wire:
    return dict(ANTHROPIC_CACHE_CONTROL)


def _source(source: Source, model: str) -> Wire:
    if source.kind == "base64":
        return {"type": "base64", "media_type": source.media_type, "data": source.data}
    if source.kind == "url":
        return {"type": "url", "url": source.url}
    raise GatewayError(
        MSG_BLOB_SOURCE,
        kind=ErrorKind.INVALID_REQUEST,
        provider=ProviderId.ANTHROPIC,
        model=model,
    )


def _block(block: ContentBlock, model: str) -> Wire | None:
    """Convert one content block; returns ``None`` for blocks the API cannot accept back."""
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ImageBlock):
        return {"type": "image", "source": _source(block.source, model)}
    if isinstance(block, FileBlock):
        return {"type": "document", "source": _source(block.source, model)}
    if isinstance(block, ToolCallBlock):
        return {
            "type": "tool_use",
            "id": block.id,
            "name": to_wire_name(block.name),
            "input": block.input,
        }
    if isinstance(block, ToolResultBlock):
        inner = [wire for item in block.content if (wire := _block(item, model)) is not None]
        return {
            "type": "tool_result",
            "tool_use_id": block.call_id,
            "content": cast("JsonValue", inner),
            "is_error": block.is_error,
        }
    if block.signature is None:
        return None  # unsigned thinking cannot be replayed
    return {"type": "thinking", "thinking": block.text, "signature": block.signature}


def _blocks(blocks: list[ContentBlock], model: str) -> list[Wire]:
    return [wire for block in blocks if (wire := _block(block, model)) is not None]


def _mark_cached(blocks: list[Wire]) -> None:
    if blocks:
        blocks[-1]["cache_control"] = _cache_control()


def _system(req: ModelRequest, cached: bool) -> JsonValue:
    if req.system is None:
        return None
    if isinstance(req.system, str):
        blocks: list[Wire] = [{"type": "text", "text": req.system}] if req.system else []
    else:
        blocks = _blocks(req.system, req.model)
    if cached:
        _mark_cached(blocks)
    return cast("JsonValue", blocks) if blocks else None


def _tool_choice(req: ModelRequest) -> Wire:
    choice = req.tool_choice
    if isinstance(choice, str):
        return dict(ANTHROPIC_TOOL_CHOICE[choice])
    name = choice.get("name")
    if isinstance(name, str):
        return {"type": "tool", "name": to_wire_name(name)}
    return dict(choice)


def _thinking(req: ModelRequest, max_tokens: int) -> Wire | None:
    if req.thinking is None or not req.thinking.enabled:
        return None
    budget = min(req.thinking.budget_tokens, max_tokens - 1)
    if budget < ANTHROPIC_MIN_THINKING_BUDGET:
        return None
    return {"type": "enabled", "budget_tokens": budget}


def to_provider(req: ModelRequest) -> Wire:
    """Build the body of ``POST /v1/messages`` (without ``stream``)."""
    hints = req.cache or []
    max_tokens = req.max_tokens or ANTHROPIC_DEFAULT_MAX_TOKENS
    messages: list[Wire] = []
    contents: list[list[Wire]] = []
    for message in req.messages:
        content = _blocks(message.content, req.model)
        contents.append(content)
        messages.append({"role": message.role, "content": cast("JsonValue", content)})
    for hint in hints:
        if hint.position == "messages" and contents:
            index = hint.index if hint.index is not None else len(contents) - 1
            if -len(contents) <= index < len(contents):
                _mark_cached(contents[index])
    body: Wire = {
        "model": req.model,
        "max_tokens": max_tokens,
        "messages": cast("JsonValue", messages),
    }
    system = _system(req, any(hint.position == "system" for hint in hints))
    if system is not None:
        body["system"] = system
    if req.tools:
        tools: list[Wire] = [
            {
                "name": to_wire_name(tool.name),
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in req.tools
        ]
        if any(hint.position == "tools" for hint in hints):
            _mark_cached(tools)
        body["tools"] = cast("JsonValue", tools)
        body["tool_choice"] = _tool_choice(req)
    thinking = _thinking(req, max_tokens)
    if thinking is not None:
        body["thinking"] = thinking
    elif req.temperature is not None:
        body["temperature"] = req.temperature
    if req.response_format is not None:
        body["output_config"] = {"format": {"type": "json_schema", "schema": req.response_format}}
    return body


def _map(value: object) -> Mapping[str, object]:
    return cast("Mapping[str, object]", value) if isinstance(value, Mapping) else {}


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _count(value: object) -> int:
    return value if isinstance(value, int) else 0


def parse_usage(raw: Mapping[str, object], previous: Usage | None = None) -> Usage:
    """Read an Anthropic ``usage`` object; missing fields keep the previous values."""
    base = previous or Usage()
    return Usage(
        input=_count(raw.get("input_tokens")) or base.input,
        output=_count(raw.get("output_tokens")) or base.output,
        cache_read=_count(raw.get("cache_read_input_tokens")) or base.cache_read,
        cache_write=_count(raw.get("cache_creation_input_tokens")) or base.cache_write,
    )


def parse_tool_input(raw_json: str) -> dict[str, JsonValue]:
    """Parse the accumulated tool input; invalid JSON is kept under a marker key."""
    if not raw_json.strip():
        return {}
    try:
        parsed = cast("JsonValue", json.loads(raw_json))
    except json.JSONDecodeError:
        return {INVALID_JSON_KEY: raw_json}
    return parsed if isinstance(parsed, dict) else {INVALID_JSON_KEY: raw_json}


def stop_reason(raw: object) -> StopReason:
    """Map an Anthropic stop reason to the internal one."""
    return ANTHROPIC_STOP_REASONS.get(_text(raw), StopReason.END)


def content_block(raw: Mapping[str, object], wire_map: dict[str, str]) -> ContentBlock | None:
    """Convert one complete response block (non-streamed path)."""
    kind = raw.get("type")
    if kind == "text":
        return TextBlock(text=_text(raw.get("text")))
    if kind == "thinking":
        signature = raw.get("signature")
        return ThinkingBlock(
            text=_text(raw.get("thinking")),
            signature=signature if isinstance(signature, str) else None,
        )
    if kind == "tool_use":
        tool_input = cast("JsonValue", raw.get("input"))
        return ToolCallBlock(
            id=_text(raw.get("id")),
            name=from_wire_name(_text(raw.get("name")), wire_map),
            input=tool_input if isinstance(tool_input, dict) else {},
        )
    return None


@dataclass(slots=True)
class _Open:
    """A content block that is still streaming."""

    kind: str
    call_id: str = ""
    name: str = ""
    parts: list[str] = field(default_factory=list[str])
    signature: str = ""


class StreamParser:
    """Turns raw Anthropic stream events into ``StreamEvent`` objects and a final response."""

    def __init__(self, model: str, wire_map: dict[str, str]) -> None:
        self._model = model
        self._wire_map = wire_map
        self._open: dict[int, _Open] = {}
        self._content: dict[int, ContentBlock] = {}
        self._usage = Usage()
        self._stop: StopReason | None = None
        self.finished = False

    def feed(self, event: Mapping[str, object]) -> list[StreamEvent]:
        """Consume one raw event and return the internal events it produces."""
        kind = event.get("type")
        if kind == "message_start":
            usage = _map(_map(event.get("message")).get("usage"))
            self._usage = parse_usage(usage)
            return [StreamEvent(type=StreamEventType.START)]
        if kind == "content_block_start":
            return self._start(_count(event.get("index")), _map(event.get("content_block")))
        if kind == "content_block_delta":
            return self._delta(_count(event.get("index")), _map(event.get("delta")))
        if kind == "content_block_stop":
            return self._stop_block(_count(event.get("index")))
        if kind == "message_delta":
            self._stop = stop_reason(_map(event.get("delta")).get("stop_reason"))
            self._usage = parse_usage(_map(event.get("usage")), self._usage)
            return [StreamEvent(type=StreamEventType.USAGE, usage=self._usage)]
        if kind == "message_stop":
            self.finished = True
            return [StreamEvent(type=StreamEventType.END, response=self.response())]
        return []

    def _start(self, index: int, block: Mapping[str, object]) -> list[StreamEvent]:
        kind = _text(block.get("type"))
        opened = _Open(kind=kind)
        self._open[index] = opened
        if kind != "tool_use":
            return []
        opened.call_id = _text(block.get("id"))
        opened.name = from_wire_name(_text(block.get("name")), self._wire_map)
        return [
            StreamEvent(
                type=StreamEventType.TOOL_CALL_START,
                call_id=opened.call_id,
                tool_name=opened.name,
            )
        ]

    def _delta(self, index: int, delta: Mapping[str, object]) -> list[StreamEvent]:
        opened = self._open.get(index)
        if opened is None:
            return []
        kind = delta.get("type")
        if kind == "text_delta":
            text = _text(delta.get("text"))
            opened.parts.append(text)
            return [StreamEvent(type=StreamEventType.TEXT_DELTA, text=text)]
        if kind == "thinking_delta":
            text = _text(delta.get("thinking"))
            opened.parts.append(text)
            return [StreamEvent(type=StreamEventType.THINKING_DELTA, text=text)]
        if kind == "signature_delta":
            opened.signature += _text(delta.get("signature"))
            return []
        if kind == "input_json_delta":
            partial = _text(delta.get("partial_json"))
            opened.parts.append(partial)
            return [
                StreamEvent(
                    type=StreamEventType.TOOL_CALL_DELTA,
                    call_id=opened.call_id,
                    input_json_delta=partial,
                )
            ]
        return []

    def _stop_block(self, index: int) -> list[StreamEvent]:
        opened = self._open.pop(index, None)
        if opened is None:
            return []
        joined = "".join(opened.parts)
        if opened.kind == "text":
            self._content[index] = TextBlock(text=joined)
        elif opened.kind == "thinking":
            self._content[index] = ThinkingBlock(text=joined, signature=opened.signature or None)
        elif opened.kind == "tool_use":
            call = ToolCallBlock(
                id=opened.call_id, name=opened.name, input=parse_tool_input(joined)
            )
            self._content[index] = call
            return [
                StreamEvent(type=StreamEventType.TOOL_CALL_END, call_id=call.id, tool_call=call)
            ]
        return []

    def response(self) -> ModelResponse:
        """Return the assembled response.

        Raises:
            GatewayError: The stream ended before the provider sent a stop reason.
        """
        if self._stop is None:
            raise GatewayError(
                MSG_STREAM_ENDED_EARLY,
                kind=ErrorKind.OVERLOADED,
                provider=ProviderId.ANTHROPIC,
                model=self._model,
            )
        content = [self._content[index] for index in sorted(self._content)]
        return ModelResponse(
            content=content, stop_reason=self._stop, usage=self._usage, model_used=self._model
        )
