"""OpenAI Chat Completions wire format: request building and stream parsing (no network).

The same format is spoken by every OpenAI-compatible provider (DeepSeek among them);
``WireOptions`` holds the few points where they differ.
"""

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
    Message,
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
from harness.models.adapters.anthropic_wire import parse_tool_input
from harness.models.consts import (
    DATA_URI,
    MSG_BLOB_SOURCE,
    MSG_FILE_BY_URL,
    MSG_STREAM_ENDED_EARLY,
    MSG_TOOL_ERROR_PREFIX,
    OPENAI_EFFORT_LOW_MAX_TOKENS,
    OPENAI_EFFORT_MEDIUM_MAX_TOKENS,
    OPENAI_MAX_TOKENS_KEY,
    OPENAI_REASONING_KEYS,
    OPENAI_SCHEMA_NAME,
    OPENAI_STOP_REASONS,
)

type Wire = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class WireOptions:
    """Where OpenAI-compatible providers differ from OpenAI itself."""

    provider: ProviderId = ProviderId.OPENAI
    max_tokens_key: str = OPENAI_MAX_TOKENS_KEY
    reasoning_effort: bool = True
    json_schema: bool = True


def _map(value: object) -> Mapping[str, object]:
    return cast("Mapping[str, object]", value) if isinstance(value, Mapping) else {}


def _list(value: object) -> list[object]:
    return cast("list[object]", value) if isinstance(value, list) else []


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _count(value: object) -> int:
    return value if isinstance(value, int) else 0


# --- request -----------------------------------------------------------------------------


def _url(source: Source, req: ModelRequest, options: WireOptions) -> str:
    if source.kind == "base64":
        return DATA_URI.format(media_type=source.media_type, data=source.data or "")
    if source.kind == "url":
        return source.url or ""
    raise GatewayError(
        MSG_BLOB_SOURCE, kind=ErrorKind.INVALID_REQUEST, provider=options.provider, model=req.model
    )


def _part(block: ContentBlock, req: ModelRequest, options: WireOptions) -> Wire | None:
    """One part of a user message; ``None`` for blocks that are not user content."""
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ImageBlock):
        return {"type": "image_url", "image_url": {"url": _url(block.source, req, options)}}
    if isinstance(block, FileBlock):
        if block.source.kind == "url":
            return {"type": "text", "text": MSG_FILE_BY_URL.format(url=block.source.url)}
        return {"type": "file", "file": {"file_data": _url(block.source, req, options)}}
    return None


def _plain(blocks: list[ContentBlock]) -> str:
    return "\n".join(block.text for block in blocks if isinstance(block, TextBlock))


def _tool_message(block: ToolResultBlock) -> Wire:
    text = _plain(block.content)
    if block.is_error:
        text = f"{MSG_TOOL_ERROR_PREFIX}{text}"
    return {"role": "tool", "tool_call_id": block.call_id, "content": text}


def _user(message: Message, req: ModelRequest, options: WireOptions) -> list[Wire]:
    """Tool results first (they must follow the assistant's calls), then the user content."""
    out: list[Wire] = []
    parts: list[Wire] = []
    for block in message.content:
        if isinstance(block, ToolResultBlock):
            out.append(_tool_message(block))
            # a tool message carries text only: other results travel as user content
            inner = [b for b in block.content if not isinstance(b, TextBlock)]
            parts += [p for b in inner if (p := _part(b, req, options)) is not None]
        elif (part := _part(block, req, options)) is not None:
            parts.append(part)
    if parts:
        out.append({"role": "user", "content": cast("JsonValue", parts)})
    return out


def _assistant(message: Message) -> Wire | None:
    calls: list[Wire] = [
        {
            "id": block.id,
            "type": "function",
            "function": {
                "name": to_wire_name(block.name),
                "arguments": json.dumps(block.input, ensure_ascii=False),
            },
        }
        for block in message.content
        if isinstance(block, ToolCallBlock)
    ]
    text = _plain(message.content)
    if not calls and not text:
        return None  # only thinking: nothing the API accepts back
    wire: Wire = {"role": "assistant", "content": text or None}
    if calls:
        wire["tool_calls"] = cast("JsonValue", calls)
    return wire


def _messages(req: ModelRequest, options: WireOptions) -> list[Wire]:
    out: list[Wire] = []
    system = req.system if isinstance(req.system, str) else _plain(req.system or [])
    if system:
        out.append({"role": "system", "content": system})
    for message in req.messages:
        if message.role == "user":
            out += _user(message, req, options)
        elif (wire := _assistant(message)) is not None:
            out.append(wire)
    return out


def _tool_choice(req: ModelRequest) -> JsonValue:
    choice = req.tool_choice
    if isinstance(choice, str):
        return choice
    name = choice.get("name")
    if isinstance(name, str):
        return {"type": "function", "function": {"name": to_wire_name(name)}}
    return dict(choice)


def _effort(budget_tokens: int) -> str:
    if budget_tokens <= OPENAI_EFFORT_LOW_MAX_TOKENS:
        return "low"
    if budget_tokens <= OPENAI_EFFORT_MEDIUM_MAX_TOKENS:
        return "medium"
    return "high"


def to_provider(req: ModelRequest, options: WireOptions | None = None) -> Wire:
    """Build the body of ``POST /chat/completions`` (without ``stream``)."""
    options = options or WireOptions()
    body: Wire = {"model": req.model, "messages": cast("JsonValue", _messages(req, options))}
    if req.max_tokens is not None:
        body[options.max_tokens_key] = req.max_tokens
    if req.tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": to_wire_name(tool.name),
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in req.tools
        ]
        body["tool_choice"] = _tool_choice(req)
    thinking = req.thinking is not None and req.thinking.enabled
    if thinking and req.thinking is not None and options.reasoning_effort:
        body["reasoning_effort"] = _effort(req.thinking.budget_tokens)
    elif req.temperature is not None and not thinking:
        body["temperature"] = req.temperature
    if req.response_format is not None:
        body["response_format"] = (
            {
                "type": "json_schema",
                "json_schema": {"name": OPENAI_SCHEMA_NAME, "schema": req.response_format},
            }
            if options.json_schema
            else {"type": "json_object"}
        )
    return body


# --- response ----------------------------------------------------------------------------


def parse_usage(raw: Mapping[str, object]) -> Usage:
    """Read a ``usage`` object; cached prompt tokens are counted apart from fresh input."""
    prompt = _count(raw.get("prompt_tokens"))
    details = _map(raw.get("prompt_tokens_details"))
    cached = _count(details.get("cached_tokens")) or _count(raw.get("prompt_cache_hit_tokens"))
    return Usage(
        input=max(prompt - cached, 0),
        output=_count(raw.get("completion_tokens")),
        cache_read=cached,
    )


def stop_reason(raw: object) -> StopReason:
    """Map a ``finish_reason`` to the internal stop reason."""
    return OPENAI_STOP_REASONS.get(_text(raw), StopReason.END)


def reasoning_text(raw: Mapping[str, object]) -> str:
    """The reasoning text of a message or delta, under whichever key the provider uses."""
    for key in OPENAI_REASONING_KEYS:
        if text := _text(raw.get(key)):
            return text
    return ""


def message_content(raw: Mapping[str, object], wire_map: dict[str, str]) -> list[ContentBlock]:
    """Convert one complete response message (non-streamed path)."""
    content: list[ContentBlock] = []
    if thinking := reasoning_text(raw):
        content.append(ThinkingBlock(text=thinking))
    if text := _text(raw.get("content")):
        content.append(TextBlock(text=text))
    for item in _list(raw.get("tool_calls")):
        call = _map(item)
        function = _map(call.get("function"))
        content.append(
            ToolCallBlock(
                id=_text(call.get("id")),
                name=from_wire_name(_text(function.get("name")), wire_map),
                input=parse_tool_input(_text(function.get("arguments"))),
            )
        )
    return content


@dataclass(slots=True)
class _OpenCall:
    """A tool call that is still streaming."""

    call_id: str = ""
    name: str = ""
    started: bool = False
    parts: list[str] = field(default_factory=list[str])


class StreamParser:
    """Turns chat-completion chunks into ``StreamEvent`` objects and a final response."""

    def __init__(self, model: str, wire_map: dict[str, str], provider: ProviderId) -> None:
        self._model = model
        self._wire_map = wire_map
        self._provider = provider
        self._started = False
        self._text: list[str] = []
        self._thinking: list[str] = []
        self._open: dict[int, _OpenCall] = {}
        self._calls: list[ToolCallBlock] = []
        self._usage = Usage()
        self._stop: StopReason | None = None

    def feed(self, chunk: Mapping[str, object]) -> list[StreamEvent]:
        """Consume one chunk and return the internal events it produces."""
        events: list[StreamEvent] = []
        if not self._started:
            self._started = True
            events.append(StreamEvent(type=StreamEventType.START))
        for raw_choice in _list(chunk.get("choices")):
            choice = _map(raw_choice)
            events += self._delta(_map(choice.get("delta")))
            if choice.get("finish_reason") is not None:
                self._stop = stop_reason(choice.get("finish_reason"))
                events += self._close_calls()
        usage = _map(chunk.get("usage"))
        if usage:
            self._usage = parse_usage(usage)
            events.append(StreamEvent(type=StreamEventType.USAGE, usage=self._usage))
        return events

    def _delta(self, delta: Mapping[str, object]) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        if thinking := reasoning_text(delta):
            self._thinking.append(thinking)
            events.append(StreamEvent(type=StreamEventType.THINKING_DELTA, text=thinking))
        if text := _text(delta.get("content")):
            self._text.append(text)
            events.append(StreamEvent(type=StreamEventType.TEXT_DELTA, text=text))
        for item in _list(delta.get("tool_calls")):
            events += self._call_delta(_map(item))
        return events

    def _call_delta(self, raw: Mapping[str, object]) -> list[StreamEvent]:
        opened = self._open.setdefault(_count(raw.get("index")), _OpenCall())
        function = _map(raw.get("function"))
        opened.call_id = opened.call_id or _text(raw.get("id"))
        opened.name += _text(function.get("name"))
        events: list[StreamEvent] = []
        if not opened.started and opened.call_id and opened.name:
            opened.started = True
            name = from_wire_name(opened.name, self._wire_map)
            events.append(
                StreamEvent(
                    type=StreamEventType.TOOL_CALL_START, call_id=opened.call_id, tool_name=name
                )
            )
        if arguments := _text(function.get("arguments")):
            opened.parts.append(arguments)
            events.append(
                StreamEvent(
                    type=StreamEventType.TOOL_CALL_DELTA,
                    call_id=opened.call_id,
                    input_json_delta=arguments,
                )
            )
        return events

    def _close_calls(self) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        for index in sorted(self._open):
            opened = self._open[index]
            call = ToolCallBlock(
                id=opened.call_id,
                name=from_wire_name(opened.name, self._wire_map),
                input=parse_tool_input("".join(opened.parts)),
            )
            self._calls.append(call)
            events.append(
                StreamEvent(type=StreamEventType.TOOL_CALL_END, call_id=call.id, tool_call=call)
            )
        self._open.clear()
        return events

    def finish(self) -> StreamEvent:
        """The ``end`` event, once the provider has closed the stream."""
        return StreamEvent(type=StreamEventType.END, response=self.response())

    def response(self) -> ModelResponse:
        """Return the assembled response.

        Raises:
            GatewayError: The stream ended before the provider sent a finish reason.
        """
        if self._stop is None:
            raise GatewayError(
                MSG_STREAM_ENDED_EARLY,
                kind=ErrorKind.OVERLOADED,
                provider=self._provider,
                model=self._model,
            )
        content: list[ContentBlock] = []
        if self._thinking:
            content.append(ThinkingBlock(text="".join(self._thinking)))
        if self._text:
            content.append(TextBlock(text="".join(self._text)))
        content += self._calls
        stop = StopReason.TOOL_CALL if self._calls and self._stop is StopReason.END else self._stop
        return ModelResponse(
            content=content, stop_reason=stop, usage=self._usage, model_used=self._model
        )
