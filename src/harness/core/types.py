"""Shared Pydantic types: ids, content blocks, model request / response, small shared types."""

from datetime import datetime
from typing import Annotated, Literal, NewType, Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from harness.core.consts import (
    AddressType,
    ErrorKind,
    ProviderId,
    RequestPurpose,
    StopReason,
    StreamEventType,
)

AgentId = NewType("AgentId", str)
RunId = NewType("RunId", str)
TaskId = NewType("TaskId", str)
CallId = NewType("CallId", str)
MessageId = NewType("MessageId", str)
MemoryItemId = NewType("MemoryItemId", str)
SummaryId = NewType("SummaryId", str)
PermissionRequestId = NewType("PermissionRequestId", str)
BackupId = NewType("BackupId", str)
ToolResultId = NewType("ToolResultId", str)
InboxItemId = NewType("InboxItemId", str)
ToolCallRecordId = NewType("ToolCallRecordId", str)
RequestId = NewType("RequestId", str)
BlobId = NewType("BlobId", str)


class Clock(Protocol):
    """Injected time source; tests use a fake."""

    def now(self) -> datetime:
        """Return the current timezone-aware UTC time."""
        ...

    async def sleep(self, seconds: float) -> None:
        """Wait for ``seconds``."""
        ...


class IdGenerator(Protocol):
    """Injected id source; tests use a deterministic one."""

    def new(self, prefix: str) -> str:
        """Return a new id starting with ``prefix``."""
        ...


class Address(BaseModel):
    """A message endpoint, rendered as ``<type>:<id>``."""

    model_config = ConfigDict(frozen=True)
    type: AddressType
    id: str

    def __str__(self) -> str:
        return f"{self.type.value}:{self.id}"


class MessageOrigin(BaseModel):
    """Where an inbox message came from; set by the core, never by content."""

    model_config = ConfigDict(frozen=True)
    source: Address
    targets: list[Address] = []
    trigger_id: str | None = None
    conversation_id: str | None = None
    hop: int = 0
    reply_to: MessageId | None = None
    message_id: str | None = None
    """The id the sender got back for this message; what a reply refers to."""
    expects_reply: bool = False
    source_name: str | None = None
    """The sender's name when the message was sent (the id in ``source`` never changes)."""


class Source(BaseModel):
    """Payload source of an image / file block."""

    kind: Literal["base64", "url", "blob"]
    media_type: str
    data: str | None = None
    url: str | None = None
    blob_id: str | None = None


class TextBlock(BaseModel):
    """Plain text."""

    type: Literal["text"] = "text"
    text: str


class ImageBlock(BaseModel):
    """An image."""

    type: Literal["image"] = "image"
    source: Source


class FileBlock(BaseModel):
    """A file attachment."""

    type: Literal["file"] = "file"
    source: Source


class ToolCallBlock(BaseModel):
    """A tool call requested by the model; ``input`` is kept exactly as the model sent it."""

    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    input: dict[str, JsonValue]
    reason: str | None = None


class ThinkingBlock(BaseModel):
    """Model reasoning."""

    type: Literal["thinking"] = "thinking"
    text: str
    signature: str | None = None


class ToolResultBlock(BaseModel):
    """The result of one tool call."""

    type: Literal["tool_result"] = "tool_result"
    call_id: str
    content: list["ContentBlock"]
    is_error: bool = False
    result_id: str | None = None


ContentBlock = Annotated[
    TextBlock | ImageBlock | FileBlock | ToolCallBlock | ToolResultBlock | ThinkingBlock,
    Field(discriminator="type"),
]
ToolResultBlock.model_rebuild()


class Message(BaseModel):
    """A message as sent to the model."""

    role: Literal["user", "assistant"]
    content: list[ContentBlock]


class ToolSpec(BaseModel):
    """Provider-neutral tool description, after injected fields were added to the schema."""

    model_config = ConfigDict(frozen=True)
    name: str
    description: str
    input_schema: dict[str, JsonValue]


class InjectedFields(BaseModel):
    """A tool input split into the clean input and the injected fields."""

    input: dict[str, JsonValue]
    reason: str | None
    run_in_background: bool = False


class ThinkingConfig(BaseModel):
    """Extended thinking settings for one request."""

    enabled: bool = True
    budget_tokens: int


class CacheHint(BaseModel):
    """A prompt-caching breakpoint."""

    position: Literal["system", "tools", "messages"]
    index: int | None = None


class RequestMeta(BaseModel):
    """Identifies who issued a model request."""

    request_id: str
    agent_id: AgentId | None = None
    run_id: RunId | None = None
    step: int | None = None
    purpose: RequestPurpose = RequestPurpose.STEP


class ModelRequest(BaseModel):
    """A provider-neutral model request."""

    model: str
    system: str | list[ContentBlock] | None = None
    messages: list[Message]
    tools: list[ToolSpec] | None = None
    tool_choice: Literal["auto", "none", "required"] | dict[str, JsonValue] = "auto"
    max_tokens: int | None = None
    temperature: float | None = None
    thinking: ThinkingConfig | None = None
    response_format: dict[str, JsonValue] | None = None
    cache: list[CacheHint] | None = None
    metadata: RequestMeta


class Usage(BaseModel):
    """Token usage of one model call."""

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input=self.input + other.input,
            output=self.output + other.output,
            cache_read=self.cache_read + other.cache_read,
            cache_write=self.cache_write + other.cache_write,
        )


class ModelResponse(BaseModel):
    """A provider-neutral model response."""

    content: list[ContentBlock]
    stop_reason: StopReason
    usage: Usage
    cost_usd: float = 0.0
    model_used: str
    latency_ms: int = 0


class StreamEvent(BaseModel):
    """One event of a model stream (gateway -> agent manager only)."""

    type: StreamEventType
    text: str | None = None
    call_id: str | None = None
    tool_name: str | None = None
    input_json_delta: str | None = None
    tool_call: ToolCallBlock | None = None
    usage: Usage | None = None
    response: ModelResponse | None = None
    error_kind: ErrorKind | None = None
    message: str | None = None


class Capabilities(BaseModel):
    """What a model supports."""

    tools: bool = True
    images: bool = False
    files: bool = False
    thinking: bool = False
    json_schema: bool = False
    prompt_cache: Literal["explicit", "auto", "none"] = "none"


class ModelPrices(BaseModel):
    """Prices in USD per million tokens."""

    input: float
    output: float
    cache_read: float = 0
    cache_write: float = 0


class ModelProfile(BaseModel):
    """A resolved model: catalog entry plus alias overrides."""

    alias: str | None = None
    provider: ProviderId
    id: str
    context_window: int
    max_output_tokens: int
    capabilities: Capabilities
    prices: ModelPrices
    fallback: list[str] = []
