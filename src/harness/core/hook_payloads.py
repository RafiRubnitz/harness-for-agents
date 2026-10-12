"""Hook payload models, one per HookEvent, and the fields ``updated_payload`` may change."""

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from harness.core.consts import (
    CompactReason,
    ErrorKind,
    HookEvent,
    MemoryKind,
    MemoryScope,
    MemorySource,
    RiskLevel,
)
from harness.core.records import AgentSpec, AgentState, ToolOutput
from harness.core.types import (
    ContentBlock,
    MemoryItemId,
    MessageOrigin,
    ModelRequest,
    ModelResponse,
    RunId,
    Usage,
)


class HookPayload(BaseModel):
    """Base of every hook payload."""

    model_config = ConfigDict(extra="forbid")


class AgentCreatedPayload(HookPayload):
    """``agent_created``."""

    spec: AgentSpec


class AgentDeletedPayload(HookPayload):
    """``agent_deleted``."""

    agent: AgentState


class MessageReceivedPayload(HookPayload):
    """``message_received``."""

    content: list[ContentBlock]
    origin: MessageOrigin


class RunStartPayload(HookPayload):
    """``run_start``."""

    run_id: RunId
    content: list[ContentBlock]
    origin: MessageOrigin


class BeforeModelCallPayload(HookPayload):
    """``before_model_call``."""

    request: ModelRequest


class AfterModelCallPayload(HookPayload):
    """``after_model_call``."""

    response: ModelResponse


class ModelErrorPayload(HookPayload):
    """``model_error``."""

    kind: ErrorKind
    retryable: bool
    message: str
    model: str
    attempts: int


class BeforeToolCallPayload(HookPayload):
    """``before_tool_call``; ``input`` excludes the injected fields."""

    tool: str
    call_id: str
    input: dict[str, JsonValue]
    reason: str | None
    read_only: bool
    risk: RiskLevel
    run_in_background: bool


class AfterToolCallPayload(HookPayload):
    """``after_tool_call``."""

    tool: str
    call_id: str
    input: dict[str, JsonValue]
    reason: str | None
    result: ToolOutput


class BeforeMemoryWritePayload(HookPayload):
    """``before_memory_write``."""

    op: Literal["remember", "update", "supersede"]
    item_id: MemoryItemId | None
    content: str
    kind: MemoryKind
    tags: list[str]
    scope: MemoryScope
    source: MemorySource


class ContextCompactPayload(HookPayload):
    """``context_compact``."""

    from_seq: int
    to_seq: int
    reason: CompactReason
    instructions: str | None = None


class RunLimitReachedPayload(HookPayload):
    """``run_limit_reached``."""

    steps: int
    max_steps: int


class RunEndPayload(HookPayload):
    """``run_end``."""

    stop_reason: str
    steps: int
    usage: Usage


PAYLOAD_BY_HOOK_EVENT: Final[dict[HookEvent, type[HookPayload]]] = {
    HookEvent.AGENT_CREATED: AgentCreatedPayload,
    HookEvent.AGENT_DELETED: AgentDeletedPayload,
    HookEvent.MESSAGE_RECEIVED: MessageReceivedPayload,
    HookEvent.RUN_START: RunStartPayload,
    HookEvent.BEFORE_MODEL_CALL: BeforeModelCallPayload,
    HookEvent.AFTER_MODEL_CALL: AfterModelCallPayload,
    HookEvent.MODEL_ERROR: ModelErrorPayload,
    HookEvent.BEFORE_TOOL_CALL: BeforeToolCallPayload,
    HookEvent.AFTER_TOOL_CALL: AfterToolCallPayload,
    HookEvent.BEFORE_MEMORY_WRITE: BeforeMemoryWritePayload,
    HookEvent.CONTEXT_COMPACT: ContextCompactPayload,
    HookEvent.RUN_LIMIT_REACHED: RunLimitReachedPayload,
    HookEvent.RUN_END: RunEndPayload,
}

# Fields a hook's ``updated_payload`` may change; anything else is dropped with a warning.
MUTABLE_PAYLOAD_FIELDS: Final[dict[HookEvent, frozenset[str]]] = {
    HookEvent.MESSAGE_RECEIVED: frozenset({"content"}),
    HookEvent.BEFORE_MODEL_CALL: frozenset({"request"}),
    HookEvent.BEFORE_TOOL_CALL: frozenset({"input"}),
    HookEvent.AFTER_TOOL_CALL: frozenset({"result"}),
    HookEvent.BEFORE_MEMORY_WRITE: frozenset({"content", "kind", "tags"}),
    HookEvent.CONTEXT_COMPACT: frozenset({"instructions"}),
}
