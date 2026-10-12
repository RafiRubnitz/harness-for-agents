"""Event envelope and payload models: everything the core tells the GUI / CLI."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, JsonValue

from harness.core.consts import (
    AgentStatus,
    Decision,
    ErrorCode,
    ErrorKind,
    EventName,
    HookEvent,
    PermissionResolution,
    PermissionScope,
    TaskStatus,
)
from harness.core.records import (
    AgentEventRecord,
    AgentState,
    ContextReport,
    MemoryItem,
    MessageRecord,
    PermissionRequestPayload,
)
from harness.core.types import (
    AgentId,
    ContentBlock,
    PermissionRequestId,
    TaskId,
    Usage,
)


class _Payload(BaseModel):
    model_config = ConfigDict(frozen=True)


class AgentCreated(_Payload):
    """``agent.created``."""

    agent: AgentState


class AgentUpdated(_Payload):
    """``agent.updated``."""

    agent: AgentState


class AgentDeleted(_Payload):
    """``agent.deleted``."""

    agent_id: AgentId


class AgentListed(_Payload):
    """``agent.listed`` (reply to ``agent.list``)."""

    agents: list[AgentState]


class HistoryLoaded(_Payload):
    """``history.loaded`` (reply to ``agent.load_history``)."""

    messages: list[MessageRecord]
    events: list[AgentEventRecord]
    from_seq: int


class AgentStatusChanged(_Payload):
    """``agent.status``."""

    status: AgentStatus
    previous: AgentStatus
    recovered: bool = False


class MessageAppended(_Payload):
    """``message.appended``."""

    message: MessageRecord


class TextDelta(_Payload):
    """``text.delta``."""

    text: str


class ThinkingDelta(_Payload):
    """``thinking.delta``."""

    text: str


class ToolCall(_Payload):
    """``tool.call``."""

    call_id: str
    tool: str
    input: dict[str, JsonValue]
    reason: str | None
    background: bool = False


class ToolProgress(_Payload):
    """``tool.progress``."""

    call_id: str
    task_id: TaskId | None = None
    chunk: str


class ToolResult(_Payload):
    """``tool.result``."""

    call_id: str
    tool: str
    reason: str | None
    content: list[ContentBlock]
    is_error: bool
    duration_ms: int
    metadata: dict[str, JsonValue] = {}


class PermissionRequest(_Payload):
    """``permission.request``."""

    request: PermissionRequestPayload


class PermissionResolved(_Payload):
    """``permission.resolved``."""

    request_id: PermissionRequestId
    decision: PermissionResolution
    scope: PermissionScope | None = None


class RunLimitReached(_Payload):
    """``run.limit_reached``."""

    steps: int
    max_steps: int


class UsageUpdate(_Payload):
    """``usage.update``."""

    usage: Usage
    cost_usd_total: float
    context_tokens: int
    context_window: int
    context_pct: float
    model: str


class TaskStatusChanged(_Payload):
    """``task.status``."""

    task_id: TaskId
    call_id: str
    tool: str
    status: TaskStatus
    duration_ms: int


class ContextCompacted(_Payload):
    """``context.compacted``."""

    from_seq: int
    to_seq: int
    summary_id: str


class ContextReported(_Payload):
    """``context.report`` (reply to ``context.inspect``)."""

    report: ContextReport
    full_text: str | None = None


class MemoryListed(_Payload):
    """``memory.listed`` (reply to ``memory.list``)."""

    items: list[MemoryItem]


class HookFired(_Payload):
    """``hook.fired``."""

    hook: str
    event: HookEvent
    decision: Decision | None
    reason: str | None
    duration_ms: int


class HookLoadFailed(_Payload):
    """``hook.load_error``."""

    hook: str
    file: str
    message: str


class ConfigErrorRaised(_Payload):
    """``config.error``."""

    file: str
    line: int | None
    message: str


class ConfigChanged(_Payload):
    """``config.changed``."""

    keys: list[str]
    restart_required: bool = False


class Warning(_Payload):  # noqa: A001 - event name fixed by the spec
    """``warning``."""

    code: ErrorCode
    message: str


class Error(_Payload):
    """``error``."""

    code: ErrorCode
    kind: ErrorKind | None = None
    message: str
    recoverable: bool
    component: str
    call_id: str | None = None


EventPayload = (
    AgentCreated
    | AgentUpdated
    | AgentDeleted
    | AgentListed
    | HistoryLoaded
    | AgentStatusChanged
    | MessageAppended
    | TextDelta
    | ThinkingDelta
    | ToolCall
    | ToolProgress
    | ToolResult
    | PermissionRequest
    | PermissionResolved
    | RunLimitReached
    | UsageUpdate
    | TaskStatusChanged
    | ContextCompacted
    | ContextReported
    | MemoryListed
    | HookFired
    | HookLoadFailed
    | ConfigErrorRaised
    | ConfigChanged
    | Warning
    | Error
)

PAYLOAD_BY_EVENT: dict[EventName, type[BaseModel]] = {
    EventName.AGENT_CREATED: AgentCreated,
    EventName.AGENT_UPDATED: AgentUpdated,
    EventName.AGENT_DELETED: AgentDeleted,
    EventName.AGENT_LISTED: AgentListed,
    EventName.HISTORY_LOADED: HistoryLoaded,
    EventName.AGENT_STATUS: AgentStatusChanged,
    EventName.MESSAGE_APPENDED: MessageAppended,
    EventName.TEXT_DELTA: TextDelta,
    EventName.THINKING_DELTA: ThinkingDelta,
    EventName.TOOL_CALL: ToolCall,
    EventName.TOOL_PROGRESS: ToolProgress,
    EventName.TOOL_RESULT: ToolResult,
    EventName.PERMISSION_REQUEST: PermissionRequest,
    EventName.PERMISSION_RESOLVED: PermissionResolved,
    EventName.RUN_LIMIT_REACHED: RunLimitReached,
    EventName.USAGE_UPDATE: UsageUpdate,
    EventName.TASK_STATUS: TaskStatusChanged,
    EventName.CONTEXT_COMPACTED: ContextCompacted,
    EventName.CONTEXT_REPORT: ContextReported,
    EventName.MEMORY_LISTED: MemoryListed,
    EventName.HOOK_FIRED: HookFired,
    EventName.HOOK_LOAD_ERROR: HookLoadFailed,
    EventName.CONFIG_ERROR: ConfigErrorRaised,
    EventName.CONFIG_CHANGED: ConfigChanged,
    EventName.WARNING: Warning,
    EventName.ERROR: Error,
}

EVENT_BY_PAYLOAD: dict[type[BaseModel], EventName] = {v: k for k, v in PAYLOAD_BY_EVENT.items()}


class Event(BaseModel):
    """The envelope every event travels in."""

    model_config = ConfigDict(frozen=True)
    name: EventName
    agent_id: AgentId | None
    seq: int
    ts: datetime
    payload: EventPayload
