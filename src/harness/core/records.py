"""Shared data records that cross component boundaries (agents, history, memory, tools, context).

The spec defines these in the component documents; they live in ``core`` so that events,
commands, and hook payloads can reference them without ``core`` importing a component.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from harness.core.consts import (
    DEFAULT_MAX_STEPS,
    DEFAULT_MODEL_ALIAS,
    DEFAULT_TOOL_TIMEOUT_S,
    AgentEventType,
    AgentStatus,
    AskedBy,
    CompactReason,
    Decision,
    DenySource,
    ErrorCode,
    HookEvent,
    McpServerState,
    MemoryKind,
    MemoryScope,
    MemorySource,
    MessageRole,
    PermissionAnswer,
    ProviderId,
    RequestPurpose,
    RiskLevel,
    StopReason,
    TaskStatus,
    ToolCallStatus,
    ToolSource,
)
from harness.core.types import (
    AgentId,
    BackupId,
    ContentBlock,
    MemoryItemId,
    MessageId,
    MessageOrigin,
    ModelRequest,
    PermissionRequestId,
    RunId,
    TaskId,
    ToolCallRecordId,
    Usage,
)

# --- Agents (04 §8, 03 §4) ---


class AgentSettings(BaseModel):
    """Per-agent run settings, fixed at creation from ``agents.defaults``."""

    model_config = ConfigDict(extra="forbid")
    max_steps: int | None = DEFAULT_MAX_STEPS
    max_tokens_per_step: int | None = None
    temperature: float | None = None
    thinking: bool = False
    thinking_budget_tokens: int | None = None
    working_dir: str | None = None
    allowed_paths: list[str] = []


class AgentSpec(BaseModel):
    """What is needed to create an agent."""

    model_config = ConfigDict(extra="forbid")
    name: str
    description: str = ""
    model: str = DEFAULT_MODEL_ALIAS
    system_prompt: str = ""
    tools: list[str] = []
    template: str | None = None
    settings: AgentSettings = AgentSettings()


class AgentState(BaseModel):
    """The agent as the GUI / CLI sees it."""

    model_config = ConfigDict(frozen=True)
    id: AgentId
    name: str
    model: str
    template: str | None
    status: AgentStatus
    settings: AgentSettings
    tools: list[str]
    run_id: RunId | None
    step: int | None
    pending_approvals: int
    background_tasks: int
    cost_usd_total: float
    context_tokens: int | None
    recovered: bool = False
    description: str = ""
    waiting_agent_messages: int = 0
    """Messages from other agents that sit in the inbox and were not read yet."""
    created_at: datetime
    updated_at: datetime


class AgentRecord(BaseModel):
    """The persisted agent row."""

    id: AgentId
    name: str
    description: str = ""
    model: str
    system_prompt: str
    tools: list[str]
    template: str | None = None
    status: AgentStatus
    settings: AgentSettings
    config_overrides: dict[str, JsonValue] = {}
    pending_hook_context: list[str] = []
    created_at: datetime
    updated_at: datetime


# --- History (03 §4) ---


class MessageRecord(BaseModel):
    """One persisted history message."""

    id: MessageId
    agent_id: AgentId
    seq: int
    run_id: RunId | None
    role: MessageRole
    content: list[ContentBlock]
    step: int | None = None
    usage: Usage | None = None
    cost_usd: float | None = None
    stop_reason: str | None = None
    origin: MessageOrigin | None = None
    created_at: datetime


class InboxItem(BaseModel):
    """A received message that was not yet injected into history (no seq)."""

    id: str
    agent_id: AgentId
    content: list[ContentBlock]
    origin: MessageOrigin
    received_at: datetime
    priority: int = 0
    """Set by the core from the source type, never from the content (phase2-03 §7a)."""


class RunRecord(BaseModel):
    """One run: from an inbox message until the model stops calling tools."""

    id: RunId
    agent_id: AgentId
    first_seq: int
    started_at: datetime
    ended_at: datetime | None = None
    step_count: int = 0
    stop_reason: str | None = None
    retrieved_memory_ids: list[MemoryItemId] = []


class ApprovalPreview(BaseModel):
    """What the approval panel shows before a tool runs."""

    diff: str | None = None
    path: str | None = None
    size: int | None = None
    mtime: datetime | None = None
    head: str | None = None
    permanent: bool = False
    command: str | None = None


class PermissionRequestPayload(BaseModel):
    """Payload of ``permission.request``; also persisted as the pending approval."""

    request_id: PermissionRequestId
    subject: HookEvent
    call_id: str | None = None
    tool: str | None = None
    reason: str | None = None
    input: dict[str, JsonValue] = {}
    risk: RiskLevel | None = None
    preview: ApprovalPreview | None = None
    asked_by: list[AskedBy] = []
    hook_reasons: list[str] = []
    summary: str | None = None


class PendingApproval(BaseModel):
    """An open approval request; re-sent after a crash."""

    request_id: PermissionRequestId
    agent_id: AgentId | None
    run_id: RunId | None
    request: PermissionRequestPayload
    created_at: datetime


class BackgroundTaskRecord(BaseModel):
    """A persisted background tool run."""

    id: TaskId
    agent_id: AgentId
    call_id: str
    tool: str
    status: TaskStatus
    started_at: datetime
    ended_at: datetime | None = None
    output_blob: str | None = None
    interrupted: bool = False
    result_summary: str | None = None
    error: str | None = None
    notified_at: datetime | None = None


class AgentEventRecord(BaseModel):
    """A persisted display event that is not a message."""

    agent_id: AgentId
    run_id: RunId | None
    after_seq: int
    type: AgentEventType
    payload: dict[str, JsonValue]
    created_at: datetime


class SummaryRecord(BaseModel):
    """A compaction summary covering ``from_seq..to_seq``."""

    id: str
    agent_id: str
    from_seq: int
    to_seq: int
    text: str
    level: int = 0
    superseded_by: str | None = None
    reason: CompactReason = CompactReason.THRESHOLD
    model_used: str | None = None
    token_count: int | None = None
    created_at: datetime


# --- Semantic memory (03 §4) ---


class MemoryVersion(BaseModel):
    """A previous value of a memory item."""

    content: str
    observed_at: datetime
    replaced_at: datetime
    replaced_by: MemorySource


class MemoryItem(BaseModel):
    """A semantic memory item (agent or global scope)."""

    id: str
    scope: MemoryScope
    agent_id: str | None
    kind: MemoryKind
    content: str
    content_norm: str
    vector: list[float] | None = None
    embed_model: str | None = None
    tags: list[str] = []
    source: MemorySource
    source_msg_seq: int | None = None
    pinned: bool = False
    observed_at: datetime
    created_at: datetime
    updated_at: datetime
    valid_until: datetime | None = None
    superseded_by: str | None = None
    versions: list[MemoryVersion] = []
    last_accessed_at: datetime | None = None
    access_count: int = 0


class ScoredItem(BaseModel):
    """One semantic search hit."""

    item: MemoryItem
    score: float
    vector_rank: int | None
    bm25_rank: int | None


# --- Tools (08 §3, §7) ---


class McpServerStatus(BaseModel):
    """What the user sees about one MCP server."""

    state: McpServerState
    error: str | None = None
    tools: list[str] = []


class ToolDef(BaseModel):
    """What a tool is: its internal name, schema (without injected fields) and policy defaults."""

    model_config = ConfigDict(frozen=True)
    name: str
    description: str
    input_schema: dict[str, JsonValue]
    source: ToolSource
    group: str | None = None
    read_only: bool = False
    reason_required: bool | None = None
    exclusive: bool = False
    risk: RiskLevel = RiskLevel.MEDIUM
    default_permission: Decision = Decision.ASK
    timeout_s: int = DEFAULT_TOOL_TIMEOUT_S


class ToolOutput(BaseModel):
    """The normalized result of a tool run; errors are values, not exceptions."""

    content: list[ContentBlock]
    is_error: bool = False
    metadata: dict[str, JsonValue] = {}


class DeletedFile(BaseModel):
    """A ``delete_file`` backup: where the file was and what happened to it (13 §2.9)."""

    id: BackupId
    agent_id: AgentId | None
    blob_id: str
    original_path: str
    size_bytes: int
    file_mtime: datetime | None = None
    reason: str = ""
    permanent: bool = False
    trashed: bool = False
    deleted_at: datetime
    restored_at: datetime | None = None


class PermissionDecision(BaseModel):
    """Outcome of the permission pipeline (stages 0-2) for one tool call."""

    decision: Decision
    input: dict[str, JsonValue]
    denied_by: DenySource | None = None
    asked_by: list[AskedBy] = []
    hook_reasons: list[str] = []
    hook_context: str | None = None
    preview: ApprovalPreview | None = None
    user_answer: PermissionAnswer | None = None
    deny_message: str | None = None
    error_code: ErrorCode | None = None


class BackgroundTask(BaseModel):
    """Display view of a background task."""

    id: TaskId
    call_id: str
    tool: str
    status: TaskStatus
    started_at: datetime
    duration_ms: int | None


# --- Context (07 §11) ---

SectionName = Literal[
    "system", "profile", "pinned", "tools", "summaries", "tail", "memories", "hook_context"
]


class SectionUsage(BaseModel):
    """Token usage of one context section."""

    name: SectionName
    tokens: int
    items: int
    budget_tokens: int | None


class ContextReport(BaseModel):
    """What ``/context`` shows."""

    model: str
    window: int
    total_tokens: int
    sections: list[SectionUsage]
    cache_breakpoints: list[int]
    compacted_ranges: list[tuple[int, int]]
    retrieved_memory_ids: list[MemoryItemId]
    full_text: str | None = None


class BuiltContext(BaseModel):
    """A model request built for one step, with its report."""

    request: ModelRequest
    report: ContextReport


class ModelCallRecord(BaseModel):
    """One model call in the cost ledger (``model_calls``); ``id`` is the request id."""

    id: str
    agent_id: AgentId | None = None
    run_id: RunId | None = None
    step: int | None = None
    message_id: MessageId | None = None
    purpose: RequestPurpose
    model_alias: str
    model_used: str
    provider: ProviderId
    usage: Usage = Usage()
    cost_usd: float = 0.0
    latency_ms: int | None = None
    stop_reason: StopReason | None = None
    created_at: datetime


class ToolCallRecord(BaseModel):
    """One tool call in the audit log (``tool_calls``); also the recovery state of the call."""

    id: ToolCallRecordId
    agent_id: AgentId
    run_id: RunId | None = None
    step: int | None = None
    call_id: str
    request_message_id: MessageId
    result_message_id: MessageId | None = None
    tool_name: str
    input: dict[str, JsonValue] = {}
    reason: str | None = None
    run_in_background: bool = False
    status: ToolCallStatus = ToolCallStatus.PENDING
    decision: Decision | None = None
    decided_by: str | None = None
    asked_by: list[AskedBy] = []
    result_content: dict[str, JsonValue] | None = None
    is_error: bool | None = None
    metadata: dict[str, JsonValue] = {}
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: int | None = None
