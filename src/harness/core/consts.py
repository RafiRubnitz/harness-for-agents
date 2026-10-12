"""Shared constants and StrEnums: the single source of truth for cross-component names."""

from enum import StrEnum
from typing import Final


class CommandName(StrEnum):
    """Commands sent from the GUI / CLI to the core."""

    AGENT_CREATE = "agent.create"
    AGENT_UPDATE = "agent.update"
    AGENT_SEND = "agent.send"
    AGENT_STOP = "agent.stop"
    AGENT_PAUSE = "agent.pause"
    AGENT_RESUME = "agent.resume"
    AGENT_ARCHIVE = "agent.archive"
    AGENT_UNARCHIVE = "agent.unarchive"
    AGENT_DELETE = "agent.delete"
    AGENT_RETRY = "agent.retry"
    AGENT_LIST = "agent.list"
    AGENT_LOAD_HISTORY = "agent.load_history"
    PERMISSION_RESPOND = "permission.respond"
    CONTEXT_INSPECT = "context.inspect"
    CONTEXT_COMPACT = "context.compact"
    CONFIG_SET = "config.set"
    CONFIG_SET_SECRET = "config.set_secret"
    PROFILE_UPDATE = "profile.update"
    TEMPLATE_APPLY = "template.apply"
    MEMORY_LIST = "memory.list"
    MEMORY_ADD = "memory.add"
    MEMORY_UPDATE = "memory.update"
    MEMORY_DELETE = "memory.delete"
    TASK_CANCEL = "task.cancel"
    FILE_RESTORE = "file.restore"
    APP_SHUTDOWN = "app.shutdown"


class EventName(StrEnum):
    """Events emitted from the core to the GUI / CLI."""

    AGENT_CREATED = "agent.created"
    AGENT_UPDATED = "agent.updated"
    AGENT_DELETED = "agent.deleted"
    AGENT_LISTED = "agent.listed"
    HISTORY_LOADED = "history.loaded"
    AGENT_STATUS = "agent.status"
    MESSAGE_APPENDED = "message.appended"
    TEXT_DELTA = "text.delta"
    THINKING_DELTA = "thinking.delta"
    TOOL_CALL = "tool.call"
    TOOL_PROGRESS = "tool.progress"
    TOOL_RESULT = "tool.result"
    PERMISSION_REQUEST = "permission.request"
    PERMISSION_RESOLVED = "permission.resolved"
    RUN_LIMIT_REACHED = "run.limit_reached"
    USAGE_UPDATE = "usage.update"
    TASK_STATUS = "task.status"
    CONTEXT_COMPACTED = "context.compacted"
    CONTEXT_REPORT = "context.report"
    MEMORY_LISTED = "memory.listed"
    HOOK_FIRED = "hook.fired"
    HOOK_LOAD_ERROR = "hook.load_error"
    CONFIG_ERROR = "config.error"
    CONFIG_CHANGED = "config.changed"
    WARNING = "warning"
    ERROR = "error"


class HookEvent(StrEnum):
    """Points in the agent lifecycle where hooks can run."""

    AGENT_CREATED = "agent_created"
    AGENT_DELETED = "agent_deleted"
    MESSAGE_RECEIVED = "message_received"
    RUN_START = "run_start"
    BEFORE_MODEL_CALL = "before_model_call"
    AFTER_MODEL_CALL = "after_model_call"
    MODEL_ERROR = "model_error"
    BEFORE_TOOL_CALL = "before_tool_call"
    AFTER_TOOL_CALL = "after_tool_call"
    BEFORE_MEMORY_WRITE = "before_memory_write"
    CONTEXT_COMPACT = "context_compact"
    RUN_LIMIT_REACHED = "run_limit_reached"
    RUN_END = "run_end"
    # Reserved (phase 2 / future): never fired in v1.
    BEFORE_REFLECTION = "before_reflection"
    AFTER_REFLECTION = "after_reflection"
    TRIGGER_FIRED = "trigger_fired"
    PERMISSION_DECIDED = "permission_decided"


class AgentStatus(StrEnum):
    """Agent state machine states."""

    IDLE = "idle"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    PAUSED = "paused"
    ERROR = "error"
    ARCHIVED = "archived"
    REFLECTING = "reflecting"  # reserved (phase 2); no v1 transition reaches it


class ErrorKind(StrEnum):
    """Gateway error classification."""

    RATE_LIMIT = "rate_limit"
    OVERLOADED = "overloaded"
    TIMEOUT = "timeout"
    CONTEXT_OVERFLOW = "context_overflow"
    AUTH = "auth"
    INVALID_REQUEST = "invalid_request"


RETRYABLE_KINDS: Final = frozenset({ErrorKind.RATE_LIMIT, ErrorKind.OVERLOADED, ErrorKind.TIMEOUT})


class DenySource(StrEnum):
    """Who blocked a tool call; written to ``metadata[META_DENIED_BY]``."""

    POLICY = "policy"
    PROTECTED_PATH = "protected_path"
    HOOK = "hook"
    USER = "user"
    UNATTENDED_TIMEOUT = "unattended_timeout"  # reserved (phase 2)
    PERMISSIONS_AGENT = "permissions_agent"  # reserved (future)


class ErrorCode(StrEnum):
    """Stable ``<area>.<code>`` error identifiers."""

    MODEL_RATE_LIMIT = "model.rate_limit"
    MODEL_OVERLOADED = "model.overloaded"
    MODEL_TIMEOUT = "model.timeout"
    MODEL_CONTEXT_OVERFLOW = "model.context_overflow"
    MODEL_AUTH = "model.auth"
    MODEL_INVALID_REQUEST = "model.invalid_request"
    MODEL_BUDGET_EXCEEDED = "model.budget_exceeded"
    MODEL_BUDGET_WARNING = "model.budget_warning"
    MODEL_PROVIDER_UNAVAILABLE = "model.provider_unavailable"
    TOOL_FAILED = "tool.failed"
    TOOL_TIMEOUT = "tool.timeout"
    TOOL_CANCELLED = "tool.cancelled"
    TOOL_INTERRUPTED = "tool.interrupted"
    TOOL_REASON_MISSING = "tool.reason_missing"
    TOOL_INVALID_INPUT = "tool.invalid_input"
    TOOL_NOT_AVAILABLE = "tool.not_available"
    PERMISSION_DENIED = "permission.denied"
    CONFIG_INVALID = "config.invalid"
    CONFIG_SECRET_MISSING = "config.secret_missing"
    MEMORY_STORE_FAILED = "memory.store_failed"
    MEMORY_EMBEDDER_UNAVAILABLE = "memory.embedder_unavailable"
    MEMORY_ITEM_NOT_FOUND = "memory.item_not_found"
    HOOK_FAILED = "hook.failed"
    HOOK_TIMEOUT = "hook.timeout"
    HOOK_INVALID_OUTPUT = "hook.invalid_output"
    HOOK_LOAD_FAILED = "hook.load_failed"
    HOOK_DENIED = "hook.denied"
    AGENT_NOT_FOUND = "agent.not_found"
    AGENT_INVALID_TRANSITION = "agent.invalid_transition"
    CONTEXT_COMPACTION_FAILED = "context.compaction_failed"
    CONTEXT_PINNED_TRIMMED = "context.pinned_trimmed"
    APP_CORE_LOCKED = "app.core_locked"
    PROJECT_UNTRUSTED = "config.project_untrusted"
    SLASH_UNKNOWN = "app.slash_unknown"
    APP_INTERNAL = "app.internal"


ERROR_CODE_BY_KIND: Final = {
    ErrorKind.RATE_LIMIT: ErrorCode.MODEL_RATE_LIMIT,
    ErrorKind.OVERLOADED: ErrorCode.MODEL_OVERLOADED,
    ErrorKind.TIMEOUT: ErrorCode.MODEL_TIMEOUT,
    ErrorKind.CONTEXT_OVERFLOW: ErrorCode.MODEL_CONTEXT_OVERFLOW,
    ErrorKind.AUTH: ErrorCode.MODEL_AUTH,
    ErrorKind.INVALID_REQUEST: ErrorCode.MODEL_INVALID_REQUEST,
}


class SlashCommand(StrEnum):
    """Slash commands typed into an agent's input box."""

    CONTEXT = "/context"
    COMPACT = "/compact"
    REFLECT = "/reflect"  # reserved (phase 2)
    PERMISSIONS = "/permissions"  # reserved (future)


SLASH_PREFIX: Final = "/"
SLASH_CONTEXT_FULL_ARG: Final = "full"
SLASH_RESERVED: Final = frozenset({SlashCommand.REFLECT, SlashCommand.PERMISSIONS})


class Decision(StrEnum):
    """Permission decision shared by hooks, rules, and tool defaults."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class PermissionAnswer(StrEnum):
    """The user's answer to a permission request."""

    APPROVE = "approve"
    DENY = "deny"


class PermissionResolution(StrEnum):
    """How a permission request was closed."""

    APPROVE = "approve"
    DENY = "deny"
    CANCELLED = "cancelled"


class PermissionScope(StrEnum):
    """How long an approval lasts."""

    ONCE = "once"
    AGENT = "agent"


class AskedBy(StrEnum):
    """Why a permission request was raised."""

    POLICY = "policy"
    HOOK = "hook"
    OUTSIDE_WORKING_DIR = "outside_working_dir"
    PERMANENT_DELETE = "permanent_delete"
    RUN_LIMIT = "run_limit"


class RiskLevel(StrEnum):
    """Tool risk level."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ToolSource(StrEnum):
    """Where a tool definition came from."""

    BUILTIN = "builtin"
    CUSTOM = "custom"
    MCP = "mcp"


class McpServerState(StrEnum):
    """The connection state of one MCP server."""

    CONNECTING = "connecting"
    CONNECTED = "connected"
    FAILED = "failed"
    STOPPED = "stopped"


class ReasonRequiredFor(StrEnum):
    """Which tools must carry a non-empty ``reason``."""

    NON_READ_ONLY = "non_read_only"
    ALL = "all"
    NONE = "none"


class TaskStatus(StrEnum):
    """Background task status."""

    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ToolCallStatus(StrEnum):
    """Lifecycle of a persisted tool call row."""

    PENDING = "pending"
    AWAITING_APPROVAL = "awaiting_approval"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    DENIED = "denied"
    INVALID = "invalid"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class MemoryScope(StrEnum):
    """Semantic memory scope."""

    AGENT = "agent"
    GLOBAL = "global"


class MemoryKind(StrEnum):
    """Semantic memory item kind."""

    FACT = "fact"
    PREFERENCE = "preference"
    DECISION = "decision"
    NOTE = "note"
    TODO = "todo"
    LESSON = "lesson"  # reserved (phase 2)


class MemorySource(StrEnum):
    """Who wrote a memory item."""

    AGENT = "agent"
    USER = "user"
    REFLECTION = "reflection"  # written only in phase 2


class SearchMode(StrEnum):
    """Semantic search mode."""

    HYBRID = "hybrid"
    VECTOR = "vector"
    LEXICAL = "lexical"


class ConfigLayer(StrEnum):
    """Config layers, lowest to highest precedence."""

    DEFAULT = "default"
    GLOBAL = "global"
    PROJECT = "project"
    AGENT = "agent"
    ENV = "env"


class HookScope(StrEnum):
    """Where a hook was defined."""

    GLOBAL = "global"
    PROJECT = "project"
    AGENT = "agent"


class HookOnError(StrEnum):
    """What a failing hook resolves to."""

    DENY = "deny"
    ASK = "ask"
    IGNORE = "ignore"


class CompactReason(StrEnum):
    """Why a compaction ran."""

    THRESHOLD = "threshold"
    OVERFLOW = "overflow"
    MANUAL = "manual"


class ProviderId(StrEnum):
    """Supported model providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    DEEPSEEK = "deepseek"


class AddressType(StrEnum):
    """Message source / target kinds."""

    USER = "user"
    TASK = "task"
    SCHEDULE = "schedule"  # reserved (phase 2)
    FILE_WATCH = "file_watch"  # reserved (phase 2)
    CLI = "cli"  # reserved (phase 2)
    HOOK = "hook"  # reserved (phase 2)
    API = "api"  # reserved (future)
    AGENT = "agent"  # reserved (future)


class MessageRole(StrEnum):
    """Role of a persisted history message."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class StopReason(StrEnum):
    """Why a model response ended."""

    END = "end"
    MAX_TOKENS = "max_tokens"
    TOOL_CALL = "tool_call"
    STOP_SEQUENCE = "stop_sequence"
    REFUSAL = "refusal"
    ERROR = "error"


class RunStopReason(StrEnum):
    """Why a run was closed (``RunRecord.stop_reason``)."""

    END = "end"
    STOP = "stop"
    LIMIT = "limit"
    ERROR = "error"
    BUDGET = "budget"
    HOOK_DENIED = "hook_denied"
    ARCHIVED = "archived"
    DELETED = "deleted"


class StreamEventType(StrEnum):
    """Gateway stream event types (internal to gateway <-> agent manager)."""

    START = "start"
    TEXT_DELTA = "text_delta"
    THINKING_DELTA = "thinking_delta"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_END = "tool_call_end"
    USAGE = "usage"
    END = "end"
    ERROR = "error"


class AgentEventType(StrEnum):
    """Persisted display events that are not messages."""

    STATUS = "status"
    COMPACTED = "compacted"
    HOOK_RUN = "hook_run"
    ERROR = "error"
    RECOVERED = "recovered"
    LIMIT_REACHED = "limit_reached"


class BlobKind(StrEnum):
    """Blob store sub-trees."""

    ATTACHMENT = "attachment"
    DELETED_BACKUP = "deleted_backup"
    TASK_OUTPUT = "task_output"
    TOOL_OUTPUT = "tool_output"


class ApprovalStatus(StrEnum):
    """State of a persisted permission request."""

    PENDING = "pending"
    ANSWERED = "answered"
    CANCELLED = "cancelled"


class RequestPurpose(StrEnum):
    """Why a model call was made."""

    STEP = "step"
    COMPACTION = "compaction"


# --- Id prefixes (prefix + ULID) ---
ID_PREFIX_AGENT: Final = "agt_"
ID_PREFIX_RUN: Final = "run_"
ID_PREFIX_TASK: Final = "task_"
ID_PREFIX_CALL: Final = "call_"
ID_PREFIX_MESSAGE: Final = "msg_"
ID_PREFIX_MEMORY: Final = "mem_"
ID_PREFIX_MEMORY_VERSION: Final = "ver_"
ID_PREFIX_SUMMARY: Final = "sum_"
ID_PREFIX_PERMISSION: Final = "perm_"
ID_PREFIX_BACKUP: Final = "bak_"
ID_PREFIX_TOOL_RESULT: Final = "res_"
ID_PREFIX_INBOX: Final = "inb_"
ID_PREFIX_TOOL_CALL_RECORD: Final = "tc_"
ID_PREFIX_REQUEST: Final = "req_"
ID_PREFIX_BLOB: Final = "blob_"
ID_PREFIX_MODEL_CALL: Final = "mc_"
ID_PREFIX_EVENT: Final = "evt_"
ULID_LENGTH: Final = 26

# --- Shared ``metadata`` keys ---
META_DENIED_BY: Final = "denied_by"
META_ERROR_CODE: Final = "error_code"
META_HOOK_CONTEXT: Final = "hook_context"
META_DIFF: Final = "diff"
META_EXIT_CODE: Final = "exit_code"
META_BACKUP_ID: Final = "backup_id"
META_TASK_ID: Final = "task_id"
META_TRUNCATED: Final = "truncated"
META_RESULT_ID: Final = "result_id"

# --- Tool wire names (provider-facing) ---
TOOL_NAME_SEPARATOR: Final = "."
TOOL_WIRE_SEPARATOR: Final = "__"
TOOL_WIRE_MAX_LENGTH: Final = 64
TOOL_WIRE_TRUNCATED_LENGTH: Final = 55
TOOL_WIRE_HASH_LENGTH: Final = 8
TOOL_WIRE_HASH_JOINER: Final = "_"

# --- Injected tool schema fields ---
INJECTED_REASON: Final = "reason"
INJECTED_RUN_IN_BACKGROUND: Final = "run_in_background"
MSG_TASK_DONE: Final = (
    '<background_task_done id="{task_id}" tool="{tool}" status="{status}">'
    "{summary}</background_task_done>"
)
"""The inbox message that tells an agent its background task ended (08 §6a)."""
INJECTED_COLLISION_PREFIX: Final = "_"

# --- Defaults referenced by shared models ---
DEFAULT_MODEL_ALIAS: Final = "main"
DEFAULT_MAX_STEPS: Final = 50
DEFAULT_TOOL_TIMEOUT_S: Final = 120
DEFAULT_USER_ID: Final = "local"

# --- Home directory layout ---
HARNESS_HOME_ENV: Final = "HARNESS_HOME"
HARNESS_HOME_DIRNAME: Final = ".harness"
CONFIG_FILENAME: Final = "config.yaml"
HOOKS_FILENAME: Final = "hooks.yaml"
USER_PROFILE_FILENAME: Final = "user_profile.md"
DB_FILENAME: Final = "harness.db"
CORE_LOCK_FILENAME: Final = "core.lock"
TEMPLATES_DIRNAME: Final = "templates"
HOOKS_DIRNAME: Final = "hooks"
TOOLS_DIRNAME: Final = "tools"
SEMANTIC_DIRNAME: Final = "semantic"
BLOBS_DIRNAME: Final = "blobs"
WORKSPACES_DIRNAME: Final = "workspaces"
LOGS_DIRNAME: Final = "logs"
BACKUPS_DIRNAME: Final = "backups"
THEMES_DIRNAME: Final = "themes"
PROJECT_DIRNAME: Final = ".harness"

# --- Logging ---
LOG_FILENAME: Final = "harness.jsonl"
LOG_ROTATION: Final = "10 MB"
LOG_RETENTION: Final = "14 days"
LOG_LEVEL_DEFAULT: Final = "INFO"
LOG_FILE_LEVEL: Final = "DEBUG"
CONSOLE_FORMAT: Final = (
    "<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <7}</level> | "
    "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)
REDACTED: Final = "***REDACTED***"
SECRET_PATTERNS: Final = (
    r"sk-ant-[A-Za-z0-9_\-]{8,}",
    r"sk-[A-Za-z0-9_\-]{16,}",
    r"fc-[A-Za-z0-9]{16,}",
    r"[Bb]earer\s+[A-Za-z0-9._\-]{16,}",
)

# --- Error message templates (English, internal) ---
MSG_NAIVE_DATETIME: Final = "naive datetime is not allowed; use timezone-aware UTC"
MSG_BAD_ID: Final = "invalid id {value!r}: expected prefix {prefix!r} followed by a ULID"
MSG_SLASH_UNKNOWN: Final = "unknown slash command: {command}"
MSG_SLASH_RESERVED: Final = "slash command {command} is not available in this version"
MSG_CORE_LOCKED: Final = "another harness core is already running (lock: {path})"
MSG_SEMANTIC_MEMORY_UNAVAILABLE: Final = "semantic memory is not available in this build"
MSG_COMMAND_FAILED: Final = "command {command} failed unexpectedly"
COMPONENT_COMMANDS: Final = "commands"
CORE_LOCK_OFFSET: Final = 1 << 30
APP_NAME: Final = "harness"
MSG_CORE_NOT_READY: Final = "the core is still starting"
