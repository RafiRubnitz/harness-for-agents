"""Constants of the desktop GUI: timing, symbols, labels, and limits."""

from enum import StrEnum
from typing import Final

from harness.core.consts import AgentStatus

POLL_INTERVAL_MS: Final = 50
MAX_EVENTS_PER_TICK: Final = 400
RESIZE_DEBOUNCE_MS: Final = 150
BRIDGE_START_TIMEOUT_S: Final = 60.0
BRIDGE_STOP_TIMEOUT_S: Final = 20.0
BRIDGE_THREAD_NAME: Final = "harness-core"
EXIT_OK: Final = 0
EXIT_CORE_LOCKED: Final = 3

TOOL_OUTPUT_MAX_CHARS: Final = 20_000
TOOL_SUMMARY_MAX_CHARS: Final = 80
NOTICE_MAX_CHARS: Final = 4_000
TRUNCATION_MARK: Final = "…"
MS_PER_SECOND: Final = 1000
TOKENS_PER_K: Final = 1000

SIDEBAR_WIDTH: Final = 220
APPROVAL_MAX_DIFF_LINES: Final = 40
INPUT_HEIGHT: Final = 90
STREAM_PADDING: Final = 14
MIN_WRAP_WIDTH_PX: Final = 120
DIALOG_WIDTH: Final = 520
DIALOG_HEIGHT: Final = 730
SETTINGS_WIDTH: Final = 640
SETTINGS_HEIGHT: Final = 560

PROMPT_SYMBOL: Final = "❯ "
TOOL_COLLAPSED: Final = "▸ "
TOOL_EXPANDED: Final = "▾ "
ERROR_SYMBOL: Final = "✗ "
OK_SYMBOL: Final = "✓"
WAITING_SYMBOL: Final = "?"
RUNNING_SYMBOL: Final = "…"
BULLET: Final = "• "
RULE_CHAR: Final = "─"
RULE_WIDTH: Final = 40
QUOTE_PREFIX: Final = "│ "
LIST_INDENT: Final = "  "
TABLE_SEPARATOR: Final = " │ "
THINKING_LABEL: Final = "thinking"
STATUS_DOT: Final = "●"
ORIGIN_LABEL: Final = "[{source}] "
AGENT_ORIGIN_LABEL: Final = "🤖 מ-{name} "
SEPARATOR: Final = " · "


class Region(StrEnum):
    """Parts of the window that a state change can make stale."""

    SIDEBAR = "sidebar"
    STREAM = "stream"
    STATUS = "status"
    APPROVAL = "approval"
    SETTINGS = "settings"


class ToolStatus(StrEnum):
    """Where one tool call stands, as shown in its row."""

    RUNNING = "running"
    WAITING = "waiting"
    OK = "ok"
    FAILED = "failed"


class NoticeLevel(StrEnum):
    """How loud a notice line is."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class Tag(StrEnum):
    """Text tags of the stream; each maps to theme tokens in the widget."""

    PROMPT = "prompt"
    USER = "user"
    MUTED = "muted"
    ERROR = "error"
    WARNING = "warning"
    SUCCESS = "success"
    ACCENT = "accent"
    BOLD = "bold"
    ITALIC = "italic"
    STRIKE = "strike"
    CODE = "code"
    CODE_BLOCK = "code_block"
    LINK = "link"
    QUOTE = "quote"
    HEADING_1 = "h1"
    HEADING_2 = "h2"
    HEADING_3 = "h3"
    MONO = "mono"
    TOOL = "tool"
    RTL = "rtl"
    LTR = "ltr"
    RTL_RUN = "rtl_run"


HEADING_TAGS: Final = (Tag.HEADING_1, Tag.HEADING_2, Tag.HEADING_3)

# Statuses in which a run is in progress, so "stop" makes sense and close asks first.
ACTIVE_STATUSES: Final = frozenset({AgentStatus.RUNNING, AgentStatus.WAITING_APPROVAL})

STATUS_COLOR_TOKENS: Final = {
    AgentStatus.IDLE: "status_idle",
    AgentStatus.RUNNING: "status_running",
    AgentStatus.WAITING_APPROVAL: "status_waiting",
    AgentStatus.PAUSED: "status_waiting",
    AgentStatus.ERROR: "status_error",
    AgentStatus.ARCHIVED: "muted",
    AgentStatus.REFLECTING: "status_running",
}

# --- Labels (the interface language is English; content may be any language) ---
TITLE: Final = "harness"
LABEL_AGENTS: Final = "AGENTS"
LABEL_NEW_AGENT: Final = "+ agent"
LABEL_SETTINGS: Final = "settings"
LABEL_ARCHIVED: Final = "archived"
LABEL_SEND: Final = "send"
LABEL_STOP: Final = "stop"
LABEL_PAUSE: Final = "pause"
LABEL_RESUME: Final = "resume"
LABEL_RETRY: Final = "retry"
LABEL_EDIT: Final = "edit"
LABEL_ARCHIVE: Final = "archive"
LABEL_UNARCHIVE: Final = "unarchive"
LABEL_DELETE: Final = "delete"
LABEL_SAVE: Final = "save"
LABEL_CANCEL: Final = "cancel"
LABEL_CREATE: Final = "create"
LABEL_APPROVE_ONCE: Final = "approve once"
LABEL_APPROVE_ALWAYS: Final = "always for this agent"
LABEL_DENY: Final = "deny"
LABEL_NAME: Final = "name"
LABEL_DESCRIPTION: Final = "description (what other agents see)"
LABEL_MODEL: Final = "model"
LABEL_TEMPLATE: Final = "template"
LABEL_NO_TEMPLATE: Final = "(none)"
LABEL_SYSTEM_PROMPT: Final = "system prompt"
LABEL_TOOLS: Final = "tools"
LABEL_WORKING_DIR: Final = "working directory (empty = private workspace)"
LABEL_ALLOWED_PATHS: Final = "extra allowed paths (one per line)"
LABEL_THEME: Final = "theme"
LABEL_API_KEY: Final = "{provider} API key: {status}"
LABEL_PROVIDER: Final = "provider"
LABEL_TAB_MEMORY: Final = "memory"
LABEL_EMBEDDER_URL: Final = "embeddings server URL (empty = the harness runs its own local server)"
LABEL_PROFILE: Final = "user profile (user_profile.md)"
LABEL_TAB_LOOK: Final = "appearance"
LABEL_TAB_KEYS: Final = "models and keys"
LABEL_TAB_PROFILE: Final = "profile"
LABEL_STOP_AGENTS: Final = "stop them"
LABEL_LEAVE_AGENTS: Final = "leave them to continue next time"
LABEL_COPY: Final = "copy"
LABEL_COPY_ALL: Final = "copy conversation"

MSG_NO_AGENT: Final = "no agent selected: create one with “+ agent”"
MSG_APPROVAL_TITLE: Final = "{agent} asks for approval ({index} of {total})"
MSG_APPROVAL_REASON: Final = "why: {reason}"
MSG_APPROVAL_TOOL: Final = "tool: {tool}   risk: {risk}"
MSG_APPROVAL_SUMMARY: Final = "{summary}"
MSG_APPROVAL_PATH: Final = "path: {path}"
MSG_APPROVAL_COMMAND: Final = "command: {command}"
MSG_APPROVAL_SIZE: Final = "size: {size} bytes"
MSG_APPROVAL_PERMANENT: Final = "PERMANENT delete: the file will not go to the recycle bin"
MSG_APPROVAL_HOOK: Final = "hook: {reason}"
MSG_APPROVAL_INPUT: Final = "input: {input}"
MSG_APPROVAL_DIFF_CUT: Final = "… {lines} more lines"
MSG_RECOVERED: Final = "recovered after an unexpected shutdown; the run continues"
MSG_COMPACTED: Final = "context compacted (messages {from_seq}-{to_seq})"
MSG_LIMIT: Final = "step limit reached ({steps}/{max_steps})"
MSG_HOOK: Final = "hook {hook}: {decision} {reason}"
MSG_ERROR: Final = "{message} [{code}]"
MSG_CONFIG_ERROR: Final = "config error in {file}{line}: {message}"
MSG_HOOK_LOAD_ERROR: Final = "hook {hook} failed to load ({file}): {message}"
MSG_CONFIG_LINE: Final = " line {line}"
MSG_CONTEXT_HEADER: Final = "context: {total} / {window} tokens, model {model}"
MSG_CONTEXT_SECTION: Final = "  {name:<20} {tokens:>8} tokens  {items:>4} items"
MSG_THEME_PROBLEM: Final = "theme {name!r}: {problem}; using {fallback}"
MSG_THEME_NOT_FOUND: Final = "not found"
MSG_THEME_LOOP: Final = "extends itself"
MSG_THEME_INVALID: Final = "invalid file ({detail})"
MSG_CLOSE_TITLE: Final = "agents are still working"
MSG_CLOSE_QUESTION: Final = (
    "{count} agent(s) are in the middle of a run. What should happen to them?"
)
MSG_DELETE_QUESTION: Final = (
    "Delete agent {name} with its history and memory? This cannot be undone."
)
MSG_LOCKED: Final = "harness is already open in another window or terminal.\n\n{detail}"
MSG_START_FAILED: Final = "harness could not start:\n\n{detail}"
MSG_KEY_SAVED: Final = "key saved to the system keyring"
MSG_KEY_SET: Final = "already set ({masked})"
MSG_KEY_NOT_SET: Final = "not set"
MSG_KEY_JUST_SAVED: Final = "already set (just saved)"
MSG_KEY_REPLACE: Final = "type a new key to replace it"
MSG_URL_INVALID: Final = "the URL must start with http:// or https://"
MSG_EMBEDDER_SAVED: Final = "saved; restart the harness to apply"
URL_SCHEMES: Final = ("http://", "https://")
KEY_EMBEDDER_URL: Final = "embedder.url"
MSG_NAME_REQUIRED: Final = "a name is required"
MSG_BLOCK_IMAGE: Final = "[image]"
MSG_BLOCK_FILE: Final = "[file]"

STATUS_MODEL: Final = "{model}"
STATUS_STEP: Final = "step {step}/{max_steps}"
STATUS_STEP_UNBOUNDED: Final = "step {step}"
STATUS_TOKENS: Final = "{used}/{window}"
STATUS_COST: Final = "${cost:.2f}"
STATUS_TASKS: Final = "{count} background task(s)"
STATUS_K: Final = "{value:.1f}k"
TOOL_DURATION: Final = "{seconds:.1f}s"

SHORTCUT_SEND: Final = "send"
SHORTCUT_STOP: Final = "stop"
SHORTCUT_NEW_AGENT: Final = "new_agent"
SHORTCUT_NEXT_AGENT: Final = "next_agent"
SHORTCUT_DEFAULTS: Final = {
    SHORTCUT_SEND: "<Control-Return>",
    SHORTCUT_STOP: "<Escape>",
    SHORTCUT_NEW_AGENT: "<Control-n>",
    SHORTCUT_NEXT_AGENT: "<Control-Tab>",
}
