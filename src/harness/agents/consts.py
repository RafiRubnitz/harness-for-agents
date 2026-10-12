"""Agent manager constants: updatable fields, meta keys, messages."""

from typing import Final

COMPONENT_AGENTS: Final = "agents"
COMPONENT_MODELS: Final = "models"
TASK_NAME_PREFIX: Final = "agent-"

# ``harness_meta`` key: False while the core runs, True after an orderly shutdown.
META_CLEAN_SHUTDOWN: Final = "clean_shutdown"

# Fields ``agent.update`` may change.
FIELD_SETTINGS: Final = "settings"
FIELD_TEMPLATE: Final = "template"
FIELD_NAME: Final = "name"
FIELD_DESCRIPTION: Final = "description"
UPDATABLE_FIELDS: Final = frozenset(
    {FIELD_NAME, FIELD_DESCRIPTION, "model", "system_prompt", "tools", FIELD_SETTINGS}
)

# Keys of persisted payloads.
RESULT_CONTENT_KEY: Final = "content"
EVENT_STATUS_KEY: Final = "status"
EVENT_PREVIOUS_KEY: Final = "previous"
DECIDED_BY_USER: Final = "user"
DECIDED_BY_POLICY: Final = "policy"

# ``stop_reason`` of an assistant message cut short by ``agent.stop``.
MESSAGE_STOPPED: Final = "stop"

# --- Texts returned to the model (English, part of the tool protocol) ---
MSG_TOOL_CANCELLED: Final = "The tool call was cancelled because the run was stopped."
MSG_TOOL_INTERRUPTED: Final = (
    "The tool was interrupted by a crash; it is unknown whether it finished. "
    "Check the current state before running it again."
)
MSG_TOOL_CRASHED: Final = "The tool failed unexpectedly: {error}"
MSG_TASK_INTERRUPTED: Final = (
    '<background_task_done id="{task_id}" status="failed">'
    "The task was interrupted by a restart; it is unknown whether it finished."
    "</background_task_done>"
)
TASK_INTERRUPTED_ERROR: Final = "interrupted by a restart"

# --- Error messages (English, internal) ---
MSG_AGENT_NOT_FOUND: Final = "no agent with id {agent_id!r}"
MSG_INVALID_TRANSITION: Final = "agent cannot go from {current} to {target}"
MSG_NOT_ALLOWED_IN: Final = "{action} is not possible while the agent is {status}"
MSG_NO_OPEN_RUN: Final = "agent {agent_id!r} has no open run"
MSG_UNKNOWN_REQUEST: Final = "no open permission request with id {request_id!r}"
MSG_UNKNOWN_FIELDS: Final = "agent.update cannot change: {fields}"
MSG_INVALID_CHANGES: Final = "invalid agent changes: {detail}"
MSG_TEMPLATE_NOT_FOUND: Final = "template {name!r} not found"
MSG_HOOK_DENIED: Final = "{event} was denied by a hook: {reasons}"
MSG_STREAM_INCOMPLETE: Final = "the model stream ended without a response"
MSG_RUN_FAILED: Final = "run failed: {error}"
MSG_NAME_INVALID: Final = (
    "agent name {name!r} is not valid: use lowercase letters, digits and '-' (e.g. code-reviewer)"
)
MSG_NAME_TAKEN: Final = "another agent is already called {name!r}"

# --- Messaging between agents (phase 2, 03) ---
INBOX_WATCH_TASK_NAME: Final = "inbox-watch"
SECONDS_PER_HOUR: Final = 3600.0
FIRST_HOP: Final = 1
MSG_CONVERSATION_LIMIT: Final = (
    "conversation {conversation_id} between agents reached its limit of {limit} messages "
    "and was stopped; message from {sender} to {target} was not delivered"
)
MSG_REPLY_UNKNOWN: Final = "you received no message with the id {message_id!r}"
MSG_REPLY_NOT_AGENT: Final = "message {message_id!r} did not come from an agent; just answer it"
