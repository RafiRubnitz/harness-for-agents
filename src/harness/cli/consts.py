"""Constants of the command line interface."""

from typing import Final

from harness.core.consts import AgentStatus

PROG: Final = "harness"
CMD_CHAT: Final = "chat"
CMD_AGENTS: Final = "agents"

EXIT_OK: Final = 0
EXIT_ERROR: Final = 1
EXIT_CORE_LOCKED: Final = 3
EXIT_INTERRUPTED: Final = 130

DEFAULT_AGENT_NAME: Final = "chat"
DEFAULT_AGENT_TOOLS: Final = ("files", "shell")
PROMPT: Final = "> "
APPROVAL_PROMPT: Final = "[y] once / [a] always for this agent / [n] deny: "
ANSWER_ONCE: Final = "y"
ANSWER_ALWAYS: Final = "a"
ANSWER_DENY: Final = "n"
LOCAL_EXIT: Final = frozenset({"/exit", "/quit"})
LOCAL_RETRY: Final = "/retry"

# Statuses in which the agent is waiting for the user, so the prompt comes back.
RESTING: Final = frozenset(
    {AgentStatus.IDLE, AgentStatus.ERROR, AgentStatus.PAUSED, AgentStatus.ARCHIVED}
)

ANSI_RESET: Final = "\x1b[0m"
ANSI_RED: Final = "\x1b[31m"
ANSI_YELLOW: Final = "\x1b[33m"
ANSI_DIM: Final = "\x1b[2m"
OUTPUT_ENCODING: Final = "utf-8"
INPUT_PREVIEW_CHARS: Final = 2000

HELP_DESCRIPTION: Final = "Local multi-agent harness."
HELP_CHAT: Final = "talk to one agent in the terminal"
HELP_AGENTS: Final = "list agents and their status"
HELP_AGENT_ARG: Final = "agent name or id (default: the first agent, or a new one)"
HELP_NEW: Final = "create a new agent"
HELP_TEMPLATE: Final = "template for the new agent"
HELP_MODEL: Final = "model alias for the new agent"
HELP_ARCHIVED: Final = "include archived agents"

MSG_LOCKED: Final = "harness is already open (GUI or another CLI); close it or use it. {detail}"
MSG_AGENT_NOT_FOUND: Final = "no agent named {name!r}; use --new to create it"
MSG_CORE_STOPPED: Final = "the core stopped before answering"
MSG_CHATTING: Final = "agent {name} ({agent_id}), model {model}. /exit to quit."
MSG_ERROR: Final = "error [{code}]: {message}"
MSG_WARNING: Final = "warning [{code}]: {message}"
MSG_RETRY_HINT: Final = "the run stopped on an error; type /retry or send a new message"
MSG_PAUSED_HINT: Final = "the run is paused"
MSG_LIMIT: Final = "step limit reached ({steps}/{max_steps})"
MSG_NO_AGENTS: Final = "no agents"
MSG_TOOL_CALL: Final = "[tool] {tool} {input}"
MSG_TOOL_REASON: Final = "       reason: {reason}"
MSG_TOOL_RESULT: Final = "[{status}] {tool} ({duration_ms} ms)"
MSG_TOOL_OK: Final = "ok"
MSG_TOOL_FAILED: Final = "failed"
MSG_APPROVAL_HEADER: Final = "--- approval needed ---"
MSG_APPROVAL_FIELD: Final = "{label}: {value}"
MSG_COMPACTED: Final = "context compacted (messages {from_seq}-{to_seq})"
MSG_CONTEXT_HEADER: Final = "context: {total} / {window} tokens, model {model}"
MSG_CONTEXT_SECTION: Final = "  {name:<20} {tokens:>8} tokens  {items:>4} items"
MSG_AGENT_ROW: Final = "{agent_id}  {status:<17} {model:<12} {name}"

LABEL_REASON: Final = "reason"
LABEL_TOOL: Final = "tool"
LABEL_INPUT: Final = "input"
LABEL_SUMMARY: Final = "summary"
LABEL_COMMAND: Final = "command"
LABEL_PATH: Final = "path"
LABEL_DIFF: Final = "diff"
LABEL_HOOKS: Final = "hooks"
LABEL_PERMANENT: Final = "permanent delete"
