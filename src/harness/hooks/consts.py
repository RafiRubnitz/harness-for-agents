"""Constants of the hooks manager (05)."""

import re
from typing import Final

from harness.core.consts import Decision, HookEvent, HookOnError

# --- ``run`` forms (05 §6) ---
RUN_PYTHON_PREFIX: Final = "python:"
RUN_PYTHON_PARTS: Final = 2  # <module>:<function>
PYTHON_SUFFIX: Final = ".py"
USER_MODULE_PREFIX: Final = "harness_user_hooks."
MODULE_SEPARATOR: Final = "."

# --- placeholders in a command line (05 §6) ---
PLACEHOLDER: Final = re.compile(r"\{\{|\}\}|\{([^{}]*)\}")
OPEN_BRACE: Final = "{{"
CLOSE_BRACE: Final = "}}"
PLACEHOLDER_PAYLOAD: Final = "payload"
PLACEHOLDER_FIELDS: Final = frozenset({"event", "agent_id", "agent_name", "working_dir"})
PATH_SEPARATOR: Final = "."

# --- match keys (05 §6) ---
MATCH_TOOL: Final = "tool"
MATCH_AGENT: Final = "agent"
MATCH_TEMPLATE: Final = "template"
MATCH_PATH_GLOB: Final = "path_glob"
MATCH_SOURCE_TYPE: Final = "source_type"
MATCH_KEYS: Final = frozenset(
    {MATCH_TOOL, MATCH_AGENT, MATCH_TEMPLATE, MATCH_PATH_GLOB, MATCH_SOURCE_TYPE}
)
PAYLOAD_TOOL: Final = "tool"
PAYLOAD_INPUT: Final = "input"
PAYLOAD_PATH: Final = "path"
PAYLOAD_SPEC: Final = "spec"
PAYLOAD_TEMPLATE: Final = "template"
PAYLOAD_ORIGIN: Final = "origin"
PAYLOAD_SOURCE: Final = "source"
PAYLOAD_TYPE: Final = "type"

# --- merging (05 §4) ---
DECISION_RANK: Final[dict[Decision | None, int]] = {
    None: 0,
    Decision.ALLOW: 1,
    Decision.ASK: 2,
    Decision.DENY: 3,
}
CONTEXT_JOINER: Final = "\n"
DECISION_ON_ERROR: Final[dict[HookOnError, Decision | None]] = {
    HookOnError.DENY: Decision.DENY,
    HookOnError.ASK: Decision.ASK,
    HookOnError.IGNORE: None,
}
# Hooks on these events are usually safety guards: a failing one blocks (05 §7).
DENY_ON_ERROR_EVENTS: Final = frozenset({HookEvent.BEFORE_TOOL_CALL})

# --- running ---
MS_PER_S: Final = 1000
KILL_WAIT_S: Final = 2.0
OUTPUT_ENCODING: Final = "utf-8"
STDERR_SHOWN_CHARS: Final = 500
ENV_IO_ENCODING: Final = "PYTHONIOENCODING"

# --- messages ---
MSG_BAD_PYTHON_REF: Final = "'{run}' is not of the form python:<module>:<function>"
MSG_MODULE_MISSING: Final = "hook module {path} does not exist"
MSG_MODULE_BROKEN: Final = "hook module {path} failed to import: {error}"
MSG_FUNCTION_MISSING: Final = "hook module {path} has no async function '{function}'"
MSG_FILE_MISSING: Final = "hook file {path} does not exist"
MSG_PROGRAM_MISSING: Final = "hook program '{program}' was not found"
MSG_EMPTY_RUN: Final = "the hook has nothing to run"
MSG_NO_RUNNER: Final = "no runner is configured for '{suffix}' files (runner '{runner}')"
MSG_UNKNOWN_MATCH: Final = "unknown match keys: {keys}"
MSG_BAD_COMMAND: Final = "the command line cannot be split into arguments: {error}"
MSG_UNKNOWN_PLACEHOLDER: Final = "unknown placeholder {{{name}}}"
MSG_MISSING_FIELD: Final = "placeholder {{{name}}} names a field the event does not have"
MSG_TIMEOUT: Final = "hook '{hook}' did not answer within {ms} ms"
MSG_RAISED: Final = "hook '{hook}' raised {error}"
MSG_EXIT_CODE: Final = "hook '{hook}' exited with code {code}: {stderr}"
MSG_INVALID_OUTPUT: Final = "hook '{hook}' returned something that is not a HookOutput: {error}"
MSG_INVALID_PAYLOAD: Final = "hook '{hook}' changed the payload into an invalid one: {error}"
MSG_FIELD_DROPPED: Final = "hook '{}' tried to change '{}' of {}, which a hook may not change"
MSG_DUPLICATE_HOOK: Final = "a hook named '{name}' is already registered"
