"""Tools constants: injected schema fields and messages returned to the model."""

from typing import Final

REASON_MAX_CHARS: Final = 300
REASON_DESCRIPTION: Final = (
    "One short sentence: why you are calling this tool now and what you expect to learn or change."
)
RUN_IN_BACKGROUND_DESCRIPTION: Final = (
    "Run the tool in the background and return a task id immediately."
)

# --- Messages returned to the model (English, part of the tool protocol) ---
MSG_TOOL_NOT_AVAILABLE: Final = "tool {tool!r} is not available to this agent"
MSG_REASON_MISSING: Final = (
    "tool {tool!r} needs a reason: send the call again with a short 'reason' field "
    "saying why you are calling it now"
)
MSG_DENIED_POLICY: Final = "tool {tool!r} is blocked by the permission policy"
MSG_DENIED_PROTECTED: Final = "tool {tool!r} may not touch the protected path {path}"
MSG_DENIED_HOOK: Final = "tool {tool!r} was blocked by a hook: {reasons}"
MSG_RESULT_DENIED_HOOK: Final = "the result of {tool!r} was withheld by a hook: {reasons}"
HOOK_REASON_JOINER: Final = "; "
MSG_DENIED_USER: Final = "the user denied tool {tool!r}"
MSG_DENIED_USER_NOTE: Final = "the user denied tool {tool!r}: {note}"
MSG_TOOL_TIMEOUT: Final = "tool {tool!r} timed out after {seconds} s"
MSG_TOOL_RAISED: Final = "tool {tool!r} failed: {error}"
MSG_BAD_INPUT: Final = "invalid input for tool {tool!r}: {error}"
MSG_RESULT_STORED: Final = (
    "[output of {chars} characters; the full text is stored as {result_id}, "
    "read it with read_tool_result]"
)

# --- Registration ---
TOOL_NAME_PATTERN: Final = r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$"
MCP_NAME_PREFIX: Final = "mcp."
MSG_BAD_TOOL_NAME: Final = "tool name {name!r} must be snake_case without dots or '__'"
MSG_DUPLICATE_TOOL: Final = "tool {name!r} is already registered"
MSG_NOT_ASYNC: Final = "tool {name!r} must be an async function"
CONTEXT_PARAM: Final = "ctx"

# --- Paths ---
PROTECTED_DIR_NAMES: Final = frozenset({".git", ".harness"})
PROTECTED_SYSTEM_DIRS: Final = ("C:/Windows", "C:/Program Files", "C:/Program Files (x86)")
WORKING_DIR_PLACEHOLDER: Final = "{working_dir}"
GLOB_ANY_DEPTH: Final = "**"

# --- Shared tool metadata keys ---
META_PATH: Final = "path"
META_SIZE: Final = "size"
META_TRASHED: Final = "trashed"
META_TIMED_OUT: Final = "timed_out"
META_MATCHES: Final = "matches"

# --- Groups ---
GROUP_FILES: Final = "files"
GROUP_SHELL: Final = "shell"
GROUP_CONTEXT: Final = "context"
GROUP_MEMORY: Final = "memory"
GROUP_TASKS: Final = "tasks"
GROUP_WEB: Final = "web"
GROUP_MESSAGING: Final = "messaging"
ALWAYS_ON_GROUPS: Final = frozenset({GROUP_CONTEXT, GROUP_TASKS, GROUP_MESSAGING})
"""Groups every agent that has at least one tool gets without asking for them."""

# --- Background tasks (08 §6a) ---
META_STATUS: Final = "status"
MS_PER_S: Final = 1000
TASK_SUMMARY_CHARS: Final = 2_000
TASK_OUTPUT_LIMIT: Final = 12_000
TASK_LINE: Final = "{task_id}  {tool}  {status}  {seconds} s"
MSG_TASK_CANCELLED: Final = "the task was cancelled"
MSG_TASK_NOT_FOUND: Final = "you have no background task with the id {task_id!r}"
MSG_NO_TASKS: Final = "no background tasks"

# --- Messaging between agents (phase 2, 03 §4-§5) ---
META_MESSAGE_ID: Final = "message_id"
META_CONVERSATION_ID: Final = "conversation_id"
META_DELIVERED: Final = "delivered"
META_REJECTED: Final = "rejected"
MSG_NO_AGENTS: Final = "there are no other agents"
MSG_NO_TARGETS: Final = "'to' must name at least one agent"

# --- Custom tools (08 §4.2) ---
CUSTOM_TOOL_GLOB: Final = "*.py"
CUSTOM_PRIVATE_PREFIX: Final = "_"
CUSTOM_MODULE_PREFIX: Final = "harness_user_tools_"
MSG_CUSTOM_BROKEN: Final = "{path} could not be imported: {error}"
MSG_CUSTOM_NOT_CUSTOM: Final = "tool {name!r} in a user file must keep the source 'custom'"

# --- Web (08 §4.4) ---
WEB_PROVIDER_FIRECRAWL: Final = "firecrawl"
WEB_SECRET_NAME: Final = "firecrawl"
FIRECRAWL_BASE_URL: Final = "https://api.firecrawl.dev/v2"
FIRECRAWL_SCRAPE: Final = "/scrape"
FIRECRAWL_SEARCH: Final = "/search"
WEB_FORMAT_DEFAULT: Final = "markdown"
WEB_VIA_FETCH: Final = "fetch"
WEB_SCHEMES: Final = frozenset({"http", "https"})
WEB_HTML_MARK: Final = "html"
WEB_USER_AGENT: Final = "harness-for-agents"
WEB_TIMEOUT_S: Final = 60.0
WEB_TOOL_TIMEOUT_S: Final = 90
WEB_FETCH_MAX_CHARS: Final = 2_000_000
WEB_SEARCH_LIMIT: Final = 5
WEB_SEARCH_LIMIT_MAX: Final = 20
WEB_RESULT_LINE: Final = "{title}\n{url}\n{description}"
META_URL: Final = "url"
META_VIA: Final = "via"
MSG_WEB_BAD_URL: Final = "not an http(s) URL: {url}"
MSG_WEB_FAILED: Final = "the web request for {url} failed: {error}"
MSG_WEB_BAD_ANSWER: Final = "the web provider returned an unexpected answer"
MSG_WEB_SEARCH_NEEDS_KEY: Final = (
    "web_search needs a Firecrawl API key (secret 'firecrawl'); web_scrape works without one"
)
MD_SKIPPED_TAGS: Final = frozenset({"script", "style", "noscript", "head", "svg", "template"})
MD_BLOCK_TAGS: Final = frozenset(
    {"p", "div", "section", "article", "ul", "ol", "table", "tr", "blockquote", "header", "footer"}
)
MD_HEADINGS: Final = {f"h{level}": "#" * level for level in range(1, 7)}
MD_INLINE: Final = {"strong": "**", "b": "**", "em": "*", "i": "*", "code": "`"}
MD_BULLET: Final = "- "
MD_FENCE: Final = "```"

# --- MCP servers (08 §4.3) ---
MCP_TOOL_NAME: Final = "mcp.{server}.{tool}"
MCP_CONNECT_TIMEOUT_S: Final = 30
MCP_PING_TIMEOUT_S: Final = 5
MCP_LOG_NAME: Final = "mcp-{server}.log"
MCP_OTHER_CONTENT: Final = "[non-text content]"
MCP_CONFIG_KEY: Final = "mcp_servers"
MSG_MCP_DOWN: Final = "the MCP server {server!r} is not connected ({error})"
MSG_MCP_FAILED: Final = "the MCP tool {tool!r} failed: {error}"
MSG_MCP_LOST: Final = "the connection failed: {error}"
MSG_MCP_TIMEOUT: Final = "no answer within {seconds} s"

# --- Stored results ---
RESULT_READ_CHARS_DEFAULT: Final = 12_000
RESULT_READ_CHARS_MAX: Final = 12_000
META_NEXT_OFFSET: Final = "next_offset"
MSG_RESULT_NOT_FOUND: Final = "no stored tool result has the id {result_id!r}"
MSG_RESULT_PART: Final = "[characters {start}-{end} of {total}]"

# --- File tools ---
READ_LINE_LIMIT_DEFAULT: Final = 2000
READ_MAX_BYTES: Final = 5_000_000
LIST_MAX_ENTRIES: Final = 1000
GLOB_MAX_RESULTS: Final = 1000
GREP_MAX_MATCHES: Final = 500
GREP_MAX_FILE_BYTES: Final = 2_000_000
GREP_LINE_MAX_CHARS: Final = 400
BINARY_PROBE_BYTES: Final = 8192
PREVIEW_HEAD_LINES: Final = 20
DIFF_CONTEXT_LINES: Final = 3
DIFF_MAX_CHARS: Final = 20_000
TEXT_ENCODING: Final = "utf-8"
DIR_SUFFIX: Final = "/"
BACKUP_META_ORIGINAL_PATH: Final = "original_path"
BACKUP_META_REASON: Final = "reason"
BACKUP_META_DELETED_AT: Final = "deleted_at"
BACKUP_META_AGENT_ID: Final = "agent_id"
BACKUP_META_FILENAME: Final = "filename"

MSG_NOT_FOUND: Final = "file not found: {path}"
MSG_IS_DIRECTORY: Final = "path is a directory: {path}"
MSG_NOT_DIRECTORY: Final = "not a directory: {path}"
MSG_BINARY_FILE: Final = "{path} looks like a binary file"
MSG_TOO_LARGE: Final = (
    "{path} is too large to read at once ({size} bytes); use offset/limit or grep"
)
MSG_WROTE: Final = "wrote {path} ({size} bytes)"
MSG_EDITED: Final = "edited {path} ({count} replacement(s))"
MSG_OLD_NOT_FOUND: Final = "old_string was not found in {path}"
MSG_OLD_AMBIGUOUS: Final = (
    "old_string appears {count} times in {path}; add context to make it unique or set replace_all"
)
MSG_SAME_STRING: Final = "old_string and new_string are the same"
MSG_NO_WILDCARDS: Final = "wildcards are not supported; delete one file per call: {path}"
MSG_DELETED: Final = "deleted {path} ({how}); backup_id={backup_id}"
MSG_HOW_TRASHED: Final = "moved to the recycle bin"
MSG_HOW_PERMANENT: Final = "permanently"
MSG_RESTORED: Final = "restored {path}"
MSG_RESTORE_EXISTS: Final = "cannot restore: a file already exists at {path}"
MSG_BACKUP_NOT_FOUND: Final = "no deletion backup with id {backup_id}"
MSG_NO_MATCHES: Final = "no matches"
MSG_EMPTY_DIR: Final = "(empty)"
MSG_BAD_REGEX: Final = "invalid regular expression: {error}"
MSG_MORE_RESULTS: Final = "... ({count} more not shown)"

# --- Shell ---
SHELL_MAX_TIMEOUT_S: Final = 3600
SHELL_CANDIDATES: Final = ("pwsh", "powershell")
SHELL_POSIX_FALLBACK: Final = "sh"
POWERSHELL_ARGS: Final = ("-NoProfile", "-NonInteractive", "-Command")
POSIX_SHELL_ARGS: Final = ("-c",)
TASKKILL: Final = "taskkill"
TASKKILL_ARGS: Final = ("/F", "/T", "/PID")
KILL_WAIT_S: Final = 5
MSG_SHELL_EXIT: Final = "[exit code {code}]"
MSG_SHELL_TIMEOUT: Final = "[killed: no result after {seconds} s]"
MSG_SHELL_NOT_FOUND: Final = "no shell executable found (tried: {tried})"

# --- Memory tools (03 §5) ---
META_ITEM_ID: Final = "item_id"
MEMORY_LINE: Final = "[{date}] ({item_id}, {kind}{tags}) {content}"
MEMORY_TAGS: Final = ", tags: {tags}"
HISTORY_LINE: Final = "#{seq} [{date}] {role}: {text}"
HISTORY_HIT_MAX_CHARS: Final = 600
HISTORY_HIT_CUT: Final = " [...]"
MSG_MEMORY_SAVED: Final = "saved as {item_id}"
MSG_MEMORY_UPDATED: Final = "memory updated; its id is now {item_id}"
MSG_MEMORY_FORGOTTEN: Final = "forgot {item_id}"
MSG_MEMORY_NOT_YOURS: Final = (
    "{item_id} is not one of your own memories; only the user can change it"
)
MSG_NO_MEMORIES: Final = "no memories found"
MSG_NO_HISTORY: Final = "nothing in the conversation history matches"
