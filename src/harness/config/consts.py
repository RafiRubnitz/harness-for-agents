"""Config constants: schema version, defaults referenced by the schema, layer rules."""

from typing import Final

CONFIG_SCHEMA_VERSION: Final = 1

# --- Models / limits (component 1) ---
MODEL_RETRIES_DEFAULT: Final = 3
MODEL_TIMEOUT_MS_DEFAULT: Final = 120_000
BUDGET_PER_AGENT_USD_DEFAULT: Final = 5.0
BUDGET_PER_DAY_USD_DEFAULT: Final = 30.0
BUDGET_WARN_AT_PCT_DEFAULT: Final = 80.0
MODEL_ALIAS_MAIN: Final = "main"
MODEL_ALIAS_FAST: Final = "fast"
MODEL_ALIAS_CHEAP: Final = "cheap"
MODEL_ID_MAIN: Final = "claude-opus-5-5"
MODEL_ID_FAST: Final = "claude-sonnet-5-5"
MODEL_ID_CHEAP: Final = "claude-haiku-5-5"
MODEL_ALIAS_GPT: Final = "gpt"
MODEL_ALIAS_GPT_MINI: Final = "gpt-mini"
MODEL_ALIAS_DEEPSEEK: Final = "deepseek"
MODEL_ALIAS_DEEPSEEK_REASONER: Final = "deepseek-reasoner"
MODEL_ID_GPT: Final = "gpt-5"
MODEL_ID_GPT_MINI: Final = "gpt-5-mini"
MODEL_ID_DEEPSEEK: Final = "deepseek-chat"
MODEL_ID_DEEPSEEK_REASONER: Final = "deepseek-reasoner"
SECRET_ANTHROPIC: Final = "anthropic"
SECRET_OPENAI: Final = "openai"
SECRET_DEEPSEEK: Final = "deepseek"
DEEPSEEK_BASE_URL: Final = "https://api.deepseek.com"

# --- GUI (component 2) ---
GUI_THEME_DEFAULT: Final = "basic-dark"
GUI_FONT_FAMILY_DEFAULT: Final = "Consolas"
GUI_FONT_SIZE_DEFAULT: Final = 13
GUI_WINDOW_WIDTH_DEFAULT: Final = 1280
GUI_WINDOW_HEIGHT_DEFAULT: Final = 800
GUI_SHORTCUTS_DEFAULT: Final = {
    "send": "<Control-Return>",
    "stop": "<Escape>",
    "new_agent": "<Control-n>",
    "next_agent": "<Control-Tab>",
}

# --- Memory (component 3) ---
DEDUP_SIMILARITY_DEFAULT: Final = 0.92
MEMORY_SEARCH_K_DEFAULT: Final = 10
RECENCY_WEIGHT_DEFAULT: Final = 0.1
EMBED_TIMEOUT_S_DEFAULT: Final = 10.0
EMBEDDER_MODEL_DEFAULT: Final = "BAAI/bge-m3"
EMBEDDER_PORT_DEFAULT: Final = 8765

# --- Context (component 7) ---
COMPACTION_THRESHOLD_DEFAULT: Final = 0.8
AUTO_RETRIEVE_K_DEFAULT: Final = 8
SUMMARIZER_MODEL_DEFAULT: Final = MODEL_ALIAS_CHEAP
RESERVE_MIN_TOKENS_DEFAULT: Final = 16_000
RESERVE_PCT_DEFAULT: Final = 0.10
SYSTEM_PCT_DEFAULT: Final = 0.15
SUMMARIES_PCT_DEFAULT: Final = 0.10
MEMORIES_PCT_DEFAULT: Final = 0.05
TAIL_KEEP_PCT_DEFAULT: Final = 0.30
TOOL_RESULT_TRUNCATE_TOKENS_DEFAULT: Final = 8_000
TOOL_RESULT_KEEP_STEPS_DEFAULT: Final = 3

# --- watching the files (06 §7) ---
WATCH_POLL_MS: Final = 250
WATCH_DEBOUNCE_MS: Final = 300
WATCHED_SUFFIXES: Final = frozenset({".yaml", ".yml", ".md"})

# --- Hooks (component 5) ---
HOOK_TIMEOUT_MS_DEFAULT: Final = 5_000
HOOK_ASYNC_TIMEOUT_MS_DEFAULT: Final = 30_000
HOOK_RUNNER_PYTHON: Final = "python"
HOOK_RUNNER_DIRECT: Final = "direct"
HOOK_RUNNERS_DEFAULT: Final = {".py": HOOK_RUNNER_PYTHON, ".exe": HOOK_RUNNER_DIRECT}

# --- Tools (component 8) ---
SHELL_EXECUTABLE_AUTO: Final = "auto"
SHELL_TIMEOUT_S_DEFAULT: Final = 120
WEB_PROVIDER_DEFAULT: Final = "firecrawl"

# --- Messaging between agents and the inbox (phase 2: 03, 02 §5a) ---
MESSAGING_MAX_PER_HOUR_DEFAULT: Final = 30
MESSAGING_MAX_PER_CONVERSATION_DEFAULT: Final = 20
MESSAGING_MAX_AGENT_HOPS_DEFAULT: Final = 5
INBOX_PRIORITIES_DEFAULT: Final = {
    "task": 80,
    "cli": 60,
    "api": 60,
    "agent": 50,
    "hook": 40,
    "schedule": 30,
    "file_watch": 30,
}
INBOX_AGING_S_DEFAULT: Final = 300.0
INBOX_POLL_INTERVAL_S_DEFAULT: Final = 2.0
LOCAL_SERVER_PORT_DEFAULT: Final = 8766
MSG_PRIORITY_USER: Final = "the priority of 'user' is fixed (always the highest)"
MSG_PRIORITY_RANGE: Final = "priority of {source!r} must be between 0 and {highest}"

# --- Layering ---
ENV_PREFIX: Final = "HARNESS_"
ENV_NESTING_SEPARATOR: Final = "__"
ENV_RESERVED: Final = frozenset({"HARNESS_HOME"})
KEY_SEPARATOR: Final = "."
ACCUMULATING_KEYS: Final = frozenset({"hooks.specs", "permissions.rules"})
HOOKS_KEY: Final = "hooks"
HOOK_SPECS_KEY: Final = "specs"
PERMISSIONS_KEY: Final = "permissions"
RULES_KEY: Final = "rules"
RULE_LAYER_FIELD: Final = "layer"
SPEC_SCOPE_FIELD: Final = "scope"
SPEC_SOURCE_FIELD: Final = "source_file"
PROJECTS_TRUSTED_KEY: Final = "projects.trusted"
# Keys of an untrusted project's config.yaml that would run code; they are dropped.
UNTRUSTED_PROJECT_KEYS: Final = frozenset({"hooks", "mcp_servers"})
# String values under these top-level keys are not scanned for ${..} references.
REF_SCAN_SKIP_KEYS: Final = frozenset({"hooks"})
GLOBAL_ONLY_KEYS: Final = frozenset(
    {
        "version",
        "user",
        "providers",
        "limits",
        "gui",
        "embedder",
        "projects",
        "logging",
        "web",
        "local_server",
    }
)
RESTART_KEYS: Final = frozenset(
    {"version", "memory.paths", "embedder", "logging.level", "local_server"}
)

# --- Secrets ---
KEYRING_SERVICE: Final = "harness"
SECRET_REF_PREFIX: Final = "secret:"
ENV_REF_PREFIX: Final = "env:"
REF_PATTERN: Final = r"\$\{([^}]*)\}"
SECRET_MASK_VISIBLE: Final = 3
SECRET_MASK_FILL: Final = "..."

# --- Files ---
YAML_SUFFIX: Final = ".yaml"
FILE_ENCODING: Final = "utf-8"

# --- Message templates ---
MSG_YAML_SYNTAX: Final = "YAML syntax error: {detail}"
MSG_NOT_A_MAPPING: Final = "config file must contain a mapping at the top level"
MSG_BAD_REF: Final = (
    "invalid reference ${{{ref}}} at {key}: use ${{secret:<name>}} or ${{env:<NAME>}}"
)
MSG_SECRET_MISSING: Final = "secret {name!r} is not set in the keyring"
MSG_ENV_MISSING: Final = "environment variable {name!r} is not set"
MSG_VALIDATION: Final = "{key}: {detail}"
MSG_LAYER_NOT_WRITABLE: Final = "config layer {layer!r} cannot be written"
MSG_AGENT_ID_REQUIRED: Final = "agent_id is required for the agent layer"
MSG_GLOBAL_ONLY: Final = "key {key!r} can only be set in the global layer"
MSG_UNKNOWN_KEY: Final = "unknown config key {key!r}"
MSG_TEMPLATE_INVALID: Final = "invalid template {file}: {detail}"
MSG_TEMPLATE_NOT_FOUND: Final = "template {name!r} not found"
MSG_NO_PROJECT: Final = "agent {agent_id} has no project directory"
MSG_HOOKS_FILE_SHAPE: Final = "hooks.yaml must contain only a top-level 'hooks' list"
MSG_PROJECT_UNTRUSTED: Final = (
    "project {root} is not trusted: its hooks, tools and MCP servers were not loaded"
)
MSG_GLOBAL_ONLY_DROPPED: Final = "key {key!r} is global-only and was ignored in the {layer} layer"
MSG_NO_STORE: Final = "no agent layer store is attached"
MSG_NOT_A_SECTION: Final = "{model} is not a config section"
MSG_ENV_VALUE: Final = "environment variable {name} has an unparsable value"
