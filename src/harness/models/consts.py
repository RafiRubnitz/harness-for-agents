"""Model gateway constants."""

from typing import Final

from harness.core.consts import StopReason

CATALOG_FILENAME: Final = "catalog.yaml"
CATALOG_MODELS_KEY: Final = "models"
CALL_LOG_FILENAME: Final = "model_calls.jsonl"
FILE_ENCODING: Final = "utf-8"

TOKENS_PER_MILLION: Final = 1_000_000
MS_PER_SECOND: Final = 1000

# --- Retry / circuit breaker ---
BACKOFF_BASE_S: Final = 1.0
BACKOFF_FACTOR: Final = 2.0
BACKOFF_MAX_S: Final = 30.0
BREAKER_FAILURE_THRESHOLD: Final = 5
BREAKER_COOLDOWN_S: Final = 60.0

# --- Local token estimate (used when the provider has no counting API) ---
ESTIMATE_CHARS_PER_TOKEN: Final = 3.0
ESTIMATE_TOKENS_PER_IMAGE: Final = 1_500
ESTIMATE_TOKENS_PER_MESSAGE: Final = 4

# --- Anthropic wire format ---
ANTHROPIC_DEFAULT_MAX_TOKENS: Final = 4_096
ANTHROPIC_MIN_THINKING_BUDGET: Final = 1_024
ANTHROPIC_CACHE_CONTROL: Final = {"type": "ephemeral"}
ANTHROPIC_STOP_REASONS: Final = {
    "end_turn": StopReason.END,
    "pause_turn": StopReason.END,
    "max_tokens": StopReason.MAX_TOKENS,
    "model_context_window_exceeded": StopReason.MAX_TOKENS,
    "tool_use": StopReason.TOOL_CALL,
    "stop_sequence": StopReason.STOP_SEQUENCE,
    "refusal": StopReason.REFUSAL,
}
ANTHROPIC_TOOL_CHOICE: Final = {
    "auto": {"type": "auto"},
    "none": {"type": "none"},
    "required": {"type": "any"},
}
# Substrings of a 400 body that mean "the prompt does not fit the context window".
ANTHROPIC_OVERFLOW_MARKERS: Final = ("prompt is too long", "context window", "too many tokens")
HTTP_STATUS_OVERLOADED: Final = 529
HTTP_STATUS_SERVER_ERROR_MIN: Final = 500
HTTP_STATUS_TOO_LARGE: Final = 413
HTTP_STATUS_TIMEOUT: Final = 408
RETRY_AFTER_HEADER: Final = "retry-after"

# --- Message templates ---
MSG_UNKNOWN_MODEL: Final = "unknown model {name!r}: not an alias and not in the catalog"
MSG_NO_PROFILE: Final = (
    "model {alias!r} ({provider}/{model}) has no catalog entry and no 'profile' in the config"
)
MSG_NO_ADAPTER: Final = "no adapter is registered for provider {provider!r}"
MSG_BREAKER_OPEN: Final = "provider {provider!r} is temporarily disabled after repeated failures"
MSG_IDLE_TIMEOUT: Final = "no data from the model for {seconds:.0f}s"
MSG_STREAM_ENDED_EARLY: Final = "the model stream ended without a final message"
MSG_BLOB_SOURCE: Final = "blob sources must be resolved before the request reaches an adapter"
MSG_BAD_FALLBACK: Final = "model {alias!r} lists unknown fallback {fallback!r}"
MSG_NO_API_KEY: Final = "no API key is configured for provider {provider!r}"

# --- OpenAI-compatible wire format (Chat Completions; also DeepSeek) ---
OPENAI_STOP_REASONS: Final = {
    "stop": StopReason.END,
    "length": StopReason.MAX_TOKENS,
    "tool_calls": StopReason.TOOL_CALL,
    "function_call": StopReason.TOOL_CALL,
    "content_filter": StopReason.REFUSAL,
}
OPENAI_MAX_TOKENS_KEY: Final = "max_completion_tokens"
OPENAI_LEGACY_MAX_TOKENS_KEY: Final = "max_tokens"
OPENAI_SCHEMA_NAME: Final = "response"
OPENAI_REASONING_KEYS: Final = ("reasoning_content", "reasoning")
# Thinking budgets up to these sizes map to the provider's coarse effort levels.
OPENAI_EFFORT_LOW_MAX_TOKENS: Final = 4_000
OPENAI_EFFORT_MEDIUM_MAX_TOKENS: Final = 16_000
OPENAI_OVERFLOW_CODE: Final = "context_length_exceeded"
OPENAI_QUOTA_CODE: Final = "insufficient_quota"
OPENAI_OVERFLOW_MARKERS: Final = ("maximum context length", "context length", "too many tokens")
DATA_URI: Final = "data:{media_type};base64,{data}"
MSG_FILE_BY_URL: Final = "[file: {url}]"
MSG_TOOL_ERROR_PREFIX: Final = "ERROR: "

# --- Budget ---
PERCENT: Final = 100.0
BUDGET_SCOPE_AGENT: Final = "agent"
BUDGET_SCOPE_DAY: Final = "day"
MSG_BUDGET_EXCEEDED: Final = (
    "the {scope} budget is used up (${used:.2f} of ${limit:.2f}); "
    "raise limits.budget and resume the agent"
)
MSG_BUDGET_WARNING: Final = (
    "{percent:.0f}% of the {scope} budget is used (${used:.2f} of ${limit:.2f})"
)
