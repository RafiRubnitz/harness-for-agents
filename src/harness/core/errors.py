"""The single HarnessError hierarchy."""

from harness.core.consts import (
    ERROR_CODE_BY_KIND,
    RETRYABLE_KINDS,
    DenySource,
    ErrorCode,
    ErrorKind,
    HookEvent,
)


class HarnessError(Exception):
    """Base class for every error raised by the harness."""

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode,
        recoverable: bool = False,
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.recoverable = recoverable
        self.context: dict[str, object] = context or {}


class ConfigError(HarnessError):
    """Invalid configuration (syntax, validation, missing values)."""

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode = ErrorCode.CONFIG_INVALID,
        file: str | None = None,
        line: int | None = None,
    ) -> None:
        super().__init__(message, code=code, recoverable=True)
        self.file = file
        self.line = line


class SecretMissingError(ConfigError):
    """A ``${secret:x}`` reference has no value in the keyring."""

    def __init__(self, message: str, *, file: str | None = None) -> None:
        super().__init__(message, code=ErrorCode.CONFIG_SECRET_MISSING, file=file)


class GatewayError(HarnessError):
    """A classified model provider failure."""

    def __init__(
        self,
        message: str,
        *,
        kind: ErrorKind,
        provider: str,
        model: str,
        code: ErrorCode | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(
            message,
            code=code or ERROR_CODE_BY_KIND[kind],
            recoverable=True,
        )
        self.kind = kind
        self.retryable = kind in RETRYABLE_KINDS
        self.provider = provider
        self.model = model
        self.retry_after_s = retry_after_s


class BudgetExceededError(GatewayError):
    """The agent or daily budget is exhausted."""

    def __init__(self, message: str, *, provider: str, model: str) -> None:
        super().__init__(
            message,
            kind=ErrorKind.INVALID_REQUEST,
            provider=provider,
            model=model,
            code=ErrorCode.MODEL_BUDGET_EXCEEDED,
        )
        self.retryable = False


class ToolError(HarnessError):
    """A tool failure; never escapes ``ToolsManager.execute``."""

    def __init__(
        self,
        message: str,
        *,
        tool: str = "",
        call_id: str = "",
        code: ErrorCode = ErrorCode.TOOL_FAILED,
    ) -> None:
        super().__init__(message, code=code)
        self.tool = tool
        self.call_id = call_id


class PermissionDenied(ToolError):  # noqa: N818 - name fixed by the spec
    """A tool call was blocked."""

    def __init__(
        self, message: str, *, denied_by: DenySource, tool: str = "", call_id: str = ""
    ) -> None:
        super().__init__(message, tool=tool, call_id=call_id, code=ErrorCode.PERMISSION_DENIED)
        self.denied_by = denied_by


class ReasonMissingError(ToolError):
    """A non-read-only tool call arrived without a ``reason``."""

    def __init__(self, message: str, *, tool: str = "", call_id: str = "") -> None:
        super().__init__(message, tool=tool, call_id=call_id, code=ErrorCode.TOOL_REASON_MISSING)


class HookError(HarnessError):
    """A hook failed to run."""

    def __init__(
        self,
        message: str,
        *,
        hook: str,
        event: HookEvent | None = None,
        code: ErrorCode = ErrorCode.HOOK_FAILED,
    ) -> None:
        super().__init__(message, code=code, recoverable=True)
        self.hook = hook
        self.event = event


class HookTimeoutError(HookError):
    """A hook exceeded its timeout."""

    def __init__(self, message: str, *, hook: str, event: HookEvent | None = None) -> None:
        super().__init__(message, hook=hook, event=event, code=ErrorCode.HOOK_TIMEOUT)


class HookLoadError(HookError):
    """A hook definition could not be loaded."""

    def __init__(self, message: str, *, hook: str, file: str | None = None) -> None:
        super().__init__(message, hook=hook, code=ErrorCode.HOOK_LOAD_FAILED)
        self.file = file


class MemoryStoreError(HarnessError):
    """SQLite / LanceDB failure."""

    def __init__(self, message: str, *, code: ErrorCode = ErrorCode.MEMORY_STORE_FAILED) -> None:
        super().__init__(message, code=code, recoverable=True)


class EmbedderUnavailableError(MemoryStoreError):
    """The managed embeddings server is down or timed out."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code=ErrorCode.MEMORY_EMBEDDER_UNAVAILABLE)


class MemoryItemNotFoundError(MemoryStoreError):
    """A memory item id does not exist."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code=ErrorCode.MEMORY_ITEM_NOT_FOUND)


class AgentError(HarnessError):
    """Agent manager failure."""

    def __init__(self, message: str, *, code: ErrorCode = ErrorCode.AGENT_NOT_FOUND) -> None:
        super().__init__(message, code=code, recoverable=True)


class AgentNotFoundError(AgentError):
    """No agent with the given id."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code=ErrorCode.AGENT_NOT_FOUND)


class InvalidTransitionError(AgentError):
    """A state transition that is not in the transition table."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code=ErrorCode.AGENT_INVALID_TRANSITION)


class ContextError(HarnessError):
    """Context manager failure (e.g. compaction failed)."""

    def __init__(
        self, message: str, *, code: ErrorCode = ErrorCode.CONTEXT_COMPACTION_FAILED
    ) -> None:
        super().__init__(message, code=code, recoverable=True)


class CoreLockedError(HarnessError):
    """Another process holds ``core.lock``."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code=ErrorCode.APP_CORE_LOCKED)


class SlashError(HarnessError):
    """An unknown or unavailable slash command; shown locally, never sent to the model."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code=ErrorCode.SLASH_UNKNOWN, recoverable=True)
