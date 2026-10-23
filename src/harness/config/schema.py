"""The config schema: every key any component reads, with defaults and validation."""

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from harness.config.consts import (
    AUTO_RETRIEVE_K_DEFAULT,
    BUDGET_PER_AGENT_USD_DEFAULT,
    BUDGET_PER_DAY_USD_DEFAULT,
    BUDGET_WARN_AT_PCT_DEFAULT,
    COMPACTION_THRESHOLD_DEFAULT,
    CONFIG_SCHEMA_VERSION,
    DEDUP_SIMILARITY_DEFAULT,
    DEEPSEEK_BASE_URL,
    EMBED_TIMEOUT_S_DEFAULT,
    EMBEDDER_MODEL_DEFAULT,
    EMBEDDER_PORT_DEFAULT,
    GUI_FONT_FAMILY_DEFAULT,
    GUI_FONT_SIZE_DEFAULT,
    GUI_SHORTCUTS_DEFAULT,
    GUI_THEME_DEFAULT,
    GUI_WINDOW_HEIGHT_DEFAULT,
    GUI_WINDOW_WIDTH_DEFAULT,
    HOOK_ASYNC_TIMEOUT_MS_DEFAULT,
    HOOK_RUNNERS_DEFAULT,
    HOOK_TIMEOUT_MS_DEFAULT,
    INBOX_AGING_S_DEFAULT,
    INBOX_POLL_INTERVAL_S_DEFAULT,
    INBOX_PRIORITIES_DEFAULT,
    LOCAL_SERVER_PORT_DEFAULT,
    MEMORIES_PCT_DEFAULT,
    MEMORY_SEARCH_K_DEFAULT,
    MESSAGING_MAX_AGENT_HOPS_DEFAULT,
    MESSAGING_MAX_PER_CONVERSATION_DEFAULT,
    MESSAGING_MAX_PER_HOUR_DEFAULT,
    MODEL_ALIAS_CHEAP,
    MODEL_ALIAS_DEEPSEEK,
    MODEL_ALIAS_DEEPSEEK_REASONER,
    MODEL_ALIAS_FAST,
    MODEL_ALIAS_GPT,
    MODEL_ALIAS_GPT_MINI,
    MODEL_ALIAS_MAIN,
    MODEL_ID_CHEAP,
    MODEL_ID_DEEPSEEK,
    MODEL_ID_DEEPSEEK_REASONER,
    MODEL_ID_FAST,
    MODEL_ID_GPT,
    MODEL_ID_GPT_MINI,
    MODEL_ID_MAIN,
    MODEL_RETRIES_DEFAULT,
    MODEL_TIMEOUT_MS_DEFAULT,
    MSG_PRIORITY_RANGE,
    MSG_PRIORITY_USER,
    RECENCY_WEIGHT_DEFAULT,
    RESERVE_MIN_TOKENS_DEFAULT,
    RESERVE_PCT_DEFAULT,
    SECRET_ANTHROPIC,
    SECRET_DEEPSEEK,
    SECRET_OPENAI,
    SECRET_REF_PREFIX,
    SHELL_EXECUTABLE_AUTO,
    SHELL_TIMEOUT_S_DEFAULT,
    SUMMARIES_PCT_DEFAULT,
    SUMMARIZER_MODEL_DEFAULT,
    SYSTEM_PCT_DEFAULT,
    TAIL_KEEP_PCT_DEFAULT,
    TOOL_RESULT_KEEP_STEPS_DEFAULT,
    TOOL_RESULT_TRUNCATE_TOKENS_DEFAULT,
    WEB_PROVIDER_DEFAULT,
)
from harness.core.consts import (
    ADDRESS_ANY,
    LOG_LEVEL_DEFAULT,
    USER_PRIORITY,
    AddressType,
    ConfigLayer,
    Decision,
    ErrorCode,
    ProviderId,
    ReasonRequiredFor,
)
from harness.core.hook_types import HookSpec
from harness.core.records import AgentSettings
from harness.core.types import AgentId, Capabilities, ModelPrices


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UserConfig(_Section):
    """``user``: who the local user is (``user:<id>`` in message origins)."""

    id: str = ""


class ProviderConfig(_Section):
    """``providers.<id>``; ``api_key`` is always a ``${secret:..}`` / ``${env:..}`` reference."""

    api_key: str | None = None
    base_url: str | None = None


class ProfileOverride(_Section):
    """``models.<alias>.profile``: override or supply a catalog entry."""

    context_window: int | None = None
    max_output_tokens: int | None = None
    capabilities: Capabilities | None = None
    prices: ModelPrices | None = None


class ModelAliasConfig(_Section):
    """``models.<alias>``."""

    provider: ProviderId
    id: str
    fallback: list[str] = []
    profile: ProfileOverride | None = None


class BudgetConfig(_Section):
    """``limits.budget`` (off by default)."""

    enabled: bool = False
    per_agent_usd: float = BUDGET_PER_AGENT_USD_DEFAULT
    per_day_usd: float = BUDGET_PER_DAY_USD_DEFAULT
    warn_at_pct: float = BUDGET_WARN_AT_PCT_DEFAULT


class LimitsConfig(_Section):
    """``limits``."""

    retries: int = Field(default=MODEL_RETRIES_DEFAULT, ge=0)
    timeout_ms: int = Field(default=MODEL_TIMEOUT_MS_DEFAULT, gt=0)
    budget: BudgetConfig = BudgetConfig()


class FontConfig(_Section):
    """``gui.font``."""

    family: str = GUI_FONT_FAMILY_DEFAULT
    size: int = GUI_FONT_SIZE_DEFAULT


class WindowConfig(_Section):
    """``gui.window``."""

    width: int = GUI_WINDOW_WIDTH_DEFAULT
    height: int = GUI_WINDOW_HEIGHT_DEFAULT


class GuiConfig(_Section):
    """``gui``."""

    theme: str = GUI_THEME_DEFAULT
    font: FontConfig = FontConfig()
    shortcuts: dict[str, str] = dict(GUI_SHORTCUTS_DEFAULT)
    window: WindowConfig = WindowConfig()


class AgentsConfig(_Section):
    """``agents``."""

    defaults: AgentSettings = AgentSettings()


class MemoryPathsConfig(_Section):
    """``memory.paths``; ``None`` means the default location under the harness home."""

    db: str | None = None
    semantic: str | None = None
    blobs: str | None = None


class MemoryConfig(_Section):
    """``memory``."""

    dedup_similarity: float = Field(default=DEDUP_SIMILARITY_DEFAULT, ge=0, le=1)
    search_default_k: int = Field(default=MEMORY_SEARCH_K_DEFAULT, gt=0)
    recency_weight: float = Field(default=RECENCY_WEIGHT_DEFAULT, ge=0)
    embed_timeout_s: float = Field(default=EMBED_TIMEOUT_S_DEFAULT, gt=0)
    paths: MemoryPathsConfig = MemoryPathsConfig()


class EmbedderConfig(_Section):
    """``embedder``; ``url=None`` means the harness starts its own local server."""

    model: str = EMBEDDER_MODEL_DEFAULT
    url: str | None = None
    port: int = EMBEDDER_PORT_DEFAULT


class ContextBudgetConfig(_Section):
    """``context.budget``: shares of the model window."""

    reserve_min_tokens: int = RESERVE_MIN_TOKENS_DEFAULT
    reserve_pct: float = RESERVE_PCT_DEFAULT
    system_pct: float = SYSTEM_PCT_DEFAULT
    summaries_pct: float = SUMMARIES_PCT_DEFAULT
    memories_pct: float = MEMORIES_PCT_DEFAULT
    tail_keep_pct: float = TAIL_KEEP_PCT_DEFAULT


class ToolResultsConfig(_Section):
    """``context.tool_results``."""

    truncate_tokens: int = TOOL_RESULT_TRUNCATE_TOKENS_DEFAULT
    keep_recent_steps: int = TOOL_RESULT_KEEP_STEPS_DEFAULT


class ContextConfig(_Section):
    """``context``."""

    compaction_threshold: float = Field(default=COMPACTION_THRESHOLD_DEFAULT, gt=0, le=1)
    auto_retrieve_k: int = Field(default=AUTO_RETRIEVE_K_DEFAULT, ge=0)
    summarizer_model: str = SUMMARIZER_MODEL_DEFAULT
    budget: ContextBudgetConfig = ContextBudgetConfig()
    tool_results: ToolResultsConfig = ToolResultsConfig()


class HooksConfig(_Section):
    """``hooks``: runner settings plus the accumulated hook specs of every layer."""

    runners: dict[str, str] = dict(HOOK_RUNNERS_DEFAULT)
    timeout_ms: int = HOOK_TIMEOUT_MS_DEFAULT
    async_timeout_ms: int = HOOK_ASYNC_TIMEOUT_MS_DEFAULT
    specs: list[HookSpec] = []


class ReasonConfig(_Section):
    """``tools.reason``."""

    required_for: ReasonRequiredFor = ReasonRequiredFor.NON_READ_ONLY


class ShellConfig(_Section):
    """``tools.shell``."""

    executable: str = SHELL_EXECUTABLE_AUTO
    timeout_s: int = SHELL_TIMEOUT_S_DEFAULT


class ToolsConfig(_Section):
    """``tools``; ``enabled=None`` means every built-in tool."""

    enabled: list[str] | None = None
    reason: ReasonConfig = ReasonConfig()
    shell: ShellConfig = ShellConfig()
    protected_paths_extra: list[str] = []


class Rule(_Section):
    """One permission rule."""

    tool: str
    match: str | None = None
    path: str | None = None
    decision: Decision
    layer: ConfigLayer | None = None  # set while merging layers; not written by users


class PermissionsConfig(_Section):
    """``permissions``; ``rules`` accumulate across layers."""

    default: Decision = Decision.ASK
    rules: list[Rule] = []


class McpServerConfig(_Section):
    """``mcp_servers.<name>``: a stdio command or an HTTP url."""

    command: list[str] | None = None
    url: str | None = None
    env: dict[str, str] = {}

    @model_validator(mode="after")
    def _exactly_one_transport(self) -> "McpServerConfig":
        if (self.command is None) == (self.url is None):
            message = "set exactly one of 'command' or 'url'"
            raise ValueError(message)
        return self


class BackgroundConfig(_Section):
    """``background``."""

    wake_on_done: bool = True


class WebConfig(_Section):
    """``web``."""

    provider: str = WEB_PROVIDER_DEFAULT


class ProjectsConfig(_Section):
    """``projects``: directories whose hooks / tools may be loaded."""

    trusted: list[str] = []


class LoggingConfig(_Section):
    """``logging``."""

    level: str = LOG_LEVEL_DEFAULT


class MessagingConfig(_Section):
    """``messaging``: who may write to an agent, and how much (phase 2, 03 §5-§7).

    ``allow_from`` / ``deny_from`` hold addresses (``agent:coder``, ``agent:*``, ``api:*``)
    or ``*``; a match in ``deny_from`` wins.
    """

    discoverable: bool = True
    allow_from: list[str] = [ADDRESS_ANY]
    deny_from: list[str] = []
    max_per_hour: int = Field(default=MESSAGING_MAX_PER_HOUR_DEFAULT, gt=0)
    max_messages_per_conversation: int = Field(default=MESSAGING_MAX_PER_CONVERSATION_DEFAULT, gt=0)
    max_agent_hops: int = Field(default=MESSAGING_MAX_AGENT_HOPS_DEFAULT, gt=0)


class InboxConfig(_Section):
    """``inbox``: the order waiting messages are handled in (phase 2, 03 §7a).

    ``priorities`` maps a source type to its priority; the user is always the highest.
    ``aging_s=0`` turns aging off.
    """

    priorities: dict[str, int] = dict(INBOX_PRIORITIES_DEFAULT)
    aging_s: float = Field(default=INBOX_AGING_S_DEFAULT, ge=0)
    poll_interval_s: float = Field(default=INBOX_POLL_INTERVAL_S_DEFAULT, gt=0)

    @field_validator("priorities")
    @classmethod
    def _below_the_user(cls, priorities: dict[str, int]) -> dict[str, int]:
        for source, priority in priorities.items():
            if source == AddressType.USER.value:
                raise ValueError(MSG_PRIORITY_USER)
            if not 0 <= priority < USER_PRIORITY:
                raise ValueError(
                    MSG_PRIORITY_RANGE.format(source=source, highest=USER_PRIORITY - 1)
                )
        return priorities

    def priority_of(self, source: AddressType) -> int:
        """Return the priority of a message that came from ``source``."""
        if source is AddressType.USER:
            return USER_PRIORITY
        return self.priorities.get(source.value, INBOX_PRIORITIES_DEFAULT.get(source.value, 0))


class LocalServerConfig(_Section):
    """``local_server``: the HTTP entry point on 127.0.0.1 (phase 2, 02 §5a); off by default."""

    enabled: bool = False
    port: int = Field(default=LOCAL_SERVER_PORT_DEFAULT, gt=0)


def _default_providers() -> dict[str, ProviderConfig]:
    def secret(name: str) -> str:
        return f"${{{SECRET_REF_PREFIX}{name}}}"

    return {
        ProviderId.ANTHROPIC: ProviderConfig(api_key=secret(SECRET_ANTHROPIC)),
        ProviderId.OPENAI: ProviderConfig(api_key=secret(SECRET_OPENAI)),
        ProviderId.DEEPSEEK: ProviderConfig(
            api_key=secret(SECRET_DEEPSEEK), base_url=DEEPSEEK_BASE_URL
        ),
    }


def _default_models() -> dict[str, ModelAliasConfig]:
    return {
        MODEL_ALIAS_MAIN: ModelAliasConfig(
            provider=ProviderId.ANTHROPIC, id=MODEL_ID_MAIN, fallback=[MODEL_ALIAS_FAST]
        ),
        MODEL_ALIAS_FAST: ModelAliasConfig(provider=ProviderId.ANTHROPIC, id=MODEL_ID_FAST),
        MODEL_ALIAS_CHEAP: ModelAliasConfig(provider=ProviderId.ANTHROPIC, id=MODEL_ID_CHEAP),
        MODEL_ALIAS_GPT: ModelAliasConfig(provider=ProviderId.OPENAI, id=MODEL_ID_GPT),
        MODEL_ALIAS_GPT_MINI: ModelAliasConfig(provider=ProviderId.OPENAI, id=MODEL_ID_GPT_MINI),
        MODEL_ALIAS_DEEPSEEK: ModelAliasConfig(provider=ProviderId.DEEPSEEK, id=MODEL_ID_DEEPSEEK),
        MODEL_ALIAS_DEEPSEEK_REASONER: ModelAliasConfig(
            provider=ProviderId.DEEPSEEK, id=MODEL_ID_DEEPSEEK_REASONER
        ),
    }


class HarnessConfig(_Section):
    """The whole resolved configuration."""

    version: int = CONFIG_SCHEMA_VERSION
    user: UserConfig = UserConfig()
    providers: dict[str, ProviderConfig] = Field(default_factory=_default_providers)
    models: dict[str, ModelAliasConfig] = Field(default_factory=_default_models)
    limits: LimitsConfig = LimitsConfig()
    gui: GuiConfig = GuiConfig()
    agents: AgentsConfig = AgentsConfig()
    memory: MemoryConfig = MemoryConfig()
    embedder: EmbedderConfig = EmbedderConfig()
    context: ContextConfig = ContextConfig()
    hooks: HooksConfig = HooksConfig()
    tools: ToolsConfig = ToolsConfig()
    permissions: PermissionsConfig = PermissionsConfig()
    mcp_servers: dict[str, McpServerConfig] = {}
    background: BackgroundConfig = BackgroundConfig()
    web: WebConfig = WebConfig()
    projects: ProjectsConfig = ProjectsConfig()
    logging: LoggingConfig = LoggingConfig()
    messaging: MessagingConfig = MessagingConfig()
    inbox: InboxConfig = InboxConfig()
    local_server: LocalServerConfig = LocalServerConfig()


class ResolvedValue(BaseModel):
    """One resolved value and the layer it came from."""

    value: JsonValue
    layer: ConfigLayer
    file: str | None = None


class ResolvedConfig(BaseModel):
    """Dotted key -> value and source, for the GUI."""

    agent_id: AgentId | None
    values: dict[str, ResolvedValue]


class ConfigIssue(BaseModel):
    """A validation finding returned as data (not the ``ConfigError`` exception)."""

    file: str | None
    line: int | None
    key: str | None
    code: ErrorCode
    message: str


class AgentTemplate(BaseModel):
    """``templates/<name>.yaml``; copied into the agent layer at creation."""

    model_config = ConfigDict(extra="forbid")
    name: str
    description: str = ""
    model: str = MODEL_ALIAS_MAIN
    system_prompt: str
    tools: list[str] = []
    settings: dict[str, JsonValue] = {}
    hooks: list[HookSpec] = []
    permissions: dict[str, JsonValue] = {}
    config: dict[str, JsonValue] = {}


SECTION_KEYS: dict[type[BaseModel], tuple[str, ...]] = {
    HarnessConfig: (),
    UserConfig: ("user",),
    LimitsConfig: ("limits",),
    BudgetConfig: ("limits", "budget"),
    GuiConfig: ("gui",),
    AgentsConfig: ("agents",),
    AgentSettings: ("agents", "defaults"),
    MemoryConfig: ("memory",),
    EmbedderConfig: ("embedder",),
    ContextConfig: ("context",),
    HooksConfig: ("hooks",),
    ToolsConfig: ("tools",),
    PermissionsConfig: ("permissions",),
    BackgroundConfig: ("background",),
    WebConfig: ("web",),
    ProjectsConfig: ("projects",),
    LoggingConfig: ("logging",),
    MessagingConfig: ("messaging",),
    InboxConfig: ("inbox",),
    LocalServerConfig: ("local_server",),
}
