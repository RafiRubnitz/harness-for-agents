"""Config manager (component 6): layered settings, secrets, templates.

Public surface: the ``ConfigManager`` protocol, its file-backed implementation, and the
schema models other components read through ``section()``.
"""

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, JsonValue

from harness.config import consts
from harness.config.loader import template_overrides
from harness.config.manager import AgentLayerStore, FileConfigManager, requires_restart
from harness.config.schema import (
    AgentsConfig,
    AgentTemplate,
    BackgroundConfig,
    BudgetConfig,
    ConfigIssue,
    ContextBudgetConfig,
    ContextConfig,
    EmbedderConfig,
    GuiConfig,
    HarnessConfig,
    HooksConfig,
    InboxConfig,
    LimitsConfig,
    LocalServerConfig,
    LoggingConfig,
    McpServerConfig,
    MemoryConfig,
    MessagingConfig,
    ModelAliasConfig,
    PermissionsConfig,
    ProfileOverride,
    ProjectsConfig,
    ProviderConfig,
    ResolvedConfig,
    ResolvedValue,
    Rule,
    ToolsConfig,
    UserConfig,
    WebConfig,
)
from harness.config.secrets import KeyringSecretStore, SecretStore, mask
from harness.config.watcher import ConfigWatcher
from harness.core.consts import ConfigLayer
from harness.core.types import AgentId


class ConfigManager(Protocol):
    """The only way any component reads or writes settings and secrets."""

    def config(self, agent_id: AgentId | None = None) -> HarnessConfig:
        """Return the whole resolved config."""
        ...

    def get(self, key: str, agent_id: AgentId | None = None) -> JsonValue:
        """Return one value by dotted key."""
        ...

    def section[T: BaseModel](self, model: type[T], agent_id: AgentId | None = None) -> T:
        """Return one section as its Pydantic model."""
        ...

    def resolved(self, agent_id: AgentId | None = None) -> ResolvedConfig:
        """Return every value with its source layer."""
        ...

    async def set(
        self,
        key: str,
        value: JsonValue,
        layer: ConfigLayer,
        agent_id: AgentId | None = None,
    ) -> None:
        """Validate and persist one value."""
        ...

    async def set_agent_layer(self, agent_id: AgentId, overrides: dict[str, JsonValue]) -> None:
        """Replace an agent's whole layer."""
        ...

    async def load_agent(self, agent_id: AgentId, working_dir: Path | None) -> None:
        """Load an agent's layer and project."""
        ...

    def forget_agent(self, agent_id: AgentId) -> None:
        """Drop a deleted agent's layer."""
        ...

    def project_root(self, agent_id: AgentId) -> Path | None:
        """Return the project directory that applies to an agent."""
        ...

    def is_trusted(self, root: Path) -> bool:
        """Return whether a project directory is trusted."""
        ...

    def subscribe(self, keys: list[str], callback: Callable[[list[str]], Awaitable[None]]) -> None:
        """Subscribe to changes under ``keys``."""
        ...

    def subscribe_errors(self, callback: Callable[[list[ConfigIssue]], Awaitable[None]]) -> None:
        """Subscribe to rejected reloads."""
        ...

    def secret(self, name: str) -> str:
        """Return a secret from the keyring."""
        ...

    async def set_secret(self, name: str, value: str) -> None:
        """Store a secret in the keyring."""
        ...

    def expand(self, value: str) -> str:
        """Resolve ``${secret:..}`` / ``${env:..}`` references."""
        ...

    def templates(self) -> list[AgentTemplate]:
        """Return the agent templates."""
        ...

    def user_profile(self) -> str:
        """Return ``user_profile.md``."""
        ...

    async def set_user_profile(self, text: str) -> None:
        """Replace ``user_profile.md``."""
        ...

    async def reload(self) -> None:
        """Re-read every config file."""
        ...

    def validate(self) -> list[ConfigIssue]:
        """Check the files on disk."""
        ...

    def warnings(self) -> list[ConfigIssue]:
        """Return non-blocking findings of the active config."""
        ...


__all__ = [
    "AgentLayerStore",
    "AgentTemplate",
    "AgentsConfig",
    "BackgroundConfig",
    "BudgetConfig",
    "ConfigIssue",
    "ConfigManager",
    "ConfigWatcher",
    "ContextBudgetConfig",
    "ContextConfig",
    "EmbedderConfig",
    "FileConfigManager",
    "GuiConfig",
    "HarnessConfig",
    "HooksConfig",
    "InboxConfig",
    "KeyringSecretStore",
    "LimitsConfig",
    "LocalServerConfig",
    "LoggingConfig",
    "McpServerConfig",
    "MemoryConfig",
    "MessagingConfig",
    "ModelAliasConfig",
    "PermissionsConfig",
    "ProfileOverride",
    "ProjectsConfig",
    "ProviderConfig",
    "ResolvedConfig",
    "ResolvedValue",
    "Rule",
    "SecretStore",
    "ToolsConfig",
    "UserConfig",
    "WebConfig",
    "consts",
    "mask",
    "requires_restart",
    "template_overrides",
]
