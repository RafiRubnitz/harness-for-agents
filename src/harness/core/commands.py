"""Command models: everything the GUI / CLI can ask the core to do."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from harness.core.consts import (
    DEFAULT_MODEL_ALIAS,
    CommandName,
    ConfigLayer,
    MemoryKind,
    PermissionAnswer,
    PermissionScope,
)
from harness.core.records import AgentSettings
from harness.core.types import (
    AgentId,
    BackupId,
    ContentBlock,
    MemoryItemId,
    PermissionRequestId,
    TaskId,
)


class _Command(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentCreate(_Command):
    """Create an agent (optionally from a template)."""

    name: Literal[CommandName.AGENT_CREATE] = CommandName.AGENT_CREATE
    agent_name: str
    description: str = ""
    model: str = DEFAULT_MODEL_ALIAS
    system_prompt: str = ""
    tools: list[str] = []
    template: str | None = None
    settings: AgentSettings | None = None


class AgentUpdate(_Command):
    """Change an agent's name, description, model, prompt, tools, or settings."""

    name: Literal[CommandName.AGENT_UPDATE] = CommandName.AGENT_UPDATE
    agent_id: AgentId
    changes: dict[str, JsonValue]


class AgentSend(_Command):
    """Send a user message to an agent."""

    name: Literal[CommandName.AGENT_SEND] = CommandName.AGENT_SEND
    agent_id: AgentId
    content: list[ContentBlock]


class AgentStop(_Command):
    """Stop the current run."""

    name: Literal[CommandName.AGENT_STOP] = CommandName.AGENT_STOP
    agent_id: AgentId


class AgentPause(_Command):
    """Pause between steps."""

    name: Literal[CommandName.AGENT_PAUSE] = CommandName.AGENT_PAUSE
    agent_id: AgentId


class AgentResume(_Command):
    """Resume a paused run."""

    name: Literal[CommandName.AGENT_RESUME] = CommandName.AGENT_RESUME
    agent_id: AgentId


class AgentArchive(_Command):
    """Archive an agent."""

    name: Literal[CommandName.AGENT_ARCHIVE] = CommandName.AGENT_ARCHIVE
    agent_id: AgentId


class AgentUnarchive(_Command):
    """Bring an archived agent back to idle."""

    name: Literal[CommandName.AGENT_UNARCHIVE] = CommandName.AGENT_UNARCHIVE
    agent_id: AgentId


class AgentDelete(_Command):
    """Delete an agent and everything it owns."""

    name: Literal[CommandName.AGENT_DELETE] = CommandName.AGENT_DELETE
    agent_id: AgentId


class AgentRetry(_Command):
    """Re-run the failed step of a run that is in ``error``."""

    name: Literal[CommandName.AGENT_RETRY] = CommandName.AGENT_RETRY
    agent_id: AgentId


class AgentList(_Command):
    """Ask for the list of agents."""

    name: Literal[CommandName.AGENT_LIST] = CommandName.AGENT_LIST
    include_archived: bool = False


class AgentLoadHistory(_Command):
    """Ask for an agent's persisted messages and display events."""

    name: Literal[CommandName.AGENT_LOAD_HISTORY] = CommandName.AGENT_LOAD_HISTORY
    agent_id: AgentId
    from_seq: int = 0
    limit: int | None = None


class PermissionRespond(_Command):
    """Answer an open permission request."""

    name: Literal[CommandName.PERMISSION_RESPOND] = CommandName.PERMISSION_RESPOND
    request_id: PermissionRequestId
    decision: PermissionAnswer
    scope: PermissionScope = PermissionScope.ONCE
    deny_message: str | None = None


class ContextInspect(_Command):
    """Ask for the context report (``/context``)."""

    name: Literal[CommandName.CONTEXT_INSPECT] = CommandName.CONTEXT_INSPECT
    agent_id: AgentId
    full: bool = False


class ContextCompact(_Command):
    """Compact the context manually (``/compact``)."""

    name: Literal[CommandName.CONTEXT_COMPACT] = CommandName.CONTEXT_COMPACT
    agent_id: AgentId


class ConfigSet(_Command):
    """Set one config key in a layer."""

    name: Literal[CommandName.CONFIG_SET] = CommandName.CONFIG_SET
    key: str
    value: JsonValue
    layer: ConfigLayer
    agent_id: AgentId | None = None


class ConfigSetSecret(_Command):
    """Store a secret in the keyring; never logged, never echoed in events."""

    name: Literal[CommandName.CONFIG_SET_SECRET] = CommandName.CONFIG_SET_SECRET
    secret_name: str
    value: str = Field(repr=False)


class ProfileUpdate(_Command):
    """Replace ``user_profile.md``."""

    name: Literal[CommandName.PROFILE_UPDATE] = CommandName.PROFILE_UPDATE
    content: str


class TemplateApply(_Command):
    """Re-copy a template into the agent layer of existing agents."""

    name: Literal[CommandName.TEMPLATE_APPLY] = CommandName.TEMPLATE_APPLY
    template: str
    agent_ids: list[AgentId]


class MemoryList(_Command):
    """List semantic memory items (``agent_id=None`` means global)."""

    name: Literal[CommandName.MEMORY_LIST] = CommandName.MEMORY_LIST
    agent_id: AgentId | None = None
    include_expired: bool = False


class MemoryAdd(_Command):
    """Add a memory item as the user."""

    name: Literal[CommandName.MEMORY_ADD] = CommandName.MEMORY_ADD
    agent_id: AgentId | None = None
    content: str
    kind: MemoryKind
    tags: list[str] = []


class MemoryUpdate(_Command):
    """Edit or supersede a memory item as the user."""

    name: Literal[CommandName.MEMORY_UPDATE] = CommandName.MEMORY_UPDATE
    item_id: MemoryItemId
    content: str | None = None
    kind: MemoryKind | None = None
    tags: list[str] | None = None
    pinned: bool | None = None
    supersede: bool = False


class MemoryDelete(_Command):
    """Physically delete a memory item."""

    name: Literal[CommandName.MEMORY_DELETE] = CommandName.MEMORY_DELETE
    item_id: MemoryItemId


class TaskCancel(_Command):
    """Cancel a background task."""

    name: Literal[CommandName.TASK_CANCEL] = CommandName.TASK_CANCEL
    task_id: TaskId


class FileRestore(_Command):
    """Restore a file deleted by ``delete_file``."""

    name: Literal[CommandName.FILE_RESTORE] = CommandName.FILE_RESTORE
    agent_id: AgentId
    backup_id: BackupId


class AppShutdown(_Command):
    """Shut the core down."""

    name: Literal[CommandName.APP_SHUTDOWN] = CommandName.APP_SHUTDOWN
    stop_agents: bool = False


Command = Annotated[
    AgentCreate
    | AgentUpdate
    | AgentSend
    | AgentStop
    | AgentPause
    | AgentResume
    | AgentArchive
    | AgentUnarchive
    | AgentDelete
    | AgentRetry
    | AgentList
    | AgentLoadHistory
    | PermissionRespond
    | ContextInspect
    | ContextCompact
    | ConfigSet
    | ConfigSetSecret
    | ProfileUpdate
    | TemplateApply
    | MemoryList
    | MemoryAdd
    | MemoryUpdate
    | MemoryDelete
    | TaskCancel
    | FileRestore
    | AppShutdown,
    Field(discriminator="name"),
]
