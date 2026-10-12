"""Tools manager (component 8): tool registry, permissions, execution, background tasks.

Other components use the ``ToolsManager`` protocol. ``NullToolsManager`` is a manager with
no tools at all: every call comes back to the model as "not available".
"""

from typing import Protocol

from pydantic import JsonValue

from harness.core.consts import META_ERROR_CODE, Decision, ErrorCode
from harness.core.records import (
    BackgroundTask,
    McpServerStatus,
    PermissionDecision,
    ToolDef,
    ToolOutput,
)
from harness.core.types import (
    AgentId,
    BackupId,
    InjectedFields,
    TaskId,
    TextBlock,
    ToolCallBlock,
    ToolSpec,
)
from harness.tools.consts import MSG_TOOL_NOT_AVAILABLE
from harness.tools.function_tool import FunctionTool, error_output, text_output, tool
from harness.tools.injected import inject_fields, injected_names, split_injected
from harness.tools.manager import DefaultToolsManager, ToolsDeps
from harness.tools.types import AgentLookup, ProgressSink, Tool, ToolContext, ToolTraits


class ToolsManager(Protocol):
    """The single passage every tool call goes through."""

    def register(self, tool: Tool) -> None:
        """Add a tool at runtime."""
        ...

    def unregister(self, name: str) -> None:
        """Remove a tool by name."""
        ...

    def specs_for(self, agent_id: AgentId) -> list[ToolSpec]:
        """Return the tool specs the model sees (injected fields included)."""
        ...

    def split_injected(self, tool: str, raw_input: dict[str, JsonValue]) -> InjectedFields:
        """Split a model-sent input into the clean input and the injected fields."""
        ...

    async def decide(self, agent_id: AgentId, call: ToolCallBlock) -> PermissionDecision:
        """Run stages 0-2 of the permission pipeline (reason, policy, hooks)."""
        ...

    async def execute(
        self, agent_id: AgentId, call: ToolCallBlock, decision: PermissionDecision
    ) -> ToolOutput:
        """Run the tool (or start it in the background); never raises a tool failure."""
        ...

    async def allow_always(
        self, agent_id: AgentId, call: ToolCallBlock, decision: PermissionDecision
    ) -> None:
        """Write the allow rule for "always approve for this agent" into the agent layer."""
        ...

    async def start_background(
        self, agent_id: AgentId, call: ToolCallBlock, decision: PermissionDecision
    ) -> TaskId:
        """Start a tool as a background task."""
        ...

    def background_tasks(self, agent_id: AgentId) -> list[BackgroundTask]:
        """Return an agent's background tasks."""
        ...

    async def cancel_background(self, task_id: TaskId) -> None:
        """Cancel a background task."""
        ...

    async def restore_deleted(self, agent_id: AgentId, backup_id: BackupId) -> ToolOutput:
        """Restore a file removed by ``delete_file``."""
        ...

    async def reload(self) -> None:
        """Re-read custom tools and MCP servers."""
        ...

    async def mcp_status(self) -> dict[str, McpServerStatus]:
        """Return the state of every configured MCP server."""
        ...

    def list(self, agent_id: AgentId | None = None) -> list[ToolDef]:
        """Return the tool definitions (of one agent, or all registered)."""
        ...


def not_available(tool: str) -> ToolOutput:
    """The result of calling a tool the agent does not have."""
    return ToolOutput(
        content=[TextBlock(text=MSG_TOOL_NOT_AVAILABLE.format(tool=tool))],
        is_error=True,
        metadata={META_ERROR_CODE: ErrorCode.TOOL_NOT_AVAILABLE.value},
    )


class NullToolsManager:
    """A ``ToolsManager`` without tools."""

    def register(self, tool: Tool) -> None:
        """Nothing is kept."""

    def unregister(self, name: str) -> None:
        """Nothing to remove."""

    def specs_for(self, agent_id: AgentId) -> list[ToolSpec]:
        """No tools."""
        return []

    def split_injected(self, tool: str, raw_input: dict[str, JsonValue]) -> InjectedFields:
        """Split using the default injected names."""
        return split_injected(raw_input)

    async def decide(self, agent_id: AgentId, call: ToolCallBlock) -> PermissionDecision:
        """Deny: the tool does not exist."""
        return PermissionDecision(
            decision=Decision.DENY,
            input=split_injected(call.input).input,
            error_code=ErrorCode.TOOL_NOT_AVAILABLE,
        )

    async def execute(
        self, agent_id: AgentId, call: ToolCallBlock, decision: PermissionDecision
    ) -> ToolOutput:
        """Return the "not available" error."""
        return not_available(call.name)

    async def allow_always(
        self, agent_id: AgentId, call: ToolCallBlock, decision: PermissionDecision
    ) -> None:
        """Nothing to remember."""

    async def start_background(
        self, agent_id: AgentId, call: ToolCallBlock, decision: PermissionDecision
    ) -> TaskId:
        """No tool can be started."""
        raise NotImplementedError

    def background_tasks(self, agent_id: AgentId) -> list[BackgroundTask]:
        """No tasks."""
        return []

    async def cancel_background(self, task_id: TaskId) -> None:
        """Nothing to cancel."""

    async def restore_deleted(self, agent_id: AgentId, backup_id: BackupId) -> ToolOutput:
        """No backups exist."""
        return not_available(str(backup_id))

    async def reload(self) -> None:
        """Nothing to reload."""

    async def mcp_status(self) -> dict[str, McpServerStatus]:
        """No servers."""
        return {}

    def list(self, agent_id: AgentId | None = None) -> list[ToolDef]:
        """No tools."""
        return []


__all__ = [
    "AgentLookup",
    "DefaultToolsManager",
    "FunctionTool",
    "NullToolsManager",
    "ProgressSink",
    "Tool",
    "ToolContext",
    "ToolTraits",
    "ToolsDeps",
    "ToolsManager",
    "error_output",
    "inject_fields",
    "injected_names",
    "not_available",
    "split_injected",
    "text_output",
    "tool",
]
