"""FakeToolsManager: in-memory tools with scripted permission decisions."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from pydantic import JsonValue

from harness.core.consts import (
    META_DENIED_BY,
    META_ERROR_CODE,
    Decision,
    DenySource,
    ErrorCode,
    PermissionAnswer,
    ToolSource,
)
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
from harness.tools import inject_fields, not_available, split_injected

type Handler = Callable[[dict[str, JsonValue]], Awaitable[ToolOutput]]

DENIED_TEXT = "denied"


def text_output(text: str, *, is_error: bool = False) -> ToolOutput:
    """A tool output with one text block."""
    return ToolOutput(content=[TextBlock(text=text)], is_error=is_error)


@dataclass
class FakeTool:
    """One registered fake tool."""

    definition: ToolDef
    handler: Handler
    decision: Decision = Decision.ALLOW


@dataclass
class FakeToolsManager:
    """Decides from a per-tool scripted decision and runs plain async handlers."""

    tools: dict[str, FakeTool] = field(default_factory=dict[str, FakeTool])
    executed: list[str] = field(default_factory=list[str])
    always_allowed: list[str] = field(default_factory=list[str])
    cancelled_tasks: list[TaskId] = field(default_factory=list[TaskId])
    tasks: dict[AgentId, list[BackgroundTask]] = field(
        default_factory=dict[AgentId, list[BackgroundTask]]
    )

    def add(
        self,
        name: str,
        handler: Handler | None = None,
        *,
        decision: Decision = Decision.ALLOW,
        exclusive: bool = False,
    ) -> None:
        """Register a tool; the default handler echoes its name."""

        async def echo(_: dict[str, JsonValue]) -> ToolOutput:
            return text_output(name)

        definition = ToolDef(
            name=name,
            description=name,
            input_schema={"type": "object"},
            source=ToolSource.BUILTIN,
            exclusive=exclusive,
        )
        self.tools[name] = FakeTool(definition, handler or echo, decision)

    def add_blocking(self, name: str, **kwargs: bool) -> tuple[asyncio.Event, asyncio.Event]:
        """Register a tool that signals ``started`` and then waits for ``release``."""
        started, release = asyncio.Event(), asyncio.Event()

        async def block(_: dict[str, JsonValue]) -> ToolOutput:
            started.set()
            await release.wait()
            return text_output(name)

        self.add(name, block, **kwargs)
        return started, release

    def specs_for(self, agent_id: AgentId) -> list[ToolSpec]:
        """Every registered tool, with the injected fields."""
        return [
            ToolSpec(
                name=tool.definition.name,
                description=tool.definition.description,
                input_schema=inject_fields(tool.definition.input_schema),
            )
            for tool in self.tools.values()
        ]

    def split_injected(self, tool: str, raw_input: dict[str, JsonValue]) -> InjectedFields:
        """Split with the default injected names."""
        return split_injected(raw_input)

    async def decide(self, agent_id: AgentId, call: ToolCallBlock) -> PermissionDecision:
        """Return the tool's scripted decision."""
        clean = split_injected(call.input).input
        if call.name not in self.tools:
            return PermissionDecision(
                decision=Decision.DENY, input=clean, error_code=ErrorCode.TOOL_NOT_AVAILABLE
            )
        decision = self.tools[call.name].decision
        denied_by = DenySource.POLICY if decision is Decision.DENY else None
        return PermissionDecision(decision=decision, input=clean, denied_by=denied_by)

    async def execute(
        self, agent_id: AgentId, call: ToolCallBlock, decision: PermissionDecision
    ) -> ToolOutput:
        """Run the handler when the decision allows it; otherwise return the denial."""
        if call.name not in self.tools:
            return not_available(call.name)
        approved = decision.user_answer is PermissionAnswer.APPROVE
        if decision.decision is Decision.DENY or (
            decision.decision is Decision.ASK and not approved
        ):
            source = decision.denied_by or DenySource.USER
            return ToolOutput(
                content=[TextBlock(text=decision.deny_message or DENIED_TEXT)],
                is_error=True,
                metadata={
                    META_DENIED_BY: source.value,
                    META_ERROR_CODE: ErrorCode.PERMISSION_DENIED.value,
                },
            )
        self.executed.append(call.name)
        return await self.tools[call.name].handler(decision.input)

    async def allow_always(
        self, agent_id: AgentId, call: ToolCallBlock, decision: PermissionDecision
    ) -> None:
        """Allow the tool from now on."""
        self.always_allowed.append(call.name)
        self.tools[call.name].decision = Decision.ALLOW

    async def start_background(
        self, agent_id: AgentId, call: ToolCallBlock, decision: PermissionDecision
    ) -> TaskId:
        """Not scripted."""
        raise NotImplementedError

    def background_tasks(self, agent_id: AgentId) -> list[BackgroundTask]:
        """The tasks a test placed for the agent."""
        return list(self.tasks.get(agent_id, []))

    async def cancel_background(self, task_id: TaskId) -> None:
        """Record the cancellation."""
        self.cancelled_tasks.append(task_id)

    async def restore_deleted(self, agent_id: AgentId, backup_id: BackupId) -> ToolOutput:
        """No backups."""
        return not_available(str(backup_id))

    async def reload(self) -> None:
        """Nothing to reload."""

    async def mcp_status(self) -> dict[str, McpServerStatus]:
        """No servers."""
        return {}

    def list(self, agent_id: AgentId | None = None) -> list[ToolDef]:
        """Every registered definition."""
        return [tool.definition for tool in self.tools.values()]
