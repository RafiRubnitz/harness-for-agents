"""``DefaultToolsManager``: the one passage every tool call goes through (08 T1)."""

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from loguru import logger
from pydantic import JsonValue

from harness.config import ConfigManager, ToolsConfig
from harness.core.consts import META_TASK_ID, ConfigLayer, Decision, ErrorCode, TaskStatus
from harness.core.errors import AgentNotFoundError, ToolError
from harness.core.messaging import Messenger
from harness.core.records import (
    AgentState,
    BackgroundTask,
    BackgroundTaskRecord,
    McpServerStatus,
    PermissionDecision,
    ToolDef,
    ToolOutput,
)
from harness.core.types import (
    AgentId,
    BackupId,
    Clock,
    IdGenerator,
    InjectedFields,
    TaskId,
    TextBlock,
    ToolCallBlock,
    ToolSpec,
)
from harness.core.utils.paths import HarnessHome
from harness.hooks import HooksManager
from harness.memory import BlobStore, MemoryStore, SemanticMemory
from harness.tools.background import BackgroundTasks, Emit, TaskSinks
from harness.tools.builtin import builtin_tools
from harness.tools.builtin.memory import memory_tools
from harness.tools.builtin.messaging import messaging_tools
from harness.tools.builtin.tasks import task_tools
from harness.tools.consts import MCP_CONFIG_KEY, META_STATUS, MSG_TOOL_NOT_AVAILABLE
from harness.tools.custom import CustomTools
from harness.tools.executor import ToolExecutor, denial, may_run
from harness.tools.function_tool import error_output
from harness.tools.injected import inject_fields, split_injected
from harness.tools.mcp import McpServers
from harness.tools.permissions import PermissionPipeline
from harness.tools.policy import exact_rule, subject_of
from harness.tools.registry import ToolRegistry
from harness.tools.types import AgentLookup, ProgressSink, Tool, ToolContext

_RULES_KEY = "permissions.rules"
_RULE_LAYER_FIELD = "layer"


@dataclass(frozen=True, slots=True)
class ToolsDeps:
    """Everything the tools manager talks to.

    ``agent``, ``progress`` and the task sinks are plain callables so the manager does not
    depend on the agent manager (which depends on it). Background tasks need ``history``.
    """

    config: ConfigManager
    hooks: HooksManager
    blobs: BlobStore
    clock: Clock
    ids: IdGenerator
    home: HarnessHome
    agent: AgentLookup
    progress: ProgressSink
    semantic: SemanticMemory | None = None
    history: MemoryStore | None = None
    task_status: Callable[[BackgroundTaskRecord], None] | None = None
    task_done: Callable[[BackgroundTaskRecord, str], Awaitable[None]] | None = None
    messenger: Messenger | None = None


def _no_status(record: BackgroundTaskRecord) -> None:
    """Nobody listens to task status changes."""


async def _no_done(record: BackgroundTaskRecord, summary: str) -> None:
    """Nobody is told that the task ended."""


def _started(task_id: TaskId) -> ToolOutput:
    status = TaskStatus.RUNNING.value
    return ToolOutput(
        content=[TextBlock(text=json.dumps({META_TASK_ID: task_id, META_STATUS: status}))],
        metadata={META_TASK_ID: task_id, META_STATUS: status},
    )


def _unavailable(tool: str) -> ToolOutput:
    return error_output(MSG_TOOL_NOT_AVAILABLE.format(tool=tool), ErrorCode.TOOL_NOT_AVAILABLE)


class DefaultToolsManager:
    """Registry, permission pipeline and safe execution of tools."""

    def __init__(self, deps: ToolsDeps) -> None:
        self._deps = deps
        self._registry = ToolRegistry()
        self._pipeline = PermissionPipeline(deps.config, deps.hooks, deps.home, deps.clock)
        self._executor = ToolExecutor(deps.config, deps.hooks, deps.blobs, deps.clock)
        builtin, self._deleter = builtin_tools(deps.config, deps.blobs, deps.ids)
        if deps.semantic is not None and deps.history is not None:
            builtin.extend(memory_tools(deps.semantic, deps.history))
        self._background: BackgroundTasks | None = None
        if deps.history is not None:
            sinks = TaskSinks(deps.task_status or _no_status, deps.task_done or _no_done)
            self._background = BackgroundTasks(
                deps.history, deps.blobs, deps.clock, deps.ids, sinks
            )
            builtin.extend(task_tools(self._background))
        if deps.messenger is not None:
            builtin.extend(messaging_tools(deps.messenger))
        for tool in builtin:
            self._registry.register(tool)
        self._custom = CustomTools(deps.home.tools_dir)
        self._mcp = McpServers(deps.config, deps.home.logs_dir)
        deps.config.subscribe([MCP_CONFIG_KEY], self._servers_changed)

    # --- registry ---

    def register(self, tool: Tool) -> None:
        """Add a tool at runtime."""
        self._registry.register(tool)

    def unregister(self, name: str) -> None:
        """Remove a tool by name."""
        self._registry.unregister(name)

    def _agent_tools(self, agent: AgentState) -> list[Tool]:
        self._custom.refresh(self._registry)
        enabled = self._deps.config.section(ToolsConfig, agent.id).enabled
        return self._registry.for_agent(agent.tools, enabled)

    def _find(self, agent_id: AgentId, name: str) -> tuple[AgentState, Tool] | None:
        try:
            agent = self._deps.agent(agent_id)
        except AgentNotFoundError:
            return None
        for tool in self._agent_tools(agent):
            if tool.definition.name == name:
                return agent, tool
        return None

    def specs_for(self, agent_id: AgentId) -> list[ToolSpec]:
        """Return the specs of the agent's tools, with the injected fields in the schema."""
        return [
            ToolSpec(
                name=tool.name,
                description=tool.description,
                input_schema=inject_fields(tool.input_schema),
            )
            for tool in self.list(agent_id)
        ]

    def split_injected(self, tool: str, raw_input: dict[str, JsonValue]) -> InjectedFields:
        """Split a model-sent input, honouring a tool's own ``reason`` field (08 §3a)."""
        registered = self._registry.get(tool)
        schema = registered.definition.input_schema if registered else None
        return split_injected(raw_input, schema)

    # --- the pipeline ---

    def _context(
        self,
        agent: AgentState,
        call: ToolCallBlock,
        reason: str | None,
        emit: Emit | None = None,
    ) -> ToolContext:
        def show(chunk: str) -> None:
            self._deps.progress(agent.id, call.id, None, chunk)

        return ToolContext(
            agent_id=agent.id,
            run_id=agent.run_id,
            call_id=call.id,
            working_dir=str(agent.settings.working_dir),
            allowed_paths=list(agent.settings.allowed_paths),
            cancel=asyncio.Event(),
            emit_progress=emit or show,
            clock=self._deps.clock,
            reason=reason,
        )

    async def decide(self, agent_id: AgentId, call: ToolCallBlock) -> PermissionDecision:
        """Stages 0-2: reason, policy, hooks; adds the preview when the user must be asked."""
        fields = self.split_injected(call.name, call.input)
        found = self._find(agent_id, call.name)
        if found is None:
            return PermissionDecision(
                decision=Decision.DENY, input=fields.input, error_code=ErrorCode.TOOL_NOT_AVAILABLE
            )
        agent, tool = found
        decision = await self._pipeline.decide(agent, tool, fields, call.id)
        if decision.decision is not Decision.ASK:
            return decision
        try:
            preview = await tool.preview(decision.input, self._context(agent, call, fields.reason))
        except Exception:  # a broken preview must not block the approval itself
            logger.opt(exception=True).warning("preview of {} failed", call.name)
            preview = None
        return decision.model_copy(update={"preview": preview})

    async def execute(
        self, agent_id: AgentId, call: ToolCallBlock, decision: PermissionDecision
    ) -> ToolOutput:
        """Stages 4-5: run only what was allowed (or approved), then the after hooks."""
        found = self._find(agent_id, call.name)
        if found is None:
            return _unavailable(call.name)
        if not may_run(decision):
            return denial(call.name, decision)
        agent, tool = found
        fields = self.split_injected(call.name, call.input)
        if fields.run_in_background and self._background is not None:
            return _started(await self.start_background(agent_id, call, decision))
        ctx = self._context(agent, call, fields.reason)
        output = await self._executor.run(tool, decision.input, ctx)
        output = await self._executor.keep_large(agent, output)
        return await self._executor.after(agent, tool, decision.input, ctx, output)

    async def allow_always(
        self, agent_id: AgentId, call: ToolCallBlock, decision: PermissionDecision
    ) -> None:
        """Write an allow rule for exactly this call into the agent layer (08 §5.1)."""
        found = self._find(agent_id, call.name)
        if found is None:
            return
        agent, tool = found
        working_dir = str(agent.settings.working_dir)
        subject = subject_of(call.name, tool.traits, decision.input, working_dir)
        rule = exact_rule(subject, tool.traits)
        current = self._deps.config.config(agent_id).permissions.rules
        kept: list[JsonValue] = [
            existing.model_dump(mode="json", exclude={_RULE_LAYER_FIELD}, exclude_none=True)
            for existing in current
            if existing.layer is ConfigLayer.AGENT
        ]
        kept.append(rule.model_dump(mode="json", exclude={_RULE_LAYER_FIELD}, exclude_none=True))
        await self._deps.config.set(_RULES_KEY, kept, ConfigLayer.AGENT, agent_id)

    # --- background tasks (08 §6a) ---

    async def start_background(
        self, agent_id: AgentId, call: ToolCallBlock, decision: PermissionDecision
    ) -> TaskId:
        """Start an allowed tool call as a background task and return its id at once.

        The ``after_tool_call`` hooks run when the task ends, not when it starts.

        Raises:
            ToolError: The agent does not have the tool, the call was not allowed, or
                background tasks are not available (no store).
        """
        found = self._find(agent_id, call.name)
        if found is None or not may_run(decision) or self._background is None:
            raise ToolError(
                MSG_TOOL_NOT_AVAILABLE.format(tool=call.name),
                tool=call.name,
                call_id=call.id,
                code=ErrorCode.TOOL_NOT_AVAILABLE,
            )
        agent, tool = found
        reason = self.split_injected(call.name, call.input).reason

        async def work(task_id: TaskId, keep: Emit) -> ToolOutput:
            def emit(chunk: str) -> None:
                keep(chunk)
                self._deps.progress(agent.id, call.id, task_id, chunk)

            ctx = self._context(agent, call, reason, emit)
            output = await self._executor.run(tool, decision.input, ctx)
            return await self._executor.after(agent, tool, decision.input, ctx, output)

        return await self._background.start(agent, call, work)

    def background_tasks(self, agent_id: AgentId) -> list[BackgroundTask]:
        """Return the background tasks an agent started in this session."""
        return self._background.of(agent_id) if self._background else []

    async def cancel_background(self, task_id: TaskId) -> None:
        """Cancel a background task; unknown and finished tasks are left alone."""
        if self._background is not None:
            await self._background.cancel(task_id)

    async def close(self) -> None:
        """Stop running background tasks (the next start reports them) and the MCP servers."""
        if self._background is not None:
            await self._background.close()
        await self._mcp.close()

    # --- the rest of the interface ---

    async def restore_deleted(self, agent_id: AgentId, backup_id: BackupId) -> ToolOutput:
        """Restore a file removed by ``delete_file`` (the ``file.restore`` command)."""
        return await self._deleter.restore(backup_id, agent_id, self._deps.clock.now())

    async def reload(self) -> None:
        """Import every custom tool file again and (re)connect the MCP servers."""
        self._custom.forget()
        self._custom.refresh(self._registry)
        await self._mcp.sync(self._registry)

    async def _servers_changed(self, keys: list[str]) -> None:
        await self._mcp.sync(self._registry)

    async def mcp_status(self) -> dict[str, McpServerStatus]:
        """The state of every configured MCP server and the tools it brought."""
        return {
            name: McpServerStatus(state=state, error=error, tools=tools)
            for name, (state, error, tools) in self._mcp.status().items()
        }

    def list(self, agent_id: AgentId | None = None) -> list[ToolDef]:
        """Return the tool definitions of one agent, or of every registered tool."""
        if agent_id is None:
            self._custom.refresh(self._registry)
            return [tool.definition for tool in self._registry.every()]
        try:
            agent = self._deps.agent(agent_id)
        except AgentNotFoundError:
            return []
        return [tool.definition for tool in self._agent_tools(agent)]
