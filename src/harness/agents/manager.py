"""``DefaultAgentManager``: agent lifecycle, run control and crash recovery (component 4)."""

import asyncio
from contextlib import suppress
from pathlib import Path

from loguru import logger
from pydantic import JsonValue, TypeAdapter

from harness.agents import state
from harness.agents.approvals import ApprovalDesk
from harness.agents.consts import (
    FIELD_NAME,
    FIELD_TEMPLATE,
    INBOX_WATCH_TASK_NAME,
    META_CLEAN_SHUTDOWN,
    MSG_AGENT_NOT_FOUND,
    MSG_HOOK_DENIED,
    MSG_NAME_INVALID,
    MSG_NAME_TAKEN,
    MSG_NO_OPEN_RUN,
    MSG_TEMPLATE_NOT_FOUND,
    UPDATABLE_FIELDS,
)
from harness.agents.events import Subscription
from harness.agents.hooking import HookCaller
from harness.agents.loop import RunLoop
from harness.agents.recovery import fail_interrupted_tasks
from harness.agents.runs import RunController
from harness.agents.runtime import AgentDeps, AgentRuntime, RunStart, StatusBoard
from harness.agents.toolcalls import ToolRunner
from harness.agents.utils import (
    agent_state,
    changed_record,
    new_record,
    spec_from_template,
    user_origin,
)
from harness.config import AgentTemplate, InboxConfig, template_overrides
from harness.core.consts import (
    ID_PREFIX_AGENT,
    ID_PREFIX_INBOX,
    AddressType,
    AgentStatus,
    ErrorCode,
    HookEvent,
    PermissionAnswer,
    PermissionScope,
    RunStopReason,
    TaskStatus,
)
from harness.core.errors import (
    AgentError,
    AgentNotFoundError,
    HarnessError,
    InvalidTransitionError,
)
from harness.core.events import AgentCreated, AgentDeleted, AgentUpdated
from harness.core.hook_payloads import (
    AgentCreatedPayload,
    AgentDeletedPayload,
    MessageReceivedPayload,
)
from harness.core.records import AgentSpec, AgentState, InboxItem
from harness.core.types import (
    AgentId,
    ContentBlock,
    MessageOrigin,
    PermissionRequestId,
)
from harness.core.utils.names import free_name, slug, valid_name

_BLOCKS = TypeAdapter(list[ContentBlock])
_S = AgentStatus


class DefaultAgentManager:
    """Owns every agent: one asyncio task per running agent, all state persisted."""

    def __init__(self, deps: AgentDeps) -> None:
        self._deps = deps
        self._runtimes: dict[AgentId, AgentRuntime] = {}
        self._status = StatusBoard(deps)
        self._desk = ApprovalDesk(deps, self._status)
        self._hooks = HookCaller(deps, self._desk)
        self._runner = ToolRunner(deps, self._desk)
        self._loop = RunLoop(
            deps, self._hooks, self._desk, self._runner, self._status, inbox_taken=self._inbox_taken
        )
        self._runs = RunController(deps, self._status, self._desk, self._runner, self._loop)
        self._watcher: asyncio.Task[None] | None = None

    # --- startup and recovery (04 §7a) ---

    async def start(self) -> None:
        """Load every agent and continue the runs that were open when the core went down."""
        memory = self._deps.memory
        crashed = await memory.get_meta(META_CLEAN_SHUTDOWN) is False
        await memory.set_meta(META_CLEAN_SHUTDOWN, False)
        for record in await memory.list_agents(include_archived=True):
            rt = AgentRuntime(record=record)
            rt.cost_total = await memory.cost_total(record.id)
            rt.last_seq = await memory.last_seq(record.id)
            rt.waiting_agent_messages = sum(
                item.origin.source.type is AddressType.AGENT
                for item in await memory.pending_inbox(record.id)
            )
            self._runtimes[record.id] = rt
            working_dir = record.settings.working_dir
            await self._deps.config.load_agent(
                record.id, Path(working_dir) if working_dir else None
            )
        await fail_interrupted_tasks(self._deps)
        for rt in list(self._runtimes.values()):
            await self._runs.recover(rt, crashed=crashed)

    def _require(self, agent_id: AgentId) -> AgentRuntime:
        if agent_id not in self._runtimes:
            raise AgentNotFoundError(MSG_AGENT_NOT_FOUND.format(agent_id=agent_id))
        return self._runtimes[agent_id]

    def _inbox_taken(self, rt: AgentRuntime) -> None:
        """The run loop moved the inbox into the conversation: nothing waits any more."""
        if rt.waiting_agent_messages and not rt.deleted:
            rt.waiting_agent_messages = 0
            self._deps.bus.publish(rt.id, AgentUpdated(agent=self.get(rt.id)))

    # --- names (phase 2, 03 §3) ---

    def _names(self, *, skip: AgentId | None = None) -> set[str]:
        return {rt.record.name for rt in self._runtimes.values() if rt.id != skip}

    def _check_name(self, name: str, *, skip: AgentId | None = None) -> None:
        """Refuse a name that cannot be an address, or that another agent has.

        Raises:
            AgentError: ``agent.name_invalid`` / ``agent.name_taken``.
        """
        if not valid_name(name):
            raise AgentError(MSG_NAME_INVALID.format(name=name), code=ErrorCode.AGENT_NAME_INVALID)
        if name in self._names(skip=skip):
            raise AgentError(MSG_NAME_TAKEN.format(name=name), code=ErrorCode.AGENT_NAME_TAKEN)

    # --- lifecycle ---

    async def create(self, spec: AgentSpec) -> AgentId:
        """Create an agent; it is ``idle`` and has a working directory when this returns.

        Raises:
            AgentError: The name is invalid or taken, or an ``agent_created`` hook (or the
                user it asked) refused.
        """
        deps = self._deps
        self._check_name(spec.name)
        agent_id = AgentId(deps.ids.new(ID_PREFIX_AGENT))
        defaults = deps.config.config().agents.defaults
        rt = AgentRuntime(record=new_record(agent_id, spec, defaults, deps.home, deps.clock.now()))
        event = HookEvent.AGENT_CREATED
        out = await self._hooks.emit(rt, event, AgentCreatedPayload(spec=spec), persist=False)
        if not await self._hooks.allowed(rt, event, out, stored=False):
            raise AgentError(
                MSG_HOOK_DENIED.format(event=event.value, reasons="; ".join(out.reasons)),
                code=ErrorCode.HOOK_DENIED,
            )
        working_dir = Path(str(rt.record.settings.working_dir))
        await asyncio.to_thread(working_dir.mkdir, parents=True, exist_ok=True)
        await deps.memory.save_agent(rt.record)
        await deps.config.load_agent(agent_id, working_dir)
        self._runtimes[agent_id] = rt
        deps.bus.publish(agent_id, AgentCreated(agent=self.get(agent_id)))
        return agent_id

    def _template(self, name: str) -> AgentTemplate:
        for template in self._deps.config.templates():
            if template.name == name:
                return template
        raise AgentError(MSG_TEMPLATE_NOT_FOUND.format(name=name), code=ErrorCode.CONFIG_INVALID)

    async def create_from_template(
        self, template: str, *, overrides: dict[str, JsonValue] | None = None
    ) -> AgentId:
        """Create an agent from a template; the template is copied into the agent layer."""
        found = self._template(template)
        given = dict(overrides or {})
        if FIELD_NAME not in given:  # the template's own name, made unique: reviewer-2
            given[FIELD_NAME] = free_name(slug(found.name), self._names())
        agent_id = await self.create(spec_from_template(found, given))
        layer = template_overrides(found)
        if layer:
            await self._deps.config.set_agent_layer(agent_id, layer)
        return agent_id

    async def _apply(
        self, rt: AgentRuntime, changes: dict[str, JsonValue], allowed: frozenset[str]
    ) -> None:
        before = rt.record.settings.working_dir
        renamed = changes.get(FIELD_NAME)
        if isinstance(renamed, str) and renamed != rt.record.name:
            self._check_name(renamed, skip=rt.id)
        rt.record = changed_record(rt.record, changes, allowed, self._deps.clock.now())
        await self._deps.memory.save_agent(rt.record)
        after = rt.record.settings.working_dir
        if after != before and after is not None:
            await asyncio.to_thread(Path(after).mkdir, parents=True, exist_ok=True)
            await self._deps.config.load_agent(rt.id, Path(after))
        self._deps.bus.publish(rt.id, AgentUpdated(agent=self.get(rt.id)))

    async def update(self, agent_id: AgentId, changes: dict[str, JsonValue]) -> None:
        """Change an agent's name, description, model, prompt, tools or settings."""
        await self._apply(self._require(agent_id), changes, UPDATABLE_FIELDS)

    async def apply_template(self, template: str, agent_ids: list[AgentId]) -> None:
        """Copy a template again into existing agents (record fields and agent layer)."""
        found = self._template(template)
        changes: dict[str, JsonValue] = {
            "model": found.model,
            "system_prompt": found.system_prompt,
            "tools": list(found.tools),
            "settings": dict(found.settings),
            FIELD_TEMPLATE: found.name,
        }
        for agent_id in agent_ids:
            rt = self._require(agent_id)
            await self._apply(rt, changes, UPDATABLE_FIELDS | {FIELD_TEMPLATE})
            await self._deps.config.set_agent_layer(agent_id, template_overrides(found))

    async def archive(self, agent_id: AgentId) -> None:
        """Archive an idle / paused / failed agent; an open run is closed."""
        rt = self._require(agent_id)
        async with rt.lock:
            state.require_status(rt.status, state.ARCHIVABLE, "archive")
            await self._runs.halt(rt, RunStopReason.ARCHIVED, _S.ARCHIVED)

    async def unarchive(self, agent_id: AgentId) -> None:
        """Bring an archived agent back to ``idle``."""
        rt = self._require(agent_id)
        state.require_status(rt.status, frozenset({_S.ARCHIVED}), "unarchive")
        await self._status.set(rt, _S.IDLE)

    async def delete(self, agent_id: AgentId) -> None:
        """Stop the agent, cancel its background tasks and delete everything it owns.

        Raises:
            AgentError: An ``agent_deleted`` hook (or the user it asked) refused.
        """
        rt = self._require(agent_id)
        event = HookEvent.AGENT_DELETED
        out = await self._hooks.emit(rt, event, AgentDeletedPayload(agent=self.get(agent_id)))
        if not await self._hooks.allowed(rt, event, out):
            raise AgentError(
                MSG_HOOK_DENIED.format(event=event.value, reasons="; ".join(out.reasons)),
                code=ErrorCode.HOOK_DENIED,
            )
        async with rt.lock:
            await self._runs.halt(rt, RunStopReason.DELETED, None)
            rt.deleted = True
            del self._runtimes[agent_id]
        for task in self._deps.tools.background_tasks(agent_id):
            if task.status is TaskStatus.RUNNING:
                await self._deps.tools.cancel_background(task.id)
        await self._deps.memory.delete_agent(agent_id)
        if self._deps.semantic is not None:
            await self._deps.semantic.delete_agent(agent_id)
        self._deps.config.forget_agent(agent_id)
        self._deps.bus.publish(agent_id, AgentDeleted(agent_id=agent_id))
        self._deps.bus.forget(agent_id)

    # --- running ---

    async def send(
        self,
        agent_id: AgentId,
        content: list[ContentBlock],
        origin: MessageOrigin | None = None,
    ) -> bool:
        """Put a message in the agent's inbox; an idle (or failed) agent starts a run.

        The single entry point for every writer (04 §6.0). ``origin`` defaults to the
        local user. The message is committed to the database before the agent is woken.

        Returns:
            Whether the message was accepted (``False``: a ``message_received`` hook
            refused it).
        """
        rt = self._require(agent_id)
        state.require_status(rt.status, frozenset(_S) - {_S.ARCHIVED}, "send")
        source = origin or user_origin(self._deps.config.config().user.id)
        event = HookEvent.MESSAGE_RECEIVED
        payload = MessageReceivedPayload(content=content, origin=source)
        out = await self._hooks.emit(rt, event, payload)
        if not await self._hooks.allowed(rt, event, out) or agent_id not in self._runtimes:
            return False
        accepted = _BLOCKS.validate_python(out.payload.get("content", content))
        item = InboxItem(
            id=self._deps.ids.new(ID_PREFIX_INBOX),
            agent_id=agent_id,
            content=accepted,
            origin=source,
            received_at=self._deps.clock.now(),
            priority=self._deps.config.section(InboxConfig).priority_of(source.source.type),
        )
        await self._deps.memory.enqueue_inbox(item)
        if source.source.type is AddressType.AGENT:
            rt.waiting_agent_messages += 1
            self._deps.bus.publish(agent_id, AgentUpdated(agent=self.get(agent_id)))
        if not self._runs.wakes(source):
            return True
        async with rt.lock:
            if rt.status is _S.ERROR:
                await self._runs.halt(rt, RunStopReason.ERROR, None)
            if rt.status in (_S.IDLE, _S.ERROR):
                await self._runs.start_run(rt, RunStart(accepted, source))
        return True

    # --- waking on what is in the database (phase 2, 02 §5a) ---

    async def wake_pending(self) -> None:
        """Open a run for every idle agent that has inbox messages waiting.

        The safety net behind ``send``: whatever reached the ``inbox`` table without the
        in-process wake (a crash in between, another process) is still picked up.
        """
        for agent_id in await self._deps.memory.agents_with_inbox():
            rt = self._runtimes.get(agent_id)
            if rt is not None and rt.status is _S.IDLE and rt.task is None:
                await self._runs.continue_from_inbox(rt)

    async def _watch_inbox(self) -> None:
        memory = self._deps.memory
        seen = await memory.data_version()
        while True:
            await asyncio.sleep(self._deps.config.section(InboxConfig).poll_interval_s)
            try:
                version = await memory.data_version()
                if version != seen:
                    seen = version
                    await self.wake_pending()
            except HarnessError as error:  # the next round tries again
                logger.warning("checking the inbox table failed: {}", error.message)

    def watch_inbox(self) -> None:
        """Start checking ``PRAGMA data_version`` every ``inbox.poll_interval_s`` seconds."""
        if self._watcher is None:
            self._watcher = asyncio.create_task(self._watch_inbox(), name=INBOX_WATCH_TASK_NAME)

    async def stop(self, agent_id: AgentId) -> None:
        """Cancel the current run at once; what was streamed so far is kept."""
        rt = self._require(agent_id)
        async with rt.lock:
            state.require_status(rt.status, state.ACTIVE, "stop")
            await self._runs.halt(rt, RunStopReason.STOP, _S.IDLE)

    async def pause(self, agent_id: AgentId) -> None:
        """Pause after the current step (and after an open approval is answered)."""
        rt = self._require(agent_id)
        state.require_status(rt.status, state.ACTIVE, "pause")
        rt.pause_requested = True

    async def _continue_run(self, agent_id: AgentId, allowed: AgentStatus, action: str) -> None:
        rt = self._require(agent_id)
        async with rt.lock:
            state.require_status(rt.status, frozenset({allowed}), action)
            run = rt.run or await self._deps.memory.open_run(agent_id)
            if run is None:
                raise InvalidTransitionError(MSG_NO_OPEN_RUN.format(agent_id=agent_id))
            rt.run = run
            await self._status.set(rt, _S.RUNNING)
            self._runs.launch(rt, run, None)

    async def resume(self, agent_id: AgentId) -> None:
        """Continue a paused run with its next step."""
        await self._continue_run(agent_id, _S.PAUSED, "resume")

    async def retry(self, agent_id: AgentId) -> None:
        """Run the failed step of a run in ``error`` again."""
        await self._continue_run(agent_id, _S.ERROR, "retry")

    async def respond_permission(
        self,
        request_id: PermissionRequestId,
        decision: PermissionAnswer,
        scope: PermissionScope = PermissionScope.ONCE,
        deny_message: str | None = None,
    ) -> None:
        """Answer an open permission request."""
        await self._desk.respond(request_id, decision, scope, deny_message)

    async def shutdown(self, *, stop_agents: bool) -> None:
        """Stop every task; with ``stop_agents`` runs are closed, otherwise they resume later."""
        watcher, self._watcher = self._watcher, None
        if watcher is not None:
            watcher.cancel()
            with suppress(asyncio.CancelledError):
                await watcher
        for rt in list(self._runtimes.values()):
            if stop_agents and rt.status in state.ACTIVE:
                async with rt.lock:
                    await self._runs.halt(rt, RunStopReason.STOP, _S.IDLE)
            else:
                await self._runs.cancel_task(rt)
        await self._deps.memory.set_meta(META_CLEAN_SHUTDOWN, True)
        self._deps.bus.close()

    async def wait_idle(self, agent_id: AgentId) -> None:
        """Wait until the agent's current task (and any run it chains into) is over."""
        rt = self._require(agent_id)
        while rt.task is not None:
            await asyncio.gather(rt.task, return_exceptions=True)

    # --- state ---

    def get(self, agent_id: AgentId) -> AgentState:
        """Return the agent as the GUI / CLI sees it."""
        rt = self._require(agent_id)
        running = [
            task
            for task in self._deps.tools.background_tasks(agent_id)
            if task.status is TaskStatus.RUNNING
        ]
        return agent_state(rt, self._desk.count(agent_id), len(running))

    def list(self, *, include_archived: bool = False) -> list[AgentState]:
        """Return every agent, in creation order."""
        return [
            self.get(agent_id)
            for agent_id, rt in self._runtimes.items()
            if include_archived or rt.status is not _S.ARCHIVED
        ]

    def events(self) -> Subscription:
        """Open the event stream the GUI / CLI consume."""
        return self._deps.bus.subscribe()
