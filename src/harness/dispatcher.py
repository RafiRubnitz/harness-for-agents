"""The running core as the GUI / CLI see it: maps commands to components (12 §2)."""

import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass

from loguru import logger
from pydantic import JsonValue

from harness.agents import AgentManager, EventBus
from harness.config import ConfigManager, HarnessConfig, mask, requires_restart
from harness.context import ContextManager
from harness.core import commands as c
from harness.core.consts import (
    COMPONENT_COMMANDS,
    MSG_COMMAND_FAILED,
    MSG_SEMANTIC_MEMORY_UNAVAILABLE,
    CompactReason,
    ErrorCode,
    MemorySource,
    ProviderId,
)
from harness.core.errors import ConfigError, HarnessError, MemoryStoreError
from harness.core.events import (
    AgentListed,
    ConfigChanged,
    ContextReported,
    Error,
    Event,
    HistoryLoaded,
    MemoryListed,
)
from harness.core.events import Warning as WarningEvent
from harness.core.handle import FrontendInfo
from harness.core.records import AgentSpec
from harness.core.types import AgentId, MemoryItemId, TextBlock
from harness.core.utils.paths import HarnessHome
from harness.memory import MemoryStore, SemanticMemory
from harness.tools import ToolsManager
from harness.tools.consts import WEB_SECRET_NAME

_TEMPLATE_SKIPPED_FIELDS = frozenset({"name", "agent_name", "template"})
_SECRETS_KEY = "secrets"
_USER = MemorySource.USER
_CHANGE_KIND, _CHANGE_TAGS, _CHANGE_PINNED = "kind", "tags", "pinned"


@dataclass(frozen=True, slots=True)
class CoreParts:
    """The components a command can reach."""

    agents: AgentManager
    config: ConfigManager
    memory: MemoryStore
    context: ContextManager
    tools: ToolsManager
    bus: EventBus
    home: HarnessHome
    semantic: SemanticMemory | None = None


def agent_of(command: c.Command) -> AgentId | None:
    """Return the agent a command is about, if it names one."""
    agent_id: AgentId | None = getattr(command, "agent_id", None)
    return agent_id


class CommandDispatcher:
    """``CoreHandle`` over the wired components."""

    def __init__(self, parts: CoreParts) -> None:
        self._parts = parts

    def events(self) -> AsyncIterator[Event]:
        """Open a stream of every event published from now on."""
        return self._parts.bus.subscribe()

    def info(self) -> FrontendInfo:
        """The current display settings and the names a front end can offer."""
        config, definitions = self._parts.config.config(), self._parts.tools.list()
        gui = config.gui
        return FrontendInfo(
            theme=gui.theme,
            font_family=gui.font.family,
            font_size=gui.font.size,
            window_width=gui.window.width,
            window_height=gui.window.height,
            shortcuts=dict(gui.shortcuts),
            themes_dir=str(self._parts.home.themes_dir),
            models=list(config.models),
            templates=[template.name for template in self._parts.config.templates()],
            tools=[definition.name for definition in definitions],
            tool_groups=sorted({d.group for d in definitions if d.group is not None}),
            providers=[provider.value for provider in ProviderId],
            key_names=[*(provider.value for provider in ProviderId), WEB_SECRET_NAME],
            user_profile=self._parts.config.user_profile(),
            model_providers={alias: m.provider.value for alias, m in config.models.items()},
            keys=self._keys(config),
            embedder_url=config.embedder.url or "",
        )

    def _keys(self, config: HarnessConfig) -> dict[str, str]:
        """The masked API key of every provider and of the web tools, when set."""
        keys: dict[str, str] = {}
        for name, provider in config.providers.items():
            if provider.api_key is None:
                continue
            try:
                keys[name] = mask(self._parts.config.expand(provider.api_key))
            except ConfigError:  # not set yet
                continue
        with contextlib.suppress(ConfigError):  # not set yet
            keys[WEB_SECRET_NAME] = mask(self._parts.config.secret(WEB_SECRET_NAME))
        return keys

    async def submit(self, command: c.Command) -> None:
        """Run one command; a failure is published as an ``error`` event."""
        try:
            handled = await self._lifecycle(command) or await self._run_control(command)
            if not handled and not await self._memory(command):
                await self._services(command)
        except HarnessError as error:
            self._fail(command, error.code, error.message, recoverable=error.recoverable)
        except Exception:  # one bad command must not take the core down
            logger.exception("command {} failed", command.name.value)
            message = MSG_COMMAND_FAILED.format(command=command.name.value)
            self._fail(command, ErrorCode.APP_INTERNAL, message, recoverable=True)

    def _fail(
        self, command: c.Command, code: ErrorCode, message: str, *, recoverable: bool
    ) -> None:
        shown = Error(
            code=code, message=message, recoverable=recoverable, component=COMPONENT_COMMANDS
        )
        self._parts.bus.publish(agent_of(command), shown)

    async def _create(self, command: c.AgentCreate) -> None:
        agents = self._parts.agents
        if command.template is None:
            spec = AgentSpec(
                name=command.agent_name,
                description=command.description,
                model=command.model,
                system_prompt=command.system_prompt,
                tools=command.tools,
            )
            if command.settings is not None:
                spec = spec.model_copy(update={"settings": command.settings})
            await agents.create(spec)
            return
        given = command.model_fields_set - _TEMPLATE_SKIPPED_FIELDS
        overrides: dict[str, JsonValue] = command.model_dump(
            mode="json", include=set(given), exclude_unset=True
        )
        overrides["name"] = command.agent_name
        await agents.create_from_template(command.template, overrides=overrides)

    async def _lifecycle(self, command: c.Command) -> bool:
        agents = self._parts.agents
        match command:
            case c.AgentCreate():
                await self._create(command)
            case c.AgentUpdate():
                await agents.update(command.agent_id, command.changes)
            case c.AgentArchive():
                await agents.archive(command.agent_id)
            case c.AgentUnarchive():
                await agents.unarchive(command.agent_id)
            case c.AgentDelete():
                await agents.delete(command.agent_id)
            case c.TemplateApply():
                await agents.apply_template(command.template, command.agent_ids)
            case c.AgentList():
                listed = agents.list(include_archived=command.include_archived)
                self._parts.bus.publish(None, AgentListed(agents=listed))
            case c.AgentLoadHistory():
                await self._load_history(command)
            case c.AppShutdown():
                await agents.shutdown(stop_agents=command.stop_agents)
            case _:
                return False
        return True

    async def _run_control(self, command: c.Command) -> bool:
        agents = self._parts.agents
        match command:
            case c.AgentSend():
                await agents.send(command.agent_id, command.content)
            case c.AgentStop():
                await agents.stop(command.agent_id)
            case c.AgentPause():
                await agents.pause(command.agent_id)
            case c.AgentResume():
                await agents.resume(command.agent_id)
            case c.AgentRetry():
                await agents.retry(command.agent_id)
            case c.PermissionRespond():
                await agents.respond_permission(
                    command.request_id, command.decision, command.scope, command.deny_message
                )
            case _:
                return False
        return True

    async def _load_history(self, command: c.AgentLoadHistory) -> None:
        agent_id = command.agent_id
        self._parts.agents.get(agent_id)
        memory = self._parts.memory
        loaded = HistoryLoaded(
            messages=await memory.get_messages(
                agent_id, from_seq=command.from_seq, limit=command.limit
            ),
            events=await memory.get_events(agent_id, from_seq=command.from_seq),
            from_seq=command.from_seq,
        )
        self._parts.bus.publish(agent_id, loaded)

    async def _restore(self, command: c.FileRestore) -> None:
        output = await self._parts.tools.restore_deleted(command.agent_id, command.backup_id)
        if output.is_error:
            text = " ".join(b.text for b in output.content if isinstance(b, TextBlock))
            shown = WarningEvent(code=ErrorCode.TOOL_FAILED, message=text)
            self._parts.bus.publish(command.agent_id, shown)

    async def _services(self, command: c.Command) -> None:
        parts = self._parts
        match command:
            case c.ContextInspect():
                report = await parts.context.inspect(command.agent_id, full=command.full)
                shown = ContextReported(report=report, full_text=report.full_text)
                parts.bus.publish(command.agent_id, shown)
            case c.ContextCompact():
                await parts.context.compact(command.agent_id, CompactReason.MANUAL)
            case c.ConfigSet():
                await parts.config.set(command.key, command.value, command.layer, command.agent_id)
                changed = ConfigChanged(
                    keys=[command.key], restart_required=requires_restart([command.key])
                )
                parts.bus.publish(command.agent_id, changed)
            case c.ConfigSetSecret():
                await parts.config.set_secret(command.secret_name, command.value)
                # no value in the event: front ends re-read which keys are set
                changed = ConfigChanged(keys=[f"{_SECRETS_KEY}.{command.secret_name}"])
                parts.bus.publish(None, changed)
            case c.ProfileUpdate():
                await parts.config.set_user_profile(command.content)
            case c.TaskCancel():
                await parts.tools.cancel_background(command.task_id)
            case c.FileRestore():
                await self._restore(command)
            case _:
                raise MemoryStoreError(MSG_SEMANTIC_MEMORY_UNAVAILABLE)

    async def _memory(self, command: c.Command) -> bool:
        """Run a ``memory.*`` command as the user, then show the owner's refreshed list."""
        if not isinstance(command, c.MemoryList | c.MemoryAdd | c.MemoryUpdate | c.MemoryDelete):
            return False
        semantic = self._parts.semantic
        if semantic is None:
            raise MemoryStoreError(MSG_SEMANTIC_MEMORY_UNAVAILABLE)
        include_expired = False
        match command:
            case c.MemoryList():
                owner, include_expired = command.agent_id, command.include_expired
            case c.MemoryAdd():
                owner = command.agent_id
                await semantic.remember(
                    owner, command.content, command.kind, tags=command.tags, source=_USER
                )
            case c.MemoryUpdate():
                owner = await self._owner(semantic, command.item_id)
                await self._edit_memory(semantic, command)
            case c.MemoryDelete():
                owner = await self._owner(semantic, command.item_id)
                await semantic.delete(command.item_id)
        items = await semantic.list(owner, include_expired=include_expired)
        self._parts.bus.publish(owner, MemoryListed(items=items))
        return True

    @staticmethod
    async def _owner(semantic: SemanticMemory, item_id: MemoryItemId) -> AgentId | None:
        owner = (await semantic.get(item_id)).agent_id
        return AgentId(owner) if owner else None

    @staticmethod
    async def _edit_memory(semantic: SemanticMemory, command: c.MemoryUpdate) -> None:
        """Apply ``memory.update``: a correction in place, or a superseding new item."""
        item_id, content = command.item_id, command.content
        if command.supersede and content is not None:
            item_id, content = await semantic.supersede(item_id, content, actor=_USER), None
        changes: dict[str, JsonValue] = {}
        if command.kind is not None:
            changes[_CHANGE_KIND] = command.kind.value
        if command.tags is not None:
            changes[_CHANGE_TAGS] = list(command.tags)
        if command.pinned is not None:
            changes[_CHANGE_PINNED] = command.pinned
        if content is not None or changes:
            await semantic.update(item_id, actor=_USER, content=content, **changes)
