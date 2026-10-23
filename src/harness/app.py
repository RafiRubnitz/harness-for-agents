"""Composition root: the only module that wires the components together."""

import secrets
import sys
from collections.abc import AsyncGenerator, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from importlib import metadata

from loguru import logger

from harness.agents import (
    AgentDeps,
    AgentManager,
    DefaultAgentManager,
    EventBus,
    MessagingService,
)
from harness.cli import run
from harness.config import (
    ConfigIssue,
    ConfigWatcher,
    FileConfigManager,
    LocalServerConfig,
    LoggingConfig,
    requires_restart,
)
from harness.context import ContextDeps, DefaultContextManager
from harness.core.consts import (
    APP_NAME,
    MSG_CORE_NOT_READY,
    MSG_TASK_DONE,
    AddressType,
    ErrorCode,
    HookEvent,
)
from harness.core.errors import (
    AgentNotFoundError,
    HarnessError,
    MemoryStoreError,
    SecretMissingError,
)
from harness.core.events import (
    ConfigChanged,
    ConfigErrorRaised,
    ContextCompacted,
    HookFired,
    HookLoadFailed,
    TaskStatusChanged,
    ToolProgress,
)
from harness.core.events import Warning as WarningEvent
from harness.core.handle import CoreHandle
from harness.core.hook_types import HookInput, HookRunRecord, HookSpec
from harness.core.lock import CoreLock
from harness.core.logging import setup_logging
from harness.core.messaging import AgentInfo, Messenger, SendMessageInput, SendMessageOutput
from harness.core.records import (
    AgentState,
    BackgroundTaskRecord,
    MemoryItem,
    SummaryRecord,
)
from harness.core.types import (
    Address,
    AgentId,
    Clock,
    IdGenerator,
    MemoryItemId,
    MessageOrigin,
    TaskId,
    TextBlock,
)
from harness.core.utils.ids import UlidGenerator
from harness.core.utils.paths import HarnessHome, resolve_home
from harness.core.utils.time import SystemClock, elapsed_ms
from harness.dispatcher import CommandDispatcher, CoreParts
from harness.gui import run_gui
from harness.hooks import DefaultHooksManager, HooksDeps, HooksManager
from harness.local_server import LocalServer, create_app
from harness.local_server import consts as local_server
from harness.memory import (
    Embedder,
    LedgerCallSink,
    Memory,
    Semantic,
    SemanticEdges,
    SemanticMemory,
    open_memory,
    open_semantic,
)
from harness.models import BudgetGuard, ModelGateway, build_gateway
from harness.tools import DefaultToolsManager, ToolsDeps, ToolsManager

_CMD_GUI = "gui"
# The agent manager announces the hooks of its own events (and stores them in the
# conversation); hooks fired from inside the other components are announced here.
_HOOKS_SHOWN_BY_AGENTS = frozenset(
    {
        HookEvent.AGENT_CREATED,
        HookEvent.AGENT_DELETED,
        HookEvent.MESSAGE_RECEIVED,
        HookEvent.RUN_START,
        HookEvent.BEFORE_MODEL_CALL,
        HookEvent.AFTER_MODEL_CALL,
        HookEvent.MODEL_ERROR,
        HookEvent.RUN_LIMIT_REACHED,
        HookEvent.RUN_END,
    }
)


@dataclass(frozen=True, slots=True)
class CoreOptions:
    """What a caller may replace when starting the core (tests swap the edges)."""

    home: HarnessHome | None = None
    config: FileConfigManager | None = None
    gateway: ModelGateway | None = None
    tools: ToolsManager | None = None
    hooks: HooksManager | None = None
    clock: Clock | None = None
    ids: IdGenerator | None = None
    embedder: Embedder | None = None
    console_logs: bool = False
    watch_config: bool = True


class _AgentDirectory:
    """Lets the tools manager look agents up; the agent manager is set once it exists."""

    def __init__(self) -> None:
        self.agents: AgentManager | None = None

    def __call__(self, agent_id: AgentId) -> AgentState:
        if self.agents is None:
            raise AgentNotFoundError(MSG_CORE_NOT_READY)
        return self.agents.get(agent_id)


class _Messenger:
    """Lets the tools reach the messaging service, which is built after them."""

    def __init__(self) -> None:
        self.service: Messenger | None = None

    def _ready(self) -> Messenger:
        if self.service is None:
            raise AgentNotFoundError(MSG_CORE_NOT_READY)
        return self.service

    def list_agents(self, viewer: Address) -> list[AgentInfo]:
        return self._ready().list_agents(viewer)

    async def send(self, sender: Address, message: SendMessageInput) -> SendMessageOutput:
        return await self._ready().send(sender, message)

    async def reply(self, sender: AgentId, message_id: str, content: str) -> SendMessageOutput:
        return await self._ready().reply(sender, message_id, content)


async def _local_server(
    stack: AsyncExitStack, config: FileConfigManager, messenger: Messenger
) -> None:
    """Start the HTTP entry point on 127.0.0.1 when ``local_server.enabled`` is on.

    The token is made on first use and kept in the keyring (02 §5a).
    """
    settings = config.section(LocalServerConfig)
    if not settings.enabled:
        return
    try:
        token = config.secret(local_server.TOKEN_SECRET)
    except SecretMissingError:
        token = secrets.token_urlsafe(local_server.TOKEN_BYTES)
        await config.set_secret(local_server.TOKEN_SECRET, token)
    server = LocalServer(create_app(messenger, token), settings.port)
    try:
        await server.start()
    except HarnessError as error:  # a busy port must not keep the harness from starting
        logger.error("{}", error.message)
        return
    stack.push_async_callback(server.stop)


class _ContextNotices:
    """Shows what the context manager did on the event bus."""

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    def compacted(self, agent_id: AgentId, summary: SummaryRecord) -> None:
        shown = ContextCompacted(
            from_seq=summary.from_seq, to_seq=summary.to_seq, summary_id=summary.id
        )
        self._bus.publish(agent_id, shown)

    def warning(self, agent_id: AgentId, code: ErrorCode, message: str) -> None:
        self._bus.publish(agent_id, WarningEvent(code=code, message=message))


class _MemoryWarnings:
    """Shows a degraded semantic memory (no embeddings server) to every front end."""

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    def __call__(self, code: ErrorCode, message: str) -> None:
        self._bus.publish(None, WarningEvent(code=code, message=message))


class _Recall:
    """The semantic memory as the context manager reads it (07 §3, §6)."""

    def __init__(self, semantic: SemanticMemory) -> None:
        self._semantic = semantic

    async def pinned(self, agent_id: AgentId) -> list[MemoryItem]:
        return await self._semantic.pinned(agent_id)

    async def relevant(self, agent_id: AgentId, query: str, *, k: int) -> list[MemoryItem]:
        """Search for the run's opening message; a failing memory never fails the request."""
        try:
            return [hit.item for hit in await self._semantic.search(agent_id, query, k=k)]
        except MemoryStoreError as error:
            logger.warning("memory retrieval for {} was skipped: {}", agent_id, error)
            return []

    async def by_ids(self, ids: Sequence[MemoryItemId]) -> list[MemoryItem]:
        return await self._semantic.by_ids(ids)


def _watch_config(
    options: CoreOptions, stack: AsyncExitStack, config: FileConfigManager, bus: EventBus
) -> None:
    """Reload the config when its files change and show the outcome (06 §7)."""

    def changed(keys: list[str]) -> None:
        bus.publish(None, ConfigChanged(keys=keys, restart_required=requires_restart(keys)))

    async def rejected(issues: list[ConfigIssue]) -> None:
        for issue in issues:
            shown = ConfigErrorRaised(file=issue.file or "", line=issue.line, message=issue.message)
            bus.publish(None, shown)

    config.subscribe_errors(rejected)
    if options.watch_config:
        watcher = ConfigWatcher(config, changed)
        watcher.start()
        stack.push_async_callback(watcher.stop)


class _HookNotices:
    """Shows hook runs and hooks that failed to load on the event bus (05 §7)."""

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    def load_failed(self, spec: HookSpec, message: str) -> None:
        shown = HookLoadFailed(hook=spec.name, file=spec.source_file or "", message=message)
        self._bus.publish(None, shown)

    def fired(self, inp: HookInput, record: HookRunRecord) -> None:
        if record.suppressed or record.event in _HOOKS_SHOWN_BY_AGENTS:
            return
        shown = HookFired(
            hook=record.hook,
            event=record.event,
            decision=record.decision,
            reason=record.reason,
            duration_ms=record.duration_ms,
        )
        self._bus.publish(inp.agent_id, shown)


async def _hooks(
    options: CoreOptions,
    stack: AsyncExitStack,
    *,
    config: FileConfigManager,
    clock: Clock,
    home: HarnessHome,
    bus: EventBus,
    directory: _AgentDirectory,
) -> HooksManager:
    if options.hooks is not None:
        return options.hooks
    notices = _HookNotices(bus)
    hooks = DefaultHooksManager(
        HooksDeps(
            config=config,
            clock=clock,
            home=home,
            agent=directory,
            load_failed=notices.load_failed,
            observe=notices.fired,
        )
    )
    stack.push_async_callback(hooks.close)
    await hooks.reload()
    return hooks


@dataclass(frozen=True, slots=True)
class _Edges:
    """What the tools manager is built from."""

    config: FileConfigManager
    hooks: HooksManager
    memory: Memory
    clock: Clock
    ids: IdGenerator
    home: HarnessHome
    bus: EventBus
    directory: _AgentDirectory
    semantic: Semantic
    stack: AsyncExitStack
    messenger: Messenger


def _tools(options: CoreOptions, edges: _Edges) -> ToolsManager:
    if options.tools is not None:
        return options.tools
    bus = edges.bus

    def progress(agent_id: AgentId, call_id: str, task_id: TaskId | None, chunk: str) -> None:
        bus.publish(agent_id, ToolProgress(call_id=call_id, task_id=task_id, chunk=chunk))

    def task_status(record: BackgroundTaskRecord) -> None:
        end = record.ended_at or edges.clock.now()
        shown = TaskStatusChanged(
            task_id=record.id,
            call_id=record.call_id,
            tool=record.tool,
            status=record.status,
            duration_ms=elapsed_ms(record.started_at, end),
        )
        bus.publish(record.agent_id, shown)

    async def task_done(record: BackgroundTaskRecord, summary: str) -> None:
        agents = edges.directory.agents
        if agents is None:
            return
        text = MSG_TASK_DONE.format(
            task_id=record.id, tool=record.tool, status=record.status.value, summary=summary
        )
        origin = MessageOrigin(source=Address(type=AddressType.TASK, id=record.id))
        try:
            await agents.send(record.agent_id, [TextBlock(text=text)], origin)
        except HarnessError as error:  # deleted or archived meanwhile: nobody to tell
            logger.info("task {} ended without a reader: {}", record.id, error.message)

    tools = DefaultToolsManager(
        ToolsDeps(
            config=edges.config,
            hooks=edges.hooks,
            blobs=edges.memory.blobs,
            clock=edges.clock,
            ids=edges.ids,
            home=edges.home,
            agent=edges.directory,
            progress=progress,
            semantic=edges.semantic.memory,
            history=edges.memory.store,
            task_status=task_status,
            task_done=task_done,
            messenger=edges.messenger,
        )
    )
    edges.stack.push_async_callback(tools.close)
    return tools


def _gateway(
    options: CoreOptions,
    *,
    config: FileConfigManager,
    clock: Clock,
    home: HarnessHome,
    memory: Memory,
    bus: EventBus,
) -> ModelGateway:
    if options.gateway is not None:
        return options.gateway

    def warn(agent_id: AgentId | None, message: str) -> None:
        bus.publish(agent_id, WarningEvent(code=ErrorCode.MODEL_BUDGET_WARNING, message=message))

    gateway = build_gateway(config, clock, home.logs_dir)
    gateway.add_sink(LedgerCallSink(memory.store))
    gateway.set_budget(BudgetGuard(config, memory.store, clock, warn))
    return gateway


@asynccontextmanager
async def open_core(options: CoreOptions | None = None) -> AsyncGenerator[CoreHandle]:
    """Start the core under ``core.lock`` and shut it down in order on exit.

    Raises:
        CoreLockedError: Another process already runs the core; nothing was touched.
    """
    options = options or CoreOptions()
    home = options.home or resolve_home()
    with CoreLock(home.lock_file):
        home.ensure()
        clock, ids = options.clock or SystemClock(), options.ids or UlidGenerator()
        config = options.config or FileConfigManager(home)
        level = config.section(LoggingConfig).level
        setup_logging(level, home.logs_dir, console=options.console_logs)
        async with AsyncExitStack() as stack:
            memory = await open_memory(home, clock, ids, app_version=metadata.version(APP_NAME))
            stack.push_async_callback(memory.close)
            config.attach_store(memory.store)
            bus = EventBus(clock)
            gateway = _gateway(
                options, config=config, clock=clock, home=home, memory=memory, bus=bus
            )
            _watch_config(options, stack, config, bus)
            directory = _AgentDirectory()
            hooks = await _hooks(
                options,
                stack,
                config=config,
                clock=clock,
                home=home,
                bus=bus,
                directory=directory,
            )
            semantic = await open_semantic(
                home,
                SemanticEdges(
                    config=config,
                    hooks=hooks,
                    history=memory.store,
                    clock=clock,
                    ids=ids,
                    warn=_MemoryWarnings(bus),
                ),
                embedder=options.embedder,
            )
            stack.push_async_callback(semantic.close)
            messenger = _Messenger()
            tools = _tools(
                options,
                _Edges(
                    config,
                    hooks,
                    memory,
                    clock,
                    ids,
                    home,
                    bus,
                    directory,
                    semantic,
                    stack,
                    messenger,
                ),
            )
            await tools.reload()  # custom tool files and MCP servers
            context = DefaultContextManager(
                ContextDeps(
                    memory=memory.store,
                    config=config,
                    gateway=gateway,
                    tools=tools,
                    hooks=hooks,
                    clock=clock,
                    ids=ids,
                    recall=_Recall(semantic.memory),
                    notify=_ContextNotices(bus),
                )
            )
            deps = AgentDeps(
                memory=memory.store,
                config=config,
                gateway=gateway,
                hooks=hooks,
                context=context,
                tools=tools,
                clock=clock,
                ids=ids,
                home=home,
                bus=bus,
                semantic=semantic.memory,
            )
            agents = DefaultAgentManager(deps)
            directory.agents = agents
            messenger.service = MessagingService(
                agents, memory=memory.store, config=config, clock=clock, ids=ids, bus=bus
            )
            await agents.start()
            agents.watch_inbox()
            await _local_server(stack, config, messenger)
            semantic.start()
            try:
                yield CommandDispatcher(
                    CoreParts(
                        agents, config, memory.store, context, tools, bus, home, semantic.memory
                    )
                )
            finally:
                await agents.shutdown(stop_agents=False)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``harness`` command: the GUI by default, the CLI for subcommands."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args == [_CMD_GUI]:
        return run_gui(open_core)
    return run(args, open_core)


if __name__ == "__main__":
    sys.exit(main())
