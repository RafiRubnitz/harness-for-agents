"""Fixtures for agent manager tests: a manager over fakes, on both memory stores."""

import asyncio
import dataclasses
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

import pytest
from pydantic import JsonValue

from harness.agents import AgentDeps, DefaultAgentManager, EventBus, Subscription
from harness.agents.consts import TASK_NAME_PREFIX
from harness.config import FileConfigManager
from harness.context import ContextDeps, DefaultContextManager
from harness.core.consts import AgentStatus, EventName, HookEvent
from harness.core.events import Event
from harness.core.records import AgentSettings, AgentSpec
from harness.core.types import AgentId, TextBlock
from harness.core.utils.names import free_name
from harness.core.utils.paths import HarnessHome
from harness.memory import Memory, MemoryStore
from tests.consts import AGENT_NAME, USER_ID
from tests.fakes.clock import FakeClock
from tests.fakes.gateway import FakeGateway
from tests.fakes.hooks import FakeHooksManager
from tests.fakes.ids import FakeIds
from tests.fakes.memory import FakeMemoryStore
from tests.fakes.secrets import FakeSecretStore
from tests.fakes.tools import FakeToolsManager

WAIT_TIMEOUT_S = 5.0
POLL_S = 0.005


async def until(predicate: Callable[[], bool]) -> None:
    """Wait until ``predicate`` holds (the agent runs in its own task)."""
    async with asyncio.timeout(WAIT_TIMEOUT_S):
        while not predicate():
            await asyncio.sleep(POLL_S)


@dataclass
class AgentEnv:
    """A manager with every collaborator reachable by the test."""

    deps: AgentDeps
    manager: DefaultAgentManager
    events: Subscription
    gateway: FakeGateway
    hooks: FakeHooksManager
    tools: FakeToolsManager
    store: MemoryStore
    seen: list[Event] = field(default_factory=list[Event])
    managers: list[DefaultAgentManager] = field(default_factory=list[DefaultAgentManager])

    async def create(self, **settings: JsonValue) -> AgentId:
        """Create an agent with the given settings."""
        taken = {agent.name for agent in self.manager.list(include_archived=True)}
        spec = AgentSpec(
            name=free_name(AGENT_NAME, taken), settings=AgentSettings.model_validate(settings)
        )
        return await self.manager.create(spec)

    async def say(self, agent_id: AgentId, text: str) -> None:
        """Send a user text message."""
        await self.manager.send(agent_id, [TextBlock(text=text)])

    async def converse(self, agent_id: AgentId, text: str) -> None:
        """Send a message and wait until the agent's task is over."""
        await self.say(agent_id, text)
        await self.manager.wait_idle(agent_id)

    async def until_status(self, agent_id: AgentId, status: AgentStatus) -> None:
        """Wait until the agent reaches ``status``."""
        await until(lambda: self.manager.get(agent_id).status is status)

    def collected(self) -> list[Event]:
        """Every event published so far."""
        self.seen.extend(self.events.drain())
        return self.seen

    def payloads[T](self, kind: type[T]) -> list[T]:
        """The payloads of one type published so far, in order."""
        return [e.payload for e in self.collected() if isinstance(e.payload, kind)]

    def names(self) -> list[EventName]:
        """The names of the events published so far."""
        return [event.name for event in self.collected()]

    def stop_reasons(self) -> list[JsonValue]:
        """The ``stop_reason`` of every ``run_end`` hook fired so far."""
        return [
            inp.payload["stop_reason"]
            for inp in self.hooks.emitted
            if inp.event is HookEvent.RUN_END
        ]

    async def crash(self) -> None:
        """Kill every agent task without any cleanup, as a dying process would."""
        tasks = [t for t in asyncio.all_tasks() if t.get_name().startswith(TASK_NAME_PREFIX)]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def restart(self) -> DefaultAgentManager:
        """Build a fresh manager (and event stream) over the same stores."""
        self.deps = dataclasses.replace(self.deps, bus=EventBus(self.deps.clock))
        self.manager = DefaultAgentManager(self.deps)
        self.managers.append(self.manager)
        self.events = self.deps.bus.subscribe()
        self.seen = []
        return self.manager


@pytest.fixture(params=["fake", "sqlite"])
def store(
    request: pytest.FixtureRequest, memory: Memory, clock: FakeClock, ids: FakeIds
) -> MemoryStore:
    """The memory store under the manager: the fake and the real one."""
    return memory.store if request.param == "sqlite" else FakeMemoryStore(clock, ids)


@pytest.fixture
async def env(
    store: MemoryStore, tmp_harness_home: HarnessHome, clock: FakeClock, ids: FakeIds
) -> AsyncIterator[AgentEnv]:
    """A started manager over fake collaborators."""
    config = FileConfigManager(
        tmp_harness_home, secrets=FakeSecretStore(), env={}, store=store, username=USER_ID
    )
    gateway, hooks, tools = FakeGateway(), FakeHooksManager(), FakeToolsManager()
    deps = AgentDeps(
        memory=store,
        config=config,
        gateway=gateway,
        hooks=hooks,
        context=DefaultContextManager(
            ContextDeps(
                memory=store,
                config=config,
                gateway=gateway,
                tools=tools,
                hooks=hooks,
                clock=clock,
                ids=ids,
            )
        ),
        tools=tools,
        clock=clock,
        ids=ids,
        home=tmp_harness_home,
        bus=EventBus(clock),
    )
    manager = DefaultAgentManager(deps)
    await manager.start()
    built = AgentEnv(
        deps=deps,
        manager=manager,
        events=deps.bus.subscribe(),
        gateway=gateway,
        hooks=hooks,
        tools=tools,
        store=store,
        managers=[manager],
    )
    yield built
    await built.crash()
