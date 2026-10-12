"""Fixtures for tools tests: the real tools manager over fakes and a temporary work tree."""

from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import JsonValue

from harness.config import FileConfigManager
from harness.core.consts import AgentStatus, ConfigLayer, Decision, PermissionAnswer
from harness.core.errors import AgentNotFoundError
from harness.core.records import AgentSettings, AgentState, PermissionDecision, ToolOutput
from harness.core.types import AgentId, TaskId, TextBlock, ToolCallBlock
from harness.core.utils.paths import HarnessHome
from harness.tools import DefaultToolsManager, ToolsDeps
from harness.tools.builtin import delete as delete_module
from tests.consts import AGENT_ID, AGENT_NAME, RUN_ID, TEST_MODEL_ALIAS, USER_ID
from tests.fakes.agent_layers import FakeAgentLayerStore
from tests.fakes.clock import FakeClock
from tests.fakes.hooks import FakeHooksManager
from tests.fakes.ids import FakeIds
from tests.fakes.memory import FakeBlobStore
from tests.fakes.secrets import FakeSecretStore

REASON = "the test needs it"
RULES_KEY = "permissions.rules"
ALL_TOOLS = ["files", "shell"]


@dataclass
class ToolsEnv:
    """A tools manager with every collaborator reachable by the test."""

    manager: DefaultToolsManager
    deps: ToolsDeps
    config: FileConfigManager
    hooks: FakeHooksManager
    blobs: FakeBlobStore
    work: Path
    clock: FakeClock
    agents: dict[AgentId, AgentState] = field(default_factory=dict[AgentId, AgentState])
    progress: list[str] = field(default_factory=list[str])
    trashed: list[str] = field(default_factory=list[str])
    calls: int = 0

    def agent(self, tools: list[str] | None = None, **settings: JsonValue) -> AgentState:
        """Register (or replace) the test agent."""
        merged: dict[str, JsonValue] = {"working_dir": str(self.work), **settings}
        state = AgentState(
            id=AGENT_ID,
            name=AGENT_NAME,
            model=TEST_MODEL_ALIAS,
            template=None,
            status=AgentStatus.RUNNING,
            settings=AgentSettings.model_validate(merged),
            tools=ALL_TOOLS if tools is None else tools,
            run_id=RUN_ID,
            step=1,
            pending_approvals=0,
            background_tasks=0,
            cost_usd_total=0.0,
            context_tokens=None,
            created_at=self.clock.now(),
            updated_at=self.clock.now(),
        )
        self.agents[state.id] = state
        return state

    def lookup(self, agent_id: AgentId) -> AgentState:
        if agent_id not in self.agents:
            raise AgentNotFoundError(agent_id)
        return self.agents[agent_id]

    def call(self, tool: str, *, reason: str | None = REASON, **args: JsonValue) -> ToolCallBlock:
        """Build a tool call the way the model sends it (reason inside the input)."""
        self.calls += 1
        raw: dict[str, JsonValue] = dict(args)
        if reason is not None:
            raw["reason"] = reason
        return ToolCallBlock(id=f"call_{self.calls}", name=tool, input=raw, reason=reason)

    async def decide(
        self, tool: str, *, reason: str | None = REASON, **args: JsonValue
    ) -> PermissionDecision:
        return await self.manager.decide(AGENT_ID, self.call(tool, reason=reason, **args))

    async def run(
        self, tool: str, *, approve: bool = True, reason: str | None = REASON, **args: JsonValue
    ) -> ToolOutput:
        """Decide and execute; an ``ask`` is answered with ``approve``."""
        call = self.call(tool, reason=reason, **args)
        decision = await self.manager.decide(AGENT_ID, call)
        if decision.decision is Decision.ASK:
            answer = PermissionAnswer.APPROVE if approve else PermissionAnswer.DENY
            decision = decision.model_copy(update={"user_answer": answer})
        return await self.manager.execute(AGENT_ID, call, decision)

    async def rules(self, *rules: dict[str, JsonValue], layer: ConfigLayer) -> None:
        """Replace the permission rules of one layer."""
        agent_id = AGENT_ID if layer is ConfigLayer.AGENT else None
        await self.config.set(RULES_KEY, list(rules), layer, agent_id)

    def write(self, relative: str, text: str) -> Path:
        """Create a file in the working directory."""
        path = self.work / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
        return path


def text_of(output: ToolOutput) -> str:
    """The text blocks of a tool output, joined."""
    return "".join(block.text for block in output.content if isinstance(block, TextBlock))


@pytest.fixture
async def env(
    tmp_harness_home: HarnessHome,
    tmp_path: Path,
    clock: FakeClock,
    ids: FakeIds,
    monkeypatch: pytest.MonkeyPatch,
) -> ToolsEnv:
    """The real tools manager; the recycle bin is replaced so tests leave nothing behind."""
    work = tmp_path / "work"
    work.mkdir()
    config = FileConfigManager(
        tmp_harness_home,
        secrets=FakeSecretStore(),
        env={},
        store=FakeAgentLayerStore(),
        username=USER_ID,
    )
    hooks, blobs = FakeHooksManager(), FakeBlobStore(ids)
    built: list[ToolsEnv] = []

    def progress(agent_id: AgentId, call_id: str, task_id: TaskId | None, chunk: str) -> None:
        built[0].progress.append(chunk)

    def lookup(agent_id: AgentId) -> AgentState:
        return built[0].lookup(agent_id)

    def fake_trash(path: str) -> None:
        built[0].trashed.append(path)
        Path(path).unlink()

    monkeypatch.setattr(delete_module, "send2trash", fake_trash)
    deps = ToolsDeps(
        config=config,
        hooks=hooks,
        blobs=blobs,
        clock=clock,
        ids=ids,
        home=tmp_harness_home,
        agent=lookup,
        progress=progress,
    )
    built.append(
        ToolsEnv(
            manager=DefaultToolsManager(deps),
            deps=deps,
            config=config,
            hooks=hooks,
            blobs=blobs,
            work=work,
            clock=clock,
        )
    )
    built[0].agent()
    await config.load_agent(AGENT_ID, work)
    return built[0]
