"""Fixtures for context manager tests: the real manager over the real store and fake edges."""

from dataclasses import dataclass, field

import pytest
from pydantic import JsonValue

from harness.config import FileConfigManager
from harness.context import ContextDeps, DefaultContextManager
from harness.core.consts import ConfigLayer, ErrorCode, MessageRole
from harness.core.records import AgentRecord, MessageRecord, RunRecord, SummaryRecord
from harness.core.types import (
    AgentId,
    ContentBlock,
    ModelRequest,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
)
from harness.core.utils.paths import HarnessHome
from harness.memory import Memory, MemoryStore
from tests.consts import AGENT_ID, RUN_ID, USER_ID
from tests.fakes.adapter import text_response
from tests.fakes.clock import FakeClock
from tests.fakes.gateway import FakeGateway
from tests.fakes.hooks import FakeHooksManager
from tests.fakes.ids import FakeIds
from tests.fakes.memory import FakeMemoryStore
from tests.fakes.recall import FakeRecall
from tests.fakes.secrets import FakeSecretStore
from tests.fakes.tools import FakeToolsManager
from tests.utils import make_agent, make_message

SMALL_WINDOW = 10_000
SUMMARY_TEXT = "SUMMARY: the user wants a report; the draft is in report.md"
TOOL_NAME = "lookup"
BYTES_PER_TOKEN = 4


def request_tokens(request: ModelRequest) -> int:
    """What a provider would count: the size of everything that is sent."""
    return len(request.model_dump_json(exclude={"metadata"}).encode("utf-8")) // BYTES_PER_TOKEN


@dataclass
class Notices:
    """What the context manager told the front ends."""

    summaries: list[SummaryRecord] = field(default_factory=list[SummaryRecord])
    warnings: list[tuple[ErrorCode, str]] = field(default_factory=list[tuple[ErrorCode, str]])

    def compacted(self, agent_id: AgentId, summary: SummaryRecord) -> None:
        self.summaries.append(summary)

    def warning(self, agent_id: AgentId, code: ErrorCode, message: str) -> None:
        self.warnings.append((code, message))


@dataclass
class ContextEnv:
    """A context manager with every collaborator reachable by the test."""

    manager: DefaultContextManager
    store: MemoryStore
    config: FileConfigManager
    gateway: FakeGateway
    hooks: FakeHooksManager
    tools: FakeToolsManager
    recall: FakeRecall
    notices: Notices
    clock: FakeClock
    count: int = 0

    async def agent(self, **updates: JsonValue) -> AgentRecord:
        """Save the test agent (again, with changes)."""
        record = make_agent().model_copy(update=updates)
        await self.store.save_agent(record)
        return record

    async def add(
        self,
        role: MessageRole,
        text: str = "",
        *,
        content: list[ContentBlock] | None = None,
        in_run: bool = False,
    ) -> MessageRecord:
        """Append one history message."""
        self.count += 1
        blocks = content if content is not None else [TextBlock(text=text)]
        message = make_message(
            f"msg_{self.count}", role=role, content=blocks, run_id=RUN_ID if in_run else None
        )
        return await self.store.append_message(message)

    async def user(self, text: str, *, in_run: bool = False) -> MessageRecord:
        return await self.add(MessageRole.USER, text, in_run=in_run)

    async def assistant(self, text: str, *, in_run: bool = False) -> MessageRecord:
        return await self.add(MessageRole.ASSISTANT, text, in_run=in_run)

    async def tool_step(self, result: str, *, result_id: str | None = None) -> str:
        """One tool-use step: the assistant calls a tool and the result follows."""
        self.count += 1
        call_id = f"call_{self.count}"
        call = ToolCallBlock(id=call_id, name=TOOL_NAME, input={"n": self.count})
        await self.add(MessageRole.ASSISTANT, content=[call])
        answer = ToolResultBlock(
            call_id=call_id, content=[TextBlock(text=result)], result_id=result_id
        )
        await self.add(MessageRole.TOOL, content=[answer])
        return call_id

    async def talk(self, turns: int, *, size: int = 1) -> None:
        """``turns`` question/answer pairs; ``size`` repeats the text to make them heavier."""
        for turn in range(turns):
            await self.user(f"question {turn} " * size)
            await self.assistant(f"answer {turn} " * size)

    async def open_run(self, first_seq: int) -> RunRecord:
        """Open the run that starts at ``first_seq``."""
        run = RunRecord(
            id=RUN_ID, agent_id=AGENT_ID, first_seq=first_seq, started_at=self.clock.now()
        )
        await self.store.save_run(run)
        return run

    async def set(self, key: str, value: JsonValue) -> None:
        """Set one global config value."""
        await self.config.set(key, value, ConfigLayer.GLOBAL)

    def shrink_window(self, window: int = SMALL_WINDOW) -> None:
        """Switch the agent's model to one with a small window."""
        self.gateway.profile = self.gateway.profile.model_copy(update={"context_window": window})

    def script_summaries(self, count: int = 1, text: str = SUMMARY_TEXT) -> None:
        """Let the summary model answer ``count`` times."""
        self.gateway.outcomes.extend(text_response(text) for _ in range(count))


async def _build(
    store: MemoryStore, home: HarnessHome, clock: FakeClock, ids: FakeIds
) -> ContextEnv:
    config = FileConfigManager(
        home, secrets=FakeSecretStore(), env={}, store=store, username=USER_ID
    )
    gateway, hooks, tools = FakeGateway(), FakeHooksManager(), FakeToolsManager()
    gateway.counter = request_tokens
    recall, notices = FakeRecall(), Notices()
    manager = DefaultContextManager(
        ContextDeps(
            memory=store,
            config=config,
            gateway=gateway,
            tools=tools,
            hooks=hooks,
            clock=clock,
            ids=ids,
            recall=recall,
            notify=notices,
        )
    )
    built = ContextEnv(
        manager=manager,
        store=store,
        config=config,
        gateway=gateway,
        hooks=hooks,
        tools=tools,
        recall=recall,
        notices=notices,
        clock=clock,
    )
    await built.agent()
    return built


@pytest.fixture
async def env(
    memory: Memory, tmp_harness_home: HarnessHome, clock: FakeClock, ids: FakeIds
) -> ContextEnv:
    """The real context manager over SQLite, with an agent already saved."""
    return await _build(memory.store, tmp_harness_home, clock, ids)


@pytest.fixture
async def bulk_env(tmp_harness_home: HarnessHome, clock: FakeClock, ids: FakeIds) -> ContextEnv:
    """The same over the in-memory store, for tests that write thousands of messages."""
    return await _build(FakeMemoryStore(clock, ids), tmp_harness_home, clock, ids)
