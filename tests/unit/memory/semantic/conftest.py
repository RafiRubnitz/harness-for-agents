"""Fixtures for semantic memory tests: the real memory over real LanceDB tables in tmp."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime

import pytest
from pydantic import JsonValue

from harness.config import FileConfigManager
from harness.core.consts import ConfigLayer, ErrorCode, MemoryKind, SearchMode
from harness.core.types import AgentId, MemoryItemId
from harness.core.utils.paths import HarnessHome
from harness.memory import DefaultSemanticMemory, Semantic, SemanticEdges, open_semantic
from tests.consts import AGENT_ID, OTHER_AGENT_ID, USER_ID
from tests.fakes.clock import FakeClock
from tests.fakes.embedder import FakeEmbedder
from tests.fakes.hooks import FakeHooksManager
from tests.fakes.ids import FakeIds
from tests.fakes.memory import FakeMemoryStore
from tests.fakes.secrets import FakeSecretStore
from tests.utils import make_agent

Warnings = list[tuple[ErrorCode, str]]


@dataclass
class SemEnv:
    """A semantic memory with every collaborator reachable by the test."""

    home: HarnessHome
    edges: SemanticEdges
    semantic: Semantic
    embedder: FakeEmbedder
    hooks: FakeHooksManager
    history: FakeMemoryStore
    config: FileConfigManager
    clock: FakeClock
    warnings: Warnings

    @property
    def memory(self) -> DefaultSemanticMemory:
        return self.semantic.memory

    async def reopen(self) -> None:
        """Close the tables and open them again, as an app restart does."""
        await self.semantic.close()
        self.semantic = await open_semantic(self.home, self.edges, embedder=self.embedder)

    async def save(
        self, content: str, *, agent_id: AgentId | None = AGENT_ID, tags: tuple[str, ...] = ()
    ) -> MemoryItemId:
        """Remember a fact for the test agent (or another owner)."""
        return await self.memory.remember(agent_id, content, MemoryKind.FACT, tags=tags)

    async def found(
        self,
        query: str,
        *,
        agent_id: AgentId = AGENT_ID,
        mode: SearchMode = SearchMode.HYBRID,
        include_expired: bool = False,
        as_of: datetime | None = None,
    ) -> list[str]:
        """The contents a search returns, best first."""
        hits = await self.memory.search(
            agent_id, query, mode=mode, include_expired=include_expired, as_of=as_of
        )
        return [hit.item.content for hit in hits]

    async def set(self, key: str, value: JsonValue) -> None:
        await self.config.set(key, value, ConfigLayer.GLOBAL)


@pytest.fixture
async def sem(
    tmp_harness_home: HarnessHome, clock: FakeClock, ids: FakeIds
) -> AsyncIterator[SemEnv]:
    """The real semantic memory with a fake embedder and two saved agents."""
    history = FakeMemoryStore(clock, ids)
    for agent_id in (AGENT_ID, OTHER_AGENT_ID):
        await history.save_agent(make_agent(agent_id))
    config = FileConfigManager(
        tmp_harness_home, secrets=FakeSecretStore(), env={}, store=history, username=USER_ID
    )
    embedder, hooks, warnings = FakeEmbedder(), FakeHooksManager(), Warnings()
    edges = SemanticEdges(
        config=config,
        hooks=hooks,
        history=history,
        clock=clock,
        ids=ids,
        warn=lambda code, message: warnings.append((code, message)),
    )
    env = SemEnv(
        home=tmp_harness_home,
        edges=edges,
        semantic=await open_semantic(tmp_harness_home, edges, embedder=embedder),
        embedder=embedder,
        hooks=hooks,
        history=history,
        config=config,
        clock=clock,
        warnings=warnings,
    )
    yield env
    await env.semantic.close()
