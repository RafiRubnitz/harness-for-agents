"""Fixtures for integration tests: the real core with a scripted model and tools."""

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field

import pytest

from harness.app import CoreOptions, open_core
from harness.config import FileConfigManager
from harness.core.events import Event
from harness.core.handle import CoreHandle
from harness.core.utils.paths import HarnessHome
from tests.consts import USER_ID
from tests.fakes.embedder import FakeEmbedder
from tests.fakes.gateway import FakeGateway
from tests.fakes.secrets import FakeSecretStore
from tests.fakes.tools import FakeToolsManager


@dataclass
class CoreKit:
    """How to start the real core over a temporary home, plus its scripted edges."""

    home: HarnessHome
    gateway: FakeGateway
    tools: FakeToolsManager
    embedder: FakeEmbedder = field(default_factory=FakeEmbedder)
    secrets: FakeSecretStore = field(default_factory=FakeSecretStore)

    def launch(self) -> AbstractAsyncContextManager[CoreHandle]:
        """Start a core; every call builds a fresh config manager, like a new process.

        The secret store outlives the core, as the OS keyring does.
        """
        config = FileConfigManager(self.home, secrets=self.secrets, env={}, username=USER_ID)
        options = CoreOptions(
            home=self.home,
            config=config,
            gateway=self.gateway,
            tools=self.tools,
            embedder=self.embedder,
        )
        return open_core(options)


@pytest.fixture
def kit(tmp_harness_home: HarnessHome) -> CoreKit:
    return CoreKit(home=tmp_harness_home, gateway=FakeGateway(), tools=FakeToolsManager())


@pytest.fixture
async def core(kit: CoreKit) -> AsyncIterator[CoreHandle]:
    async with kit.launch() as handle:
        yield handle


async def next_payload[T](events: AsyncIterator[Event], kind: type[T]) -> T:
    """Read the stream until a payload of ``kind`` arrives."""
    async for event in events:
        if isinstance(event.payload, kind):
            return event.payload
    message = f"stream ended before {kind.__name__}"
    raise AssertionError(message)
