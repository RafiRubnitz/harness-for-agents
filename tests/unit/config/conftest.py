"""Fixtures for config tests."""

from collections.abc import Callable
from pathlib import Path

import pytest

from harness.config import FileConfigManager
from harness.core.utils.paths import HarnessHome
from tests.consts import TEST_API_KEY, USER_ID
from tests.fakes.agent_layers import FakeAgentLayerStore
from tests.fakes.secrets import FakeSecretStore

SECRET_NAME = "anthropic"

type ManagerFactory = Callable[..., FileConfigManager]


@pytest.fixture
def secrets() -> FakeSecretStore:
    """A secret store holding one provider key."""
    return FakeSecretStore({SECRET_NAME: TEST_API_KEY})


@pytest.fixture
def layer_store() -> FakeAgentLayerStore:
    """An in-memory agent layer store."""
    return FakeAgentLayerStore()


@pytest.fixture
def make_manager(
    tmp_harness_home: HarnessHome, secrets: FakeSecretStore, layer_store: FakeAgentLayerStore
) -> ManagerFactory:
    """Build a manager over the temporary home with an explicit environment."""

    def build(env: dict[str, str] | None = None) -> FileConfigManager:
        return FileConfigManager(
            tmp_harness_home,
            secrets=secrets,
            env=env or {},
            store=layer_store,
            username=USER_ID,
        )

    return build


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """A working directory that has a ``.harness/`` project folder."""
    root = tmp_path / "project"
    (root / ".harness").mkdir(parents=True)
    return root
