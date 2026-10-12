"""The config watcher: a changed file is reloaded, a broken one keeps the old config."""

import asyncio
from collections.abc import AsyncGenerator, Sequence
from pathlib import Path

import pytest

from harness.config import ConfigIssue, ConfigWatcher, FileConfigManager
from harness.core.utils.paths import HarnessHome
from tests.consts import AGENT_ID
from tests.unit.config.conftest import ManagerFactory

MODEL_KEY = "models.main.id"
NEW_MODEL = "claude-test-next"
OLD_RETRIES = 6
NEW_RETRIES = 7
ENCODING = "utf-8"
WAIT_S = 8.0


class FakeWatch:
    """Stands in for the file-system watcher: the test says which files changed."""

    def __init__(self) -> None:
        self.rounds: list[list[Path]] = []
        self._changes: asyncio.Queue[set[Path]] = asyncio.Queue()
        self._handled = asyncio.Event()

    def __call__(self, paths: Sequence[Path], stop: asyncio.Event) -> AsyncGenerator[set[Path]]:
        self.rounds.append(list(paths))
        return self._round(stop)

    async def _round(self, stop: asyncio.Event) -> AsyncGenerator[set[Path]]:
        while not stop.is_set():
            changed = await self._changes.get()
            try:
                yield changed
            finally:  # resumed or closed: the watcher is done with this batch
                self._handled.set()

    async def changed(self, *paths: Path) -> None:
        """Report changed files and wait until the watcher dealt with them."""
        self._handled.clear()
        await self._changes.put(set(paths))
        await self._handled.wait()


class Watched:
    """A config manager under a watcher, with what the watcher announced."""

    def __init__(self, manager: FileConfigManager, home: HarnessHome) -> None:
        self.manager = manager
        self.home = home
        self.watch = FakeWatch()
        self.announced: list[list[str]] = []
        self.rejected: list[ConfigIssue] = []
        self.watcher = ConfigWatcher(manager, self.announced.append, watch=self.watch)
        manager.subscribe_errors(self._on_error)

    async def _on_error(self, issues: list[ConfigIssue]) -> None:
        self.rejected.extend(issues)

    async def write(self, text: str, path: Path | None = None) -> None:
        """Change a config file and let the watcher see it."""
        target = path or self.home.config_file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding=ENCODING)
        await self.watch.changed(target)


@pytest.fixture
async def watched(
    make_manager: ManagerFactory, tmp_harness_home: HarnessHome
) -> AsyncGenerator[Watched]:
    env = Watched(make_manager(), tmp_harness_home)
    env.watcher.start()
    yield env
    await env.watcher.stop()


@pytest.mark.acceptance("06-11.1")
async def test_a_changed_model_alias_is_live_without_a_restart(watched: Watched) -> None:
    before = watched.manager.get(MODEL_KEY)

    await watched.write(f"models:\n  main:\n    provider: anthropic\n    id: {NEW_MODEL}\n")

    assert before != NEW_MODEL
    assert watched.manager.get(MODEL_KEY) == NEW_MODEL
    assert watched.manager.get(MODEL_KEY, AGENT_ID) == NEW_MODEL  # every agent reads it next step
    assert watched.announced == [[MODEL_KEY]]


@pytest.mark.acceptance("06-11.2")
async def test_a_syntax_error_is_reported_with_its_line_and_the_old_config_stays(
    watched: Watched,
) -> None:
    await watched.write(f"limits:\n  retries: {OLD_RETRIES}\n")

    await watched.write(f"limits:\n  retries: {NEW_RETRIES}\n  budget: [unclosed\n")

    (issue,) = watched.rejected
    assert watched.manager.get("limits.retries") == OLD_RETRIES
    assert issue.file == str(watched.home.config_file)
    assert issue.line is not None
    assert len(watched.announced) == 1  # only the valid change was announced

    await watched.write(f"limits:\n  retries: {NEW_RETRIES}\n")  # fixing the file recovers

    assert watched.manager.get("limits.retries") == NEW_RETRIES


async def test_files_that_are_not_config_do_not_cause_a_reload(watched: Watched) -> None:
    watched.home.config_file.write_text(f"limits:\n  retries: {NEW_RETRIES}\n", encoding=ENCODING)

    await watched.watch.changed(watched.home.root / "core.lock")

    assert watched.manager.get("limits.retries") != NEW_RETRIES
    assert watched.announced == []


async def test_a_project_is_watched_once_an_agent_works_in_it(
    watched: Watched, project_dir: Path
) -> None:
    await watched.manager.load_agent(AGENT_ID, project_dir)

    await watched.watch.changed()

    first, second = watched.watch.rounds
    assert project_dir / ".harness" not in first
    assert project_dir / ".harness" in second
    assert watched.home.root in second


async def test_a_reload_that_changes_nothing_announces_nothing(watched: Watched) -> None:
    await watched.watcher.reload()

    assert watched.announced == []


async def test_the_real_file_watcher_sees_a_saved_file(
    make_manager: ManagerFactory, tmp_harness_home: HarnessHome
) -> None:
    manager = make_manager()
    seen = asyncio.Event()
    watcher = ConfigWatcher(manager, lambda keys: seen.set())
    watcher.start()
    await asyncio.sleep(0.5)  # let the OS watch start before the file is written

    tmp_harness_home.config_file.write_text(
        f"limits:\n  retries: {NEW_RETRIES}\n", encoding=ENCODING
    )
    async with asyncio.timeout(WAIT_S):
        await seen.wait()
    await watcher.stop()

    assert manager.get("limits.retries") == NEW_RETRIES
