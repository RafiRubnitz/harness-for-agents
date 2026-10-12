"""Reloads the config when one of its files changes on disk (06 §7)."""

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import aclosing
from pathlib import Path
from typing import Protocol

from loguru import logger
from watchfiles import awatch  # pyright: ignore[reportUnknownVariableType]

from harness.config.consts import WATCH_DEBOUNCE_MS, WATCH_POLL_MS, WATCHED_SUFFIXES
from harness.config.schema import HarnessConfig

Watch = Callable[[Sequence[Path], asyncio.Event], AsyncGenerator[set[Path]]]


class Reloadable(Protocol):
    """The part of the config manager the watcher drives."""

    def watch_dirs(self) -> list[Path]:
        """The directories whose files make up the config right now."""
        ...

    def subscribe(self, keys: list[str], callback: Callable[[list[str]], Awaitable[None]]) -> None:
        """Register for changes."""
        ...

    async def reload(self) -> None:
        """Re-read every file."""
        ...


async def watch_files(paths: Sequence[Path], stop: asyncio.Event) -> AsyncGenerator[set[Path]]:
    """Yield the changed files of ``paths``; an empty set is a periodic wake-up."""
    async for changes in awatch(
        *paths,
        stop_event=stop,
        recursive=False,
        debounce=WATCH_DEBOUNCE_MS,
        rust_timeout=WATCH_POLL_MS,
        yield_on_timeout=True,
    ):
        yield {Path(path) for _, path in changes}


class ConfigWatcher:
    """Watches the config directories and reloads on every relevant change.

    The set of directories follows the agents: a project joins when an agent starts
    working in it.
    """

    def __init__(
        self,
        config: Reloadable,
        on_changed: Callable[[list[str]], None],
        *,
        watch: Watch = watch_files,
    ) -> None:
        self._config = config
        self._on_changed = on_changed
        self._watch = watch
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._changed: list[str] | None = None
        config.subscribe(list(HarnessConfig.model_fields), self._collect)

    async def _collect(self, keys: list[str]) -> None:
        if self._changed is not None:
            self._changed.extend(keys)

    def start(self) -> None:
        """Begin watching; a second call changes nothing."""
        if self._task is None:
            self._stop.clear()
            self._task = asyncio.create_task(self._run(), name="config-watcher")

    async def stop(self) -> None:
        """Stop watching."""
        task, self._task = self._task, None
        if task is None:
            return
        self._stop.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def reload(self) -> None:
        """Reload now and announce what changed because of it."""
        self._changed = []
        try:
            await self._config.reload()
            changed = sorted(set(self._changed))
        finally:
            self._changed = None
        if changed:
            self._on_changed(changed)

    def _dirs(self) -> list[Path]:
        return sorted(path for path in self._config.watch_dirs() if path.is_dir())

    async def _run(self) -> None:
        while not self._stop.is_set():
            dirs = self._dirs()
            round_stop = asyncio.Event()
            async with aclosing(self._watch(dirs, round_stop)) as rounds:
                async for changed in rounds:
                    if any(path.suffix.lower() in WATCHED_SUFFIXES for path in changed):
                        await self._reload_safely()
                    if self._stop.is_set() or self._dirs() != dirs:
                        round_stop.set()  # stopping, or an agent entered or left a project
                        break

    async def _reload_safely(self) -> None:
        try:
            await self.reload()
        except asyncio.CancelledError:
            raise
        except Exception as error:  # the watcher must survive a bad file
            logger.warning("config reload after a file change failed: {}", error)
