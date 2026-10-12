"""Supervisor of the managed embeddings server process (03 §6.4, 11 §4a).

Starts ``python -m harness.embedder_server`` on the loopback interface, starts it again
when it exits, and stops it when the app closes. Nothing here waits for the model to
load: the memory layer works lexically until ``/health`` says ``ok``.
"""

import asyncio
import contextlib
import sys
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Protocol

from loguru import logger

from harness.core.consts import ErrorCode
from harness.core.embedding import EMBEDDER_ARG_MODEL, EMBEDDER_ARG_PORT, EMBEDDER_MODULE
from harness.memory import consts
from harness.memory.semantic.types import Warn


class ServerProcess(Protocol):
    """The part of a child process the supervisor uses."""

    @property
    def returncode(self) -> int | None: ...
    async def wait(self) -> int: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...


Spawn = Callable[[Sequence[str], Path], Awaitable[ServerProcess]]


async def spawn_process(args: Sequence[str], log_file: Path) -> ServerProcess:
    """Start the server as a child process, its output appended to ``log_file``."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("ab") as log:
        return await asyncio.create_subprocess_exec(
            *args, stdin=asyncio.subprocess.DEVNULL, stdout=log, stderr=asyncio.subprocess.STDOUT
        )


class EmbedderServer:
    """Keeps one embeddings server process alive."""

    def __init__(
        self,
        *,
        port: int,
        model: str,
        log_file: Path,
        warn: Warn | None = None,
        spawn: Spawn = spawn_process,
        restart_delay_s: float = consts.EMBEDDER_WATCH_INTERVAL_S,
    ) -> None:
        self._args = [
            sys.executable,
            "-m",
            EMBEDDER_MODULE,
            EMBEDDER_ARG_PORT,
            str(port),
            EMBEDDER_ARG_MODEL,
            model,
        ]
        self._log_file = log_file
        self._warn = warn
        self._spawn = spawn
        self._restart_delay_s = restart_delay_s
        self._process: ServerProcess | None = None
        self._watcher: asyncio.Task[None] | None = None
        self.starts = 0

    def start(self) -> None:
        """Start the server in the background; returns at once."""
        if self._watcher is None:
            self._watcher = asyncio.create_task(self._watch())

    async def _watch(self) -> None:
        """Run the process; start it again when it exits, a limited number of times."""
        while self.starts <= consts.EMBEDDER_MAX_RESTARTS:
            try:
                self._process = await self._spawn(self._args, self._log_file)
            except OSError as error:
                logger.error("the embeddings server could not be started: {}", error)
                break
            self.starts += 1
            code = await self._process.wait()
            logger.warning("the embeddings server exited with code {}", code)
            await asyncio.sleep(self._restart_delay_s)
        logger.error(consts.MSG_EMBEDDER_GAVE_UP)
        if self._warn is not None:
            self._warn(ErrorCode.MEMORY_EMBEDDER_UNAVAILABLE, consts.MSG_EMBEDDER_GAVE_UP)

    async def wait(self) -> None:
        """Wait until the supervisor gave up restarting (used by tests)."""
        if self._watcher is not None:
            await self._watcher

    async def stop(self) -> None:
        """Stop supervising and end the process."""
        watcher, self._watcher = self._watcher, None
        if watcher is not None:
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher
        process, self._process = self._process, None
        if process is None or process.returncode is not None:
            return
        process.terminate()
        try:
            async with asyncio.timeout(consts.EMBEDDER_STOP_TIMEOUT_S):
                await process.wait()
        except TimeoutError:
            process.kill()
            await process.wait()
