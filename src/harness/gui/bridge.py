"""Runs the core on a background thread and carries commands and events across (02 §3).

Tk owns the main thread and is not thread-safe, so the asyncio core lives on its own
thread. Events reach the window through a ``queue.Queue`` that it polls; commands go the
other way with ``run_coroutine_threadsafe``. Nothing here blocks the main thread except
``start`` and ``stop``, which run before the first frame and after the last one.
"""

import asyncio
import queue
import threading
from collections.abc import AsyncIterator
from concurrent.futures import Future

from loguru import logger

from harness.core.commands import AppShutdown, Command
from harness.core.events import ConfigChanged, Event
from harness.core.handle import CoreHandle, CoreLauncher, FrontendInfo
from harness.gui import consts

_MSG_NOT_RUNNING = "the core is not running"
_MSG_START_TIMEOUT = "the core did not start in time"


class BridgeError(RuntimeError):
    """The bridge was used while the core is not running."""


class CoreBridge:
    """The window's handle on a core that runs on another thread."""

    def __init__(self, launcher: CoreLauncher) -> None:
        self._launcher = launcher
        self._events: queue.Queue[Event] = queue.Queue()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=consts.BRIDGE_THREAD_NAME, daemon=True
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self._core: CoreHandle | None = None
        self._stop: asyncio.Event | None = None
        self._info: FrontendInfo | None = None
        self.failure: BaseException | None = None

    # --- lifecycle (main thread) ---

    def start(self) -> None:
        """Start the core and wait until it is ready; raises what stopped it from starting."""
        self._thread.start()
        if not self._ready.wait(consts.BRIDGE_START_TIMEOUT_S):
            raise BridgeError(_MSG_START_TIMEOUT)
        if self.failure is not None:
            raise self.failure

    def stop(self, *, stop_agents: bool = False) -> None:
        """Shut the core down and wait for its thread."""
        loop, stop = self._loop, self._stop
        if loop is None or stop is None or not self._thread.is_alive():
            return
        if stop_agents:
            pending = self.submit(AppShutdown(stop_agents=True))
            try:
                pending.result(consts.BRIDGE_STOP_TIMEOUT_S)
            except (TimeoutError, BridgeError):
                logger.warning("agents did not stop in time")
        loop.call_soon_threadsafe(stop.set)
        self._thread.join(consts.BRIDGE_STOP_TIMEOUT_S)

    @property
    def running(self) -> bool:
        """Whether the core is up."""
        return self._core is not None and self._thread.is_alive()

    # --- traffic (main thread) ---

    def submit(self, command: Command) -> Future[None]:
        """Send a command; failures come back as ``error`` events."""
        loop, core = self._loop, self._core
        if loop is None or core is None or not self._thread.is_alive():
            raise BridgeError(_MSG_NOT_RUNNING)
        return asyncio.run_coroutine_threadsafe(core.submit(command), loop)

    def poll(self, limit: int = consts.MAX_EVENTS_PER_TICK) -> list[Event]:
        """Take the events that arrived since the last call, without waiting."""
        events: list[Event] = []
        while len(events) < limit:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                break
        return events

    def info(self) -> FrontendInfo:
        """The display settings as of the last config change."""
        if self._info is None:
            raise BridgeError(_MSG_NOT_RUNNING)
        return self._info

    # --- core thread ---

    def _run(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as error:
            self.failure = error
        finally:
            self._core = None
            self._ready.set()

    async def _serve(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        async with self._launcher() as core:
            self._info = core.info()
            pump = asyncio.create_task(self._pump(core, core.events()))
            self._core = core
            self._ready.set()
            try:
                await self._stop.wait()
            finally:
                pump.cancel()
                await asyncio.gather(pump, return_exceptions=True)

    async def _pump(self, core: CoreHandle, events: AsyncIterator[Event]) -> None:
        async for event in events:
            if isinstance(event.payload, ConfigChanged):
                self._info = core.info()
            self._events.put(event)
