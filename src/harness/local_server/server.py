"""Runs the local server inside the core's event loop, on 127.0.0.1 only."""

import asyncio
from collections.abc import Generator
from contextlib import contextmanager

import uvicorn
from fastapi import FastAPI
from loguru import logger

from harness.core.consts import ErrorCode
from harness.core.errors import HarnessError
from harness.local_server import consts


class _Server(uvicorn.Server):
    """uvicorn without its signal handlers: Ctrl-C belongs to the core."""

    @contextmanager
    def capture_signals(self) -> Generator[None]:
        yield


class LocalServer:
    """Starts and stops the HTTP server."""

    def __init__(self, app: FastAPI, port: int) -> None:
        self._port = port
        config = uvicorn.Config(
            app, host=consts.LOCALHOST, port=port, log_level=consts.LOG_LEVEL, lifespan="off"
        )
        self._server = _Server(config)
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start listening; returns once the port is open.

        Raises:
            HarnessError: The port could not be opened.
        """
        self._task = asyncio.create_task(self._serve(), name=consts.APP_TITLE)
        # uvicorn only exposes a flag for "the socket is open", no event to wait on
        while not self._server.started and not self._task.done():  # noqa: ASYNC110
            await asyncio.sleep(consts.START_POLL_S)
        if not self._server.started:
            raise HarnessError(
                consts.MSG_START_FAILED.format(host=consts.LOCALHOST, port=self._port),
                code=ErrorCode.APP_INTERNAL,
            )
        logger.info("local server listens on {}:{}", consts.LOCALHOST, self._port)

    async def _serve(self) -> None:
        try:
            await self._server.serve()
        except SystemExit:  # uvicorn exits this way when the port is taken
            logger.error("local server could not start on port {}", self._port)

    async def stop(self) -> None:
        """Stop listening and wait until the server is gone."""
        task, self._task = self._task, None
        if task is None:
            return
        self._server.should_exit = True
        await asyncio.gather(task, return_exceptions=True)
