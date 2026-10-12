"""The terminal the CLI talks to; a protocol so tests can script it."""

import asyncio
import io
import sys
import threading
from typing import Protocol

from harness.cli import consts


class Terminal(Protocol):
    """Line input and text output."""

    def write(self, text: str) -> None:
        """Write ``text`` as it is (no newline added)."""
        ...

    def line(self, text: str = "", style: str | None = None) -> None:
        """Write one line, optionally in an ANSI style."""
        ...

    async def read(self, prompt: str) -> str | None:
        """Ask for one line; ``None`` means the input ended."""
        ...


class StdTerminal:
    """``Terminal`` over the process's standard streams."""

    def __init__(self) -> None:
        for stream in (sys.stdout, sys.stderr):
            if isinstance(stream, io.TextIOWrapper):
                stream.reconfigure(encoding=consts.OUTPUT_ENCODING, errors="replace")
        self._colour = sys.stdout.isatty()

    def write(self, text: str) -> None:
        """Write ``text`` and flush, so streamed tokens show at once."""
        sys.stdout.write(text)
        sys.stdout.flush()

    def line(self, text: str = "", style: str | None = None) -> None:
        """Write one line; the style is dropped when output is not a terminal."""
        if style and self._colour:
            text = f"{style}{text}{consts.ANSI_RESET}"
        self.write(f"{text}\n")

    async def read(self, prompt: str) -> str | None:
        """Read one line on a daemon thread, so Ctrl-C never waits for Enter."""
        loop = asyncio.get_running_loop()
        answer: asyncio.Future[str | None] = loop.create_future()

        def settle(value: str | None) -> None:
            if not answer.done():
                answer.set_result(value)

        def ask() -> None:
            try:
                value: str | None = input(prompt)
            except EOFError:
                value = None
            loop.call_soon_threadsafe(settle, value)

        threading.Thread(target=ask, daemon=True).start()
        return await answer
