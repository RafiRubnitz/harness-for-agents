"""What the GUI and the CLI see of the core: commands in, events out (02 §5)."""

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from harness.core.commands import Command
from harness.core.events import Event


class FrontendInfo(BaseModel):
    """What a front end needs to draw itself and to offer choices; a snapshot of the config."""

    model_config = ConfigDict(frozen=True)
    theme: str
    font_family: str
    font_size: int
    window_width: int
    window_height: int
    shortcuts: dict[str, str]
    themes_dir: str
    models: list[str]
    templates: list[str]
    tools: list[str]
    tool_groups: list[str]
    providers: list[str]
    user_profile: str
    model_providers: dict[str, str] = {}
    """Model alias -> the provider that serves it."""
    key_names: list[str] = []
    """Every key the settings dialog can set: the providers, then the web tools' key."""
    keys: dict[str, str] = {}
    """Key name -> the key in masked form (``sk-...a3f``); only names that have one."""
    embedder_url: str = ""
    """The configured embeddings server; empty when the harness runs its own."""


class CoreHandle(Protocol):
    """A running core."""

    def info(self) -> FrontendInfo:
        """The current display settings and the names a front end can offer."""
        ...

    async def submit(self, command: Command) -> None:
        """Run one command. Failures arrive as ``error`` events, never as exceptions."""
        ...

    def events(self) -> AsyncIterator[Event]:
        """Open a stream of every event published from now on."""
        ...


CoreLauncher = Callable[[], AbstractAsyncContextManager[CoreHandle]]
"""Starts the core (taking ``core.lock``) and shuts it down on exit.

Entering raises ``CoreLockedError`` when another process already runs the core.
"""
