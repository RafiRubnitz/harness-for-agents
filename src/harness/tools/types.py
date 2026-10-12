"""What a tool is and what it receives when it runs (08 §3)."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from pydantic import JsonValue

from harness.core.records import AgentState, ApprovalPreview, ToolDef, ToolOutput
from harness.core.types import AgentId, Clock, RunId, TaskId


@dataclass(frozen=True, slots=True)
class ToolContext:
    """The per-call surroundings of a tool run."""

    agent_id: AgentId
    run_id: RunId | None
    call_id: str
    working_dir: str
    allowed_paths: list[str]
    cancel: asyncio.Event
    emit_progress: Callable[[str], None]
    clock: Clock
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ToolTraits:
    """How the permission policy reads a tool's input.

    Attributes:
        path_param: Input field holding a file path; enables ``path`` rules, the
            working-directory check and protected paths.
        match_param: Input field a rule's ``match`` glob applies to (the shell command).
            Without it the glob applies to the whole input as JSON.
        confirm_param: Boolean input field that always needs the user's approval when
            true, whatever rules and hooks say (``delete_file`` with ``permanent``).
    """

    path_param: str | None = None
    match_param: str | None = None
    confirm_param: str | None = None


class Tool(Protocol):
    """One callable tool."""

    @property
    def definition(self) -> ToolDef:
        """Name, description, schema (without injected fields) and policy defaults."""
        ...

    @property
    def traits(self) -> ToolTraits:
        """How rules read this tool's input."""
        ...

    async def run(self, args: dict[str, JsonValue], ctx: ToolContext) -> ToolOutput:
        """Run with the clean input (no injected fields)."""
        ...

    async def preview(self, args: dict[str, JsonValue], ctx: ToolContext) -> ApprovalPreview | None:
        """Describe what the call would do, without side effects; shown before approval."""
        ...


ProgressSink = Callable[[AgentId, str, TaskId | None, str], None]
"""``(agent_id, call_id, task_id, chunk)``: live output of a running tool."""

AgentLookup = Callable[[AgentId], AgentState]
"""Returns the current state of an agent; raises ``AgentNotFoundError``."""
