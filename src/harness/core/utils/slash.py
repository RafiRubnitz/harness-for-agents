"""Parse agent input text into core commands; shared by the GUI and the CLI."""

from harness.core.commands import AgentSend, ContextCompact, ContextInspect
from harness.core.consts import (
    MSG_SLASH_RESERVED,
    MSG_SLASH_UNKNOWN,
    SLASH_CONTEXT_FULL_ARG,
    SLASH_PREFIX,
    SLASH_RESERVED,
    SlashCommand,
)
from harness.core.errors import SlashError
from harness.core.types import AgentId, TextBlock

SlashResult = AgentSend | ContextInspect | ContextCompact


def is_slash(text: str) -> bool:
    """Return whether ``text`` is a single-line ``/word ...`` command (not a path like ``/a/b``)."""
    stripped = text.strip()
    if not stripped.startswith(SLASH_PREFIX) or len(stripped) == 1 or "\n" in stripped:
        return False
    head = stripped.split(maxsplit=1)[0]
    return SLASH_PREFIX not in head[1:]


def parse_input(text: str, agent_id: AgentId) -> SlashResult:
    """Turn input text into a command.

    Plain text becomes ``agent.send``. A slash command becomes its core command and is
    never sent to the model; an unknown one raises ``SlashError`` so a typo cannot reach
    the agent.
    """
    if not is_slash(text):
        return AgentSend(agent_id=agent_id, content=[TextBlock(text=text)])
    parts = text.strip().split()
    name, args = parts[0].lower(), [part.lower() for part in parts[1:]]
    if name in {member.value for member in SLASH_RESERVED}:
        raise SlashError(MSG_SLASH_RESERVED.format(command=name))
    if name == SlashCommand.CONTEXT and args in ([], [SLASH_CONTEXT_FULL_ARG]):
        return ContextInspect(agent_id=agent_id, full=bool(args))
    if name == SlashCommand.COMPACT and not args:
        return ContextCompact(agent_id=agent_id)
    raise SlashError(MSG_SLASH_UNKNOWN.format(command=text.strip()))
