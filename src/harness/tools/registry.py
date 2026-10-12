"""The tool registry: what exists, and which tools one agent gets (08 §4, §7)."""

import re
from fnmatch import fnmatchcase

from harness.core.consts import TOOL_WIRE_SEPARATOR, ToolSource
from harness.core.errors import ToolError
from harness.tools.consts import (
    ALWAYS_ON_GROUPS,
    MCP_NAME_PREFIX,
    MSG_BAD_TOOL_NAME,
    MSG_DUPLICATE_TOOL,
    TOOL_NAME_PATTERN,
)
from harness.tools.types import Tool

_NAME_RE = re.compile(TOOL_NAME_PATTERN)


def check_name(tool: Tool) -> None:
    """Reject a name that could not be mapped to a wire name and back (01 §5).

    Raises:
        ToolError: The name breaks the registration rules.
    """
    name = tool.definition.name
    if tool.definition.source is ToolSource.MCP:
        valid = name.startswith(MCP_NAME_PREFIX) and TOOL_WIRE_SEPARATOR not in name
    else:
        valid = _NAME_RE.fullmatch(name) is not None
    if not valid:
        raise ToolError(MSG_BAD_TOOL_NAME.format(name=name), tool=name)


def selected(tool: Tool, selectors: list[str]) -> bool:
    """Return whether a list of names, groups and wildcards picks ``tool``."""
    definition = tool.definition
    return any(
        selector in (definition.name, definition.group) or fnmatchcase(definition.name, selector)
        for selector in selectors
    )


class ToolRegistry:
    """Registered tools by internal name, in registration order."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Add a tool.

        Raises:
            ToolError: The name is invalid or already taken.
        """
        check_name(tool)
        name = tool.definition.name
        if name in self._tools:
            raise ToolError(MSG_DUPLICATE_TOOL.format(name=name), tool=name)
        self._tools[name] = tool

    def unregister(self, name: str) -> None:
        """Remove a tool; unknown names are ignored."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Return a tool by its internal name."""
        return self._tools.get(name)

    def every(self) -> list[Tool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def for_agent(self, selectors: list[str], enabled: list[str] | None) -> list[Tool]:
        """Return the tools an agent has: picked by its selectors and not switched off.

        ``enabled`` (``tools.enabled``) filters built-in tools only; ``None`` keeps all.
        An agent with any tool also gets the always-on groups (reading stored results).
        """
        picked = [
            tool
            for tool in self._tools.values()
            if selected(tool, selectors)
            and (
                enabled is None
                or tool.definition.source is not ToolSource.BUILTIN
                or selected(tool, enabled)
            )
        ]
        if not picked:
            return []
        names = {tool.definition.name for tool in picked}
        extra = [
            tool
            for tool in self._tools.values()
            if tool.definition.group in ALWAYS_ON_GROUPS and tool.definition.name not in names
        ]
        return [*picked, *extra]
