"""A tiny MCP server over stdio for the tests: one tool echoes, one kills the server."""

import os

from mcp.server.mcpserver import MCPServer

server = MCPServer("echo")


@server.tool()
def echo(text: str) -> str:
    """Return the text in upper case."""
    return text.upper()


@server.tool()
def fail(text: str) -> str:
    """Always fails."""
    raise ValueError(text)


@server.tool()
def crash() -> str:
    """Kill the server process."""
    os._exit(1)


if __name__ == "__main__":
    server.run()
