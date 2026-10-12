"""MCP servers: every tool of a configured server becomes ``mcp.<server>.<tool>`` (08 §4.3).

Each server lives in its own asyncio task, because an MCP connection must be opened and
closed by the same task. A server that fails or dies never raises into the agent: its
tools answer with an error the model can read.
"""

# pyright: basic
import asyncio
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger
from mcp import ClientSession, StdioServerParameters, stdio_client, types
from mcp.client.stdio import get_default_environment
from mcp.client.streamable_http import streamable_http_client
from pydantic import JsonValue, TypeAdapter

from harness.config import ConfigManager, McpServerConfig
from harness.core.consts import Decision, ErrorCode, McpServerState, ToolSource
from harness.core.errors import ToolError
from harness.core.records import ApprovalPreview, ToolDef, ToolOutput
from harness.tools import consts
from harness.tools.function_tool import error_output, text_output
from harness.tools.registry import ToolRegistry
from harness.tools.types import ToolContext, ToolTraits

_SCHEMA = TypeAdapter(dict[str, JsonValue])


@dataclass(slots=True)
class _Server:
    """One configured server and its connection."""

    name: str
    config: McpServerConfig
    state: McpServerState = McpServerState.CONNECTING
    error: str | None = None
    session: ClientSession | None = None
    tools: list[types.Tool] = field(default_factory=list[types.Tool])
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    stop: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None
    registered: list[str] = field(default_factory=list[str])


class McpTool:
    """A ``Tool`` that forwards the call to its server."""

    def __init__(self, server: _Server, remote: types.Tool) -> None:
        self._server = server
        self._remote = remote.name
        self._definition = ToolDef(
            name=consts.MCP_TOOL_NAME.format(server=server.name, tool=remote.name),
            description=remote.description or remote.name,
            input_schema=_SCHEMA.validate_python(remote.input_schema),
            source=ToolSource.MCP,
            group=consts.MCP_NAME_PREFIX + server.name,
            default_permission=Decision.ASK,
        )

    @property
    def definition(self) -> ToolDef:
        """The tool definition under its internal name."""
        return self._definition

    @property
    def traits(self) -> ToolTraits:
        """Rules match the whole input."""
        return ToolTraits()

    async def preview(self, args: dict[str, JsonValue], ctx: ToolContext) -> ApprovalPreview | None:
        """MCP tools cannot be previewed."""
        return None

    async def run(self, args: dict[str, JsonValue], ctx: ToolContext) -> ToolOutput:
        """Call the server; a dead server is an error value, never an exception."""
        server, name = self._server, self._definition.name
        if server.session is None or server.state is not McpServerState.CONNECTED:
            return error_output(
                consts.MSG_MCP_DOWN.format(server=server.name, error=server.error or server.state),
                ErrorCode.TOOL_FAILED,
            )
        try:
            result = await server.session.call_tool(self._remote, dict(args))
        except asyncio.CancelledError:
            raise
        except Exception as error:  # whatever the transport raises, the agent goes on
            logger.warning("MCP tool {} failed: {!r}", name, error)
            await _check_alive(server, error)
            return error_output(consts.MSG_MCP_FAILED.format(tool=name, error=error))
        if not isinstance(result, types.CallToolResult):
            return error_output(
                consts.MSG_MCP_FAILED.format(tool=name, error=type(result).__name__)
            )
        text = "\n".join(
            block.text if isinstance(block, types.TextContent) else consts.MCP_OTHER_CONTENT
            for block in result.content
        )
        return error_output(text) if result.is_error else text_output(text)


async def _check_alive(server: _Server, cause: Exception) -> None:
    """After a failed call: a server that no longer answers a ping is marked as failed."""
    session = server.session
    if session is None:
        return
    try:
        async with asyncio.timeout(consts.MCP_PING_TIMEOUT_S):
            await session.send_ping()
    except asyncio.CancelledError:
        raise
    except Exception:  # no answer in any form means the connection is gone
        server.state, server.session = McpServerState.FAILED, None
        server.error = consts.MSG_MCP_LOST.format(error=cause)
        server.stop.set()
        logger.warning("MCP server {} stopped answering", server.name)


class McpServers:
    """Connects the configured servers and keeps their tools in the registry."""

    def __init__(self, config: ConfigManager, logs_dir: Path) -> None:
        self._config = config
        self._logs_dir = logs_dir
        self._servers: dict[str, _Server] = {}

    def status(self) -> dict[str, tuple[McpServerState, str | None, list[str]]]:
        """State, failure reason and registered tool names of every configured server."""
        return {
            name: (server.state, server.error, list(server.registered))
            for name, server in self._servers.items()
        }

    async def sync(self, registry: ToolRegistry) -> None:
        """Connect new and changed servers, reconnect failed ones, drop removed ones."""
        wanted = self._config.config().mcp_servers
        for name in [n for n, s in self._servers.items() if self._stale(s, wanted.get(n))]:
            await self._drop(name, registry)
        fresh = [name for name in wanted if name not in self._servers]
        for name in fresh:
            server = _Server(name, wanted[name])
            self._servers[name] = server
            server.task = asyncio.create_task(self._serve(server), name=f"mcp:{name}")
        for name in fresh:
            await self._await_ready(self._servers[name], registry)

    @staticmethod
    def _stale(server: _Server, wanted: McpServerConfig | None) -> bool:
        return wanted != server.config or server.state is not McpServerState.CONNECTED

    async def _await_ready(self, server: _Server, registry: ToolRegistry) -> None:
        try:
            async with asyncio.timeout(consts.MCP_CONNECT_TIMEOUT_S):
                await server.ready.wait()
        except TimeoutError:
            self._failed(
                server, consts.MSG_MCP_TIMEOUT.format(seconds=consts.MCP_CONNECT_TIMEOUT_S)
            )
            server.stop.set()
            return
        for remote in server.tools:
            tool = McpTool(server, remote)
            try:
                registry.register(tool)
            except ToolError as error:
                logger.warning("MCP tool skipped: {}", error.message)
                continue
            server.registered.append(tool.definition.name)

    async def _drop(self, name: str, registry: ToolRegistry) -> None:
        server = self._servers.pop(name)
        for tool in server.registered:
            registry.unregister(tool)
        server.stop.set()
        if server.task is not None:
            server.task.cancel()
            await asyncio.gather(server.task, return_exceptions=True)

    async def close(self) -> None:
        """Disconnect every server."""
        for server in self._servers.values():
            server.stop.set()
        tasks = [server.task for server in self._servers.values() if server.task is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._servers.clear()

    @staticmethod
    def _failed(server: _Server, reason: str) -> None:
        server.state, server.error, server.session = McpServerState.FAILED, reason, None
        logger.warning("MCP server {}: {}", server.name, reason)

    @asynccontextmanager
    async def _connect(self, server: _Server) -> AsyncGenerator[ClientSession]:
        async with AsyncExitStack() as stack:
            if server.config.command:
                self._logs_dir.mkdir(parents=True, exist_ok=True)
                log_path = self._logs_dir / consts.MCP_LOG_NAME.format(server=server.name)
                log = stack.enter_context(log_path.open("a", encoding=consts.TEXT_ENCODING))
                command, *args = server.config.command
                params = StdioServerParameters(
                    command=command,
                    args=args,
                    env={**get_default_environment(), **server.config.env},
                )
                read, write = await stack.enter_async_context(stdio_client(params, errlog=log))
            else:
                streams = await stack.enter_async_context(
                    streamable_http_client(server.config.url or "")
                )
                read, write = streams[0], streams[1]
            yield await stack.enter_async_context(ClientSession(read, write))

    async def _serve(self, server: _Server) -> None:
        try:
            async with self._connect(server) as session:
                await session.initialize()
                server.tools = list((await session.list_tools()).tools)
                server.session, server.state = session, McpServerState.CONNECTED
                server.ready.set()
                await server.stop.wait()
        except asyncio.CancelledError:
            raise
        except BaseException as error:  # the SDK reports failures as exception groups
            self._failed(server, consts.MSG_MCP_LOST.format(error=_describe(error)))
        finally:
            server.session = None
            if server.state is McpServerState.CONNECTED:
                server.state = McpServerState.STOPPED
            server.ready.set()


def _describe(error: BaseException) -> str:
    """The first real cause inside (possibly nested) exception groups."""
    while isinstance(error, BaseExceptionGroup) and error.exceptions:
        error = error.exceptions[0]
    return repr(error)
