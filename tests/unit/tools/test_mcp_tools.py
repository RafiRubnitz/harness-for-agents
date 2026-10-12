"""MCP servers: a real stdio server whose tools the agent calls, and which may die (08 §4.3)."""

import sys
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from pydantic import JsonValue

from harness.core.consts import ConfigLayer, Decision, McpServerState, ToolSource
from tests.consts import AGENT_ID
from tests.unit.tools.conftest import ToolsEnv, text_of

SERVER = "echo"
SCRIPT = Path(__file__).parents[2] / "data" / "mcp" / "echo_server.py"
ECHO = f"mcp.{SERVER}.echo"
FAIL = f"mcp.{SERVER}.fail"
CRASH = f"mcp.{SERVER}.crash"
WORD = "quiet words"
MISSING_COMMAND = "harness-test-no-such-mcp-server"
KEY = "mcp_servers"
SLOW = pytest.mark.timeout(90)


def stdio(*command: str) -> dict[str, JsonValue]:
    return {"command": list(command)}


@pytest.fixture
async def mcp(env: ToolsEnv) -> AsyncGenerator[ToolsEnv]:
    """The tools manager with the echo server connected."""
    env.agent(tools=[f"mcp.{SERVER}.*", "files"])
    servers: dict[str, JsonValue] = {SERVER: stdio(sys.executable, str(SCRIPT))}
    await env.config.set(KEY, servers, ConfigLayer.GLOBAL)
    await env.manager.reload()
    yield env
    await env.manager.close()


@SLOW
@pytest.mark.acceptance("08-10.8")
async def test_a_tool_of_an_mcp_server_is_called_and_returns_its_result(mcp: ToolsEnv) -> None:
    output = await mcp.run(ECHO, text=WORD)

    (definition,) = [d for d in mcp.manager.list(AGENT_ID) if d.name == ECHO]
    status = (await mcp.manager.mcp_status())[SERVER]
    assert text_of(output) == WORD.upper()
    assert not output.is_error
    assert (definition.source, definition.default_permission) == (ToolSource.MCP, Decision.ASK)
    assert definition.input_schema["required"] == ["text"]
    assert status.state is McpServerState.CONNECTED
    assert {ECHO, FAIL, CRASH} == set(status.tools)


@SLOW
async def test_an_error_of_the_remote_tool_is_an_error_the_model_reads(mcp: ToolsEnv) -> None:
    output = await mcp.run(FAIL, text=WORD)

    assert output.is_error
    assert text_of(output)


@SLOW
@pytest.mark.acceptance("08-10.8")
async def test_a_server_that_dies_does_not_take_the_agent_down(mcp: ToolsEnv) -> None:
    mcp.write("note.txt", WORD)

    died = await mcp.run(CRASH)
    after = await mcp.run(ECHO, text=WORD)
    other = await mcp.run("read_file", path="note.txt")

    assert died.is_error
    assert after.is_error
    assert SERVER in text_of(after) or ECHO in text_of(after)
    assert WORD in text_of(other)
    assert not other.is_error


@SLOW
async def test_reload_reconnects_a_server_that_died(mcp: ToolsEnv) -> None:
    await mcp.run(CRASH)
    await mcp.run(ECHO, text=WORD)  # lets the dead connection be noticed

    await mcp.manager.reload()
    output = await mcp.run(ECHO, text=WORD)

    assert text_of(output) == WORD.upper()


@SLOW
async def test_a_server_that_cannot_start_is_reported_and_brings_no_tools(env: ToolsEnv) -> None:
    env.agent(tools=[f"mcp.{SERVER}.*", "files"])
    await env.config.set(KEY, {SERVER: stdio(MISSING_COMMAND)}, ConfigLayer.GLOBAL)

    await env.manager.reload()

    status = (await env.manager.mcp_status())[SERVER]
    names = [definition.name for definition in env.manager.list(AGENT_ID)]
    await env.manager.close()
    assert status.state is McpServerState.FAILED
    assert status.error
    assert status.tools == []
    assert "read_file" in names
    assert not [name for name in names if name.startswith("mcp.")]


@SLOW
async def test_a_server_removed_from_the_config_takes_its_tools_away(mcp: ToolsEnv) -> None:
    before = [definition.name for definition in mcp.manager.list(AGENT_ID)]
    empty: dict[str, JsonValue] = {}

    await mcp.config.set(KEY, empty, ConfigLayer.GLOBAL)
    await mcp.manager.reload()

    assert ECHO in before
    assert ECHO not in [definition.name for definition in mcp.manager.list(AGENT_ID)]
    assert await mcp.manager.mcp_status() == {}
