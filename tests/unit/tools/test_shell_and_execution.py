"""The shell tool and safe execution: output, exit codes, timeouts, stop (08 §6)."""

import asyncio
import os
import subprocess
import sys

import pytest

from harness.core.consts import META_ERROR_CODE, META_EXIT_CODE, Decision, ErrorCode, HookEvent
from harness.core.records import ToolOutput
from harness.core.types import ToolCallBlock
from harness.tools import text_output, tool
from harness.tools.consts import META_TIMED_OUT
from tests.consts import AGENT_ID, HEBREW_PLAIN
from tests.unit.tools.conftest import ToolsEnv, text_of

pytestmark = pytest.mark.timeout(60)

WORD = "harness-shell-test"
EXIT_CODE = 3
SHORT_TIMEOUT_S = 1
LONG_SLEEP_S = 120
PID_FILE = "child.pid"
SLOW_TOOL = "slow_tool"
BROKEN_TOOL = "broken_tool"
FAILURE = "the tool exploded"
PYTHON = sys.executable.replace("\\", "/")
CHILD_SCRIPT = (
    "import os, time, pathlib; "
    f"pathlib.Path('{PID_FILE}').write_text(str(os.getpid())); "
    f"time.sleep({LONG_SLEEP_S})"
)


def pid_alive(pid: int) -> bool:
    """Whether a process exists (never ``os.kill`` on Windows: that terminates it)."""
    if sys.platform == "win32":
        listed = subprocess.run(  # noqa: S603 - fixed system tool
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        )
        return str(pid) in listed.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def wait_for_pid(env: ToolsEnv) -> int:
    path = env.work / PID_FILE
    async with asyncio.timeout(30):
        while not (path.exists() and path.read_text().strip()):
            await asyncio.sleep(0.05)
    return int(path.read_text())


@pytest.mark.slow
async def test_shell_runs_in_the_working_dir_and_streams_output(env: ToolsEnv) -> None:
    env.write("marker.txt", WORD)

    output = await env.run(
        "shell", command=f"Get-Content marker.txt; Write-Output '{HEBREW_PLAIN}'"
    )

    assert not output.is_error
    assert text_of(output).split() == [WORD, *HEBREW_PLAIN.split()]
    assert "".join(env.progress) == text_of(output)
    assert output.metadata[META_EXIT_CODE] == 0


@pytest.mark.slow
async def test_shell_reports_a_failing_exit_code(env: ToolsEnv) -> None:
    output = await env.run("shell", command=f"exit {EXIT_CODE}")

    assert output.is_error
    assert output.metadata[META_EXIT_CODE] == EXIT_CODE
    assert str(EXIT_CODE) in text_of(output)


@pytest.mark.slow
async def test_shell_timeout_kills_the_process_tree(env: ToolsEnv) -> None:
    command = f'& "{PYTHON}" -c "{CHILD_SCRIPT}"'
    running = asyncio.create_task(env.run("shell", command=command, timeout_s=SHORT_TIMEOUT_S))
    pid = await wait_for_pid(env)

    output = await running

    assert output.is_error
    assert output.metadata[META_TIMED_OUT] is True
    assert not pid_alive(pid)


@pytest.mark.slow
@pytest.mark.acceptance("08-10.6")
async def test_stop_kills_a_long_shell_command_and_its_children(env: ToolsEnv) -> None:
    command = f'& "{PYTHON}" -c "{CHILD_SCRIPT}"'
    running = asyncio.create_task(env.run("shell", command=command))
    pid = await wait_for_pid(env)
    alive_before = pid_alive(pid)

    running.cancel()
    await asyncio.gather(running, return_exceptions=True)

    assert alive_before
    assert running.cancelled()
    assert not pid_alive(pid)


async def test_shell_preview_shows_the_command(env: ToolsEnv) -> None:
    command = f"Write-Output {WORD}"

    decision = await env.decide("shell", command=command)

    assert decision.decision is Decision.ASK
    assert decision.preview is not None
    assert decision.preview.command == command


async def test_slow_tool_times_out_as_an_error_value(env: ToolsEnv) -> None:
    @tool(name=SLOW_TOOL, default_permission=Decision.ALLOW, timeout_s=SHORT_TIMEOUT_S)
    async def slow_tool() -> ToolOutput:
        """Never finishes in time."""
        await asyncio.sleep(LONG_SLEEP_S)
        return text_output(WORD)

    env.manager.register(slow_tool)
    env.agent(tools=[SLOW_TOOL])

    output = await env.run(SLOW_TOOL)

    assert output.metadata[META_ERROR_CODE] == ErrorCode.TOOL_TIMEOUT.value


async def test_tool_exception_and_bad_input_become_error_values(env: ToolsEnv) -> None:
    @tool(name=BROKEN_TOOL, default_permission=Decision.ALLOW)
    async def broken_tool(count: int = 0) -> ToolOutput:
        """Always fails."""
        raise RuntimeError(FAILURE)

    env.manager.register(broken_tool)
    env.agent(tools=[BROKEN_TOOL])

    raised = await env.run(BROKEN_TOOL)
    invalid = await env.run(BROKEN_TOOL, count="many", extra=True)

    assert FAILURE in text_of(raised)
    assert raised.metadata[META_ERROR_CODE] == ErrorCode.TOOL_FAILED.value
    assert invalid.is_error
    assert "count" in text_of(invalid)


async def test_after_hook_can_replace_the_result(env: ToolsEnv) -> None:
    env.write("a.txt", WORD)
    replaced = text_output(HEBREW_PLAIN).model_dump(mode="json")
    call = ToolCallBlock(id="call_x", name="read_file", input={"path": "a.txt"})
    decision = await env.manager.decide(AGENT_ID, call)
    env.hooks.script(HookEvent.AFTER_TOOL_CALL, updated={"result": replaced})

    output = await env.manager.execute(AGENT_ID, call, decision)

    assert text_of(output) == HEBREW_PLAIN
    original = env.hooks.emitted[-1].payload["result"]
    assert isinstance(original, dict)
    assert original["content"] == text_output(WORD).model_dump(mode="json")["content"]
