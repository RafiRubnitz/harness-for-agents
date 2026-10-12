"""The ``shell`` tool: PowerShell commands with live output and tree kill (08 §6, §9.3)."""

import asyncio
import codecs
import contextlib
import os
import shutil
import signal
import subprocess
import sys
from asyncio.subprocess import Process
from pathlib import Path

from harness.config import ConfigManager, ToolsConfig
from harness.core.consts import META_EXIT_CODE, Decision, ToolSource
from harness.core.records import ApprovalPreview, ToolOutput
from harness.core.types import TextBlock
from harness.tools import consts
from harness.tools.function_tool import FunctionTool, error_output, tool
from harness.tools.types import ToolContext, ToolTraits

_READ_CHUNK = 4096
_POWERSHELL_UTF8 = "[Console]::OutputEncoding=[Text.Encoding]::UTF8; "
_AUTO = "auto"


def find_executable(configured: str) -> str | None:
    """Return the shell to run: the configured one, else pwsh, else Windows PowerShell."""
    if configured != _AUTO:
        return shutil.which(configured) or (configured if Path(configured).is_file() else None)
    for candidate in consts.SHELL_CANDIDATES:
        found = shutil.which(candidate)
        if found:
            return found
    return None if sys.platform == "win32" else shutil.which(consts.SHELL_POSIX_FALLBACK)


def command_line(executable: str, command: str) -> list[str]:
    """Build the argument vector; never goes through another shell."""
    stem = Path(executable).stem.lower()
    if stem in consts.SHELL_CANDIDATES:
        return [executable, *consts.POWERSHELL_ARGS, _POWERSHELL_UTF8 + command]
    return [executable, *consts.POSIX_SHELL_ARGS, command]


async def kill_tree(process: Process) -> None:
    """Kill a process and everything it started, then reap it."""
    if process.returncode is not None:
        return
    if sys.platform == "win32":
        killer = await asyncio.create_subprocess_exec(
            consts.TASKKILL,
            *consts.TASKKILL_ARGS,
            str(process.pid),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await killer.wait()
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    with contextlib.suppress(ProcessLookupError):
        if process.returncode is None:
            process.kill()
    with contextlib.suppress(TimeoutError):
        async with asyncio.timeout(consts.KILL_WAIT_S):
            await process.wait()


async def _start(argv: list[str], cwd: str) -> Process:
    if sys.platform == "win32":
        return await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    return await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )


async def _pump(process: Process, ctx: ToolContext, chunks: list[str]) -> int:
    """Forward output as it arrives; return the exit code."""
    decoder = codecs.getincrementaldecoder(consts.TEXT_ENCODING)(errors="replace")
    stream = process.stdout
    while stream is not None:
        data = await stream.read(_READ_CHUNK)
        text = decoder.decode(data, final=not data)
        if text:
            chunks.append(text)
            ctx.emit_progress(text)
        if not data:
            break
    return await process.wait()


class Shell:
    """Runs one command in the agent's working directory."""

    def __init__(self, config: ConfigManager) -> None:
        self._config = config

    async def run(self, command: str, ctx: ToolContext, timeout_s: int | None) -> ToolOutput:
        """Run ``command``; a timeout or a stop kills the whole process tree."""
        settings = self._config.section(ToolsConfig, ctx.agent_id).shell
        executable = find_executable(settings.executable)
        if executable is None:
            tried = settings.executable if settings.executable != _AUTO else consts.SHELL_CANDIDATES
            return error_output(consts.MSG_SHELL_NOT_FOUND.format(tried=tried))
        seconds = min(timeout_s or settings.timeout_s, consts.SHELL_MAX_TIMEOUT_S)
        process = await _start(command_line(executable, command), ctx.working_dir)
        chunks: list[str] = []
        code: int | None = None
        try:
            async with asyncio.timeout(seconds):
                code = await _pump(process, ctx, chunks)
        except TimeoutError:
            await kill_tree(process)
            chunks.append(f"\n{consts.MSG_SHELL_TIMEOUT.format(seconds=seconds)}")
        except asyncio.CancelledError:
            await asyncio.shield(kill_tree(process))
            raise
        if code not in (None, 0):
            chunks.append(f"\n{consts.MSG_SHELL_EXIT.format(code=code)}")
        return ToolOutput(
            content=[TextBlock(text="".join(chunks))],
            is_error=code != 0,
            metadata={META_EXIT_CODE: code, consts.META_TIMED_OUT: code is None},
        )


def shell_tools(config: ConfigManager) -> list[FunctionTool]:
    """Build the ``shell`` tool."""
    runner = Shell(config)

    async def shell_preview(
        command: str, ctx: ToolContext, timeout_s: int | None = None
    ) -> ApprovalPreview:
        return ApprovalPreview(command=command, path=ctx.working_dir)

    @tool(
        group=consts.GROUP_SHELL,
        default_permission=Decision.ASK,
        timeout_s=consts.SHELL_MAX_TIMEOUT_S,
        source=ToolSource.BUILTIN,
        traits=ToolTraits(match_param="command"),
        preview=shell_preview,
    )
    async def shell(command: str, ctx: ToolContext, timeout_s: int | None = None) -> ToolOutput:
        """Run a PowerShell command in the working directory and return its output.

        ``timeout_s`` overrides the default time limit for this command; when it passes,
        the command and every process it started are killed.
        """
        return await runner.run(command, ctx, timeout_s)

    return [shell]
