"""Argument parsing and the sub-commands of ``harness`` (12 §7.2)."""

import argparse
import asyncio
from collections.abc import Sequence

from harness.cli import consts
from harness.cli.chat import ChatError, ChatOptions, ChatSession
from harness.cli.terminal import StdTerminal, Terminal
from harness.cli.utils import format_agents
from harness.core.errors import CoreLockedError, HarnessError
from harness.core.handle import CoreLauncher


def build_parser() -> argparse.ArgumentParser:
    """Build the ``harness`` argument parser."""
    parser = argparse.ArgumentParser(prog=consts.PROG, description=consts.HELP_DESCRIPTION)
    commands = parser.add_subparsers(dest="command")
    chat = commands.add_parser(consts.CMD_CHAT, help=consts.HELP_CHAT)
    chat.add_argument("agent", nargs="?", help=consts.HELP_AGENT_ARG)
    chat.add_argument("--new", action="store_true", help=consts.HELP_NEW)
    chat.add_argument("--template", help=consts.HELP_TEMPLATE)
    chat.add_argument("--model", help=consts.HELP_MODEL)
    agents = commands.add_parser(consts.CMD_AGENTS, help=consts.HELP_AGENTS)
    agents.add_argument("--archived", action="store_true", help=consts.HELP_ARCHIVED)
    return parser


async def _chat(launcher: CoreLauncher, terminal: Terminal, options: ChatOptions) -> None:
    async with launcher() as core:
        await ChatSession(core, terminal).run(options)


async def _agents(launcher: CoreLauncher, terminal: Terminal, *, archived: bool) -> None:
    async with launcher() as core:
        listed = await ChatSession(core, terminal).list_agents(include_archived=archived)
    for line in format_agents(listed):
        terminal.line(line)


def run(
    argv: Sequence[str] | None, launcher: CoreLauncher, terminal: Terminal | None = None
) -> int:
    """Run one ``harness`` command line and return the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    out = terminal or StdTerminal()
    if args.command == consts.CMD_CHAT:
        options = ChatOptions(
            agent=args.agent,
            new=args.new or args.template is not None,
            template=args.template,
            model=args.model,
        )
        job = _chat(launcher, out, options)
    elif args.command == consts.CMD_AGENTS:
        job = _agents(launcher, out, archived=args.archived)
    else:
        out.line(parser.format_help())
        return consts.EXIT_OK
    try:
        asyncio.run(job)
    except CoreLockedError as error:
        out.line(consts.MSG_LOCKED.format(detail=error.message), consts.ANSI_RED)
        return consts.EXIT_CORE_LOCKED
    except (ChatError, HarnessError) as error:
        out.line(str(error), consts.ANSI_RED)
        return consts.EXIT_ERROR
    except KeyboardInterrupt:
        out.line()
        return consts.EXIT_INTERRUPTED
    return consts.EXIT_OK
