"""Read-only file tools: ``read_file``, ``list_dir``, ``glob``, ``grep`` (08 §4.1)."""

import asyncio
import re
from fnmatch import fnmatchcase
from functools import partial
from pathlib import Path

from harness.core.consts import META_TRUNCATED, Decision, RiskLevel, ToolSource
from harness.core.records import ToolOutput
from harness.tools import consts
from harness.tools.builtin.utils import looks_binary, read_text
from harness.tools.function_tool import FunctionTool, error_output, text_output, tool
from harness.tools.paths import resolve_path
from harness.tools.types import ToolContext, ToolTraits

_PATH = ToolTraits(path_param="path")


_reader = partial(
    tool,
    group=consts.GROUP_FILES,
    read_only=True,
    risk=RiskLevel.LOW,
    default_permission=Decision.ALLOW,
    source=ToolSource.BUILTIN,
    traits=_PATH,
)


def _slice(path: Path, offset: int, limit: int | None) -> ToolOutput:
    if not path.exists():
        return error_output(consts.MSG_NOT_FOUND.format(path=path))
    if path.is_dir():
        return error_output(consts.MSG_IS_DIRECTORY.format(path=path))
    size = path.stat().st_size
    if size > consts.READ_MAX_BYTES and limit is None:
        return error_output(consts.MSG_TOO_LARGE.format(path=path, size=size))
    with path.open("rb") as handle:
        probe = handle.read(consts.BINARY_PROBE_BYTES)
    if looks_binary(probe):
        return error_output(consts.MSG_BINARY_FILE.format(path=path))
    lines = read_text(path).splitlines(keepends=True)
    count = consts.READ_LINE_LIMIT_DEFAULT if limit is None else max(limit, 0)
    start = max(offset, 0)
    chosen = lines[start : start + count]
    return text_output(
        "".join(chosen),
        metadata={
            consts.META_PATH: str(path),
            consts.META_SIZE: size,
            META_TRUNCATED: start + len(chosen) < len(lines),
        },
    )


@_reader()
async def read_file(
    path: str, ctx: ToolContext, offset: int = 0, limit: int | None = None
) -> ToolOutput:
    """Read a text file. ``offset`` is the first line (0-based) and ``limit`` the number of lines.

    Relative paths start at the working directory.
    """
    return await asyncio.to_thread(_slice, resolve_path(path, ctx.working_dir), offset, limit)


def _capped(lines: list[str], limit: int) -> str:
    shown = lines[:limit]
    if len(lines) > limit:
        shown.append(consts.MSG_MORE_RESULTS.format(count=len(lines) - limit))
    return "\n".join(shown)


def _entries(path: Path) -> ToolOutput:
    if not path.exists():
        return error_output(consts.MSG_NOT_FOUND.format(path=path))
    if not path.is_dir():
        return error_output(consts.MSG_NOT_DIRECTORY.format(path=path))
    names = sorted(
        f"{entry.name}{consts.DIR_SUFFIX}" if entry.is_dir() else entry.name
        for entry in path.iterdir()
    )
    text = _capped(names, consts.LIST_MAX_ENTRIES) if names else consts.MSG_EMPTY_DIR
    return text_output(text, metadata={consts.META_PATH: str(path)})


@_reader()
async def list_dir(ctx: ToolContext, path: str = ".") -> ToolOutput:
    """List a directory: one entry per line, directories end with ``/``."""
    return await asyncio.to_thread(_entries, resolve_path(path, ctx.working_dir))


def _hidden(relative: Path) -> bool:
    return any(part in consts.PROTECTED_DIR_NAMES for part in relative.parts)


def _find(base: Path, pattern: str) -> ToolOutput:
    if not base.is_dir():
        return error_output(consts.MSG_NOT_DIRECTORY.format(path=base))
    found = sorted(
        match.relative_to(base).as_posix()
        for match in base.glob(pattern)
        if not _hidden(match.relative_to(base))
    )
    text = _capped(found, consts.GLOB_MAX_RESULTS) if found else consts.MSG_NO_MATCHES
    return text_output(text, metadata={consts.META_MATCHES: len(found)})


@_reader()
async def glob(pattern: str, ctx: ToolContext, path: str = ".") -> ToolOutput:
    """Find files by glob pattern (``**/*.py``) under ``path``; returns relative paths."""
    return await asyncio.to_thread(_find, resolve_path(path, ctx.working_dir), pattern)


def _candidates(base: Path, name_glob: str | None) -> list[Path]:
    if base.is_file():
        return [base]
    return [
        path
        for path in sorted(base.rglob("*"))
        if path.is_file()
        and not _hidden(path.relative_to(base))
        and (name_glob is None or fnmatchcase(path.name, name_glob))
    ]


def _search(base: Path, expression: re.Pattern[str], name_glob: str | None) -> ToolOutput:
    if not base.exists():
        return error_output(consts.MSG_NOT_FOUND.format(path=base))
    root = base.parent if base.is_file() else base
    hits: list[str] = []
    for path in _candidates(base, name_glob):
        if path.stat().st_size > consts.GREP_MAX_FILE_BYTES:
            continue
        data = path.read_bytes()
        if looks_binary(data):
            continue
        text = data.decode(consts.TEXT_ENCODING, errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):
            if expression.search(line):
                shown = line[: consts.GREP_LINE_MAX_CHARS]
                hits.append(f"{path.relative_to(root).as_posix()}:{number}: {shown}")
    text = _capped(hits, consts.GREP_MAX_MATCHES) if hits else consts.MSG_NO_MATCHES
    return text_output(text, metadata={consts.META_MATCHES: len(hits)})


@_reader()
async def grep(
    pattern: str,
    ctx: ToolContext,
    path: str = ".",
    include: str | None = None,
    ignore_case: bool = False,
) -> ToolOutput:
    """Search file contents with a regular expression; returns ``file:line: text`` lines.

    ``include`` filters file names with a glob (``*.py``).
    """
    try:
        expression = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as error:
        return error_output(consts.MSG_BAD_REGEX.format(error=error))
    base = resolve_path(path, ctx.working_dir)
    return await asyncio.to_thread(_search, base, expression, include)


READ_ONLY_FILE_TOOLS: list[FunctionTool] = [read_file, list_dir, glob, grep]
