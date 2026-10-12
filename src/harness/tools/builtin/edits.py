"""File writing tools: ``write_file`` and ``edit_file`` (08 §4.1, T5)."""

import asyncio
from functools import partial
from pathlib import Path

from harness.core.consts import META_DIFF, Decision, ToolSource
from harness.core.records import ApprovalPreview, ToolOutput
from harness.tools import consts
from harness.tools.builtin.utils import read_text, read_text_or_empty, unified_diff, write_text
from harness.tools.function_tool import FunctionTool, error_output, text_output, tool
from harness.tools.paths import resolve_path
from harness.tools.types import ToolContext, ToolTraits

_writer = partial(
    tool,
    group=consts.GROUP_FILES,
    exclusive=True,
    default_permission=Decision.ASK,
    source=ToolSource.BUILTIN,
    traits=ToolTraits(path_param="path"),
)


def _write(path: Path, content: str) -> ToolOutput:
    if path.is_dir():
        return error_output(consts.MSG_IS_DIRECTORY.format(path=path))
    diff = unified_diff(read_text_or_empty(path), content, path)
    size = write_text(path, content)
    return text_output(
        consts.MSG_WROTE.format(path=path, size=size),
        metadata={consts.META_PATH: str(path), consts.META_SIZE: size, META_DIFF: diff},
    )


async def _write_preview(path: str, content: str, ctx: ToolContext) -> ApprovalPreview:
    target = resolve_path(path, ctx.working_dir)
    before = await asyncio.to_thread(read_text_or_empty, target)
    return ApprovalPreview(diff=unified_diff(before, content, target), path=str(target))


@_writer(preview=_write_preview)
async def write_file(path: str, content: str, ctx: ToolContext) -> ToolOutput:
    """Create a file or replace its whole content. Parent directories are created.

    For a small change to an existing file prefer ``edit_file``.
    """
    return await asyncio.to_thread(_write, resolve_path(path, ctx.working_dir), content)


def _replaced(
    path: Path, old_string: str, new_string: str, *, replace_all: bool
) -> tuple[str, str, int] | str:
    """Return ``(before, after, count)``, or the error text when the edit cannot apply."""
    if not path.exists():
        return consts.MSG_NOT_FOUND.format(path=path)
    if path.is_dir():
        return consts.MSG_IS_DIRECTORY.format(path=path)
    if old_string == new_string:
        return consts.MSG_SAME_STRING
    before = read_text(path)
    count = before.count(old_string) if old_string else 0
    if count == 0:
        return consts.MSG_OLD_NOT_FOUND.format(path=path)
    if count > 1 and not replace_all:
        return consts.MSG_OLD_AMBIGUOUS.format(path=path, count=count)
    return before, before.replace(old_string, new_string), count


def _edit(path: Path, old_string: str, new_string: str, *, replace_all: bool) -> ToolOutput:
    outcome = _replaced(path, old_string, new_string, replace_all=replace_all)
    if isinstance(outcome, str):
        return error_output(outcome)
    before, after, count = outcome
    size = write_text(path, after)
    return text_output(
        consts.MSG_EDITED.format(path=path, count=count),
        metadata={
            consts.META_PATH: str(path),
            consts.META_SIZE: size,
            META_DIFF: unified_diff(before, after, path),
        },
    )


async def _edit_preview(
    path: str, old_string: str, new_string: str, ctx: ToolContext, replace_all: bool = False
) -> ApprovalPreview:
    target = resolve_path(path, ctx.working_dir)
    outcome = await asyncio.to_thread(
        partial(_replaced, target, old_string, new_string, replace_all=replace_all)
    )
    if isinstance(outcome, str):
        return ApprovalPreview(path=str(target))
    before, after, _ = outcome
    return ApprovalPreview(diff=unified_diff(before, after, target), path=str(target))


@_writer(preview=_edit_preview)
async def edit_file(
    path: str, old_string: str, new_string: str, ctx: ToolContext, replace_all: bool = False
) -> ToolOutput:
    """Replace an exact string in a file.

    ``old_string`` must appear exactly once unless ``replace_all`` is true; include
    enough surrounding text to make it unique.
    """
    return await asyncio.to_thread(
        partial(
            _edit,
            resolve_path(path, ctx.working_dir),
            old_string,
            new_string,
            replace_all=replace_all,
        )
    )


WRITE_FILE_TOOLS: list[FunctionTool] = [write_file, edit_file]
