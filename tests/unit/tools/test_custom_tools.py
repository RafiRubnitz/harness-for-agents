"""Custom tools: a Python file in the tools directory becomes a tool without a restart."""

import os
from pathlib import Path

import pytest

from harness.core.consts import Decision, ToolSource
from tests.consts import AGENT_ID
from tests.unit.tools.conftest import ToolsEnv, text_of

GROUP = "dev"
TOOL = "count_words"
OTHER_TOOL = "shout"
FILE = "wordtools.py"
SENTENCE = "four words are here"
FIRST_SUFFIX = "words"
SECOND_SUFFIX = "tokens"
STAMP_STEP_NS = 1_000_000_000


def source(name: str, suffix: str) -> str:
    """A tool file the way the spec shows it: a decorated async function."""
    return (
        "from harness.tools import tool\n\n\n"
        f'@tool(group="{GROUP}", read_only=True, default_permission="allow")\n'
        f"async def {name}(text: str) -> str:\n"
        '    """Count the words of a text."""\n'
        f'    return f"{{len(text.split())}} {suffix}"\n'
    )


def write(env: ToolsEnv, text: str, name: str = FILE) -> Path:
    """Save a tool file with a modification time later than any earlier save."""
    path = env.deps.home.tools_dir / name
    earlier = path.stat().st_mtime_ns if path.exists() else None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if earlier is not None:
        os.utime(path, ns=(earlier + STAMP_STEP_NS, earlier + STAMP_STEP_NS))
    return path


def names(env: ToolsEnv) -> list[str]:
    return [definition.name for definition in env.manager.list(AGENT_ID)]


@pytest.mark.acceptance("08-10.7")
async def test_a_new_tool_file_is_usable_without_a_restart(env: ToolsEnv) -> None:
    env.agent(tools=[GROUP])
    before = names(env)

    write(env, source(TOOL, FIRST_SUFFIX))
    output = await env.run(TOOL, text=SENTENCE)

    (definition,) = [d for d in env.manager.list(AGENT_ID) if d.name == TOOL]
    assert TOOL not in before
    assert text_of(output) == f"{len(SENTENCE.split())} {FIRST_SUFFIX}"
    assert (definition.source, definition.group) == (ToolSource.CUSTOM, GROUP)
    assert definition.default_permission is Decision.ALLOW
    assert definition.input_schema["required"] == ["text"]


async def test_a_changed_file_replaces_its_tools(env: ToolsEnv) -> None:
    env.agent(tools=[GROUP])
    write(env, source(TOOL, FIRST_SUFFIX))
    first = await env.run(TOOL, text=SENTENCE)

    write(env, source(TOOL, SECOND_SUFFIX) + source(OTHER_TOOL, SECOND_SUFFIX).split("\n\n\n")[1])
    second = await env.run(TOOL, text=SENTENCE)

    assert text_of(first).endswith(FIRST_SUFFIX)
    assert text_of(second).endswith(SECOND_SUFFIX)
    assert {TOOL, OTHER_TOOL} <= set(names(env))


async def test_a_removed_file_takes_its_tools_away(env: ToolsEnv) -> None:
    env.agent(tools=[GROUP])
    path = write(env, source(TOOL, FIRST_SUFFIX))
    offered = names(env)

    path.unlink()

    assert TOOL in offered
    assert TOOL not in names(env)


async def test_a_broken_file_is_skipped_and_the_other_tools_stay(env: ToolsEnv) -> None:
    env.agent(tools=[GROUP, "files"])
    write(env, source(TOOL, FIRST_SUFFIX))
    write(env, "raise RuntimeError('not today')\n", "broken.py")

    offered = names(env)
    write(env, source(OTHER_TOOL, FIRST_SUFFIX), "broken.py")  # fixing the file recovers

    assert TOOL in offered
    assert "read_file" in offered
    assert OTHER_TOOL not in offered
    assert OTHER_TOOL in names(env)


async def test_a_file_cannot_take_the_name_of_a_built_in_tool(env: ToolsEnv) -> None:
    env.agent(tools=["files"])
    builtin = next(d for d in env.manager.list(AGENT_ID) if d.name == "read_file")

    write(env, source("read_file", FIRST_SUFFIX))

    (kept,) = [d for d in env.manager.list(AGENT_ID) if d.name == "read_file"]
    assert kept == builtin
    assert kept.source is ToolSource.BUILTIN


async def test_reload_imports_every_file_again(env: ToolsEnv) -> None:
    env.agent(tools=[GROUP])
    path = write(env, source(TOOL, FIRST_SUFFIX))
    await env.run(TOOL, text=SENTENCE)
    stat = path.stat()
    path.write_text(source(TOOL, SECOND_SUFFIX).replace(SECOND_SUFFIX, "x" * len(FIRST_SUFFIX)))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))  # same size, same time: unseen

    unseen = await env.run(TOOL, text=SENTENCE)
    await env.manager.reload()
    seen = await env.run(TOOL, text=SENTENCE)

    assert text_of(unseen).endswith(FIRST_SUFFIX)
    assert text_of(seen).endswith("x" * len(FIRST_SUFFIX))
