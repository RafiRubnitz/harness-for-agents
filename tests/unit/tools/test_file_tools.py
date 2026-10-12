"""Built-in file tools: read, list, glob, grep, write, edit, delete, restore (08 §4.1)."""

import pytest
from pydantic import JsonValue

from harness.core.consts import (
    META_BACKUP_ID,
    META_DENIED_BY,
    META_DIFF,
    META_RESULT_ID,
    META_TRUNCATED,
    AskedBy,
    ConfigLayer,
    Decision,
    DenySource,
    HookEvent,
)
from harness.core.types import BackupId
from harness.core.utils.paths import HarnessHome
from tests.consts import AGENT_ID, HEBREW_MIXED
from tests.unit.tools.conftest import REASON, ToolsEnv, text_of

FILE = "notes.txt"
LINES = ["first line", "second line", HEBREW_MIXED, "last line"]
TEXT = "\n".join(LINES) + "\n"
NEEDLE = "second"
REPLACEMENT = "2nd"
MISSING = "missing.txt"
SUBDIR = "pkg"
LIMIT = 2


async def test_read_file_returns_text_and_slices_lines(env: ToolsEnv) -> None:
    env.write(FILE, TEXT)

    whole = await env.run("read_file", path=FILE)
    part = await env.run("read_file", path=FILE, offset=1, limit=LIMIT)

    assert text_of(whole) == TEXT
    assert text_of(part).splitlines() == LINES[1 : 1 + LIMIT]
    assert (whole.metadata[META_TRUNCATED], part.metadata[META_TRUNCATED]) == (False, True)


@pytest.mark.parametrize("target", [MISSING, SUBDIR, "blob.bin"])
async def test_read_file_reports_what_it_cannot_read(env: ToolsEnv, target: str) -> None:
    (env.work / SUBDIR).mkdir()
    (env.work / "blob.bin").write_bytes(b"\x00\x01\x02")

    output = await env.run("read_file", path=target)

    assert output.is_error
    assert target in text_of(output)


async def test_list_glob_and_grep_find_files(env: ToolsEnv) -> None:
    env.write(FILE, TEXT)
    env.write(f"{SUBDIR}/mod.py", f"x = '{NEEDLE}'\n")
    env.write(".git/config", NEEDLE)

    listed = text_of(await env.run("list_dir")).splitlines()
    globbed = text_of(await env.run("glob", pattern="**/*.py")).splitlines()
    found = text_of(await env.run("grep", pattern=NEEDLE.upper(), ignore_case=True)).splitlines()
    filtered = text_of(await env.run("grep", pattern=NEEDLE, include="*.py")).splitlines()
    nothing = await env.run("grep", pattern="(")

    assert listed == [".git/", FILE, f"{SUBDIR}/"]
    assert globbed == [f"{SUBDIR}/mod.py"]
    assert found == [f"{FILE}:2: {LINES[1]}", f"{SUBDIR}/mod.py:1: x = '{NEEDLE}'"]
    assert filtered == found[1:]
    assert nothing.is_error


async def test_write_file_creates_and_reports_the_diff(env: ToolsEnv) -> None:
    target = f"{SUBDIR}/{FILE}"

    output = await env.run("write_file", path=target, content=TEXT)

    diff = output.metadata[META_DIFF]
    assert (env.work / target).read_bytes() == TEXT.encode("utf-8")
    assert isinstance(diff, str)
    assert f"+{LINES[0]}" in diff
    assert not output.is_error


async def test_user_refusal_leaves_the_file_alone(env: ToolsEnv) -> None:
    output = await env.run("write_file", approve=False, path=FILE, content=TEXT)

    assert output.metadata[META_DENIED_BY] == DenySource.USER.value
    assert not (env.work / FILE).exists()


async def test_edit_file_replaces_an_exact_string(env: ToolsEnv) -> None:
    env.write(FILE, TEXT)

    output = await env.run("edit_file", path=FILE, old_string=NEEDLE, new_string=REPLACEMENT)

    assert (env.work / FILE).read_text(encoding="utf-8") == TEXT.replace(NEEDLE, REPLACEMENT)
    assert f"+{REPLACEMENT} line" in str(output.metadata[META_DIFF])


@pytest.mark.parametrize(
    ("old", "new", "replace_all", "changed"),
    [
        ("line", "row", False, False),
        ("line", "row", True, True),
        ("absent", "row", False, False),
        (NEEDLE, NEEDLE, False, False),
        ("", "row", False, False),
    ],
)
async def test_edit_file_refuses_unclear_edits(
    env: ToolsEnv, old: str, new: str, replace_all: bool, changed: bool
) -> None:
    env.write(FILE, TEXT)

    output = await env.run(
        "edit_file", path=FILE, old_string=old, new_string=new, replace_all=replace_all
    )

    after = (env.work / FILE).read_text(encoding="utf-8")
    assert output.is_error is not changed
    assert (after != TEXT) is changed


@pytest.mark.acceptance("08-10.16")
async def test_delete_file_trashes_backs_up_and_restores(env: ToolsEnv) -> None:
    path = env.write(FILE, TEXT)

    deleted = await env.run("delete_file", path=FILE)
    backup_id = BackupId(str(deleted.metadata[META_BACKUP_ID]))
    record = await env.blobs.get_deleted(backup_id)
    gone = not path.exists()
    restored = await env.manager.restore_deleted(AGENT_ID, backup_id)
    again = await env.manager.restore_deleted(AGENT_ID, backup_id)

    assert gone
    assert env.trashed == [str(path)]
    assert (record.original_path, record.reason, record.trashed) == (str(path), REASON, True)
    assert str(backup_id) in text_of(deleted)
    assert not restored.is_error
    assert path.read_bytes() == TEXT.encode("utf-8")
    assert again.is_error


async def test_restore_deleted_tool_and_unknown_backup(env: ToolsEnv) -> None:
    path = env.write(FILE, TEXT)
    deleted = await env.run("delete_file", path=FILE)

    restored = await env.run("restore_deleted", backup_id=str(deleted.metadata[META_BACKUP_ID]))
    unknown = await env.run("restore_deleted", backup_id="bak_unknown")

    assert not restored.is_error
    assert path.exists()
    assert unknown.is_error


@pytest.mark.acceptance("08-10.17")
@pytest.mark.parametrize("target", [SUBDIR, MISSING, ".git/config", "*.txt"])
async def test_delete_file_refuses_and_deletes_nothing(env: ToolsEnv, target: str) -> None:
    (env.work / SUBDIR).mkdir()
    kept = [env.write(".git/config", TEXT), env.write(FILE, TEXT)]

    output = await env.run("delete_file", path=target)

    assert output.is_error
    assert all(path.exists() for path in kept)
    assert (env.work / SUBDIR).is_dir()
    assert env.trashed == []


async def test_protected_paths_cannot_be_written_but_can_be_read(env: ToolsEnv) -> None:
    env.write(".git/config", TEXT)
    env.hooks.script(HookEvent.BEFORE_TOOL_CALL, Decision.ALLOW)

    written = await env.decide("write_file", path=".harness/config.yaml", content=TEXT)
    read = await env.decide("read_file", path=".git/config")

    assert (written.decision, written.denied_by) == (Decision.DENY, DenySource.PROTECTED_PATH)
    assert read.decision is Decision.ALLOW


@pytest.mark.acceptance("08-10.18")
@pytest.mark.acceptance("08-10.25")
async def test_permanent_delete_always_asks(env: ToolsEnv) -> None:
    path = env.write(FILE, TEXT)
    await env.rules({"tool": "delete_file", "decision": "allow"}, layer=ConfigLayer.GLOBAL)
    env.hooks.script(HookEvent.BEFORE_TOOL_CALL, Decision.ALLOW)

    ordinary = await env.decide("delete_file", path=FILE)
    env.hooks.script(HookEvent.BEFORE_TOOL_CALL, Decision.ALLOW)
    permanent = await env.decide("delete_file", path=FILE, permanent=True)
    output = await env.run("delete_file", path=FILE, permanent=True)

    assert ordinary.decision is Decision.ALLOW
    assert (permanent.decision, permanent.asked_by) == (Decision.ASK, [AskedBy.PERMANENT_DELETE])
    assert permanent.preview is not None
    assert (permanent.preview.permanent, permanent.preview.size) == (True, len(TEXT.encode()))
    assert not output.is_error
    assert not path.exists()
    assert env.trashed == []


async def test_agent_workspace_under_the_harness_home_is_writable(
    env: ToolsEnv, tmp_harness_home: HarnessHome
) -> None:
    home_root = tmp_harness_home.root
    default_dir = tmp_harness_home.workspace_for(AGENT_ID)
    default_dir.mkdir(parents=True)
    env.agent(working_dir=str(default_dir))

    inside = await env.decide("write_file", path=FILE, content=TEXT)
    home_file = await env.decide("write_file", path=str(home_root / "config.yaml"), content=TEXT)

    assert inside.decision is Decision.ASK
    assert home_file.denied_by is DenySource.PROTECTED_PATH


@pytest.mark.acceptance("08-10.9")
async def test_only_changing_tools_are_exclusive(env: ToolsEnv) -> None:
    flags = {definition.name: definition.exclusive for definition in env.manager.list(AGENT_ID)}

    assert (flags["read_file"], flags["grep"]) == (False, False)
    assert (flags["write_file"], flags["edit_file"], flags["delete_file"]) == (True, True, True)


async def test_large_result_is_stored_whole(env: ToolsEnv) -> None:
    limit = env.config.config().context.tool_results.truncate_tokens
    big = "word " * (limit * 2)
    env.write(FILE, big)

    output = await env.run("read_file", path=FILE)

    result_id = output.metadata[META_RESULT_ID]
    assert isinstance(result_id, str)
    assert (await env.blobs.get(result_id)).decode("utf-8") == big


async def test_stored_result_is_read_back_in_parts(env: ToolsEnv) -> None:
    limit = env.config.config().context.tool_results.truncate_tokens
    big = "".join(f"word{number} " for number in range(limit * 2))
    env.write(FILE, big)
    stored = await env.run("read_file", path=FILE)
    result_id = stored.metadata[META_RESULT_ID]

    parts: list[str] = []
    offset: JsonValue = 0
    more: JsonValue = True
    while more:
        part = await env.run("read_tool_result", result_id=result_id, offset=offset)
        parts.append(text_of(part).split("\n", 1)[1])
        offset, more = part.metadata["next_offset"], part.metadata[META_TRUNCATED]

    assert len(parts) > 1
    assert "".join(parts) == big


async def test_reading_an_unknown_result_is_an_error_value(env: ToolsEnv) -> None:
    output = await env.run("read_tool_result", result_id="blob_missing")

    assert output.is_error
    assert "blob_missing" in text_of(output)


async def test_an_agent_with_only_the_shell_can_still_read_stored_results(env: ToolsEnv) -> None:
    env.agent(tools=["shell"])

    names = [definition.name for definition in env.manager.list(AGENT_ID)]

    assert "read_tool_result" in names
    assert "read_file" not in names


async def test_an_agent_without_tools_gets_none(env: ToolsEnv) -> None:
    env.agent(tools=[])

    assert env.manager.list(AGENT_ID) == []
