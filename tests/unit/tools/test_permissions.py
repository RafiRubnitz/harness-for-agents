"""The permission pipeline, stages 0-2, and the single passage through execute (08 §3a, §5)."""

from pathlib import Path

import pytest
from pydantic import JsonValue

from harness.core.consts import (
    INJECTED_REASON,
    INJECTED_RUN_IN_BACKGROUND,
    META_DENIED_BY,
    META_ERROR_CODE,
    AskedBy,
    ConfigLayer,
    Decision,
    DenySource,
    ErrorCode,
    HookEvent,
)
from harness.core.records import ToolOutput
from harness.tools import text_output, tool
from tests.consts import AGENT_ID, OTHER_AGENT_ID
from tests.unit.tools.conftest import REASON, ToolsEnv, text_of

CONTENT = "new content\n"
OLD_CONTENT = "old content\n"
FILE = "notes.txt"
SAFE_COMMAND = "git status"
OTHER_COMMAND = "git push"
HOOK_REASON = "the hook says so"
ECHO_TOOL = "echo_args"
ALWAYS_ON = ["read_tool_result"]

A, K, D = Decision.ALLOW, Decision.ASK, Decision.DENY
SHELL_REASON_REQUIRED_FOR = "tools.reason.required_for"


def rule(tool_glob: str, decision: Decision, **conditions: str) -> dict[str, JsonValue]:
    return {"tool": tool_glob, "decision": decision.value, **conditions}


@pytest.mark.acceptance("08-10.1")
async def test_agent_sees_only_its_tools(env: ToolsEnv) -> None:
    env.agent(tools=["read_file", "gr*"])

    names = [spec.name for spec in env.manager.specs_for(AGENT_ID)]
    foreign = await env.decide("shell", command=SAFE_COMMAND)
    output = await env.run("shell", command=SAFE_COMMAND)

    assert names == ["read_file", "grep", *ALWAYS_ON]
    assert env.manager.specs_for(OTHER_AGENT_ID) == []
    assert (foreign.decision, foreign.error_code) == (D, ErrorCode.TOOL_NOT_AVAILABLE)
    assert output.metadata[META_ERROR_CODE] == ErrorCode.TOOL_NOT_AVAILABLE.value


async def test_tools_enabled_switches_builtin_tools_off(env: ToolsEnv) -> None:
    await env.config.set("tools.enabled", ["read_file"], ConfigLayer.GLOBAL)

    listed = [definition.name for definition in env.manager.list(AGENT_ID)]
    assert listed == ["read_file", *ALWAYS_ON]


@pytest.mark.acceptance("08-10.2")
@pytest.mark.acceptance("08-10.22")
async def test_read_runs_freely_and_write_asks_with_a_diff(env: ToolsEnv) -> None:
    env.write(FILE, OLD_CONTENT)

    read = await env.decide("read_file", path=FILE)
    write = await env.decide("write_file", path=FILE, content=CONTENT)

    assert (read.decision, read.preview) == (A, None)
    assert (write.decision, write.asked_by) == (K, [AskedBy.POLICY])
    assert write.preview is not None
    assert write.preview.diff is not None
    assert f"-{OLD_CONTENT}" in write.preview.diff
    assert f"+{CONTENT}" in write.preview.diff
    assert (env.work / FILE).read_text(encoding="utf-8") == OLD_CONTENT


@pytest.mark.acceptance("08-10.3")
async def test_policy_deny_beats_a_hook_allow(env: ToolsEnv) -> None:
    await env.rules(rule("write_file", D), layer=ConfigLayer.GLOBAL)
    env.hooks.script(HookEvent.BEFORE_TOOL_CALL, A)

    output = await env.run("write_file", path=FILE, content=CONTENT)

    assert output.is_error
    assert output.metadata[META_DENIED_BY] == DenySource.POLICY.value
    assert not (env.work / FILE).exists()


@pytest.mark.acceptance("08-10.4")
async def test_hook_allow_skips_the_question(env: ToolsEnv) -> None:
    env.hooks.script(HookEvent.BEFORE_TOOL_CALL, A)

    decision = await env.decide("write_file", path=FILE, content=CONTENT)

    assert (decision.decision, decision.asked_by) == (A, [])


async def test_hook_can_tighten_an_allowed_call(env: ToolsEnv) -> None:
    env.write(FILE, CONTENT)
    env.hooks.script(HookEvent.BEFORE_TOOL_CALL, K, reason=HOOK_REASON)
    asked = await env.decide("read_file", path=FILE)
    env.hooks.script(HookEvent.BEFORE_TOOL_CALL, D, reason=HOOK_REASON)

    denied = await env.run("read_file", path=FILE)

    assert (asked.decision, asked.asked_by, asked.hook_reasons) == (
        K,
        [AskedBy.HOOK],
        [HOOK_REASON],
    )
    assert denied.metadata[META_DENIED_BY] == DenySource.HOOK.value
    assert HOOK_REASON in text_of(denied)


@pytest.mark.acceptance("08-10.5")
async def test_path_escaping_the_working_dir_needs_approval(env: ToolsEnv, tmp_path: Path) -> None:
    outside = tmp_path / "secret.txt"
    outside.write_text(CONTENT, encoding="utf-8")
    escaping = f"../{outside.name}"
    env.hooks.script(HookEvent.BEFORE_TOOL_CALL, A)

    decision = await env.decide("read_file", path=escaping)
    refused = await env.run("read_file", approve=False, path=escaping)

    assert (decision.decision, decision.asked_by) == (K, [AskedBy.OUTSIDE_WORKING_DIR])
    assert refused.metadata[META_DENIED_BY] == DenySource.USER.value


async def test_allowed_paths_and_allow_rules_open_other_directories(
    env: ToolsEnv, tmp_path: Path
) -> None:
    extra, ruled = tmp_path / "extra", tmp_path / "ruled"
    for directory in (extra, ruled):
        directory.mkdir()
        (directory / FILE).write_text(CONTENT, encoding="utf-8")
    env.agent(allowed_paths=[str(extra)])
    await env.rules(rule("read_file", A, path=f"{ruled.as_posix()}/**"), layer=ConfigLayer.GLOBAL)

    decisions = [
        (await env.decide("read_file", path=str(directory / FILE))).decision
        for directory in (extra, ruled)
    ]

    assert decisions == [A, A]


@pytest.mark.acceptance("08-10.15")
async def test_schema_carries_reason_but_the_tool_never_sees_it(env: ToolsEnv) -> None:
    seen: list[dict[str, JsonValue]] = []

    @tool(name=ECHO_TOOL, default_permission=A)
    async def echo_args(text: str) -> ToolOutput:
        """Record the arguments."""
        seen.append({"text": text})
        return text_output(text)

    env.manager.register(echo_args)
    env.agent(tools=["files", ECHO_TOOL])

    schemas = [spec.input_schema["properties"] for spec in env.manager.specs_for(AGENT_ID)]
    output = await env.run(ECHO_TOOL, text=CONTENT)

    assert all(
        isinstance(properties, dict)
        and {INJECTED_REASON, INJECTED_RUN_IN_BACKGROUND} <= properties.keys()
        for properties in schemas
    )
    assert INJECTED_REASON not in echo_args.definition.input_schema["properties"]  # type: ignore[operator]
    assert (seen, text_of(output)) == ([{"text": CONTENT}], CONTENT)


@pytest.mark.acceptance("08-10.19")
@pytest.mark.acceptance("08-10.20")
async def test_reason_is_required_only_for_changing_tools(env: ToolsEnv) -> None:
    env.write(FILE, OLD_CONTENT)

    write = await env.run("write_file", reason=None, path=FILE, content=CONTENT)
    read = await env.run("read_file", reason=None, path=FILE)

    assert write.metadata[META_ERROR_CODE] == ErrorCode.TOOL_REASON_MISSING.value
    assert INJECTED_REASON in text_of(write)
    assert env.hooks.events() == [HookEvent.BEFORE_TOOL_CALL, HookEvent.AFTER_TOOL_CALL]
    assert text_of(read) == OLD_CONTENT
    assert (env.work / FILE).read_text(encoding="utf-8") == OLD_CONTENT


@pytest.mark.parametrize(
    ("setting", "read_needs", "write_needs"), [("all", True, True), ("none", False, False)]
)
async def test_reason_requirement_is_configurable(
    env: ToolsEnv, setting: str, read_needs: bool, write_needs: bool
) -> None:
    env.write(FILE, OLD_CONTENT)
    await env.config.set(SHELL_REASON_REQUIRED_FOR, setting, ConfigLayer.GLOBAL)

    read = await env.decide("read_file", reason=None, path=FILE)
    write = await env.decide("write_file", reason=None, path=FILE, content=CONTENT)

    missing = ErrorCode.TOOL_REASON_MISSING
    assert (read.error_code is missing, write.error_code is missing) == (read_needs, write_needs)


@pytest.mark.acceptance("08-10.21")
async def test_hook_receives_the_reason_and_the_clean_input(env: ToolsEnv) -> None:
    await env.decide("write_file", path=FILE, content=CONTENT)

    payload = env.hooks.emitted[0].payload
    assert payload["reason"] == REASON
    assert payload["input"] == {"path": FILE, "content": CONTENT}


@pytest.mark.acceptance("08-10.23")
@pytest.mark.acceptance("06-11.8")
async def test_global_deny_survives_always_approve(env: ToolsEnv) -> None:
    call = env.call("shell", command=SAFE_COMMAND)
    await env.manager.allow_always(AGENT_ID, call, await env.manager.decide(AGENT_ID, call))
    allowed = await env.decide("shell", command=SAFE_COMMAND)
    await env.rules(rule("shell", D), layer=ConfigLayer.GLOBAL)

    blocked = await env.decide("shell", command=SAFE_COMMAND)

    assert allowed.decision is A
    assert (blocked.decision, blocked.denied_by) == (D, DenySource.POLICY)


@pytest.mark.acceptance("08-10.24")
async def test_always_approve_covers_only_the_exact_call(env: ToolsEnv) -> None:
    env.write(FILE, OLD_CONTENT)
    shell_call = env.call("shell", command=SAFE_COMMAND)
    write_call = env.call("write_file", path=FILE, content=CONTENT)
    for call in (shell_call, write_call):
        await env.manager.allow_always(AGENT_ID, call, await env.manager.decide(AGENT_ID, call))

    same = [
        await env.decide("shell", command=SAFE_COMMAND),
        await env.decide("write_file", path=FILE, content=OLD_CONTENT),
    ]
    other = [
        await env.decide("shell", command=OTHER_COMMAND),
        await env.decide("write_file", path=f"other-{FILE}", content=CONTENT),
    ]

    assert [d.decision for d in same] == [A, A]
    assert [d.decision for d in other] == [K, K]


@pytest.mark.acceptance("08-10.26")
async def test_hook_cannot_rewrite_the_input_past_a_deny(env: ToolsEnv) -> None:
    await env.rules(rule("shell", D, match=f"{OTHER_COMMAND}*"), layer=ConfigLayer.GLOBAL)
    env.hooks.script(HookEvent.BEFORE_TOOL_CALL, A, updated={"input": {"command": OTHER_COMMAND}})

    decision = await env.decide("shell", command=SAFE_COMMAND)

    assert (decision.decision, decision.denied_by) == (D, DenySource.POLICY)
    assert decision.input == {"command": OTHER_COMMAND}


async def test_execute_never_runs_an_unanswered_ask(env: ToolsEnv) -> None:
    call = env.call("write_file", path=FILE, content=CONTENT)
    decision = await env.manager.decide(AGENT_ID, call)

    output = await env.manager.execute(AGENT_ID, call, decision)

    assert decision.decision is K
    assert output.metadata[META_ERROR_CODE] == ErrorCode.PERMISSION_DENIED.value
    assert not (env.work / FILE).exists()


@pytest.mark.parametrize(
    ("layers", "expected"),
    [
        ({ConfigLayer.GLOBAL: [rule("shell", A)]}, A),
        ({ConfigLayer.GLOBAL: [rule("shell", A), rule("shell", K)]}, K),
        ({ConfigLayer.GLOBAL: [rule("shell", K)], ConfigLayer.AGENT: [rule("sh*", A)]}, A),
        ({ConfigLayer.GLOBAL: [rule("shell", A)], ConfigLayer.AGENT: [rule("shell", K)]}, K),
        ({ConfigLayer.GLOBAL: [rule("shell", D)], ConfigLayer.AGENT: [rule("shell", A)]}, D),
        ({ConfigLayer.GLOBAL: [rule("shell", A, match="npm *")]}, K),
        ({ConfigLayer.GLOBAL: [rule("read_file", A)]}, K),
    ],
)
async def test_rule_precedence(
    env: ToolsEnv, layers: dict[ConfigLayer, list[dict[str, JsonValue]]], expected: Decision
) -> None:
    for layer, rules in layers.items():
        await env.rules(*rules, layer=layer)

    decision = await env.decide("shell", command=SAFE_COMMAND)

    assert decision.decision is expected
