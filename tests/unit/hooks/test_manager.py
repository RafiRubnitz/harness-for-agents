"""The hooks manager: which hooks run for an event and how their answers are merged."""

import asyncio
from pathlib import Path

import pytest

from harness.core.consts import ConfigLayer, Decision, ErrorCode, HookEvent, HookScope
from harness.core.errors import HookError
from harness.core.hook_payloads import AfterToolCallPayload
from harness.core.hook_types import HookSpec
from harness.core.records import ToolOutput
from harness.core.types import TextBlock
from tests.consts import AGENT_ID, OTHER_AGENT_ID
from tests.unit.hooks.conftest import CALL_ID, COMMAND, TOOL, TOOL_REASON, HooksEnv, tool_call

REASON = "recursive deletes are blocked"
OTHER_REASON = "ask the user first"
CONTEXT = "today is Monday"
OTHER_CONTEXT = "the build is red"
SAFE_COMMAND = "git status --short"
SHORT_TIMEOUT_MS = 50
STUCK_S = 30
MARKER = "ran.txt"
POLLS = 200
POLL_S = 0.01

RAISING = "async def hook(inp: HookInput) -> HookOutput:\n    raise RuntimeError('boom')\n"
STUCK = (
    "import asyncio\n\n"
    f"async def hook(inp: HookInput) -> HookOutput:\n    await asyncio.sleep({STUCK_S})\n"
    "    return HookOutput()\n"
)


def after_tool() -> AfterToolCallPayload:
    return AfterToolCallPayload(
        tool=TOOL,
        call_id=CALL_ID,
        input={"command": COMMAND},
        reason=TOOL_REASON,
        result=ToolOutput(content=[TextBlock(text="clean")]),
    )


# --- merging ---


async def test_no_hook_means_no_opinion_and_the_payload_is_untouched(env: HooksEnv) -> None:
    payload = tool_call()

    merged = await env.emit(payload)

    assert merged.decision is None
    assert merged.payload == payload.model_dump(mode="json")
    assert merged.fired == []


@pytest.mark.acceptance("05-10.4")
async def test_the_strictest_decision_wins(env: HooksEnv) -> None:
    env.answering(f"decision='allow', reason={REASON!r}")
    env.answering(f"decision='ask', reason={OTHER_REASON!r}")
    env.answering("")
    env.answering("decision='allow'")

    merged = await env.emit()

    assert merged.decision is Decision.ASK
    assert merged.reasons == [REASON, OTHER_REASON]
    assert [fired.decision for fired in merged.fired] == [
        Decision.ALLOW,
        Decision.ASK,
        None,
        Decision.ALLOW,
    ]


async def test_a_deny_ends_the_chain(env: HooksEnv) -> None:
    env.answering("decision='ask'")
    denying = env.answering(f"decision='deny', reason={REASON!r}")
    env.answering("decision='allow'")

    merged = await env.emit()

    assert merged.decision is Decision.DENY
    assert merged.reasons == [REASON]
    assert merged.fired[-1].hook == denying.name
    assert len(merged.fired) == len(env.manager.list()) - 1


async def test_context_from_every_hook_is_kept(env: HooksEnv) -> None:
    env.answering(f"additional_context={CONTEXT!r}")
    env.answering(f"additional_context={OTHER_CONTEXT!r}")

    merged = await env.emit()

    assert merged.additional_context is not None
    assert merged.additional_context.splitlines() == [CONTEXT, OTHER_CONTEXT]


@pytest.mark.acceptance("05-10.6")
async def test_a_hook_changes_the_tool_input_but_not_the_tool_or_the_reason(
    env: HooksEnv,
) -> None:
    env.answering(
        f"updated_payload={{'input': {{'command': {SAFE_COMMAND!r}}}, "
        "'tool': 'other', 'reason': 'changed'}"
    )

    merged = await env.emit()

    assert merged.payload["input"] == {"command": SAFE_COMMAND}
    assert (merged.payload["tool"], merged.payload["reason"]) == (TOOL, TOOL_REASON)


async def test_each_hook_sees_the_payload_the_previous_one_left(env: HooksEnv) -> None:
    env.answering(f"updated_payload={{'input': {{'command': {SAFE_COMMAND!r}}}}}")
    env.answering("reason=inp.payload['input']['command']")

    merged = await env.emit()

    assert merged.reasons == [SAFE_COMMAND]


async def test_an_invalid_payload_change_is_a_hook_error_and_changes_nothing(
    env: HooksEnv,
) -> None:
    env.answering("updated_payload={'input': 'not an object'}")

    merged = await env.emit()

    assert merged.payload["input"] == {"command": COMMAND}
    assert merged.fired[0].error is ErrorCode.HOOK_INVALID_OUTPUT
    assert merged.decision is Decision.DENY  # a failing tool guard blocks


# --- failures ---


@pytest.mark.acceptance("05-10.10")
async def test_a_raising_hook_blocks_before_a_tool_and_is_ignored_after_it(
    env: HooksEnv,
) -> None:
    env.module("raising", RAISING)
    env.hook("python:raising:hook", HookEvent.BEFORE_TOOL_CALL)
    env.hook("python:raising:hook", HookEvent.AFTER_TOOL_CALL)

    before = await env.emit()
    after = await env.emit(after_tool(), HookEvent.AFTER_TOOL_CALL)

    assert before.decision is Decision.DENY
    assert before.reasons  # the model is told why
    assert after.decision is None
    assert after.reasons == []
    assert [fired.error for fired in (*before.fired, *after.fired)] == [ErrorCode.HOOK_FAILED] * 2


async def test_on_error_decides_what_a_failure_counts_as(env: HooksEnv) -> None:
    env.module("raising", RAISING)
    env.hook("python:raising:hook", on_error="ask")
    env.hook("python:raising:hook", HookEvent.AFTER_TOOL_CALL, on_error="deny")

    before = await env.emit()
    after = await env.emit(after_tool(), HookEvent.AFTER_TOOL_CALL)

    assert before.decision is Decision.ASK
    assert after.decision is Decision.DENY


@pytest.mark.acceptance("05-10.11")
async def test_a_stuck_hook_is_cut_off_at_its_timeout(env: HooksEnv) -> None:
    env.module("stuck", STUCK)
    env.hook("python:stuck:hook", timeout_ms=SHORT_TIMEOUT_MS)

    async with asyncio.timeout(STUCK_S / 10):
        merged = await env.emit()

    assert merged.fired[0].error is ErrorCode.HOOK_TIMEOUT
    assert merged.decision is Decision.DENY


async def test_an_async_hook_runs_without_being_waited_for_and_decides_nothing(
    env: HooksEnv,
) -> None:
    marker = env.home.root / MARKER
    body = (
        "from pathlib import Path\n\n"
        "async def hook(inp: HookInput) -> HookOutput:\n"
        f"    Path({str(marker)!r}).write_text(inp.agent_name, encoding='utf-8')\n"
        "    return HookOutput(decision='deny')\n"
    )
    env.module("notify", body)
    env.hook("python:notify:hook", **{"async": True})

    merged = await env.emit()
    for _ in range(POLLS):
        if marker.exists():
            break
        await asyncio.sleep(POLL_S)

    assert merged.decision is None
    assert merged.fired == []
    assert marker.exists()


async def test_closing_stops_async_hooks_that_still_run(env: HooksEnv) -> None:
    env.module("stuck", STUCK)
    env.hook("python:stuck:hook", **{"async": True})
    await env.emit()

    async with asyncio.timeout(STUCK_S / 10):
        await env.manager.close()


# --- loading ---


@pytest.mark.acceptance("05-10.9")
async def test_a_changed_hook_file_is_used_without_a_restart(env: HooksEnv) -> None:
    spec = env.answering("decision='allow'")
    module = spec.run.split(":")[1]
    first = await env.emit()

    env.module(
        module,
        "async def hook(inp: HookInput) -> HookOutput:\n"
        f"    return HookOutput(decision='deny', reason={REASON!r})\n",
    )
    second = await env.emit()

    assert first.decision is Decision.ALLOW
    assert (second.decision, second.reasons) == (Decision.DENY, [REASON])


@pytest.mark.parametrize(
    ("run", "match"),
    [
        ("python:missing:hook", {}),
        ("python:broken:hook", {}),
        ("python:plain:absent", {}),
        ("python:plain:not_async", {}),
        ("python:no-such-form", {}),
        ("python:plain:hook", {"colour": "red"}),
        ("no-such-program-anywhere --flag", {}),
        ("missing_file.py", {}),
        ("echo {unknown}", {}),
        ('echo "unbalanced', {}),
    ],
)
async def test_a_hook_that_cannot_be_loaded_is_reported_and_the_rest_still_run(
    env: HooksEnv, run: str, match: dict[str, str]
) -> None:
    env.module("broken", "raise RuntimeError('at import')\n")
    env.module(
        "plain",
        "def not_async(inp):\n    return None\n\n"
        "async def hook(inp: HookInput) -> HookOutput:\n    return HookOutput()\n",
    )
    bad = env.hook(run, match=match)
    env.answering(f"decision='ask', reason={REASON!r}")

    first = await env.emit()
    await env.emit()

    assert first.decision is Decision.ASK
    assert [fired.hook for fired in first.fired] != [bad.name]
    assert [name for name, _ in env.load_errors] == [bad.name]  # reported once, not per event


async def test_reload_checks_the_configured_hooks(env: HooksEnv) -> None:
    await env.global_hooks(
        {"name": "gone", "event": HookEvent.RUN_END.value, "run": "python:gone:hook"}
    )

    assert [name for name, _ in env.load_errors] == ["gone"]  # the config change reloads


async def test_configured_hooks_run_after_registered_ones_and_only_for_their_event(
    env: HooksEnv,
) -> None:
    env.module(
        "ctx",
        f"async def hook(inp):\n    return HookOutput(reason={REASON!r})\n",
    )
    spec = {
        "name": "from-config",
        "event": HookEvent.BEFORE_TOOL_CALL.value,
        "run": "python:ctx:hook",
    }
    await env.global_hooks(spec, {**spec, "name": "other-event", "event": HookEvent.RUN_END.value})
    registered = env.answering("")

    merged = await env.emit()

    assert [fired.hook for fired in merged.fired] == [registered.name, "from-config"]
    assert [found.name for found in env.manager.list(HookEvent.RUN_END)] == ["other-event"]
    assert env.manager.list(HookEvent.BEFORE_TOOL_CALL)[-1].scope is HookScope.GLOBAL


@pytest.mark.acceptance("05-10.7")
async def test_a_project_hook_runs_only_for_agents_working_in_that_project(
    env: HooksEnv, tmp_path: Path
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    project = env.work / ".harness"
    (project / "hooks").mkdir(parents=True)
    (project / "hooks" / "deny.py").write_text(
        f"import json\nprint(json.dumps({{'decision': 'deny', 'reason': {REASON!r}}}))\n",
        encoding="utf-8",
    )
    (project / "hooks.yaml").write_text(
        "hooks:\n  - name: project-guard\n"
        f"    event: {HookEvent.BEFORE_TOOL_CALL.value}\n    run: hooks/deny.py\n",
        encoding="utf-8",
    )
    await env.config.set("projects.trusted", [str(env.work)], ConfigLayer.GLOBAL)
    await env.config.load_agent(AGENT_ID, env.work)
    await env.config.load_agent(OTHER_AGENT_ID, elsewhere)

    inside = await env.emit(agent_id=AGENT_ID)
    outside = await env.emit(agent_id=OTHER_AGENT_ID)

    assert (inside.decision, inside.reasons) == (Decision.DENY, [REASON])
    assert outside.fired == []
    assert [found.scope for found in env.manager.list(agent_id=AGENT_ID)] == [HookScope.PROJECT]


# --- registry and log ---


async def test_registered_names_are_unique_and_can_be_removed(env: HooksEnv) -> None:
    spec = env.answering("decision='deny'")

    with pytest.raises(HookError):
        env.manager.register(spec)
    env.manager.unregister(spec.name)

    assert (await env.emit()).decision is None


async def test_every_run_is_logged_with_its_decision(env: HooksEnv) -> None:
    quiet = env.answering("suppress_output=True")
    loud = env.answering(f"decision='allow', reason={REASON!r}")

    merged = await env.emit()

    assert env.observed == merged.fired
    by_hook = {fired.hook: fired for fired in merged.fired}
    assert by_hook[quiet.name].suppressed
    assert (by_hook[loud.name].decision, by_hook[loud.name].reason) == (Decision.ALLOW, REASON)
    assert {fired.agent_id for fired in merged.fired} == {AGENT_ID}
    assert {fired.event for fired in merged.fired} == {HookEvent.BEFORE_TOOL_CALL}


async def test_hooks_are_filtered_by_their_match(env: HooksEnv) -> None:
    env.module(
        "deny",
        "async def hook(inp: HookInput) -> HookOutput:\n    return HookOutput(decision='deny')\n",
    )
    env.manager.register(
        HookSpec(
            name="writes-only",
            event=HookEvent.BEFORE_TOOL_CALL,
            match={"tool": "write_*", "path_glob": "*.py"},
            run="python:deny:hook",
        )
    )

    shell = await env.emit(tool_call())
    text = await env.emit(tool_call("write_file", path="notes/readme.md", content=""))
    code = await env.emit(tool_call("write_file", path="src/app.py", content=""))

    assert (shell.decision, text.decision) == (None, None)
    assert code.decision is Decision.DENY
