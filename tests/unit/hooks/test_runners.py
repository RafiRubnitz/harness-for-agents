"""The hook runners: Python functions, external programs, command lines and matching."""

import json
import sys
from datetime import UTC, datetime

import pytest
from pydantic import JsonValue

from harness.core.consts import AddressType, Decision, ErrorCode, HookEvent
from harness.core.hook_payloads import MessageReceivedPayload
from harness.core.hook_types import HookInput, HookSpec
from harness.core.types import Address, MessageOrigin, TextBlock
from harness.hooks.utils import matches, split_command
from tests.consts import AGENT_ID, AGENT_NAME, HEBREW_PLAIN
from tests.unit.hooks.conftest import COMMAND, HooksEnv, tool_call

TEMPLATE = "reviewer"
SPACED_PATH = "my notes/to do.txt"
EXIT_CODE = 3
SHORT_TIMEOUT_MS = 300
STUCK_S = 30

ECHO_COMMAND = (
    "import json, sys\n"
    "inp = json.load(sys.stdin)\n"
    "print(json.dumps({'decision': 'deny', 'reason': inp['payload']['input']['command']}))\n"
)
ECHO_ARGS = "import json, sys\nprint(json.dumps({'reason': json.dumps(sys.argv[1:])}))\n"


def _input(event: HookEvent, payload: dict[str, JsonValue]) -> HookInput:
    return HookInput(
        event=event,
        agent_id=AGENT_ID,
        agent_name=AGENT_NAME,
        payload=payload,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _spec(**match: str) -> HookSpec:
    return HookSpec(name="h", event=HookEvent.BEFORE_TOOL_CALL, match=match, run="x")


def _line(script: object, *arguments: str) -> str:
    return " ".join([f'"{sys.executable}"', f'"{script}"', *arguments])


# --- the same contract for both kinds ---


@pytest.mark.acceptance("05-10.1")
async def test_a_python_hook_and_a_program_answer_alike(env: HooksEnv) -> None:
    env.module(
        "echo",
        "async def hook(inp: HookInput) -> HookOutput:\n"
        "    return HookOutput(decision='deny', reason=inp.payload['input']['command'])\n",
    )
    script = env.script("echo.py", ECHO_COMMAND)
    in_process = env.hook("python:echo:hook")
    from_python = await env.emit()
    env.manager.unregister(in_process.name)
    env.hook(str(script))

    from_program = await env.emit()

    assert (from_python.decision, from_python.reasons) == (Decision.DENY, [COMMAND])
    assert (from_program.decision, from_program.reasons) == (Decision.DENY, [COMMAND])
    assert from_program.payload == from_python.payload


@pytest.mark.parametrize("answer", ["None", "{'decision': 'ask'}", "HookOutput(decision='ask')"])
async def test_a_python_hook_may_answer_with_nothing_or_a_dict(env: HooksEnv, answer: str) -> None:
    env.module("loose", f"async def hook(inp):\n    return {answer}\n")
    env.hook("python:loose:hook")

    merged = await env.emit()

    assert merged.decision is (None if answer == "None" else Decision.ASK)
    assert merged.fired[0].error is None


async def test_an_answer_that_is_not_a_hook_output_is_an_error(env: HooksEnv) -> None:
    env.module("odd", "async def hook(inp):\n    return 42\n")
    env.hook("python:odd:hook")

    merged = await env.emit()

    assert merged.fired[0].error is ErrorCode.HOOK_INVALID_OUTPUT


# --- external programs ---


@pytest.mark.parametrize(
    ("body", "error", "decision"),
    [
        ("", None, None),  # exit 0 with nothing printed: no opinion
        (f"import sys\nsys.exit({EXIT_CODE})\n", ErrorCode.HOOK_FAILED, Decision.DENY),
        ("print('not json')\n", ErrorCode.HOOK_INVALID_OUTPUT, Decision.DENY),
        ('print(\'{"decision": "maybe"}\')\n', ErrorCode.HOOK_INVALID_OUTPUT, Decision.DENY),
    ],
)
async def test_how_a_program_answer_is_read(
    env: HooksEnv, body: str, error: ErrorCode | None, decision: Decision | None
) -> None:
    env.hook(str(env.script("hook.py", body)))

    merged = await env.emit()

    assert (merged.fired[0].error, merged.decision) == (error, decision)


async def test_a_program_answers_in_utf8(env: HooksEnv) -> None:
    body = f"import json\nprint(json.dumps({{'reason': {HEBREW_PLAIN!r}}}, ensure_ascii=False))\n"
    env.hook(str(env.script("hebrew.py", body)))

    merged = await env.emit()

    assert merged.reasons == [HEBREW_PLAIN]


async def test_a_relative_file_is_found_next_to_the_home_directory(env: HooksEnv) -> None:
    env.script("echo.py", ECHO_COMMAND)
    env.hook("echo.py")

    merged = await env.emit()

    assert merged.reasons == [COMMAND]


async def test_a_stuck_program_is_cut_off(env: HooksEnv) -> None:
    script = env.script("stuck.py", f"import time\ntime.sleep({STUCK_S})\n")
    env.hook(str(script), timeout_ms=SHORT_TIMEOUT_MS)

    merged = await env.emit()

    assert merged.fired[0].error is ErrorCode.HOOK_TIMEOUT


# --- command lines ---


async def test_placeholders_are_filled_inside_single_arguments(env: HooksEnv) -> None:
    script = env.script("args.py", ECHO_ARGS)
    env.hook(_line(script, "{payload.input.path}", "--agent={agent_name}", "{event}", "{{x}}"))

    merged = await env.emit(tool_call("write_file", path=SPACED_PATH, content=""))

    assert json.loads(merged.reasons[0]) == [
        SPACED_PATH,
        f"--agent={AGENT_NAME}",
        HookEvent.BEFORE_TOOL_CALL.value,
        "{x}",
    ]


async def test_a_placeholder_for_a_missing_field_is_a_hook_error(env: HooksEnv) -> None:
    script = env.script("args.py", ECHO_ARGS)
    env.hook(_line(script, "{payload.input.path}"))

    merged = await env.emit()  # a shell call has no path

    assert merged.fired[0].error is ErrorCode.HOOK_FAILED
    assert merged.decision is Decision.DENY


def test_a_command_line_is_split_without_a_shell() -> None:
    parts = split_command(_line("a script.py", "--flag", "x;y", "&&", "z"))

    assert parts == [sys.executable, "a script.py", "--flag", "x;y", "&&", "z"]
    with pytest.raises(ValueError, match="nothing"):
        split_command("   ")


# --- matching ---


def test_every_match_key_must_fit() -> None:
    write = _input(
        HookEvent.BEFORE_TOOL_CALL,
        tool_call("mcp.github.push", path="a/b.py").model_dump(mode="json"),
    )

    assert matches(_spec(), write, None)
    assert matches(_spec(tool="mcp.github.*", agent="test*", path_glob="a/*.py"), write, None)
    assert not matches(_spec(tool="mcp.github.*", agent="other*"), write, None)
    assert not matches(_spec(tool="shell"), write, None)
    assert matches(_spec(template=TEMPLATE), write, TEMPLATE)
    assert not matches(_spec(template=TEMPLATE), write, None)
    assert not matches(_spec(colour="red"), write, None)


def test_a_path_glob_never_fits_an_event_without_a_path() -> None:
    shell = _input(HookEvent.BEFORE_TOOL_CALL, tool_call().model_dump(mode="json"))

    assert not matches(_spec(path_glob="*"), shell, None)


def test_messages_are_matched_by_where_they_came_from() -> None:
    origin = MessageOrigin(source=Address(type=AddressType.USER, id="me"))
    payload = MessageReceivedPayload(content=[TextBlock(text="hi")], origin=origin)
    message = _input(HookEvent.MESSAGE_RECEIVED, payload.model_dump(mode="json"))

    assert matches(_spec(source_type=AddressType.USER.value), message, None)
    assert not matches(_spec(source_type=AddressType.TASK.value), message, None)
