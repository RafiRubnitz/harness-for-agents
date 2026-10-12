"""``harness chat`` and ``harness agents`` against the real core (02 §3; 12 §7.2)."""

import pytest

from harness.cli import run
from harness.cli.consts import (
    ANSI_RED,
    ANSWER_ALWAYS,
    ANSWER_DENY,
    ANSWER_ONCE,
    APPROVAL_PROMPT,
    CMD_AGENTS,
    CMD_CHAT,
    EXIT_CORE_LOCKED,
    EXIT_ERROR,
    EXIT_OK,
    MSG_COMPACTED,
)
from harness.core.consts import Decision, ErrorKind, RequestPurpose, SlashCommand
from harness.core.lock import CoreLock
from harness.core.types import ToolCallBlock
from tests.consts import AGENT_NAME
from tests.fakes.adapter import gateway_error, text_response, tool_response
from tests.fakes.terminal import FakeTerminal
from tests.integration.conftest import CoreKit

QUESTION = "please do the thing"
ANSWER = "the thing is done"
SECOND_ANSWER = "and again"
SUMMARY = "SUMMARY: the thing was asked for and done"
TOOL = "risky_tool"
CALL = ToolCallBlock(id="call_1", name=TOOL, input={"target": "x"})
EXIT = "/exit"
UNKNOWN_SLASH = "/nonsense"
OTHER_NAME = "someone-else"


def chat(kit: CoreKit, terminal: FakeTerminal, *args: str) -> int:
    return run([CMD_CHAT, *args], kit.launch, terminal)


def test_chat_streams_the_answer_and_leaves_on_exit(kit: CoreKit) -> None:
    kit.gateway.outcomes = [text_response(ANSWER), text_response(SECOND_ANSWER)]
    terminal = FakeTerminal(inputs=[QUESTION, "", QUESTION, EXIT, QUESTION])

    code = chat(kit, terminal, AGENT_NAME, "--new")

    assert code == EXIT_OK
    assert terminal.output.index(ANSWER) < terminal.output.index(SECOND_ANSWER)
    assert terminal.inputs == [QUESTION]
    assert kit.gateway.outcomes == []


def test_chat_ends_with_the_input(kit: CoreKit) -> None:
    kit.gateway.outcomes = [text_response(ANSWER)]
    terminal = FakeTerminal(inputs=[QUESTION])

    code = chat(kit, terminal)

    assert code == EXIT_OK
    assert ANSWER in terminal.output


def test_chat_continues_the_same_agent_next_time(kit: CoreKit) -> None:
    kit.gateway.outcomes = [text_response(ANSWER), text_response(SECOND_ANSWER)]
    chat(kit, FakeTerminal(inputs=[QUESTION]), AGENT_NAME, "--new")
    terminal = FakeTerminal(inputs=[QUESTION])

    code = chat(kit, terminal, AGENT_NAME)

    sent = [message.content for message in kit.gateway.requests[-1].messages]
    assert code == EXIT_OK
    assert SECOND_ANSWER in terminal.output
    assert len(sent) == len([QUESTION, ANSWER, QUESTION])


@pytest.mark.parametrize(
    ("answer", "runs", "always"),
    [(ANSWER_ONCE, [TOOL], []), (ANSWER_ALWAYS, [TOOL], [TOOL]), (ANSWER_DENY, [], [])],
)
@pytest.mark.acceptance("02-10.8")
def test_inline_approval(kit: CoreKit, answer: str, runs: list[str], always: list[str]) -> None:
    kit.tools.add(TOOL, decision=Decision.ASK)
    kit.gateway.outcomes = [tool_response(CALL), text_response(ANSWER)]
    terminal = FakeTerminal(inputs=[QUESTION, "what?", answer])

    code = chat(kit, terminal, "--new")

    assert code == EXIT_OK
    assert terminal.prompts.count(APPROVAL_PROMPT) == len(["what?", answer])
    assert TOOL in terminal.output
    assert kit.tools.executed == runs
    assert kit.tools.always_allowed == always
    assert ANSWER in terminal.output


def test_end_of_input_at_an_approval_denies(kit: CoreKit) -> None:
    kit.tools.add(TOOL, decision=Decision.ASK)
    kit.gateway.outcomes = [tool_response(CALL), text_response(ANSWER)]
    terminal = FakeTerminal(inputs=[QUESTION])

    code = chat(kit, terminal, "--new")

    assert code == EXIT_OK
    assert kit.tools.executed == []


def test_unknown_slash_command_never_reaches_the_model(kit: CoreKit) -> None:
    terminal = FakeTerminal(inputs=[UNKNOWN_SLASH])

    code = chat(kit, terminal, "--new")

    assert code == EXIT_OK
    assert UNKNOWN_SLASH in terminal.output
    assert ANSI_RED in terminal.styles
    assert kit.gateway.requests == []


@pytest.mark.acceptance("07-12.8")
def test_context_slash_command_prints_the_report(kit: CoreKit) -> None:
    terminal = FakeTerminal(inputs=[SlashCommand.CONTEXT.value])

    code = chat(kit, terminal, "--new")

    assert code == EXIT_OK
    assert str(kit.gateway.profile.context_window) in terminal.output
    assert kit.gateway.requests == []


@pytest.mark.acceptance("07-12.9")
def test_compact_slash_command_summarises_the_conversation(kit: CoreKit) -> None:
    kit.gateway.outcomes = [
        text_response(ANSWER),
        text_response(SUMMARY),
        text_response(SECOND_ANSWER),
    ]
    terminal = FakeTerminal(inputs=[QUESTION, SlashCommand.COMPACT.value, QUESTION])

    code = chat(kit, terminal, "--new")

    first, summary_call, after = kit.gateway.requests
    assert code == EXIT_OK
    assert summary_call.metadata.purpose is RequestPurpose.COMPACTION
    assert QUESTION in summary_call.model_dump_json()
    assert SlashCommand.COMPACT.value not in first.model_dump_json() + after.model_dump_json()
    assert SUMMARY in after.messages[0].model_dump_json()
    assert MSG_COMPACTED.split("(")[0] in terminal.output


def test_model_error_is_shown_in_red_and_the_prompt_returns(kit: CoreKit) -> None:
    kit.gateway.outcomes = [gateway_error(ErrorKind.AUTH), text_response(ANSWER)]
    terminal = FakeTerminal(inputs=[QUESTION, "/retry"])

    code = chat(kit, terminal, "--new")

    assert code == EXIT_OK
    assert ANSI_RED in terminal.styles
    assert ANSWER in terminal.output


def test_chat_with_an_unknown_agent_fails(kit: CoreKit) -> None:
    terminal = FakeTerminal()

    code = chat(kit, terminal, OTHER_NAME)

    assert code == EXIT_ERROR
    assert OTHER_NAME in terminal.output


def test_agents_lists_names_and_status(kit: CoreKit) -> None:
    chat(kit, FakeTerminal(), AGENT_NAME, "--new")
    terminal = FakeTerminal()

    code = run([CMD_AGENTS], kit.launch, terminal)

    assert code == EXIT_OK
    assert AGENT_NAME in terminal.output


@pytest.mark.parametrize("command", [CMD_CHAT, CMD_AGENTS])
@pytest.mark.acceptance("02-10.9")
def test_cli_exits_at_once_when_the_core_is_taken(kit: CoreKit, command: str) -> None:
    terminal = FakeTerminal(inputs=[QUESTION])

    with CoreLock(kit.home.lock_file):
        code = run([command], kit.launch, terminal)

    assert code == EXIT_CORE_LOCKED
    assert terminal.inputs == [QUESTION]
    assert not kit.home.db_file.exists()


def test_no_command_prints_help(kit: CoreKit) -> None:
    terminal = FakeTerminal()

    code = run([], kit.launch, terminal)

    assert code == EXIT_OK
    assert CMD_CHAT in terminal.output
