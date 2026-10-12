"""``harness chat`` with the real built-in tools: approval, refusal, history (08 §10)."""

from contextlib import AbstractAsyncContextManager
from pathlib import Path

import pytest

from harness.app import CoreOptions, open_core
from harness.cli import run
from harness.cli.consts import ANSI_RED, ANSWER_DENY, ANSWER_ONCE, CMD_CHAT, EXIT_OK
from harness.config import FileConfigManager
from harness.core.handle import CoreHandle
from harness.core.types import ToolCallBlock, ToolResultBlock
from harness.core.utils.paths import HarnessHome
from tests.consts import USER_ID
from tests.fakes.adapter import text_response, tool_response
from tests.fakes.embedder import FakeEmbedder
from tests.fakes.gateway import FakeGateway
from tests.fakes.secrets import FakeSecretStore
from tests.fakes.terminal import FakeTerminal

QUESTION = "write the note"
ANSWER = "finished"
FILE = "note.txt"
CONTENT = "a line for the note\n"
REASON = "the user asked for a note"
WRITE = ToolCallBlock(
    id="call_1", name="write_file", input={"path": FILE, "content": CONTENT, "reason": REASON}
)
READ_MISSING = ToolCallBlock(id="call_2", name="read_file", input={"path": FILE})


def chat(home: HarnessHome, gateway: FakeGateway, terminal: FakeTerminal) -> int:
    """Run ``harness chat --new`` over a core whose tools manager is the real one."""

    def launch() -> AbstractAsyncContextManager[CoreHandle]:
        config = FileConfigManager(home, secrets=FakeSecretStore(), env={}, username=USER_ID)
        return open_core(
            CoreOptions(home=home, config=config, gateway=gateway, embedder=FakeEmbedder())
        )

    return run([CMD_CHAT, "--new"], launch, terminal)


def written(home: HarnessHome) -> list[Path]:
    return list(home.root.glob(f"workspaces/*/{FILE}"))


@pytest.mark.acceptance("08-10.21")
@pytest.mark.acceptance("08-10.22")
def test_approved_write_shows_reason_and_diff_then_writes(tmp_harness_home: HarnessHome) -> None:
    gateway = FakeGateway([tool_response(WRITE), text_response(ANSWER)])
    terminal = FakeTerminal(inputs=[QUESTION, ANSWER_ONCE])

    code = chat(tmp_harness_home, gateway, terminal)

    files = written(tmp_harness_home)
    assert code == EXIT_OK
    assert REASON in terminal.output
    assert f"+{CONTENT.strip()}" in terminal.output
    assert [path.read_bytes() for path in files] == [CONTENT.encode("utf-8")]
    assert ANSWER in terminal.output


@pytest.mark.acceptance("08-10.13")
@pytest.mark.acceptance("08-10.27")
def test_refused_and_failed_tools_are_red_and_history_keeps_the_input(
    tmp_harness_home: HarnessHome,
) -> None:
    gateway = FakeGateway(
        [tool_response(WRITE), tool_response(READ_MISSING), text_response(ANSWER)]
    )
    terminal = FakeTerminal(inputs=[QUESTION, ANSWER_DENY])

    code = chat(tmp_harness_home, gateway, terminal)

    blocks = [block for message in gateway.requests[-1].messages for block in message.content]
    calls = [block for block in blocks if isinstance(block, ToolCallBlock)]
    results = [block for block in blocks if isinstance(block, ToolResultBlock)]
    assert code == EXIT_OK
    assert written(tmp_harness_home) == []
    assert terminal.styles.count(ANSI_RED) >= len(results)
    assert [call.input for call in calls] == [WRITE.input, READ_MISSING.input]
    assert [result.is_error for result in results] == [True, True]
