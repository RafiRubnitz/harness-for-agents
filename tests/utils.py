"""Shared test helpers (builders, not assertions about production constants)."""

from datetime import datetime
from pathlib import Path

from harness.core.consts import AgentStatus, MessageRole
from harness.core.records import AgentRecord, AgentSettings, MessageRecord
from harness.core.types import (
    AgentId,
    ContentBlock,
    Message,
    MessageId,
    ModelRequest,
    RequestMeta,
    RunId,
    TextBlock,
    ToolSpec,
)
from tests.consts import AGENT_ID, AGENT_NAME, RUN_ID, TEST_MODEL_ALIAS
from tests.fakes.clock import FAKE_CLOCK_START

DATA_DIR = Path(__file__).parent / "data"
REQUEST_ID = "req_0001"


def user_message(text: str) -> Message:
    """A user message with one text block."""
    return Message(role="user", content=[TextBlock(text=text)])


def make_request(
    text: str = "hello",
    *,
    model: str = TEST_MODEL_ALIAS,
    tools: list[ToolSpec] | None = None,
    request_id: str = REQUEST_ID,
) -> ModelRequest:
    """A minimal model request."""
    return ModelRequest(
        model=model,
        messages=[user_message(text)],
        tools=tools,
        metadata=RequestMeta(request_id=request_id, agent_id=AGENT_ID, run_id=RUN_ID, step=1),
    )


def tool_spec(name: str) -> ToolSpec:
    """A tool spec with an empty object schema."""
    return ToolSpec(name=name, description=name, input_schema={"type": "object"})


def read_data(relative: str) -> str:
    """Read a file under ``tests/data``."""
    return (DATA_DIR / relative).read_text(encoding="utf-8")


def make_agent(
    agent_id: AgentId = AGENT_ID,
    *,
    status: AgentStatus = AgentStatus.IDLE,
    at: datetime = FAKE_CLOCK_START,
) -> AgentRecord:
    """A minimal agent record."""
    return AgentRecord(
        id=agent_id,
        name=AGENT_NAME if agent_id == AGENT_ID else f"{AGENT_NAME}-{agent_id}",
        model=TEST_MODEL_ALIAS,
        system_prompt="",
        tools=[],
        status=status,
        settings=AgentSettings(),
        created_at=at,
        updated_at=at,
    )


def make_message(
    message_id: str,
    text: str = "hello",
    *,
    agent_id: AgentId = AGENT_ID,
    role: MessageRole = MessageRole.ASSISTANT,
    run_id: RunId | None = None,
    content: list[ContentBlock] | None = None,
    at: datetime = FAKE_CLOCK_START,
) -> MessageRecord:
    """A history message without a seq (the store allocates it)."""
    return MessageRecord(
        id=MessageId(message_id),
        agent_id=agent_id,
        seq=0,
        run_id=run_id,
        role=role,
        content=content if content is not None else [TextBlock(text=text)],
        created_at=at,
    )
