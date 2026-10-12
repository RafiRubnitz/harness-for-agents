"""Tests for the core catalog: enums, commands, events, errors, logging redaction."""

import pytest
from loguru import logger
from pydantic import TypeAdapter, ValidationError

from harness.core.commands import AgentSend, Command, PermissionRespond
from harness.core.consts import (
    ERROR_CODE_BY_KIND,
    REDACTED,
    RETRYABLE_KINDS,
    CommandName,
    DenySource,
    ErrorKind,
    EventName,
    HookEvent,
    PermissionAnswer,
    ProviderId,
)
from harness.core.errors import GatewayError, HarnessError, PermissionDenied
from harness.core.events import EVENT_BY_PAYLOAD, PAYLOAD_BY_EVENT
from harness.core.hook_payloads import MUTABLE_PAYLOAD_FIELDS, PAYLOAD_BY_HOOK_EVENT
from harness.core.logging import redact, register_secret
from harness.core.types import TextBlock, ToolCallBlock, ToolResultBlock
from tests.consts import AGENT_ID, TEST_API_KEY, TEST_MODEL_ID

RESERVED_HOOK_EVENTS = {
    HookEvent.BEFORE_REFLECTION,
    HookEvent.AFTER_REFLECTION,
    HookEvent.TRIGGER_FIRED,
    HookEvent.PERMISSION_DECIDED,
}
COMMAND_ADAPTER: TypeAdapter[Command] = TypeAdapter(Command)


def test_every_event_name_has_a_payload_model() -> None:
    assert set(PAYLOAD_BY_EVENT) == set(EventName)


def test_event_payload_mapping_is_one_to_one() -> None:
    assert len(EVENT_BY_PAYLOAD) == len(PAYLOAD_BY_EVENT)


def test_every_v1_hook_event_has_a_payload_model() -> None:
    assert set(PAYLOAD_BY_HOOK_EVENT) == set(HookEvent) - RESERVED_HOOK_EVENTS


def test_mutable_fields_exist_on_their_payload_models() -> None:
    for event, fields in MUTABLE_PAYLOAD_FIELDS.items():
        assert fields <= set(PAYLOAD_BY_HOOK_EVENT[event].model_fields)


def test_every_error_kind_maps_to_a_code() -> None:
    assert set(ERROR_CODE_BY_KIND) == set(ErrorKind)


@pytest.mark.parametrize("command_name", list(CommandName))
def test_every_command_name_is_in_the_union(command_name: CommandName) -> None:
    with pytest.raises(ValidationError) as raised:
        COMMAND_ADAPTER.validate_python({"name": command_name, "__bogus__": 1})

    assert "union_tag_invalid" not in {error["type"] for error in raised.value.errors()}


def test_command_roundtrips_through_json() -> None:
    command = AgentSend(agent_id=AGENT_ID, content=[TextBlock(text="hi")])

    assert COMMAND_ADAPTER.validate_json(command.model_dump_json()) == command


def test_command_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        PermissionRespond.model_validate(
            {"request_id": "perm_1", "decision": PermissionAnswer.APPROVE, "origin": "user:x"}
        )


def test_nested_tool_result_content_roundtrips() -> None:
    call = ToolCallBlock(id="toolu_1", name="read_file", input={"path": "a"})
    block = ToolResultBlock(call_id=call.id, content=[TextBlock(text="x")])

    assert ToolResultBlock.model_validate_json(block.model_dump_json()) == block


@pytest.mark.parametrize("kind", list(ErrorKind))
def test_gateway_error_retryable_follows_kind(kind: ErrorKind) -> None:
    error = GatewayError("boom", kind=kind, provider=ProviderId.ANTHROPIC, model=TEST_MODEL_ID)

    assert error.retryable == (kind in RETRYABLE_KINDS)
    assert error.code == ERROR_CODE_BY_KIND[kind]


def test_permission_denied_carries_source_and_is_harness_error() -> None:
    error = PermissionDenied("no", denied_by=DenySource.POLICY, tool="shell")

    assert isinstance(error, HarnessError)
    assert error.denied_by == DenySource.POLICY


def test_registered_secret_is_redacted() -> None:
    register_secret(TEST_API_KEY)

    assert redact(f"key={TEST_API_KEY}") == f"key={REDACTED}"


def test_api_key_shaped_text_is_redacted_from_logs(log_capture: list[str]) -> None:
    key = "sk-ant-" + "a" * 32

    logger.info("calling with {}", key)

    assert all(key not in line for line in log_capture)
    assert any(REDACTED in line for line in log_capture)
