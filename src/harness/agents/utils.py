"""Pure builders for agent records, states and tool outputs."""

from datetime import datetime

from pydantic import JsonValue, TypeAdapter, ValidationError

from harness.agents.consts import (
    FIELD_NAME,
    FIELD_SETTINGS,
    FIELD_TEMPLATE,
    MSG_INVALID_CHANGES,
    MSG_UNKNOWN_FIELDS,
    RESULT_CONTENT_KEY,
)
from harness.agents.runtime import AgentRuntime
from harness.config import AgentTemplate
from harness.core.consts import (
    DEFAULT_USER_ID,
    META_ERROR_CODE,
    AddressType,
    AgentStatus,
    ErrorCode,
    ToolCallStatus,
)
from harness.core.errors import AgentError
from harness.core.records import (
    AgentRecord,
    AgentSettings,
    AgentSpec,
    AgentState,
    ToolCallRecord,
    ToolOutput,
)
from harness.core.types import Address, AgentId, ContentBlock, MessageOrigin, TextBlock
from harness.core.utils.paths import HarnessHome

_BLOCKS = TypeAdapter(list[ContentBlock])
_FINISHED = frozenset(
    {
        ToolCallStatus.DONE,
        ToolCallStatus.FAILED,
        ToolCallStatus.DENIED,
        ToolCallStatus.INVALID,
        ToolCallStatus.CANCELLED,
        ToolCallStatus.INTERRUPTED,
    }
)


def merged_settings(defaults: AgentSettings, given: AgentSettings) -> AgentSettings:
    """Return ``defaults`` overridden by the fields that were explicitly set in ``given``."""
    return AgentSettings.model_validate(
        {**defaults.model_dump(), **given.model_dump(exclude_unset=True)}
    )


def new_record(
    agent_id: AgentId, spec: AgentSpec, defaults: AgentSettings, home: HarnessHome, now: datetime
) -> AgentRecord:
    """Build the record of a new agent; a missing working dir becomes the agent's workspace."""
    settings = merged_settings(defaults, spec.settings)
    if settings.working_dir is None:
        settings = settings.model_copy(update={"working_dir": str(home.workspace_for(agent_id))})
    return AgentRecord(
        id=agent_id,
        name=spec.name,
        model=spec.model,
        system_prompt=spec.system_prompt,
        tools=list(spec.tools),
        template=spec.template,
        status=AgentStatus.IDLE,
        settings=settings,
        created_at=now,
        updated_at=now,
    )


def _invalid(error: ValidationError) -> AgentError:
    detail = error.errors()[0]["msg"]
    return AgentError(MSG_INVALID_CHANGES.format(detail=detail), code=ErrorCode.CONFIG_INVALID)


def spec_from_template(
    template: AgentTemplate, overrides: dict[str, JsonValue] | None
) -> AgentSpec:
    """Build the spec of an agent created from a template.

    Raises:
        AgentError: The overrides do not form a valid spec.
    """
    given = dict(overrides or {})
    extra = given.pop(FIELD_SETTINGS, None)
    settings = {**template.settings, **(extra if isinstance(extra, dict) else {})}
    data: dict[str, JsonValue] = {
        FIELD_NAME: template.name,
        "model": template.model,
        "system_prompt": template.system_prompt,
        "tools": list(template.tools),
        FIELD_TEMPLATE: template.name,
        **given,
        FIELD_SETTINGS: settings,
    }
    try:
        return AgentSpec.model_validate(data)
    except ValidationError as error:
        raise _invalid(error) from error


def changed_record(
    record: AgentRecord, changes: dict[str, JsonValue], allowed: frozenset[str], now: datetime
) -> AgentRecord:
    """Return ``record`` with ``changes`` applied (settings are merged key by key).

    Raises:
        AgentError: A field may not be changed, or the result is not a valid record.
    """
    unknown = sorted(set(changes) - allowed)
    if unknown:
        raise AgentError(
            MSG_UNKNOWN_FIELDS.format(fields=", ".join(unknown)), code=ErrorCode.CONFIG_INVALID
        )
    data = record.model_dump()
    extra = changes.get(FIELD_SETTINGS)
    merged: dict[str, object] = {**data, **changes, "updated_at": now}
    if FIELD_SETTINGS in changes:
        merged[FIELD_SETTINGS] = {
            **data[FIELD_SETTINGS],
            **(extra if isinstance(extra, dict) else {}),
        }
    try:
        return AgentRecord.model_validate(merged)
    except ValidationError as error:
        raise _invalid(error) from error


def agent_state(rt: AgentRuntime, pending_approvals: int, background_tasks: int) -> AgentState:
    """The agent as the GUI / CLI sees it."""
    record = rt.record
    return AgentState(
        id=record.id,
        name=record.name,
        model=record.model,
        template=record.template,
        status=record.status,
        settings=record.settings,
        tools=list(record.tools),
        run_id=rt.run.id if rt.run else None,
        step=rt.run.step_count if rt.run else None,
        pending_approvals=pending_approvals,
        background_tasks=background_tasks,
        cost_usd_total=rt.cost_total,
        context_tokens=rt.context_tokens,
        recovered=rt.recovered,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def user_origin(user_id: str) -> MessageOrigin:
    """The origin of a message typed by the local user."""
    return MessageOrigin(source=Address(type=AddressType.USER, id=user_id or DEFAULT_USER_ID))


def error_output(text: str, code: ErrorCode) -> ToolOutput:
    """A tool result that reports a failure to the model."""
    return ToolOutput(
        content=[TextBlock(text=text)], is_error=True, metadata={META_ERROR_CODE: code.value}
    )


def result_content(output: ToolOutput) -> dict[str, JsonValue]:
    """The JSON form of a tool output's content, as stored in ``tool_calls``."""
    return {RESULT_CONTENT_KEY: [block.model_dump(mode="json") for block in output.content]}


def stored_output(record: ToolCallRecord) -> ToolOutput | None:
    """Rebuild the output of a tool call that finished and was saved; ``None`` otherwise."""
    if record.status not in _FINISHED or record.result_content is None:
        return None
    blocks = _BLOCKS.validate_python(record.result_content.get(RESULT_CONTENT_KEY, []))
    return ToolOutput(
        content=blocks, is_error=bool(record.is_error), metadata=dict(record.metadata)
    )
