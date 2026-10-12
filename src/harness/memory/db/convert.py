"""Conversions between ORM rows and the Pydantic records of ``core.records``."""

import json
from collections.abc import Sequence

from pydantic import JsonValue

from harness.core.consts import ApprovalStatus
from harness.core.records import (
    AgentEventRecord,
    AgentRecord,
    BackgroundTaskRecord,
    InboxItem,
    MessageRecord,
    ModelCallRecord,
    PendingApproval,
    PermissionRequestPayload,
    RunRecord,
    SummaryRecord,
    ToolCallRecord,
)
from harness.core.types import (
    ContentBlock,
    MessageOrigin,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
    Usage,
)
from harness.memory import hebrew
from harness.memory.db.models import (
    AgentEventRow,
    AgentRow,
    BackgroundTaskRow,
    InboxRow,
    MessageRow,
    ModelCallRow,
    PermissionRequestRow,
    RunRow,
    SummaryRow,
    ToolCallRow,
)


def dump_blocks(content: Sequence[ContentBlock]) -> list[JsonValue]:
    """Serialise content blocks for a JSON column."""
    return [block.model_dump(mode="json") for block in content]


def _plain_text(content: Sequence[ContentBlock]) -> list[str]:
    parts: list[str] = []
    for block in content:
        if isinstance(block, TextBlock):
            parts.append(block.text)
        elif isinstance(block, ToolCallBlock):
            parts.append(block.name)
            if block.reason:
                parts.append(block.reason)
            parts.append(json.dumps(block.input, ensure_ascii=False))
        elif isinstance(block, ToolResultBlock):
            parts.extend(_plain_text(block.content))
    return parts


def search_text(content: Sequence[ContentBlock]) -> str:
    """Build the normalised FTS text of a message: text, tool calls, tool results; no thinking."""
    return hebrew.index_text("\n".join(_plain_text(content)))


# --- agents ---


def agent_values(record: AgentRecord) -> dict[str, object]:
    """Column values of an agent, without ``config_overrides`` and ``last_seq``."""
    return {
        "name": record.name,
        "model": record.model,
        "system_prompt": record.system_prompt,
        "tools": list(record.tools),
        "template": record.template,
        "status": record.status.value,
        "settings": record.settings.model_dump(mode="json"),
        "pending_hook_context": list(record.pending_hook_context),
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def agent_record(row: AgentRow) -> AgentRecord:
    """Convert an ``agents`` row."""
    return AgentRecord.model_validate(
        {
            "id": row.id,
            "name": row.name,
            "model": row.model,
            "system_prompt": row.system_prompt,
            "tools": row.tools,
            "template": row.template,
            "status": row.status,
            "settings": row.settings,
            "config_overrides": row.config_overrides,
            "pending_hook_context": row.pending_hook_context,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }
    )


# --- messages and inbox ---


def message_row(record: MessageRecord, seq: int) -> MessageRow:
    """Build a ``messages`` row; origin columns are all set or all NULL."""
    origin = record.origin
    return MessageRow(
        id=record.id,
        agent_id=record.agent_id,
        seq=seq,
        run_id=record.run_id,
        step=record.step,
        role=record.role.value,
        content=dump_blocks(record.content),
        search_text=search_text(record.content),
        origin_source_type=origin.source.type.value if origin else None,
        origin_source_id=origin.source.id if origin else None,
        origin_targets=[t.model_dump(mode="json") for t in origin.targets] if origin else None,
        trigger_id=origin.trigger_id if origin else None,
        conversation_id=origin.conversation_id if origin else None,
        hop=origin.hop if origin else None,
        reply_to=origin.reply_to if origin else None,
        usage=record.usage.model_dump() if record.usage else None,
        cost_usd=record.cost_usd,
        stop_reason=record.stop_reason,
        created_at=record.created_at,
    )


def _origin(row: MessageRow) -> MessageOrigin | None:
    if row.origin_source_type is None:
        return None
    return MessageOrigin.model_validate(
        {
            "source": {"type": row.origin_source_type, "id": row.origin_source_id},
            "targets": row.origin_targets or [],
            "trigger_id": row.trigger_id,
            "conversation_id": row.conversation_id,
            "hop": row.hop or 0,
            "reply_to": row.reply_to,
        }
    )


def message_record(row: MessageRow) -> MessageRecord:
    """Convert a ``messages`` row."""
    return MessageRecord.model_validate(
        {
            "id": row.id,
            "agent_id": row.agent_id,
            "seq": row.seq,
            "run_id": row.run_id,
            "role": row.role,
            "content": row.content,
            "step": row.step,
            "usage": row.usage,
            "cost_usd": row.cost_usd,
            "stop_reason": row.stop_reason,
            "origin": _origin(row),
            "created_at": row.created_at,
        }
    )


def inbox_row(item: InboxItem) -> InboxRow:
    """Build an ``inbox`` row."""
    return InboxRow(
        id=item.id,
        agent_id=item.agent_id,
        content=dump_blocks(item.content),
        origin=item.origin.model_dump(mode="json"),
        received_at=item.received_at,
    )


def inbox_item(row: InboxRow) -> InboxItem:
    """Convert an ``inbox`` row."""
    return InboxItem.model_validate(
        {
            "id": row.id,
            "agent_id": row.agent_id,
            "content": row.content,
            "origin": row.origin,
            "received_at": row.received_at,
        }
    )


# --- runs ---


def run_row(record: RunRecord) -> RunRow:
    """Build a ``runs`` row."""
    return RunRow(
        id=record.id,
        agent_id=record.agent_id,
        first_seq=record.first_seq,
        step_count=record.step_count,
        stop_reason=record.stop_reason,
        retrieved_memory_ids=list(record.retrieved_memory_ids),
        started_at=record.started_at,
        ended_at=record.ended_at,
    )


def run_record(row: RunRow) -> RunRecord:
    """Convert a ``runs`` row."""
    return RunRecord.model_validate(
        {
            "id": row.id,
            "agent_id": row.agent_id,
            "first_seq": row.first_seq,
            "started_at": row.started_at,
            "ended_at": row.ended_at,
            "step_count": row.step_count,
            "stop_reason": row.stop_reason,
            "retrieved_memory_ids": row.retrieved_memory_ids,
        }
    )


# --- approvals ---


def approval_row(pending: PendingApproval, tool_call_id: str | None) -> PermissionRequestRow:
    """Build an open ``permission_requests`` row from the request payload."""
    request = pending.request
    return PermissionRequestRow(
        id=pending.request_id,
        agent_id=pending.agent_id,
        run_id=pending.run_id,
        subject=request.subject.value,
        tool_call_id=tool_call_id,
        call_id=request.call_id,
        tool=request.tool,
        reason=request.reason,
        input=dict(request.input),
        risk=request.risk.value if request.risk else None,
        preview=request.preview.model_dump(mode="json") if request.preview else None,
        asked_by=[source.value for source in request.asked_by],
        hook_reasons=list(request.hook_reasons),
        summary=request.summary,
        status=ApprovalStatus.PENDING.value,
        decision=None,
        scope=None,
        deny_message=None,
        created_at=pending.created_at,
        answered_at=None,
    )


def pending_approval(row: PermissionRequestRow) -> PendingApproval:
    """Convert a ``permission_requests`` row back to the pending approval."""
    payload = PermissionRequestPayload.model_validate(
        {
            "request_id": row.id,
            "subject": row.subject,
            "call_id": row.call_id,
            "tool": row.tool,
            "reason": row.reason,
            "input": row.input,
            "risk": row.risk,
            "preview": row.preview,
            "asked_by": row.asked_by,
            "hook_reasons": row.hook_reasons,
            "summary": row.summary,
        }
    )
    return PendingApproval.model_validate(
        {
            "request_id": row.id,
            "agent_id": row.agent_id,
            "run_id": row.run_id,
            "request": payload,
            "created_at": row.created_at,
        }
    )


# --- background tasks, events, summaries ---


def task_row(record: BackgroundTaskRecord, tool_call_id: str) -> BackgroundTaskRow:
    """Build a ``background_tasks`` row."""
    return BackgroundTaskRow(
        id=record.id,
        agent_id=record.agent_id,
        tool_call_id=tool_call_id,
        call_id=record.call_id,
        tool=record.tool,
        status=record.status.value,
        output_blob=record.output_blob,
        interrupted=record.interrupted,
        result_summary=record.result_summary,
        error=record.error,
        notified_at=record.notified_at,
        started_at=record.started_at,
        ended_at=record.ended_at,
    )


def task_record(row: BackgroundTaskRow) -> BackgroundTaskRecord:
    """Convert a ``background_tasks`` row."""
    return BackgroundTaskRecord.model_validate(
        {
            "id": row.id,
            "agent_id": row.agent_id,
            "call_id": row.call_id,
            "tool": row.tool,
            "status": row.status,
            "started_at": row.started_at,
            "ended_at": row.ended_at,
            "output_blob": row.output_blob,
            "interrupted": row.interrupted,
            "result_summary": row.result_summary,
            "error": row.error,
            "notified_at": row.notified_at,
        }
    )


def event_row(record: AgentEventRecord) -> AgentEventRow:
    """Build an ``agent_events`` row."""
    return AgentEventRow(
        agent_id=record.agent_id,
        run_id=record.run_id,
        after_seq=record.after_seq,
        type=record.type.value,
        payload=dict(record.payload),
        created_at=record.created_at,
    )


def event_record(row: AgentEventRow) -> AgentEventRecord:
    """Convert an ``agent_events`` row."""
    return AgentEventRecord.model_validate(
        {
            "agent_id": row.agent_id,
            "run_id": row.run_id,
            "after_seq": row.after_seq,
            "type": row.type,
            "payload": row.payload,
            "created_at": row.created_at,
        }
    )


def summary_row(record: SummaryRecord) -> SummaryRow:
    """Build a ``summaries`` row."""
    return SummaryRow(
        id=record.id,
        agent_id=record.agent_id,
        from_seq=record.from_seq,
        to_seq=record.to_seq,
        text=record.text,
        level=record.level,
        superseded_by=record.superseded_by,
        reason=record.reason.value,
        model_used=record.model_used,
        token_count=record.token_count,
        created_at=record.created_at,
    )


def summary_record(row: SummaryRow) -> SummaryRecord:
    """Convert a ``summaries`` row."""
    return SummaryRecord.model_validate(
        {
            "id": row.id,
            "agent_id": row.agent_id,
            "from_seq": row.from_seq,
            "to_seq": row.to_seq,
            "text": row.text,
            "level": row.level,
            "superseded_by": row.superseded_by,
            "reason": row.reason,
            "model_used": row.model_used,
            "token_count": row.token_count,
            "created_at": row.created_at,
        }
    )


# --- tool calls and the cost ledger ---


def tool_call_row(record: ToolCallRecord) -> ToolCallRow:
    """Build a ``tool_calls`` row."""
    return ToolCallRow(
        id=record.id,
        agent_id=record.agent_id,
        run_id=record.run_id,
        step=record.step,
        call_id=record.call_id,
        request_message_id=record.request_message_id,
        result_message_id=record.result_message_id,
        tool_name=record.tool_name,
        input=dict(record.input),
        reason=record.reason,
        run_in_background=record.run_in_background,
        status=record.status.value,
        decision=record.decision.value if record.decision else None,
        decided_by=record.decided_by,
        asked_by=[source.value for source in record.asked_by],
        result_content=record.result_content,
        is_error=record.is_error,
        metadata_=dict(record.metadata),
        created_at=record.created_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
        duration_ms=record.duration_ms,
    )


def tool_call_record(row: ToolCallRow) -> ToolCallRecord:
    """Convert a ``tool_calls`` row."""
    return ToolCallRecord.model_validate(
        {
            "id": row.id,
            "agent_id": row.agent_id,
            "run_id": row.run_id,
            "step": row.step,
            "call_id": row.call_id,
            "request_message_id": row.request_message_id,
            "result_message_id": row.result_message_id,
            "tool_name": row.tool_name,
            "input": row.input,
            "reason": row.reason,
            "run_in_background": row.run_in_background,
            "status": row.status,
            "decision": row.decision,
            "decided_by": row.decided_by,
            "asked_by": row.asked_by,
            "result_content": row.result_content,
            "is_error": row.is_error,
            "metadata": row.metadata_,
            "created_at": row.created_at,
            "started_at": row.started_at,
            "finished_at": row.finished_at,
            "duration_ms": row.duration_ms,
        }
    )


def model_call_row(record: ModelCallRecord) -> ModelCallRow:
    """Build a ``model_calls`` row."""
    usage: Usage = record.usage
    return ModelCallRow(
        id=record.id,
        agent_id=record.agent_id,
        run_id=record.run_id,
        step=record.step,
        message_id=record.message_id,
        purpose=record.purpose.value,
        model_alias=record.model_alias,
        model_used=record.model_used,
        provider=record.provider.value,
        input_tokens=usage.input,
        output_tokens=usage.output,
        cache_read_tokens=usage.cache_read,
        cache_write_tokens=usage.cache_write,
        cost_usd=record.cost_usd,
        latency_ms=record.latency_ms,
        stop_reason=record.stop_reason.value if record.stop_reason else None,
        created_at=record.created_at,
    )
