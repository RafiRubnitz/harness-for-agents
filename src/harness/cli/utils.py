"""Text rendering for the CLI (pure functions)."""

import json

from pydantic import JsonValue

from harness.cli import consts
from harness.core.events import ToolCall, ToolResult
from harness.core.records import AgentState, ContextReport, PermissionRequestPayload
from harness.core.types import TextBlock


def compact_json(value: JsonValue) -> str:
    """Render ``value`` on one line, keeping non-ASCII text readable and bounded."""
    text = json.dumps(value, ensure_ascii=False)
    return text[: consts.INPUT_PREVIEW_CHARS]


def find_agent(agents: list[AgentState], wanted: str) -> AgentState | None:
    """Return the agent whose id or name is ``wanted``."""
    for agent in agents:
        if wanted in (agent.id, agent.name):
            return agent
    return None


def format_agents(agents: list[AgentState]) -> list[str]:
    """One line per agent for ``harness agents``."""
    if not agents:
        return [consts.MSG_NO_AGENTS]
    return [
        consts.MSG_AGENT_ROW.format(
            agent_id=agent.id, status=agent.status.value, model=agent.model, name=agent.name
        )
        for agent in agents
    ]


def format_tool_call(call: ToolCall) -> list[str]:
    """The lines announcing a tool call."""
    lines = [consts.MSG_TOOL_CALL.format(tool=call.tool, input=compact_json(call.input))]
    if call.reason:
        lines.append(consts.MSG_TOOL_REASON.format(reason=call.reason))
    return lines


def format_tool_result(result: ToolResult) -> list[str]:
    """The lines closing a tool call; a failure also shows what went wrong (08 §6b)."""
    status = consts.MSG_TOOL_FAILED if result.is_error else consts.MSG_TOOL_OK
    lines = [
        consts.MSG_TOOL_RESULT.format(
            status=status, tool=result.tool, duration_ms=result.duration_ms
        )
    ]
    if result.is_error:
        lines += [block.text for block in result.content if isinstance(block, TextBlock)]
    return lines


def format_request(request: PermissionRequestPayload) -> list[str]:
    """What the user must see before answering an approval: reason first (02 §3)."""
    preview = request.preview
    fields: list[tuple[str, str | None]] = [
        (consts.LABEL_REASON, request.reason),
        (consts.LABEL_SUMMARY, request.summary),
        (consts.LABEL_TOOL, request.tool),
        (consts.LABEL_INPUT, compact_json(request.input) if request.input else None),
        (consts.LABEL_HOOKS, "; ".join(request.hook_reasons) or None),
    ]
    if preview is not None:
        fields += [
            (consts.LABEL_COMMAND, preview.command),
            (consts.LABEL_PATH, preview.path),
            (consts.LABEL_PERMANENT, "yes" if preview.permanent else None),
            (consts.LABEL_DIFF, f"\n{preview.diff}" if preview.diff else None),
        ]
    lines = [consts.MSG_APPROVAL_HEADER]
    lines += [
        consts.MSG_APPROVAL_FIELD.format(label=label, value=value)
        for label, value in fields
        if value
    ]
    return lines


def format_report(report: ContextReport, full_text: str | None) -> list[str]:
    """The ``/context`` report."""
    lines = [
        consts.MSG_CONTEXT_HEADER.format(
            total=report.total_tokens, window=report.window, model=report.model
        )
    ]
    lines += [
        consts.MSG_CONTEXT_SECTION.format(
            name=str(section.name), tokens=section.tokens, items=section.items
        )
        for section in report.sections
    ]
    if full_text:
        lines.append(full_text)
    return lines
