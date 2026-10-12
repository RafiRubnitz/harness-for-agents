"""Stages 4-5: run a decided tool call safely and pass the result through hooks (08 §6)."""

import asyncio

from loguru import logger
from pydantic import JsonValue, ValidationError

from harness.config import ConfigManager, ContextConfig
from harness.core.consts import (
    META_DENIED_BY,
    META_ERROR_CODE,
    META_HOOK_CONTEXT,
    META_RESULT_ID,
    BlobKind,
    Decision,
    DenySource,
    ErrorCode,
    HookEvent,
    PermissionAnswer,
)
from harness.core.hook_payloads import AfterToolCallPayload
from harness.core.hook_types import HookInput
from harness.core.records import AgentState, PermissionDecision, ToolOutput
from harness.core.types import Clock, TextBlock
from harness.hooks import HooksManager
from harness.memory import BlobStore
from harness.tools.consts import (
    BACKUP_META_AGENT_ID,
    HOOK_REASON_JOINER,
    MSG_DENIED_POLICY,
    MSG_DENIED_USER,
    MSG_DENIED_USER_NOTE,
    MSG_RESULT_DENIED_HOOK,
    MSG_TOOL_RAISED,
    MSG_TOOL_TIMEOUT,
    TEXT_ENCODING,
)
from harness.tools.function_tool import error_output
from harness.tools.types import Tool, ToolContext

_RESULT_KEY = "result"
_CHARS_PER_TOKEN = 4


def may_run(decision: PermissionDecision) -> bool:
    """T1: only ``allow``, or ``ask`` that the user approved, lets a tool run."""
    if decision.decision is Decision.ALLOW:
        return True
    return decision.decision is Decision.ASK and decision.user_answer is PermissionAnswer.APPROVE


def denial(tool: str, decision: PermissionDecision) -> ToolOutput:
    """The error the model gets for a call that was blocked or refused."""
    code = decision.error_code or ErrorCode.PERMISSION_DENIED
    metadata: dict[str, JsonValue] = {META_ERROR_CODE: code.value}
    if decision.decision is Decision.ASK:
        note = decision.deny_message
        text = (
            MSG_DENIED_USER_NOTE.format(tool=tool, note=note)
            if note
            else MSG_DENIED_USER.format(tool=tool)
        )
        metadata[META_DENIED_BY] = DenySource.USER.value
    else:
        text = decision.deny_message or MSG_DENIED_POLICY.format(tool=tool)
        if decision.denied_by is not None:
            metadata[META_DENIED_BY] = decision.denied_by.value
    return ToolOutput(content=[TextBlock(text=text)], is_error=True, metadata=metadata)


class ToolExecutor:
    """Runs one tool with a timeout, turning every failure into a result."""

    def __init__(
        self, config: ConfigManager, hooks: HooksManager, blobs: BlobStore, clock: Clock
    ) -> None:
        self._config = config
        self._hooks = hooks
        self._blobs = blobs
        self._clock = clock

    async def run(self, tool: Tool, args: dict[str, JsonValue], ctx: ToolContext) -> ToolOutput:
        """Stage 4: run the tool; cancellation passes through, everything else is a value."""
        name, seconds = tool.definition.name, tool.definition.timeout_s
        try:
            async with asyncio.timeout(seconds):
                return await tool.run(args, ctx)
        except TimeoutError:
            ctx.cancel.set()
            return error_output(
                MSG_TOOL_TIMEOUT.format(tool=name, seconds=seconds), ErrorCode.TOOL_TIMEOUT
            )
        except asyncio.CancelledError:
            ctx.cancel.set()
            raise
        except Exception as failure:  # a tool bug is reported to the model, never raised
            logger.opt(exception=True).warning("tool {} raised", name)
            return error_output(MSG_TOOL_RAISED.format(tool=name, error=failure))

    async def keep_large(self, agent: AgentState, output: ToolOutput) -> ToolOutput:
        """Store an oversized text result whole and tag the output with its id (08 §6)."""
        text = "\n".join(block.text for block in output.content if isinstance(block, TextBlock))
        limit = self._config.section(ContextConfig, agent.id).tool_results.truncate_tokens
        if len(text) // _CHARS_PER_TOKEN <= limit:
            return output
        result_id = await self._blobs.put(
            text.encode(TEXT_ENCODING),
            kind=BlobKind.TOOL_OUTPUT,
            meta={BACKUP_META_AGENT_ID: agent.id},
        )
        return output.model_copy(
            update={"metadata": {**output.metadata, META_RESULT_ID: result_id}}
        )

    async def after(
        self,
        agent: AgentState,
        tool: Tool,
        args: dict[str, JsonValue],
        ctx: ToolContext,
        output: ToolOutput,
    ) -> ToolOutput:
        """Stage 5: ``after_tool_call`` hooks may replace the result."""
        payload = AfterToolCallPayload(
            tool=tool.definition.name,
            call_id=ctx.call_id,
            input=args,
            reason=ctx.reason,
            result=output,
        )
        merged = await self._hooks.emit(
            HookEvent.AFTER_TOOL_CALL,
            HookInput(
                event=HookEvent.AFTER_TOOL_CALL,
                agent_id=agent.id,
                agent_name=agent.name,
                run_id=agent.run_id,
                step=agent.step,
                working_dir=agent.settings.working_dir,
                payload=payload.model_dump(mode="json"),
                timestamp=self._clock.now(),
            ),
        )
        if merged.decision is Decision.DENY:
            reasons = HOOK_REASON_JOINER.join(merged.reasons)
            message = MSG_RESULT_DENIED_HOOK.format(tool=tool.definition.name, reasons=reasons)
            output = error_output(message, ErrorCode.HOOK_DENIED)
        else:
            output = _replaced(output, merged.payload.get(_RESULT_KEY))
        if merged.additional_context is None:
            return output
        metadata = {**output.metadata, META_HOOK_CONTEXT: merged.additional_context}
        return output.model_copy(update={"metadata": metadata})


def _replaced(output: ToolOutput, updated: JsonValue) -> ToolOutput:
    """The result as an ``after_tool_call`` hook left it."""
    if updated is None:
        return output
    try:
        return ToolOutput.model_validate(updated)
    except ValidationError:
        logger.warning("after_tool_call hook returned an invalid result; keeping the original")
        return output
