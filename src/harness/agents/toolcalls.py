"""Run the tool calls of one step: decide, ask, execute, save the tool message (04 §5.2)."""

import asyncio

from harness.agents.approvals import ApprovalDesk
from harness.agents.consts import (
    DECIDED_BY_POLICY,
    DECIDED_BY_USER,
    MSG_TOOL_CANCELLED,
    MSG_TOOL_CRASHED,
    MSG_TOOL_INTERRUPTED,
)
from harness.agents.runtime import AgentDeps, AgentRuntime, OpenCalls
from harness.agents.utils import error_output, result_content, stored_output
from harness.core.consts import (
    ID_PREFIX_MESSAGE,
    ID_PREFIX_TOOL_CALL_RECORD,
    META_DENIED_BY,
    META_ERROR_CODE,
    META_HOOK_CONTEXT,
    META_RESULT_ID,
    Decision,
    ErrorCode,
    HookEvent,
    MessageRole,
    PermissionScope,
    ToolCallStatus,
)
from harness.core.events import MessageAppended, ToolResult
from harness.core.records import (
    MessageRecord,
    PermissionDecision,
    PermissionRequestPayload,
    RunRecord,
    ToolCallRecord,
    ToolOutput,
)
from harness.core.types import MessageId, ToolCallBlock, ToolCallRecordId, ToolResultBlock
from harness.core.utils.time import elapsed_ms

_INVALID_CODES = frozenset(
    {ErrorCode.TOOL_INVALID_INPUT.value, ErrorCode.TOOL_REASON_MISSING.value}
)
_STATUS_BY_CODE = {
    ErrorCode.TOOL_CANCELLED.value: ToolCallStatus.CANCELLED,
    ErrorCode.TOOL_INTERRUPTED.value: ToolCallStatus.INTERRUPTED,
}
_UNSTARTED = frozenset({ToolCallStatus.PENDING, ToolCallStatus.AWAITING_APPROVAL})

type Records = dict[str, ToolCallRecord]


def _final_status(output: ToolOutput) -> ToolCallStatus:
    code = output.metadata.get(META_ERROR_CODE)
    if isinstance(code, str) and code in _STATUS_BY_CODE:
        return _STATUS_BY_CODE[code]
    if META_DENIED_BY in output.metadata:
        return ToolCallStatus.DENIED
    if code in _INVALID_CODES:
        return ToolCallStatus.INVALID
    return ToolCallStatus.FAILED if output.is_error else ToolCallStatus.DONE


def _decided_by(decision: PermissionDecision) -> str:
    if decision.user_answer is not None:
        return DECIDED_BY_USER
    return decision.denied_by.value if decision.denied_by else DECIDED_BY_POLICY


def _result_id(output: ToolOutput) -> str | None:
    stored = output.metadata.get(META_RESULT_ID)
    return stored if isinstance(stored, str) else None


class ToolRunner:
    """Executes tool calls for the run loop and keeps the ``tool_calls`` audit log."""

    def __init__(self, deps: AgentDeps, desk: ApprovalDesk) -> None:
        self._deps = deps
        self._desk = desk

    # --- audit records ---

    async def _records(
        self, rt: AgentRuntime, run: RunRecord, message_id: MessageId, calls: list[ToolCallBlock]
    ) -> Records:
        stored = await self._deps.memory.get_tool_calls(rt.id, run_id=run.id)
        records = {record.call_id: record for record in stored}
        for call in calls:
            if call.id in records:
                continue
            fields = self._deps.tools.split_injected(call.name, call.input)
            records[call.id] = ToolCallRecord(
                id=ToolCallRecordId(self._deps.ids.new(ID_PREFIX_TOOL_CALL_RECORD)),
                agent_id=rt.id,
                run_id=run.id,
                step=run.step_count,
                call_id=call.id,
                request_message_id=message_id,
                tool_name=call.name,
                input=fields.input,
                reason=fields.reason,
                run_in_background=fields.run_in_background,
                created_at=self._deps.clock.now(),
            )
            await self._deps.memory.save_tool_call(records[call.id])
        return records

    async def _update(self, records: Records, call_id: str, **changes: object) -> ToolCallRecord:
        records[call_id] = records[call_id].model_copy(update=changes)
        await self._deps.memory.save_tool_call(records[call_id])
        return records[call_id]

    async def _finish(
        self, rt: AgentRuntime, records: Records, call: ToolCallBlock, output: ToolOutput
    ) -> None:
        open_calls = rt.open_calls
        if open_calls is not None:
            open_calls.resolved[call.id] = output
        now = self._deps.clock.now()
        started = records[call.id].started_at or now
        record = await self._update(
            records,
            call.id,
            status=_final_status(output),
            result_content=result_content(output),
            is_error=output.is_error,
            metadata=dict(output.metadata),
            finished_at=now,
            duration_ms=elapsed_ms(started, now),
        )
        self._deps.bus.publish(
            rt.id,
            ToolResult(
                call_id=call.id,
                tool=call.name,
                reason=record.reason,
                content=output.content,
                is_error=output.is_error,
                duration_ms=record.duration_ms or 0,
                metadata=output.metadata,
            ),
        )

    # --- one step ---

    async def run(self, rt: AgentRuntime, run: RunRecord, pending: OpenCalls) -> MessageRecord:
        """Decide, ask, execute and store the results of one assistant message's tool calls.

        Calls already present in ``pending.resolved`` are not executed again.
        """
        rt.open_calls = pending
        records = await self._records(rt, run, pending.message_id, pending.calls)
        todo = [call for call in pending.calls if call.id not in pending.resolved]
        decided = await asyncio.gather(*(self._deps.tools.decide(rt.id, call) for call in todo))
        decisions = {call.id: decision for call, decision in zip(todo, decided, strict=True)}
        for decision in decided:
            await self._keep_context(rt, decision.hook_context)
        for call in todo:
            if decisions[call.id].decision is Decision.ASK:
                decisions[call.id] = await self._ask(rt, run, call, decisions[call.id], records)
        for batch in self._batches(rt, todo):
            await asyncio.gather(
                *(self._execute(rt, call, decisions[call.id], records) for call in batch)
            )
        return await self._save_results(rt, run, records)

    def _batches(self, rt: AgentRuntime, calls: list[ToolCallBlock]) -> list[list[ToolCallBlock]]:
        exclusive = {tool.name for tool in self._deps.tools.list(rt.id) if tool.exclusive}
        batches: list[list[ToolCallBlock]] = []
        parallel: list[ToolCallBlock] = []
        for call in calls:
            if call.name in exclusive:
                if parallel:
                    batches.append(parallel)
                    parallel = []
                batches.append([call])
            else:
                parallel.append(call)
        if parallel:
            batches.append(parallel)
        return batches

    async def _ask(
        self,
        rt: AgentRuntime,
        run: RunRecord,
        call: ToolCallBlock,
        decision: PermissionDecision,
        records: Records,
    ) -> PermissionDecision:
        risks = {tool.name: tool.risk for tool in self._deps.tools.list(rt.id)}
        record = await self._update(
            records,
            call.id,
            status=ToolCallStatus.AWAITING_APPROVAL,
            asked_by=list(decision.asked_by),
        )
        request = PermissionRequestPayload(
            request_id=self._desk.new_id(),
            subject=HookEvent.BEFORE_TOOL_CALL,
            call_id=call.id,
            tool=call.name,
            reason=record.reason,
            input=decision.input,
            risk=risks.get(call.name),
            preview=decision.preview,
            asked_by=list(decision.asked_by),
            hook_reasons=list(decision.hook_reasons),
        )
        answer = await self._desk.ask(request, rt=rt, run_id=run.id)
        answered = decision.model_copy(
            update={"user_answer": answer.decision, "deny_message": answer.deny_message}
        )
        if answer.approved and answer.scope is PermissionScope.AGENT:
            await self._deps.tools.allow_always(rt.id, call, answered)
        return answered

    async def _execute(
        self, rt: AgentRuntime, call: ToolCallBlock, decision: PermissionDecision, records: Records
    ) -> None:
        await self._update(
            records,
            call.id,
            status=ToolCallStatus.RUNNING,
            decision=decision.decision,
            decided_by=_decided_by(decision),
            started_at=self._deps.clock.now(),
        )
        try:
            output = await self._deps.tools.execute(rt.id, call, decision)
        except Exception as failure:  # a tool manager bug must not kill the run (04 A4)
            output = error_output(MSG_TOOL_CRASHED.format(error=failure), ErrorCode.TOOL_FAILED)
        context = output.metadata.get(META_HOOK_CONTEXT)
        if isinstance(context, str):
            await self._keep_context(rt, context)
            kept = {k: v for k, v in output.metadata.items() if k != META_HOOK_CONTEXT}
            output = output.model_copy(update={"metadata": kept})
        await self._finish(rt, records, call, output)

    async def _keep_context(self, rt: AgentRuntime, text: str | None) -> None:
        """Queue what a tool hook wants the model to know for the next model call (05 §4)."""
        if text:
            pending = [*rt.record.pending_hook_context, text]
            rt.record = rt.record.model_copy(update={"pending_hook_context": pending})
            await self._deps.memory.save_agent(rt.record)

    async def _save_results(
        self, rt: AgentRuntime, run: RunRecord, records: Records
    ) -> MessageRecord:
        pending = rt.open_calls
        if pending is None:
            raise RuntimeError
        blocks = [
            ToolResultBlock(
                call_id=call.id,
                content=pending.resolved[call.id].content,
                is_error=pending.resolved[call.id].is_error,
                result_id=_result_id(pending.resolved[call.id]),
            )
            for call in pending.calls
        ]
        message = await self._deps.memory.append_message(
            MessageRecord(
                id=MessageId(self._deps.ids.new(ID_PREFIX_MESSAGE)),
                agent_id=rt.id,
                seq=0,
                run_id=run.id,
                role=MessageRole.TOOL,
                content=list(blocks),
                step=run.step_count,
                created_at=self._deps.clock.now(),
            )
        )
        rt.last_seq = message.seq
        rt.open_calls = None
        for call in pending.calls:
            await self._update(records, call.id, result_message_id=message.id)
        self._deps.bus.publish(rt.id, MessageAppended(message=message))
        return message

    # --- stop and recovery ---

    async def cancel_open(self, rt: AgentRuntime, run: RunRecord) -> None:
        """Give every unfinished call of the open step a ``tool.cancelled`` result."""
        pending = rt.open_calls
        if pending is None:
            return
        records = await self._records(rt, run, pending.message_id, pending.calls)
        for call in pending.calls:
            if call.id not in pending.resolved:
                output = error_output(MSG_TOOL_CANCELLED, ErrorCode.TOOL_CANCELLED)
                await self._finish(rt, records, call, output)
        await self._save_results(rt, run, records)

    async def interrupted(self, rt: AgentRuntime, run: RunRecord) -> OpenCalls | None:
        """Find tool calls left without a tool message by a crash (04 §7a.2).

        A call that finished keeps its saved result; a call that had started gets a
        ``tool.interrupted`` error; a call that never started is left to run normally.
        """
        messages = await self._deps.memory.get_messages(rt.id, from_seq=run.first_seq)
        last = messages[-1] if messages else None
        if last is None or last.role is not MessageRole.ASSISTANT:
            return None
        calls = [block for block in last.content if isinstance(block, ToolCallBlock)]
        if not calls:
            return None
        pending = OpenCalls(message_id=last.id, calls=calls, resolved={})
        rt.open_calls = pending
        records = await self._records(rt, run, last.id, calls)
        for call in calls:
            saved = stored_output(records[call.id])
            if saved is not None:
                pending.resolved[call.id] = saved
            elif records[call.id].status not in _UNSTARTED:
                output = error_output(MSG_TOOL_INTERRUPTED, ErrorCode.TOOL_INTERRUPTED)
                await self._finish(rt, records, call, output)
        return pending
