"""The agent run loop (04 §5): steps of model call + tool calls until the model stops."""

from collections.abc import Callable
from enum import Enum, auto

from harness.agents.approvals import ApprovalDesk
from harness.agents.consts import COMPONENT_MODELS, MSG_STREAM_INCOMPLETE
from harness.agents.hooking import HookCaller
from harness.agents.runtime import AgentDeps, AgentRuntime, OpenCalls, RunStart, StatusBoard
from harness.agents.toolcalls import ToolRunner
from harness.config import InboxConfig
from harness.core.consts import (
    ID_PREFIX_MESSAGE,
    AgentEventType,
    AgentStatus,
    AskedBy,
    CompactReason,
    Decision,
    ErrorKind,
    HookEvent,
    MessageRole,
    RunStopReason,
    StreamEventType,
)
from harness.core.errors import BudgetExceededError, ContextError, GatewayError
from harness.core.events import (
    Error,
    MessageAppended,
    RunLimitReached,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    UsageUpdate,
)
from harness.core.hook_payloads import (
    AfterModelCallPayload,
    BeforeModelCallPayload,
    ModelErrorPayload,
    RunEndPayload,
    RunLimitReachedPayload,
    RunStartPayload,
)
from harness.core.records import (
    AgentEventRecord,
    MessageRecord,
    PermissionRequestPayload,
    RunRecord,
)
from harness.core.types import (
    ContentBlock,
    MessageId,
    ModelRequest,
    ModelResponse,
    StreamEvent,
    ToolCallBlock,
    Usage,
)

_PERCENT = 100.0


class _Step(Enum):
    CONTINUE = auto()  # tools ran; go on to the next step
    RETRY = auto()  # run the same step again (compaction, approved retry)
    FINISHED = auto()  # the model answered without tool calls
    HALTED = auto()  # the run was ended or parked inside the step


class RunLoop:
    """Drives one run; the manager wraps it in the agent's task."""

    def __init__(
        self,
        deps: AgentDeps,
        hooks: HookCaller,
        desk: ApprovalDesk,
        runner: ToolRunner,
        status: StatusBoard,
        *,
        inbox_taken: Callable[[AgentRuntime], None] | None = None,
    ) -> None:
        self._deps = deps
        self._hooks = hooks
        self._desk = desk
        self._runner = runner
        self._status = status
        self._inbox_taken = inbox_taken

    async def run(
        self,
        rt: AgentRuntime,
        run: RunRecord,
        start: RunStart | None,
        pending: OpenCalls | None = None,
    ) -> None:
        """Run until the model stops calling tools, or the run is parked / ended.

        Args:
            rt: The agent (already ``running``).
            run: The open run.
            start: The message that opened the run; ``None`` when continuing a run.
            pending: Tool calls of the last assistant message that still need results.
        """
        reuse_step = False
        if start is not None:
            payload = RunStartPayload(run_id=run.id, content=start.content, origin=start.origin)
            out = await self._hooks.emit(rt, HookEvent.RUN_START, payload)
            if not await self._hooks.allowed(rt, HookEvent.RUN_START, out):
                return await self.end_run(rt, run, RunStopReason.HOOK_DENIED)
        elif pending is not None:
            await self._runner.run(rt, run, pending)
        else:
            reuse_step = await self._step_unfinished(rt, run)
        window = 0
        while True:
            max_steps = rt.record.settings.max_steps
            if max_steps is not None and window >= max_steps:
                if not await self._on_limit(rt, run, max_steps):
                    return await self.end_run(rt, run, RunStopReason.LIMIT)
                window = 0
            if not reuse_step:
                window += 1
                run.step_count += 1
                await self._deps.memory.save_run(run)
            result = await self._step(rt, run)
            reuse_step = result is _Step.RETRY
            if result is _Step.HALTED:
                return None
            if result is _Step.FINISHED:
                return await self.end_run(rt, run, RunStopReason.END)

    async def _step_unfinished(self, rt: AgentRuntime, run: RunRecord) -> bool:
        """Return whether the run's current step has no saved model answer yet."""
        if run.step_count == 0:
            return False
        messages = await self._deps.memory.get_messages(rt.id, from_seq=run.first_seq)
        return not any(
            m.role is MessageRole.ASSISTANT and m.run_id == run.id and m.step == run.step_count
            for m in messages
        )

    # --- one step ---

    async def _step(self, rt: AgentRuntime, run: RunRecord) -> _Step:
        memory = self._deps.memory
        aging_s = self._deps.config.section(InboxConfig).aging_s or None
        taken = await memory.inject_inbox(rt.id, run.id, step=run.step_count, aging_s=aging_s)
        for injected in taken:
            rt.last_seq = injected.seq
            self._deps.bus.publish(rt.id, MessageAppended(message=injected))
        if taken and self._inbox_taken is not None:
            self._inbox_taken(rt)
        hook_context = list(rt.record.pending_hook_context)
        built = await self._deps.context.build(rt.id)
        # the automatic memory search stores its ids on the run; keep this copy in step
        run.retrieved_memory_ids = list(built.report.retrieved_memory_ids)
        meta = built.request.metadata.model_copy(
            update={"agent_id": rt.id, "run_id": run.id, "step": run.step_count}
        )
        request = built.request.model_copy(update={"metadata": meta})
        out = await self._hooks.emit(
            rt, HookEvent.BEFORE_MODEL_CALL, BeforeModelCallPayload(request=request)
        )
        if not await self._hooks.allowed(rt, HookEvent.BEFORE_MODEL_CALL, out):
            await self.end_run(rt, run, RunStopReason.HOOK_DENIED)
            return _Step.HALTED
        request = ModelRequest.model_validate(out.payload.get("request", request))
        try:
            response = await self._stream(rt, request)
        except GatewayError as error:
            return await self._on_gateway_error(rt, run, error)
        rt.overflow_retried = False
        message = await self._save_answer(rt, run, response, hook_context)
        calls = [block for block in response.content if isinstance(block, ToolCallBlock)]
        if calls:
            rt.open_calls = OpenCalls(message_id=message.id, calls=calls, resolved={})
        out = await self._hooks.emit(
            rt, HookEvent.AFTER_MODEL_CALL, AfterModelCallPayload(response=response)
        )
        if not await self._hooks.allowed(rt, HookEvent.AFTER_MODEL_CALL, out):
            await self._runner.cancel_open(rt, run)
            await self.end_run(rt, run, RunStopReason.HOOK_DENIED)
            return _Step.HALTED
        if rt.open_calls is None:
            return _Step.FINISHED
        await self._runner.run(rt, run, rt.open_calls)
        if rt.pause_requested:
            rt.pause_requested = False
            await self._status.set(rt, AgentStatus.PAUSED)
            return _Step.HALTED
        return _Step.CONTINUE

    def _with_reason(self, block: ContentBlock) -> ContentBlock:
        if not isinstance(block, ToolCallBlock):
            return block
        fields = self._deps.tools.split_injected(block.name, block.input)
        return block.model_copy(update={"reason": fields.reason})

    def _announce(self, rt: AgentRuntime, event: StreamEvent) -> None:
        bus = self._deps.bus
        if event.type is StreamEventType.TEXT_DELTA and event.text:
            rt.partial_text += event.text
            bus.publish(rt.id, TextDelta(text=event.text))
        elif event.type is StreamEventType.THINKING_DELTA and event.text:
            bus.publish(rt.id, ThinkingDelta(text=event.text))
        elif event.type is StreamEventType.TOOL_CALL_END and event.tool_call is not None:
            call = event.tool_call
            fields = self._deps.tools.split_injected(call.name, call.input)
            shown = ToolCall(
                call_id=call.id,
                tool=call.name,
                input=fields.input,
                reason=fields.reason,
                background=fields.run_in_background,
            )
            bus.publish(rt.id, shown)

    async def _stream(self, rt: AgentRuntime, request: ModelRequest) -> ModelResponse:
        """Call the model, forwarding every stream event to the GUI as it arrives."""
        rt.partial_text = ""
        response: ModelResponse | None = None
        async for event in self._deps.gateway.stream(request):
            self._announce(rt, event)
            if event.type is StreamEventType.END:
                response = event.response
        if response is None:
            raise GatewayError(
                MSG_STREAM_INCOMPLETE,
                kind=ErrorKind.OVERLOADED,
                provider=self._deps.gateway.resolve(request.model).provider.value,
                model=request.model,
            )
        content = [self._with_reason(block) for block in response.content]
        return response.model_copy(update={"content": content})

    async def _save_answer(
        self, rt: AgentRuntime, run: RunRecord, response: ModelResponse, used_context: list[str]
    ) -> MessageRecord:
        message = await self._deps.memory.append_message(
            MessageRecord(
                id=MessageId(self._deps.ids.new(ID_PREFIX_MESSAGE)),
                agent_id=rt.id,
                seq=0,
                run_id=run.id,
                role=MessageRole.ASSISTANT,
                content=response.content,
                step=run.step_count,
                usage=response.usage,
                cost_usd=response.cost_usd,
                stop_reason=response.stop_reason.value,
                created_at=self._deps.clock.now(),
            )
        )
        rt.partial_text = ""
        rt.last_seq = message.seq
        rt.run_usage = rt.run_usage + response.usage
        rt.cost_total += response.cost_usd
        usage = response.usage
        rt.context_tokens = usage.input + usage.cache_read + usage.cache_write
        left = [text for text in rt.record.pending_hook_context if text not in used_context]
        if len(left) != len(rt.record.pending_hook_context):
            rt.record = rt.record.model_copy(update={"pending_hook_context": left})
            await self._deps.memory.save_agent(rt.record)
        window = self._deps.gateway.resolve(rt.record.model).context_window
        self._deps.bus.publish(rt.id, MessageAppended(message=message))
        self._deps.bus.publish(
            rt.id,
            UsageUpdate(
                usage=usage,
                cost_usd_total=rt.cost_total,
                context_tokens=rt.context_tokens,
                context_window=window,
                context_pct=_PERCENT * rt.context_tokens / window if window else 0.0,
                model=response.model_used,
            ),
        )
        return message

    # --- limits and errors ---

    async def _on_limit(self, rt: AgentRuntime, run: RunRecord, max_steps: int) -> bool:
        """Handle reaching ``max_steps`` (04 §5.1); return whether the run goes on."""
        shown = RunLimitReached(steps=run.step_count, max_steps=max_steps)
        self._deps.bus.publish(rt.id, shown)
        await self._record_event(rt, run, AgentEventType.LIMIT_REACHED, shown.model_dump())
        payload = RunLimitReachedPayload(steps=run.step_count, max_steps=max_steps)
        out = await self._hooks.emit(rt, HookEvent.RUN_LIMIT_REACHED, payload)
        self._hooks.keep_context(rt, out)
        if out.decision is Decision.ALLOW:
            return True
        if out.decision is Decision.DENY:
            return False
        request = PermissionRequestPayload(
            request_id=self._desk.new_id(),
            subject=HookEvent.RUN_LIMIT_REACHED,
            asked_by=[AskedBy.RUN_LIMIT],
            hook_reasons=list(out.reasons),
            summary=rt.record.name,
        )
        return (await self._desk.ask(request, rt=rt, run_id=run.id)).approved

    async def _on_gateway_error(
        self, rt: AgentRuntime, run: RunRecord, error: GatewayError
    ) -> _Step:
        if error.kind is ErrorKind.CONTEXT_OVERFLOW and not rt.overflow_retried:
            rt.overflow_retried = True
            try:
                await self._deps.context.compact(rt.id, CompactReason.OVERFLOW)
            except ContextError:
                pass
            else:
                return _Step.RETRY
        return await self._on_model_error(rt, run, error)

    async def _on_model_error(self, rt: AgentRuntime, run: RunRecord, error: GatewayError) -> _Step:
        retries = self._deps.config.config(rt.id).limits.retries
        payload = ModelErrorPayload(
            kind=error.kind,
            retryable=error.retryable,
            message=error.message,
            model=error.model,
            attempts=1 + retries if error.retryable else 1,
        )
        out = await self._hooks.emit(rt, HookEvent.MODEL_ERROR, payload)
        shown = Error(
            code=error.code,
            kind=error.kind,
            message=error.message,
            recoverable=error.recoverable,
            component=COMPONENT_MODELS,
        )
        self._deps.bus.publish(rt.id, shown)
        await self._record_event(rt, run, AgentEventType.ERROR, shown.model_dump(mode="json"))
        if out.decision is Decision.ASK:
            if await self._hooks.allowed(rt, HookEvent.MODEL_ERROR, out):
                return _Step.RETRY
            await self.end_run(rt, run, RunStopReason.ERROR)
        elif out.decision is Decision.DENY:
            await self.end_run(rt, run, RunStopReason.ERROR)
        elif isinstance(error, BudgetExceededError):
            await self._status.set(rt, AgentStatus.PAUSED)
        else:
            await self._status.set(rt, AgentStatus.ERROR)
        return _Step.HALTED

    async def _record_event(
        self, rt: AgentRuntime, run: RunRecord, kind: AgentEventType, payload: dict[str, object]
    ) -> None:
        record = AgentEventRecord.model_validate(
            {
                "agent_id": rt.id,
                "run_id": run.id,
                "after_seq": rt.last_seq,
                "type": kind,
                "payload": payload,
                "created_at": self._deps.clock.now(),
            }
        )
        await self._deps.memory.append_event(record)

    # --- ending ---

    async def end_run(
        self,
        rt: AgentRuntime,
        run: RunRecord,
        stop_reason: RunStopReason,
        final: AgentStatus | None = AgentStatus.IDLE,
    ) -> None:
        """Fire ``run_end``, close the run, and move the agent to ``final`` (if given)."""
        payload = RunEndPayload(
            stop_reason=stop_reason.value, steps=run.step_count, usage=rt.run_usage
        )
        out = await self._hooks.emit(rt, HookEvent.RUN_END, payload)
        if self._hooks.keep_context(rt, out):
            await self._deps.memory.save_agent(rt.record)
        run.ended_at = self._deps.clock.now()
        run.stop_reason = stop_reason.value
        await self._deps.memory.save_run(run)
        rt.run = None
        rt.run_usage = Usage()
        rt.pause_requested = False
        rt.overflow_retried = False
        rt.open_calls = None
        if final is not None:
            await self._status.set(rt, final)
