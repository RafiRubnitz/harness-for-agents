"""Fire hooks for an agent and apply the merged decision (04 §5, 05 §3)."""

from harness.agents.approvals import ApprovalDesk
from harness.agents.consts import MSG_HOOK_DENIED
from harness.agents.runtime import AgentDeps, AgentRuntime
from harness.core.consts import AgentEventType, Decision, ErrorCode, HookEvent
from harness.core.events import Error, HookFired
from harness.core.hook_payloads import HookPayload
from harness.core.hook_types import HookInput, MergedOutput
from harness.core.records import AgentEventRecord, PermissionRequestPayload

_REASON_JOINER = "; "
_COMPONENT = "hooks"


class HookCaller:
    """Builds the ``HookInput``, announces what fired, and turns decisions into outcomes."""

    def __init__(self, deps: AgentDeps, desk: ApprovalDesk) -> None:
        self._deps = deps
        self._desk = desk

    async def emit(
        self, rt: AgentRuntime, event: HookEvent, payload: HookPayload, *, persist: bool = True
    ) -> MergedOutput:
        """Fire ``event`` for an agent.

        Args:
            rt: The agent the event belongs to.
            event: The hook event.
            payload: The typed payload.
            persist: Store the fired hooks as display events (off before the agent is saved).
        """
        now = self._deps.clock.now()
        out = await self._deps.hooks.emit(
            event,
            HookInput(
                event=event,
                agent_id=rt.id,
                agent_name=rt.record.name,
                run_id=rt.run.id if rt.run else None,
                step=rt.run.step_count if rt.run else None,
                working_dir=rt.record.settings.working_dir,
                payload=payload.model_dump(mode="json"),
                timestamp=now,
            ),
        )
        for fired in out.fired:
            if fired.suppressed:
                continue
            shown = HookFired(
                hook=fired.hook,
                event=fired.event,
                decision=fired.decision,
                reason=fired.reason,
                duration_ms=fired.duration_ms,
            )
            self._deps.bus.publish(rt.id, shown)
            if persist:
                await self._deps.memory.append_event(
                    AgentEventRecord(
                        agent_id=rt.id,
                        run_id=rt.run.id if rt.run else None,
                        after_seq=rt.last_seq,
                        type=AgentEventType.HOOK_RUN,
                        payload=shown.model_dump(mode="json"),
                        created_at=now,
                    )
                )
        return out

    def keep_context(self, rt: AgentRuntime, out: MergedOutput) -> bool:
        """Queue a hook's ``additional_context`` for the next model call; return if it did."""
        if not out.additional_context:
            return False
        pending = [*rt.record.pending_hook_context, out.additional_context]
        rt.record = rt.record.model_copy(update={"pending_hook_context": pending})
        return True

    async def allowed(
        self, rt: AgentRuntime, event: HookEvent, out: MergedOutput, *, stored: bool = True
    ) -> bool:
        """Apply a merged hook decision; return whether the action may go on.

        ``allow`` / no decision continues, ``deny`` refuses, ``ask`` asks the user.

        Args:
            rt: The agent.
            event: The event the decision is about.
            out: The merged hook output.
            stored: Whether the agent already exists in the store.
        """
        if self.keep_context(rt, out) and stored:
            await self._deps.memory.save_agent(rt.record)
        if out.decision is Decision.DENY:
            message = MSG_HOOK_DENIED.format(
                event=event.value, reasons=_REASON_JOINER.join(out.reasons)
            )
            shown = Error(
                code=ErrorCode.HOOK_DENIED,
                message=message,
                recoverable=True,
                component=_COMPONENT,
            )
            self._deps.bus.publish(rt.id, shown)
            return False
        if out.decision is not Decision.ASK:
            return True
        request = PermissionRequestPayload(
            request_id=self._desk.new_id(),
            subject=event,
            hook_reasons=list(out.reasons),
            summary=rt.record.name,
        )
        answer = await self._desk.ask(
            request, rt=rt if stored else None, run_id=rt.run.id if rt.run and stored else None
        )
        return answer.approved
