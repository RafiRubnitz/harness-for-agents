"""Run control: the agent's task, starting and halting runs, startup recovery (04 §7, §7a)."""

import asyncio

from loguru import logger

from harness.agents import state
from harness.agents.approvals import ApprovalDesk
from harness.agents.consts import (
    COMPONENT_AGENTS,
    MESSAGE_STOPPED,
    MSG_RUN_FAILED,
    TASK_NAME_PREFIX,
)
from harness.agents.loop import RunLoop
from harness.agents.runtime import AgentDeps, AgentRuntime, OpenCalls, RunStart, StatusBoard
from harness.agents.toolcalls import ToolRunner
from harness.core.consts import (
    ID_PREFIX_MESSAGE,
    ID_PREFIX_RUN,
    AddressType,
    AgentStatus,
    ErrorCode,
    MessageRole,
    PermissionResolution,
    RunStopReason,
)
from harness.core.errors import HarnessError
from harness.core.events import Error, MessageAppended
from harness.core.records import MessageRecord, RunRecord
from harness.core.types import MessageId, MessageOrigin, RunId, TextBlock

_S = AgentStatus


class RunController:
    """Owns the asyncio task of every running agent."""

    def __init__(
        self,
        deps: AgentDeps,
        status: StatusBoard,
        desk: ApprovalDesk,
        runner: ToolRunner,
        loop: RunLoop,
    ) -> None:
        self._deps = deps
        self._status = status
        self._desk = desk
        self._runner = runner
        self._loop = loop

    def launch(
        self,
        rt: AgentRuntime,
        run: RunRecord,
        start: RunStart | None,
        pending: OpenCalls | None = None,
    ) -> None:
        """Start the agent's task for an open run."""
        rt.run = run
        rt.task = asyncio.create_task(
            self._drive(rt, run, start, pending), name=f"{TASK_NAME_PREFIX}{rt.id}"
        )

    async def _drive(
        self, rt: AgentRuntime, run: RunRecord, start: RunStart | None, pending: OpenCalls | None
    ) -> None:
        try:
            await self._loop.run(rt, run, start, pending)
        except Exception as failure:  # the agent's task must end in a visible state
            logger.exception("run {} of agent {} failed", run.id, rt.id)
            await self._fail(rt, failure)
            return
        finally:
            if rt.task is asyncio.current_task():
                rt.task = None
        if rt.status is _S.IDLE and run.step_count > 0:
            await self.continue_from_inbox(rt)

    async def _fail(self, rt: AgentRuntime, failure: Exception) -> None:
        code = failure.code if isinstance(failure, HarnessError) else ErrorCode.APP_INTERNAL
        self._deps.bus.publish(
            rt.id,
            Error(
                code=code,
                message=MSG_RUN_FAILED.format(error=failure),
                recoverable=True,
                component=COMPONENT_AGENTS,
            ),
        )
        if rt.status is _S.WAITING_APPROVAL:
            await self._desk.cancel_for(rt.id)
            await self._status.set(rt, _S.RUNNING)
        if state.can_transition(rt.status, _S.ERROR):
            await self._status.set(rt, _S.ERROR)

    def wakes(self, origin: MessageOrigin) -> bool:
        """Return whether a message from ``origin`` may open a run for an idle agent."""
        if origin.source.type is not AddressType.TASK:
            return True
        return self._deps.config.config().background.wake_on_done

    async def continue_from_inbox(self, rt: AgentRuntime) -> None:
        """Open a run for an idle agent that has inbox messages allowed to wake it."""
        async with rt.lock:
            if rt.status is not _S.IDLE or rt.deleted:
                return
            pending = await self._deps.memory.pending_inbox(rt.id)
            waking = [item for item in pending if self.wakes(item.origin)]
            if waking:
                await self.start_run(rt, RunStart(waking[0].content, waking[0].origin))

    async def start_run(self, rt: AgentRuntime, start: RunStart) -> None:
        """Open a new run and launch the agent's task (the caller holds ``rt.lock``)."""
        run = RunRecord(
            id=RunId(self._deps.ids.new(ID_PREFIX_RUN)),
            agent_id=rt.id,
            first_seq=rt.last_seq + 1,
            started_at=self._deps.clock.now(),
        )
        await self._deps.memory.save_run(run)
        rt.run = run
        await self._status.set(rt, _S.RUNNING)
        self.launch(rt, run, start)

    async def cancel_task(self, rt: AgentRuntime) -> None:
        """Cancel the agent's task and wait until it is gone; persisted state is untouched."""
        task, rt.task = rt.task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def halt(
        self, rt: AgentRuntime, reason: RunStopReason, final: AgentStatus | None
    ) -> None:
        """Cancel the agent's task and close its open run, keeping what was produced.

        Streamed text becomes an assistant message, unfinished tool calls get a
        ``tool.cancelled`` result, and open approval requests are closed as cancelled.
        """
        await self.cancel_task(rt)
        await self._desk.cancel_for(rt.id)
        run = rt.run or await self._deps.memory.open_run(rt.id)
        if run is None:
            if final is not None:
                await self._status.set(rt, final)
            return
        rt.run = run
        if rt.partial_text:
            await self._save_partial(rt, run)
        await self._runner.cancel_open(rt, run)
        await self._loop.end_run(rt, run, reason, final)

    async def _save_partial(self, rt: AgentRuntime, run: RunRecord) -> None:
        message = await self._deps.memory.append_message(
            MessageRecord(
                id=MessageId(self._deps.ids.new(ID_PREFIX_MESSAGE)),
                agent_id=rt.id,
                seq=0,
                run_id=run.id,
                role=MessageRole.ASSISTANT,
                content=[TextBlock(text=rt.partial_text)],
                step=run.step_count,
                stop_reason=MESSAGE_STOPPED,
                created_at=self._deps.clock.now(),
            )
        )
        rt.partial_text = ""
        rt.last_seq = message.seq
        self._deps.bus.publish(rt.id, MessageAppended(message=message))

    async def recover(self, rt: AgentRuntime, *, crashed: bool) -> None:
        """Continue an agent whose stored status says it was in the middle of a run."""
        memory = self._deps.memory
        for stale in await memory.pending_approvals(rt.id):
            await memory.close_pending_approval(stale.request_id, PermissionResolution.CANCELLED)
        run = await memory.open_run(rt.id)
        rt.run = run
        if rt.status in state.ACTIVE and run is not None:
            rt.recovered = crashed
            await self._status.set(rt, _S.RUNNING, recovered=crashed, force=True)
            self.launch(rt, run, None, await self._runner.interrupted(rt, run))
            return
        if rt.status in state.ACTIVE:
            await self._status.set(rt, _S.IDLE, force=True)
        if rt.status is _S.IDLE:
            await self.continue_from_inbox(rt)
