"""Background tasks: tools that run while the agent goes on working (08 §6a)."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from loguru import logger

from harness.core.consts import ID_PREFIX_TASK, META_TASK_ID, BlobKind, ErrorCode, TaskStatus
from harness.core.errors import HarnessError, ToolError
from harness.core.records import AgentState, BackgroundTask, BackgroundTaskRecord, ToolOutput
from harness.core.types import AgentId, Clock, IdGenerator, TaskId, TextBlock, ToolCallBlock
from harness.core.utils.time import elapsed_ms
from harness.memory import BlobStore, MemoryStore
from harness.tools.consts import (
    BACKUP_META_AGENT_ID,
    MSG_TASK_CANCELLED,
    MSG_TASK_NOT_FOUND,
    TASK_SUMMARY_CHARS,
    TEXT_ENCODING,
)

Emit = Callable[[str], None]
Work = Callable[[TaskId, Emit], Awaitable[ToolOutput]]


@dataclass(frozen=True, slots=True)
class TaskSinks:
    """Where a task's life is announced.

    Attributes:
        status: Every status change (the ``task.status`` event).
        done: The finished task and a summary of its result, for the agent's inbox.
    """

    status: Callable[[BackgroundTaskRecord], None]
    done: Callable[[BackgroundTaskRecord, str], Awaitable[None]]


@dataclass(slots=True)
class _Live:
    record: BackgroundTaskRecord
    chunks: list[str] = field(default_factory=list[str])
    task: asyncio.Task[None] | None = None
    cancelled: bool = False
    final: str | None = None


def _text(output: ToolOutput) -> str:
    return "".join(block.text for block in output.content if isinstance(block, TextBlock))


class BackgroundTasks:
    """Starts, follows and cancels the background tasks of every agent."""

    def __init__(
        self,
        store: MemoryStore,
        blobs: BlobStore,
        clock: Clock,
        ids: IdGenerator,
        sinks: TaskSinks,
    ) -> None:
        self._store = store
        self._blobs = blobs
        self._clock = clock
        self._ids = ids
        self._sinks = sinks
        self._live: dict[TaskId, _Live] = {}

    async def start(self, agent: AgentState, call: ToolCallBlock, work: Work) -> TaskId:
        """Run ``work`` as its own asyncio task and return at once."""
        record = BackgroundTaskRecord(
            id=TaskId(self._ids.new(ID_PREFIX_TASK)),
            agent_id=agent.id,
            call_id=call.id,
            tool=call.name,
            status=TaskStatus.RUNNING,
            started_at=self._clock.now(),
        )
        live = _Live(record)
        self._live[record.id] = live
        await self._store.save_task(record)
        self._sinks.status(record)
        live.task = asyncio.create_task(self._follow(live, work), name=f"task:{record.id}")
        return record.id

    async def _follow(self, live: _Live, work: Work) -> None:
        try:
            output = await work(live.record.id, live.chunks.append)
        except asyncio.CancelledError:
            if not live.cancelled:
                raise  # the app is shutting down: the task stays "running" for recovery
            await self._finish(live, TaskStatus.CANCELLED, MSG_TASK_CANCELLED)
            return
        status = TaskStatus.FAILED if output.is_error else TaskStatus.DONE
        await self._finish(live, status, _text(output))

    async def _finish(self, live: _Live, status: TaskStatus, text: str) -> None:
        live.final = text
        blob = await self._keep(live.record, text)
        now = self._clock.now()
        summary = text[:TASK_SUMMARY_CHARS]
        live.record = live.record.model_copy(
            update={
                "status": status,
                "ended_at": now,
                "output_blob": blob,
                "result_summary": summary,
                "error": summary if status is TaskStatus.FAILED else None,
                "notified_at": now,
            }
        )
        await self._store.save_task(live.record)
        self._sinks.status(live.record)
        try:
            await self._sinks.done(live.record, summary)
        except asyncio.CancelledError:
            raise
        except Exception:  # the agent may be gone; the task itself is finished and stored
            logger.opt(exception=True).warning("task {} could not notify its agent", live.record.id)

    async def _keep(self, record: BackgroundTaskRecord, text: str) -> str | None:
        try:
            return await self._blobs.put(
                text.encode(TEXT_ENCODING),
                kind=BlobKind.TASK_OUTPUT,
                meta={BACKUP_META_AGENT_ID: record.agent_id, META_TASK_ID: record.id},
            )
        except HarnessError as error:  # the agent must still learn that the task ended
            logger.warning("output of task {} was not stored: {}", record.id, error.message)
            return None

    async def cancel(self, task_id: TaskId) -> None:
        """Cancel a running task; unknown and finished tasks are left alone."""
        live = self._live.get(task_id)
        if live is None or live.task is None or live.task.done():
            return
        live.cancelled = True
        live.task.cancel()
        await asyncio.gather(live.task, return_exceptions=True)

    async def close(self) -> None:
        """Stop every running task without marking it; the next start reports them."""
        tasks = [live.task for live in self._live.values() if live.task and not live.task.done()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def of(self, agent_id: AgentId) -> list[BackgroundTask]:
        """The tasks an agent started in this session, oldest first."""
        return [
            self.view(live.record)
            for live in self._live.values()
            if live.record.agent_id == agent_id
        ]

    def view(self, record: BackgroundTaskRecord) -> BackgroundTask:
        """The display view of a task record."""
        end = record.ended_at or self._clock.now()
        return BackgroundTask(
            id=record.id,
            call_id=record.call_id,
            tool=record.tool,
            status=record.status,
            started_at=record.started_at,
            duration_ms=elapsed_ms(record.started_at, end),
        )

    async def every(self, agent_id: AgentId) -> list[BackgroundTaskRecord]:
        """Every task of an agent, including those of earlier sessions."""
        stored = {record.id: record for record in await self._store.list_tasks(agent_id)}
        for live in self._live.values():
            if live.record.agent_id == agent_id:
                stored[live.record.id] = live.record
        return sorted(stored.values(), key=lambda record: record.started_at)

    async def find(self, agent_id: AgentId, task_id: str) -> BackgroundTaskRecord:
        """Return one task of the agent.

        Raises:
            ToolError: The agent has no such task.
        """
        for record in await self.every(agent_id):
            if record.id == task_id:
                return record
        raise ToolError(
            MSG_TASK_NOT_FOUND.format(task_id=task_id), code=ErrorCode.TOOL_FAILED, tool=""
        )

    async def output(self, record: BackgroundTaskRecord) -> str:
        """What the task printed so far, or its whole result once it ended."""
        live = self._live.get(record.id)
        if live is not None:
            return live.final if live.final is not None else "".join(live.chunks)
        if record.output_blob is None:
            return record.error or ""
        return (await self._blobs.get(record.output_blob)).decode(TEXT_ENCODING, "replace")
