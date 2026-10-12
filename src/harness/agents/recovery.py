"""Startup recovery of background tasks (04 §7a.5, 08 §6a)."""

from harness.agents.consts import MSG_TASK_INTERRUPTED, TASK_INTERRUPTED_ERROR
from harness.agents.runtime import AgentDeps
from harness.core.consts import ID_PREFIX_INBOX, AddressType, TaskStatus
from harness.core.records import InboxItem
from harness.core.types import Address, MessageOrigin, TextBlock


async def fail_interrupted_tasks(deps: AgentDeps) -> int:
    """Mark background tasks that were running as failed and tell their agents.

    The notice goes straight into the agent's inbox, so it is injected at the next step
    (or opens a run, when ``background.wake_on_done`` is on).

    Returns:
        The number of tasks that were marked.
    """
    now = deps.clock.now()
    interrupted = await deps.memory.list_tasks(None, status=TaskStatus.RUNNING)
    for task in interrupted:
        await deps.memory.save_task(
            task.model_copy(
                update={
                    "status": TaskStatus.FAILED,
                    "interrupted": True,
                    "ended_at": now,
                    "error": TASK_INTERRUPTED_ERROR,
                    "notified_at": now,
                }
            )
        )
        await deps.memory.enqueue_inbox(
            InboxItem(
                id=deps.ids.new(ID_PREFIX_INBOX),
                agent_id=task.agent_id,
                content=[TextBlock(text=MSG_TASK_INTERRUPTED.format(task_id=task.id))],
                origin=MessageOrigin(source=Address(type=AddressType.TASK, id=task.id)),
                received_at=now,
            )
        )
    return len(interrupted)
