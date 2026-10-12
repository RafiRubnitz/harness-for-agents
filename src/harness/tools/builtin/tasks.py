"""The ``tasks`` group: follow and cancel background tasks (08 §6a)."""

from harness.core.consts import META_TASK_ID, Decision, RiskLevel, TaskStatus, ToolSource
from harness.core.errors import ToolError
from harness.core.records import BackgroundTaskRecord, ToolOutput
from harness.core.types import TaskId
from harness.tools import consts
from harness.tools.background import BackgroundTasks
from harness.tools.function_tool import FunctionTool, error_output, text_output, tool
from harness.tools.types import ToolContext


def _line(tasks: BackgroundTasks, record: BackgroundTaskRecord) -> str:
    shown = tasks.view(record)
    return consts.TASK_LINE.format(
        task_id=record.id,
        tool=record.tool,
        status=record.status.value,
        seconds=(shown.duration_ms or 0) // consts.MS_PER_S,
    )


def task_tools(tasks: BackgroundTasks) -> list[FunctionTool]:
    """Build the tools an agent uses to manage what it started in the background."""

    @tool(
        group=consts.GROUP_TASKS,
        read_only=True,
        risk=RiskLevel.LOW,
        default_permission=Decision.ALLOW,
        source=ToolSource.BUILTIN,
    )
    async def task_status(task_id: str, ctx: ToolContext) -> ToolOutput:
        """Say whether a background task is running, done, failed or cancelled."""
        try:
            record = await tasks.find(ctx.agent_id, task_id)
        except ToolError as error:
            return error_output(error.message, error.code)
        return text_output(
            _line(tasks, record),
            metadata={META_TASK_ID: record.id, consts.META_STATUS: record.status.value},
        )

    @tool(
        group=consts.GROUP_TASKS,
        read_only=True,
        risk=RiskLevel.LOW,
        default_permission=Decision.ALLOW,
        source=ToolSource.BUILTIN,
    )
    async def task_output(
        task_id: str, ctx: ToolContext, offset: int = 0, limit: int = consts.TASK_OUTPUT_LIMIT
    ) -> ToolOutput:
        """Read what a background task printed so far, even while it is still running.

        ``offset`` and ``limit`` are in characters; ``next_offset`` in the result says
        where to continue.
        """
        try:
            record = await tasks.find(ctx.agent_id, task_id)
        except ToolError as error:
            return error_output(error.message, error.code)
        whole = await tasks.output(record)
        start = max(offset, 0)
        part = whole[start : start + max(limit, 0)]
        return text_output(
            part,
            metadata={
                META_TASK_ID: record.id,
                consts.META_STATUS: record.status.value,
                consts.META_NEXT_OFFSET: start + len(part),
                consts.META_SIZE: len(whole),
            },
        )

    @tool(
        group=consts.GROUP_TASKS,
        risk=RiskLevel.LOW,
        default_permission=Decision.ALLOW,
        source=ToolSource.BUILTIN,
    )
    async def task_cancel(task_id: str, ctx: ToolContext) -> ToolOutput:
        """Cancel a background task that is still running."""
        try:
            record = await tasks.find(ctx.agent_id, task_id)
        except ToolError as error:
            return error_output(error.message, error.code)
        if record.status is not TaskStatus.RUNNING:
            return text_output(_line(tasks, record), metadata={META_TASK_ID: record.id})
        await tasks.cancel(TaskId(record.id))
        return text_output(_line(tasks, await tasks.find(ctx.agent_id, task_id)))

    @tool(
        group=consts.GROUP_TASKS,
        read_only=True,
        risk=RiskLevel.LOW,
        default_permission=Decision.ALLOW,
        source=ToolSource.BUILTIN,
    )
    async def task_list(ctx: ToolContext) -> ToolOutput:
        """List every background task you started, oldest first."""
        records = await tasks.every(ctx.agent_id)
        lines = [_line(tasks, record) for record in records]
        return text_output(
            "\n".join(lines) if lines else consts.MSG_NO_TASKS,
            metadata={consts.META_MATCHES: len(lines)},
        )

    return [task_status, task_output, task_cancel, task_list]
