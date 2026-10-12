"""The GUI's picture of the core, built only from events (02 §5). No Tk in here."""

from dataclasses import dataclass, field

from harness.core import events as e
from harness.core.consts import AgentStatus, MessageRole, TaskStatus
from harness.core.events import Event, EventPayload
from harness.core.records import AgentState, ContextReport, MemoryItem, PermissionRequestPayload
from harness.core.types import AgentId, TaskId
from harness.gui import consts
from harness.gui.consts import NoticeLevel, Region, ToolStatus
from harness.gui.items import (
    AssistantItem,
    Conversation,
    NoticeItem,
    ThinkingItem,
    ToolItem,
    add_message,
    blocks_text,
    clip,
    from_history,
    shown_input,
)


@dataclass
class AgentView:
    """Everything shown about one agent."""

    state: AgentState
    conversation: Conversation = field(default_factory=Conversation)
    approvals: list[PermissionRequestPayload] = field(
        default_factory=list[PermissionRequestPayload]
    )
    tasks: dict[TaskId, e.TaskStatusChanged] = field(
        default_factory=dict[TaskId, e.TaskStatusChanged]
    )
    usage: e.UsageUpdate | None = None
    unread: int = 0
    loaded: bool = False

    @property
    def badge(self) -> int:
        """Open approvals, or else updates not yet seen."""
        return len(self.approvals) or self.unread

    @property
    def running_tasks(self) -> int:
        """Background tasks still running."""
        return sum(1 for task in self.tasks.values() if task.status is TaskStatus.RUNNING)


def format_report(report: ContextReport, full_text: str | None) -> str:
    """The ``/context`` report as text."""
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
    return "\n".join(lines)


class AppState:
    """Agents, their conversations, and what changed since the window last looked."""

    def __init__(self) -> None:
        self.agents: dict[AgentId, AgentView] = {}
        self.selected: AgentId | None = None
        self.banner: str | None = None
        self.memory_items: list[MemoryItem] = []
        self._changed: set[Region] = set()

    # --- reading ---

    def ordered(self, *, archived: bool = False) -> list[AgentView]:
        """Agents by creation time; archived ones only in the archived view."""
        views = [
            view
            for view in self.agents.values()
            if (view.state.status is AgentStatus.ARCHIVED) is archived
        ]
        return sorted(views, key=lambda view: view.state.created_at)

    def current(self) -> AgentView | None:
        """The selected agent."""
        return self.agents.get(self.selected) if self.selected is not None else None

    def active(self) -> list[AgentView]:
        """Agents in the middle of a run."""
        return [v for v in self.agents.values() if v.state.status in consts.ACTIVE_STATUSES]

    def take_changes(self) -> set[Region]:
        """Return the stale regions and clear them."""
        changed, self._changed = self._changed, set()
        return changed

    # --- user actions ---

    def select(self, agent_id: AgentId | None) -> None:
        """Show another agent; its updates count as seen."""
        self.selected = agent_id
        view = self.current()
        if view is not None:
            view.unread = 0
            view.conversation.touch(0)
        self._changed |= set(Region)

    def set_banner(self, text: str | None) -> None:
        """Show (or clear) a message that belongs to no agent."""
        self.banner = text
        self._changed.add(Region.STATUS)

    # --- events ---

    def apply(self, event: Event) -> None:
        """Fold one event into the picture."""
        payload = event.payload
        if self._apply_app(payload):
            return
        view = self.agents.get(event.agent_id) if event.agent_id is not None else None
        if view is None:
            self._apply_orphan(payload)
            return
        handled = (
            self._apply_stream(view, payload)
            or self._apply_tools(view, payload)
            or self._apply_run(view, payload)
        )
        if handled and event.agent_id == self.selected:
            self._changed.add(Region.STREAM)

    def _upsert(self, agent: AgentState) -> AgentView:
        view = self.agents.get(agent.id)
        if view is None:
            view = self.agents[agent.id] = AgentView(agent)
        else:
            view.state = agent
        self._changed |= {Region.SIDEBAR, Region.STATUS}
        return view

    def _apply_app(self, payload: EventPayload) -> bool:
        match payload:
            case e.AgentListed():
                for agent in payload.agents:
                    self._upsert(agent)
                shown = self.ordered()
                if self.selected is None and shown:
                    self.select(shown[0].state.id)
            case e.AgentCreated() | e.AgentUpdated():
                self._upsert(payload.agent)
            case e.AgentDeleted():
                self.agents.pop(payload.agent_id, None)
                if self.selected == payload.agent_id:
                    remaining = self.ordered()
                    self.select(remaining[0].state.id if remaining else None)
                self._changed.add(Region.SIDEBAR)
            case e.ConfigChanged():
                self._changed.add(Region.SETTINGS)
            case e.MemoryListed():
                self.memory_items = payload.items
                self._changed.add(Region.SETTINGS)
            case e.ConfigErrorRaised():
                line = consts.MSG_CONFIG_LINE.format(line=payload.line) if payload.line else ""
                self.set_banner(
                    consts.MSG_CONFIG_ERROR.format(
                        file=payload.file, line=line, message=payload.message
                    )
                )
            case e.HookLoadFailed():
                self.set_banner(
                    consts.MSG_HOOK_LOAD_ERROR.format(
                        hook=payload.hook, file=payload.file, message=payload.message
                    )
                )
            case _:
                return False
        return True

    def _apply_orphan(self, payload: EventPayload) -> None:
        """Errors and warnings that name no known agent go to the banner."""
        if isinstance(payload, e.Error | e.Warning):
            self.set_banner(
                consts.MSG_ERROR.format(message=payload.message, code=payload.code.value)
            )

    def _notify(self, view: AgentView) -> None:
        if view.state.id != self.selected:
            view.unread += 1
            self._changed.add(Region.SIDEBAR)

    def _apply_stream(self, view: AgentView, payload: EventPayload) -> bool:
        conversation = view.conversation
        match payload:
            case e.HistoryLoaded():
                live = [
                    item
                    for item in conversation.items
                    if isinstance(item, AssistantItem | ThinkingItem) and item.streaming
                ]
                view.conversation = from_history(payload.messages, payload.events)
                for item in live:
                    view.conversation.add(item)
                view.conversation.touch(0)
                view.loaded = True
            case e.MessageAppended():
                add_message(conversation, payload.message)
                if payload.message.role is MessageRole.ASSISTANT:
                    self._notify(view)
            case e.TextDelta():
                last = conversation.items[-1] if conversation.items else None
                if isinstance(last, AssistantItem) and last.streaming:
                    last.text += payload.text
                    conversation.touch(len(conversation.items) - 1)
                else:
                    conversation.add(AssistantItem(payload.text, streaming=True))
            case e.ThinkingDelta():
                last = conversation.items[-1] if conversation.items else None
                if isinstance(last, ThinkingItem) and last.streaming:
                    last.text += payload.text
                    conversation.touch(len(conversation.items) - 1)
                else:
                    conversation.add(ThinkingItem(payload.text, streaming=True))
            case _:
                return False
        return True

    def _apply_tools(self, view: AgentView, payload: EventPayload) -> bool:
        conversation = view.conversation
        match payload:
            case e.ToolCall():
                row = conversation.tool(payload.call_id)
                if row is None:
                    row = ToolItem(payload.call_id, payload.tool, shown_input(payload.input))
                    conversation.add(row)
                row.reason, row.background = payload.reason or row.reason, payload.background
                if row.status is ToolStatus.WAITING:
                    row.status = ToolStatus.RUNNING
            case e.ToolProgress():
                row = conversation.tool(payload.call_id)
                if row is not None:
                    row.output = clip(row.output + payload.chunk, consts.TOOL_OUTPUT_MAX_CHARS)
            case e.ToolResult():
                row = conversation.tool(payload.call_id)
                if row is None:
                    row = ToolItem(payload.call_id, payload.tool, {}, payload.reason)
                    conversation.add(row)
                row.status = ToolStatus.FAILED if payload.is_error else ToolStatus.OK
                row.duration_ms = payload.duration_ms
                row.output = clip(blocks_text(payload.content), consts.TOOL_OUTPUT_MAX_CHARS)
            case e.PermissionRequest():
                view.approvals.append(payload.request)
                row = conversation.tool(payload.request.call_id or "")
                if row is not None:
                    row.status = ToolStatus.WAITING
                self._notify(view)
                self._changed |= {Region.APPROVAL, Region.SIDEBAR}
            case e.PermissionResolved():
                view.approvals = [r for r in view.approvals if r.request_id != payload.request_id]
                self._changed |= {Region.APPROVAL, Region.SIDEBAR}
            case e.TaskStatusChanged():
                view.tasks[payload.task_id] = payload
                self._changed.add(Region.STATUS)
            case _:
                return False
        return True

    def _apply_run(self, view: AgentView, payload: EventPayload) -> bool:
        conversation = view.conversation
        match payload:
            case e.AgentStatusChanged():
                view.state = view.state.model_copy(update={"status": payload.status})
                if payload.recovered:
                    conversation.add(NoticeItem(consts.MSG_RECOVERED, NoticeLevel.WARNING))
                if payload.status not in consts.ACTIVE_STATUSES:
                    self._settle(view)
                self._changed |= {Region.SIDEBAR, Region.STATUS}
            case e.UsageUpdate():
                view.usage = payload
                self._changed.add(Region.STATUS)
            case e.RunLimitReached():
                text = consts.MSG_LIMIT.format(steps=payload.steps, max_steps=payload.max_steps)
                conversation.add(NoticeItem(text, NoticeLevel.WARNING))
            case e.ContextCompacted():
                text = consts.MSG_COMPACTED.format(from_seq=payload.from_seq, to_seq=payload.to_seq)
                conversation.add(NoticeItem(text))
            case e.ContextReported():
                text = format_report(payload.report, payload.full_text)
                conversation.add(NoticeItem(text, mono=True))
            case e.Error():
                text = consts.MSG_ERROR.format(message=payload.message, code=payload.code.value)
                conversation.add(NoticeItem(clip(text, consts.NOTICE_MAX_CHARS), NoticeLevel.ERROR))
                self._notify(view)
            case e.Warning():
                text = consts.MSG_ERROR.format(message=payload.message, code=payload.code.value)
                conversation.add(NoticeItem(text, NoticeLevel.WARNING))
            case _:
                return False
        return True

    def _settle(self, view: AgentView) -> None:
        """A run ended: nothing streams any more and nothing is still running."""
        for index, item in enumerate(view.conversation.items):
            if isinstance(item, AssistantItem | ThinkingItem) and item.streaming:
                item.streaming = False
                view.conversation.touch(index)
            unfinished = isinstance(item, ToolItem) and item.status is ToolStatus.RUNNING
            if isinstance(item, ToolItem) and unfinished and not item.background:
                item.status = ToolStatus.FAILED
                view.conversation.touch(index)
