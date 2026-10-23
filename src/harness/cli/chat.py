"""``harness chat``: one agent in the terminal, with streaming and inline approvals."""

from collections.abc import AsyncIterator
from dataclasses import dataclass

from harness.cli import consts
from harness.cli.terminal import Terminal
from harness.cli.utils import (
    find_agent,
    format_report,
    format_request,
    format_tool_call,
    format_tool_result,
)
from harness.core.commands import (
    AgentCreate,
    AgentList,
    AgentRetry,
    Command,
    PermissionRespond,
)
from harness.core.consts import (
    COMPONENT_COMMANDS,
    AgentStatus,
    PermissionAnswer,
    PermissionScope,
)
from harness.core.errors import SlashError
from harness.core.events import (
    AgentCreated,
    AgentListed,
    AgentStatusChanged,
    ContextCompacted,
    ContextReported,
    Error,
    Event,
    PermissionRequest,
    RunLimitReached,
    TextDelta,
    ToolCall,
    ToolProgress,
    ToolResult,
)
from harness.core.events import Warning as WarningEvent
from harness.core.handle import CoreHandle
from harness.core.records import AgentState, PermissionRequestPayload
from harness.core.utils.names import free_name
from harness.core.utils.slash import parse_input


class ChatError(Exception):
    """The chat cannot start or continue; the message is shown to the user."""


@dataclass(frozen=True, slots=True)
class ChatOptions:
    """Arguments of ``harness chat``."""

    agent: str | None = None
    new: bool = False
    template: str | None = None
    model: str | None = None


class ChatSession:
    """Drives one agent from terminal input until the user leaves."""

    def __init__(self, core: CoreHandle, terminal: Terminal) -> None:
        self._core = core
        self._terminal = terminal
        self._events: AsyncIterator[Event] = core.events()
        self._mid_line = False

    # --- choosing the agent ---

    async def _reply[T](self, command: Command, kind: type[T]) -> T:
        """Run ``command`` and return its reply payload, or raise what the core reported."""
        await self._core.submit(command)
        async for event in self._events:
            payload = event.payload
            if isinstance(payload, kind):
                return payload
            if isinstance(payload, Error) and payload.component == COMPONENT_COMMANDS:
                raise ChatError(payload.message)
        raise ChatError(consts.MSG_CORE_STOPPED)

    async def list_agents(self, *, include_archived: bool = False) -> list[AgentState]:
        """Ask the core for its agents."""
        listed = await self._reply(AgentList(include_archived=include_archived), AgentListed)
        return listed.agents

    async def _create(self, options: ChatOptions) -> AgentState:
        # The model stays unset unless given, so a template's own model is kept.
        name = options.agent
        if name is None:  # names are unique: chat, chat-2, ...
            taken = {agent.name for agent in await self.list_agents(include_archived=True)}
            name = free_name(consts.DEFAULT_AGENT_NAME, taken)
        fields: dict[str, str | list[str] | None] = {
            "agent_name": name,
            "template": options.template,
        }
        if options.template is None:
            fields["tools"] = list(consts.DEFAULT_AGENT_TOOLS)
        if options.model is not None:
            fields["model"] = options.model
        command = AgentCreate.model_validate(fields)
        return (await self._reply(command, AgentCreated)).agent

    async def pick_agent(self, options: ChatOptions) -> AgentState:
        """Find the agent to talk to, creating one when asked (or when there is none)."""
        if options.new:
            return await self._create(options)
        agents = await self.list_agents()
        if options.agent is not None:
            found = find_agent(agents, options.agent)
            if found is None:
                raise ChatError(consts.MSG_AGENT_NOT_FOUND.format(name=options.agent))
            return found
        return agents[0] if agents else await self._create(options)

    # --- the conversation ---

    def _say(self, text: str, style: str | None = None) -> None:
        if self._mid_line:
            self._terminal.line()
            self._mid_line = False
        self._terminal.line(text, style)

    def _stream(self, text: str) -> None:
        self._terminal.write(text)
        self._mid_line = not text.endswith("\n")

    async def _approve(self, request: PermissionRequestPayload) -> None:
        for line in format_request(request):
            self._say(line)
        answer: str | None = ""
        choices = (consts.ANSWER_ONCE, consts.ANSWER_ALWAYS, consts.ANSWER_DENY)
        while answer is not None and answer not in choices:
            answer = await self._terminal.read(consts.APPROVAL_PROMPT)
            answer = answer.strip().lower() if answer is not None else None
        approved = answer in (consts.ANSWER_ONCE, consts.ANSWER_ALWAYS)
        await self._core.submit(
            PermissionRespond(
                request_id=request.request_id,
                decision=PermissionAnswer.APPROVE if approved else PermissionAnswer.DENY,
                scope=(
                    PermissionScope.AGENT
                    if answer == consts.ANSWER_ALWAYS
                    else PermissionScope.ONCE
                ),
            )
        )

    def _show_tool(self, payload: ToolCall | ToolProgress | ToolResult) -> None:
        match payload:
            case ToolCall():
                for line in format_tool_call(payload):
                    self._say(line, consts.ANSI_DIM)
            case ToolProgress(chunk=chunk):
                self._stream(chunk)
            case ToolResult():
                style = consts.ANSI_RED if payload.is_error else consts.ANSI_DIM
                for line in format_tool_result(payload):
                    self._say(line, style)

    async def _show(self, event: Event) -> None:
        match event.payload:
            case TextDelta(text=text):
                self._stream(text)
            case ToolCall() | ToolProgress() | ToolResult() as tool_event:
                self._show_tool(tool_event)
            case PermissionRequest(request=request):
                await self._approve(request)
            case RunLimitReached(steps=steps, max_steps=max_steps):
                self._say(consts.MSG_LIMIT.format(steps=steps, max_steps=max_steps))
            case ContextCompacted(from_seq=from_seq, to_seq=to_seq):
                self._say(consts.MSG_COMPACTED.format(from_seq=from_seq, to_seq=to_seq))
            case ContextReported(report=report, full_text=full_text):
                for line in format_report(report, full_text):
                    self._say(line)
            case Error(code=code, message=message):
                self._say(
                    consts.MSG_ERROR.format(code=code.value, message=message), consts.ANSI_RED
                )
            case WarningEvent(code=code, message=message):
                self._say(
                    consts.MSG_WARNING.format(code=code.value, message=message),
                    consts.ANSI_YELLOW,
                )
            case _:
                pass

    def _rest(self, status: AgentStatus) -> None:
        if status is AgentStatus.ERROR:
            self._say(consts.MSG_RETRY_HINT, consts.ANSI_YELLOW)
        elif status is AgentStatus.PAUSED:
            self._say(consts.MSG_PAUSED_HINT)
        elif self._mid_line:
            self._say("")

    async def follow(self, agent: AgentState, command: Command) -> None:
        """Run ``command`` and show what happens until the agent waits for the user again.

        A trailing ``agent.list`` marks the point where the command itself is done; from
        there on, the first resting status ends the turn.
        """
        await self._core.submit(command)
        await self._core.submit(AgentList(include_archived=True))
        settled = False
        async for event in self._events:
            payload = event.payload
            status: AgentStatus | None = None
            if isinstance(payload, AgentListed):
                current = find_agent(payload.agents, agent.id)
                status = current.status if current else AgentStatus.ARCHIVED
                settled = True
            elif event.agent_id not in (agent.id, None):
                continue
            elif isinstance(payload, AgentStatusChanged):
                status = payload.status if settled else None
            else:
                await self._show(event)
            if status is not None and status in consts.RESTING:
                self._rest(status)
                return

    async def run(self, options: ChatOptions) -> None:
        """Pick the agent and talk until ``/exit`` or the end of input."""
        agent = await self.pick_agent(options)
        self._say(
            consts.MSG_CHATTING.format(name=agent.name, agent_id=agent.id, model=agent.model),
            consts.ANSI_DIM,
        )
        while (text := await self._terminal.read(consts.PROMPT)) is not None:
            typed = text.strip()
            if typed.lower() in consts.LOCAL_EXIT:
                return
            if not typed:
                continue
            try:
                command: Command = (
                    AgentRetry(agent_id=agent.id)
                    if typed.lower() == consts.LOCAL_RETRY
                    else parse_input(text, agent.id)
                )
            except SlashError as error:
                self._say(error.message, consts.ANSI_RED)
                continue
            await self.follow(agent, command)


__all__ = ["ChatError", "ChatOptions", "ChatSession"]
