"""The ``messaging`` group: find other agents and write to them (phase 2, 03 §4-§5)."""

import json

from harness.core.consts import AddressType, Decision, ErrorCode, RiskLevel, ToolSource
from harness.core.errors import HarnessError
from harness.core.messaging import Messenger, SendMessageInput, SendMessageOutput
from harness.core.records import ToolOutput
from harness.core.types import Address
from harness.tools import consts
from harness.tools.function_tool import FunctionTool, error_output, text_output, tool
from harness.tools.types import ToolContext


def _sent(output: SendMessageOutput) -> ToolOutput:
    shown = text_output(
        output.model_dump_json(),
        metadata={
            consts.META_MESSAGE_ID: output.message_id,
            consts.META_CONVERSATION_ID: output.conversation_id,
            consts.META_DELIVERED: len(output.delivered),
            consts.META_REJECTED: len(output.rejected),
        },
    )
    if output.delivered:
        return shown
    return shown.model_copy(update={"is_error": True})


def messaging_tools(messenger: Messenger) -> list[FunctionTool]:
    """Build the tools an agent uses to talk to other agents."""

    @tool(
        group=consts.GROUP_MESSAGING,
        read_only=True,
        risk=RiskLevel.LOW,
        default_permission=Decision.ALLOW,
        source=ToolSource.BUILTIN,
    )
    async def list_agents(ctx: ToolContext) -> ToolOutput:
        """List the other agents you can write to.

        Each one has an ``address`` (pass it to send_message), a ``description`` of what it
        is for, its ``status``, and ``accepts_messages`` (false: it blocked you).
        """
        found = messenger.list_agents(Address(type=AddressType.AGENT, id=ctx.agent_id))
        if not found:
            return text_output(consts.MSG_NO_AGENTS, metadata={consts.META_MATCHES: 0})
        return text_output(
            json.dumps([agent.model_dump(mode="json") for agent in found], ensure_ascii=False),
            metadata={consts.META_MATCHES: len(found)},
        )

    @tool(
        group=consts.GROUP_MESSAGING,
        risk=RiskLevel.LOW,
        default_permission=Decision.ALLOW,
        source=ToolSource.BUILTIN,
    )
    async def send_message(
        to: list[str],
        content: str,
        ctx: ToolContext,
        *,
        expects_reply: bool = False,
        conversation_id: str | None = None,
        reply_to: str | None = None,
    ) -> ToolOutput:
        """Send a message to one or more agents and carry on; this never waits for an answer.

        ``to`` holds addresses from list_agents (``agent:<name>``). Set ``expects_reply``
        when you need an answer: it arrives later as a new <agent_message>. The result
        lists who got the message (``delivered``) and who did not, with the reason
        (``rejected``).
        """
        if not to:
            return error_output(consts.MSG_NO_TARGETS, ErrorCode.TOOL_INVALID_INPUT)
        message = SendMessageInput(
            to=to,
            content=content,
            expects_reply=expects_reply,
            conversation_id=conversation_id,
            reply_to=reply_to,
        )
        sender = Address(type=AddressType.AGENT, id=ctx.agent_id)
        return _sent(await messenger.send(sender, message))

    @tool(
        group=consts.GROUP_MESSAGING,
        risk=RiskLevel.LOW,
        default_permission=Decision.ALLOW,
        source=ToolSource.BUILTIN,
    )
    async def reply(message_id: str, content: str, ctx: ToolContext) -> ToolOutput:
        """Answer an <agent_message> you received; ``message_id`` is the one on its tag.

        Nothing is answered automatically: the other agent hears from you only if you
        call this (or send_message).
        """
        try:
            return _sent(await messenger.reply(ctx.agent_id, message_id, content))
        except HarnessError as error:
            return error_output(error.message, error.code)

    return [list_agents, send_message, reply]
