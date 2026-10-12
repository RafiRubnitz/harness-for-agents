"""The memory tools an agent gets: save, search, update, forget, and history search (03 §5)."""

from harness.core.consts import (
    Decision,
    MemoryKind,
    MemorySource,
    RiskLevel,
    SearchMode,
    ToolSource,
)
from harness.core.errors import MemoryStoreError
from harness.core.records import MemoryItem, MessageRecord, ToolOutput
from harness.core.types import MemoryItemId, TextBlock
from harness.memory import MemoryStore, SemanticMemory
from harness.tools import consts
from harness.tools.function_tool import FunctionTool, error_output, text_output, tool
from harness.tools.types import ToolContext


def _line(item: MemoryItem) -> str:
    """One memory as the model reads it: date, id, kind, tags, text."""
    tags = consts.MEMORY_TAGS.format(tags=", ".join(item.tags)) if item.tags else ""
    return consts.MEMORY_LINE.format(
        date=item.observed_at.date().isoformat(),
        item_id=item.id,
        kind=item.kind.value,
        tags=tags,
        content=item.content,
    )


def _message_line(message: MessageRecord) -> str:
    text = " ".join(block.text for block in message.content if isinstance(block, TextBlock))
    if len(text) > consts.HISTORY_HIT_MAX_CHARS:
        text = text[: consts.HISTORY_HIT_MAX_CHARS] + consts.HISTORY_HIT_CUT
    return consts.HISTORY_LINE.format(
        seq=message.seq,
        date=message.created_at.date().isoformat(),
        role=message.role.value,
        text=text,
    )


def _write_tools(memory: SemanticMemory) -> list[FunctionTool]:
    async def own(item_id: str, ctx: ToolContext) -> MemoryItem:
        """Return the agent's own item; global and foreign items cannot be changed."""
        item = await memory.get(MemoryItemId(item_id))
        if item.agent_id != ctx.agent_id:
            raise MemoryStoreError(consts.MSG_MEMORY_NOT_YOURS.format(item_id=item_id))
        return item

    @tool(
        group=consts.GROUP_MEMORY,
        risk=RiskLevel.LOW,
        default_permission=Decision.ALLOW,
        source=ToolSource.BUILTIN,
    )
    async def memory_save(
        content: str, kind: MemoryKind, ctx: ToolContext, tags: list[str] | None = None
    ) -> ToolOutput:
        """Save a fact to your long-term memory so that later conversations can recall it.

        Write one self-contained fact per call. ``kind`` is one of fact, preference,
        decision, note, todo. Saving something you already know updates that memory.
        """
        try:
            item_id = await memory.remember(ctx.agent_id, content, kind, tags=tags or [])
        except MemoryStoreError as error:
            return error_output(error.message, error.code)
        return text_output(
            consts.MSG_MEMORY_SAVED.format(item_id=item_id), metadata={consts.META_ITEM_ID: item_id}
        )

    @tool(
        group=consts.GROUP_MEMORY,
        risk=RiskLevel.LOW,
        default_permission=Decision.ALLOW,
        source=ToolSource.BUILTIN,
    )
    async def memory_update(
        id: str,  # noqa: A002 - the tool's public parameter name (03 §5)
        content: str,
        ctx: ToolContext,
        supersede: bool = False,
    ) -> ToolOutput:
        """Change one of your memories.

        ``supersede=false`` corrects the same fact (the old wording is kept as a version).
        ``supersede=true`` records that the fact itself changed: the old memory expires
        and a new one replaces it.
        """
        try:
            await own(id, ctx)
            item_id = MemoryItemId(id)
            if supersede:
                item_id = await memory.supersede(item_id, content, actor=MemorySource.AGENT)
            else:
                await memory.update(item_id, actor=MemorySource.AGENT, content=content)
        except MemoryStoreError as error:
            return error_output(error.message, error.code)
        return text_output(
            consts.MSG_MEMORY_UPDATED.format(item_id=item_id),
            metadata={consts.META_ITEM_ID: item_id},
        )

    @tool(
        group=consts.GROUP_MEMORY,
        risk=RiskLevel.LOW,
        default_permission=Decision.ALLOW,
        source=ToolSource.BUILTIN,
    )
    async def memory_forget(
        id: str,  # noqa: A002 - the tool's public parameter name (03 §5)
        ctx: ToolContext,
    ) -> ToolOutput:
        """Forget one of your memories: it stops being recalled but stays in the history."""
        try:
            await own(id, ctx)
            await memory.forget(MemoryItemId(id))
        except MemoryStoreError as error:
            return error_output(error.message, error.code)
        return text_output(consts.MSG_MEMORY_FORGOTTEN.format(item_id=id))

    return [memory_save, memory_update, memory_forget]


def _read_tools(memory: SemanticMemory, history: MemoryStore) -> list[FunctionTool]:
    @tool(
        group=consts.GROUP_MEMORY,
        read_only=True,
        risk=RiskLevel.LOW,
        default_permission=Decision.ALLOW,
        source=ToolSource.BUILTIN,
    )
    async def memory_search(
        query: str, ctx: ToolContext, mode: SearchMode = SearchMode.HYBRID
    ) -> ToolOutput:
        """Search your long-term memory and the user's global facts.

        ``mode`` is hybrid (meaning and exact words, the default), vector or lexical.
        Every hit shows the date the fact was observed; weigh how current it is.
        """
        try:
            hits = await memory.search(ctx.agent_id, query, mode=mode)
        except MemoryStoreError as error:
            return error_output(error.message, error.code)
        lines = [_line(hit.item) for hit in hits]
        return text_output(
            "\n".join(lines) if lines else consts.MSG_NO_MEMORIES,
            metadata={consts.META_MATCHES: len(lines)},
        )

    @tool(
        group=consts.GROUP_MEMORY,
        read_only=True,
        risk=RiskLevel.LOW,
        default_permission=Decision.ALLOW,
        source=ToolSource.BUILTIN,
    )
    async def history_search(query: str, ctx: ToolContext) -> ToolOutput:
        """Search the full text of your own past conversation, including compacted parts.

        Matches exact words (not meaning) and returns the best matching messages.
        """
        try:
            found = await history.search_history(ctx.agent_id, query)
        except MemoryStoreError as error:
            return error_output(error.message, error.code)
        lines = [_message_line(message) for message in found]
        return text_output(
            "\n".join(lines) if lines else consts.MSG_NO_HISTORY,
            metadata={consts.META_MATCHES: len(lines)},
        )

    return [memory_search, history_search]


def memory_tools(memory: SemanticMemory, history: MemoryStore) -> list[FunctionTool]:
    """Build the ``memory`` group over the semantic memory and the agent's history."""
    return [*_write_tools(memory), *_read_tools(memory, history)]
