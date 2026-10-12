"""``read_tool_result``: read a stored oversized tool result in parts (07 §7, 08 §6)."""

from harness.core.consts import META_TRUNCATED, Decision, RiskLevel, ToolSource
from harness.core.errors import MemoryStoreError
from harness.core.records import ToolOutput
from harness.memory import BlobStore
from harness.tools import consts
from harness.tools.function_tool import FunctionTool, error_output, text_output, tool


def result_tools(blobs: BlobStore) -> list[FunctionTool]:
    """Build ``read_tool_result`` over the blob store."""

    @tool(
        group=consts.GROUP_CONTEXT,
        read_only=True,
        risk=RiskLevel.LOW,
        default_permission=Decision.ALLOW,
        source=ToolSource.BUILTIN,
    )
    async def read_tool_result(
        result_id: str, offset: int = 0, limit: int = consts.RESULT_READ_CHARS_DEFAULT
    ) -> ToolOutput:
        """Read a part of a long tool result that was cut short in the conversation.

        ``result_id`` is the id named in the cut note, ``offset`` the first character
        (0-based) and ``limit`` the number of characters to return.
        """
        try:
            data = await blobs.get(result_id)
        except MemoryStoreError:
            return error_output(consts.MSG_RESULT_NOT_FOUND.format(result_id=result_id))
        text = data.decode(consts.TEXT_ENCODING, errors="replace")
        start = min(max(offset, 0), len(text))
        count = min(max(limit, 0), consts.RESULT_READ_CHARS_MAX)
        part = text[start : start + count]
        end = start + len(part)
        header = consts.MSG_RESULT_PART.format(start=start, end=end, total=len(text))
        return text_output(
            f"{header}\n{part}",
            metadata={
                consts.META_SIZE: len(text),
                consts.META_NEXT_OFFSET: end,
                META_TRUNCATED: end < len(text),
            },
        )

    return [read_tool_result]
