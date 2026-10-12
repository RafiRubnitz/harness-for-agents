"""The built-in tools of this version (08 §4.1)."""

from harness.config import ConfigManager
from harness.core.types import IdGenerator
from harness.memory import BlobStore
from harness.tools.builtin.delete import Deleter, delete_tools
from harness.tools.builtin.edits import WRITE_FILE_TOOLS
from harness.tools.builtin.files import READ_ONLY_FILE_TOOLS
from harness.tools.builtin.results import result_tools
from harness.tools.builtin.shell import shell_tools
from harness.tools.builtin.web import web_tools
from harness.tools.function_tool import FunctionTool


def builtin_tools(
    config: ConfigManager, blobs: BlobStore, ids: IdGenerator
) -> tuple[list[FunctionTool], Deleter]:
    """Build every built-in tool; the ``Deleter`` also serves the ``file.restore`` command."""
    deleter = Deleter(blobs, ids)
    tools = [
        *READ_ONLY_FILE_TOOLS,
        *WRITE_FILE_TOOLS,
        *delete_tools(deleter),
        *shell_tools(config),
        *result_tools(blobs),
        *web_tools(config),
    ]
    return tools, deleter


__all__ = ["Deleter", "builtin_tools"]
