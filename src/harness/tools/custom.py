"""Custom tools: Python files in ``~/.harness/tools/`` (08 §4.2).

A file is imported when it appears and again whenever it changes; no restart is needed.
"""

import sys
from pathlib import Path
from types import ModuleType

from loguru import logger

from harness.core.consts import ToolSource
from harness.core.errors import ToolError
from harness.tools.consts import (
    CUSTOM_MODULE_PREFIX,
    CUSTOM_PRIVATE_PREFIX,
    CUSTOM_TOOL_GLOB,
    MSG_CUSTOM_BROKEN,
    MSG_CUSTOM_NOT_CUSTOM,
    TEXT_ENCODING,
)
from harness.tools.function_tool import FunctionTool
from harness.tools.registry import ToolRegistry

Stamp = tuple[int, int]


def _import(path: Path, tools_dir: Path) -> ModuleType:
    """Run the file as a fresh module; the source is always read, never a cached ``.pyc``."""
    if str(tools_dir) not in sys.path:  # tool files may import their neighbours
        sys.path.append(str(tools_dir))
    module = ModuleType(CUSTOM_MODULE_PREFIX + path.stem)
    module.__file__ = str(path)
    code = compile(path.read_text(encoding=TEXT_ENCODING), str(path), "exec")
    exec(code, module.__dict__)  # noqa: S102 - running the user's own tool file is the feature
    return module


class CustomTools:
    """Keeps the registry in step with the tool files of one directory."""

    def __init__(self, tools_dir: Path) -> None:
        self._dir = tools_dir
        self._stamps: dict[Path, Stamp] = {}
        self._names: dict[Path, list[str]] = {}
        self.problems: dict[Path, str] = {}

    def forget(self) -> None:
        """Make the next refresh import every file again."""
        self._stamps.clear()

    def _files(self) -> dict[Path, Stamp]:
        found: dict[Path, Stamp] = {}
        try:
            paths = sorted(self._dir.glob(CUSTOM_TOOL_GLOB))
        except OSError:
            return found
        for path in paths:
            if path.name.startswith(CUSTOM_PRIVATE_PREFIX):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            found[path] = (stat.st_mtime_ns, stat.st_size)
        return found

    def refresh(self, registry: ToolRegistry) -> None:
        """Register new and changed files and drop the tools of removed ones."""
        current = self._files()
        if current == self._stamps:
            return
        for path in [*self._stamps, *current]:
            if current.get(path) == self._stamps.get(path):
                continue
            for name in self._names.pop(path, []):
                registry.unregister(name)
            self.problems.pop(path, None)
            if path in current:
                self._load(path, registry)
        self._stamps = current

    def _load(self, path: Path, registry: ToolRegistry) -> None:
        try:
            module = _import(path, self._dir)
        except Exception as error:  # user code may fail in any way
            self._report(path, MSG_CUSTOM_BROKEN.format(path=path, error=repr(error)))
            return
        names: list[str] = []
        for value in vars(module).values():
            if not isinstance(value, FunctionTool):
                continue
            name = value.definition.name
            if value.definition.source is not ToolSource.CUSTOM:
                self._report(path, MSG_CUSTOM_NOT_CUSTOM.format(name=name))
                continue
            try:
                registry.register(value)
            except ToolError as error:
                self._report(path, error.message)
                continue
            names.append(name)
        self._names[path] = names

    def _report(self, path: Path, message: str) -> None:
        self.problems[path] = message
        logger.warning("custom tool file {}: {}", path.name, message)
