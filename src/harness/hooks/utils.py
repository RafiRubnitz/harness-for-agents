"""Pure helpers of the hooks manager: matching and command-line placeholders (05 §6)."""

import os
import re
import shlex
from fnmatch import fnmatchcase
from pathlib import PurePath

from pydantic import JsonValue

from harness.core.errors import HookError
from harness.core.hook_types import HookInput, HookSpec
from harness.hooks import consts

_QUOTES = ('"', "'")


def _dig(node: JsonValue, path: list[str]) -> JsonValue:
    """Follow ``path`` into nested JSON objects; ``KeyError`` when a step is missing."""
    for part in path:
        if not isinstance(node, dict) or part not in node:
            raise KeyError(part)
        node = node[part]
    return node


def _tool_path(payload: dict[str, JsonValue]) -> str | None:
    try:
        value = _dig(payload, [consts.PAYLOAD_INPUT, consts.PAYLOAD_PATH])
    except KeyError:
        return None
    return value if isinstance(value, str) else None


def _path_matches(pattern: str, path: str) -> bool:
    """Glob the whole path and, for a bare pattern such as ``*.py``, the file name."""
    normal = PurePath(path).as_posix()
    return fnmatchcase(normal, pattern) or fnmatchcase(PurePath(normal).name, pattern)


def _text_at(payload: dict[str, JsonValue], path: list[str]) -> str | None:
    try:
        value = _dig(payload, path)
    except KeyError:
        return None
    return value if isinstance(value, str) else None


def matches(spec: HookSpec, inp: HookInput, template: str | None) -> bool:
    """Whether every key of the hook's ``match`` fits this event (05 §6)."""
    for key, pattern in spec.match.items():
        if key == consts.MATCH_TOOL:
            tool = inp.payload.get(consts.PAYLOAD_TOOL)
            fits = isinstance(tool, str) and fnmatchcase(tool, pattern)
        elif key == consts.MATCH_AGENT:
            fits = fnmatchcase(inp.agent_name, pattern)
        elif key == consts.MATCH_TEMPLATE:
            fits = template == pattern
        elif key == consts.MATCH_PATH_GLOB:
            path = _tool_path(inp.payload)
            fits = path is not None and _path_matches(pattern, path)
        elif key == consts.MATCH_SOURCE_TYPE:
            where = [consts.PAYLOAD_ORIGIN, consts.PAYLOAD_SOURCE, consts.PAYLOAD_TYPE]
            fits = _text_at(inp.payload, where) == pattern
        else:
            fits = False
        if not fits:
            return False
    return True


def template_in(payload: dict[str, JsonValue]) -> str | None:
    """The template named by an ``agent_created`` payload, if any."""
    return _text_at(payload, [consts.PAYLOAD_SPEC, consts.PAYLOAD_TEMPLATE])


def split_command(run: str) -> list[str]:
    """Split a command line into arguments without a shell.

    Raises:
        ValueError: The line has unbalanced quotes or is empty.
    """
    posix = os.name != "nt"
    parts = shlex.split(run, posix=posix)
    if not posix:  # non-POSIX splitting keeps the quotes; backslashes stay as they are
        parts = [p[1:-1] if len(p) > 1 and p[0] == p[-1] and p[0] in _QUOTES else p for p in parts]
    if not parts:
        raise ValueError(consts.MSG_EMPTY_RUN)
    return parts


def _value(name: str, inp: HookInput, hook: str) -> str:
    head, _, rest = name.partition(consts.PATH_SEPARATOR)
    if head == consts.PLACEHOLDER_PAYLOAD and rest:
        try:
            found = _dig(inp.payload, rest.split(consts.PATH_SEPARATOR))
        except KeyError:
            raise HookError(consts.MSG_MISSING_FIELD.format(name=name), hook=hook) from None
        return found if isinstance(found, str) else str(found)
    if name in consts.PLACEHOLDER_FIELDS:
        found = getattr(inp, name)
        if found is None:
            raise HookError(consts.MSG_MISSING_FIELD.format(name=name), hook=hook)
        return str(found.value) if name == "event" else str(found)
    raise HookError(consts.MSG_UNKNOWN_PLACEHOLDER.format(name=name), hook=hook)


def fill(argument: str, inp: HookInput, hook: str) -> str:
    """Replace the placeholders inside one already-split argument.

    A value never becomes more than one argument, whatever it contains.

    Raises:
        HookError: A placeholder is unknown or names a field this event lacks.
    """

    def replace(found: re.Match[str]) -> str:
        text = found.group(0)
        if text == consts.OPEN_BRACE:
            return "{"
        if text == consts.CLOSE_BRACE:
            return "}"
        return _value(found.group(1), inp, hook)

    return consts.PLACEHOLDER.sub(replace, argument)


def placeholders(argument: str) -> list[str]:
    """The placeholder names used in one argument."""
    return [m.group(1) for m in consts.PLACEHOLDER.finditer(argument) if m.group(1) is not None]


def known_placeholder(name: str) -> bool:
    """Whether a placeholder name is one the manager can fill."""
    head, _, rest = name.partition(consts.PATH_SEPARATOR)
    return (head == consts.PLACEHOLDER_PAYLOAD and bool(rest)) or name in consts.PLACEHOLDER_FIELDS
