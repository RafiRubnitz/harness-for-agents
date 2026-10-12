"""Config file helpers: YAML reading / round-trip writing, line lookup, env parsing."""

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from pydantic import JsonValue
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.error import YAMLError

from harness.config.consts import (
    ENV_NESTING_SEPARATOR,
    ENV_PREFIX,
    ENV_RESERVED,
    FILE_ENCODING,
    MSG_ENV_VALUE,
    MSG_NOT_A_MAPPING,
    MSG_YAML_SYNTAX,
)
from harness.config.layers import Tree, set_path, split_key
from harness.config.schema import ConfigIssue
from harness.core.consts import PROJECT_DIRNAME, ErrorCode

_TMP_SUFFIX = ".tmp"


def to_plain(node: object) -> JsonValue:
    """Convert a ruamel round-trip node into plain JSON-compatible data."""
    if isinstance(node, Mapping):
        mapping = cast("Mapping[object, object]", node)
        return {str(name): to_plain(value) for name, value in mapping.items()}
    if isinstance(node, list | tuple):
        return [to_plain(item) for item in cast("Sequence[object]", node)]
    if node is None or isinstance(node, bool):
        return node
    if isinstance(node, int):
        return int(node)
    if isinstance(node, float):
        return float(node)
    return str(node)


def issue(
    message: str,
    *,
    file: str | None = None,
    line: int | None = None,
    key: str | None = None,
    code: ErrorCode = ErrorCode.CONFIG_INVALID,
) -> ConfigIssue:
    """Build a ``ConfigIssue``."""
    return ConfigIssue(file=file, line=line, key=key, code=code, message=message)


def read_yaml(path: Path) -> tuple[Tree, object | None, list[ConfigIssue]]:
    """Read a YAML mapping. Returns ``(data, round-trip document, issues)``."""
    if not path.is_file():
        return {}, None, []
    try:
        doc = cast("object", YAML().load(path.read_text(encoding=FILE_ENCODING)))  # pyright: ignore[reportUnknownMemberType]
    except YAMLError as exc:
        mark = cast("object", getattr(exc, "problem_mark", None))
        mark_line = cast("object", getattr(mark, "line", None))
        line = mark_line + 1 if isinstance(mark_line, int) else None
        detail = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        return {}, None, [issue(MSG_YAML_SYNTAX.format(detail=detail), file=str(path), line=line)]
    if doc is None:
        return {}, None, []
    plain = to_plain(doc)
    if not isinstance(plain, dict):
        return {}, None, [issue(MSG_NOT_A_MAPPING, file=str(path), line=1)]
    return plain, doc, []


def line_of(doc: object | None, parts: Sequence[str | int]) -> int | None:
    """Return the 1-based line of the deepest part of ``parts`` found in a round-trip doc."""
    node: object | None = doc
    line: int | None = None
    for part in parts:
        positions = cast("Mapping[object, Sequence[int]] | None", _positions(node))
        if positions is None or part not in positions:
            break
        line = positions[part][0] + 1
        node = _child(node, part)
    return line


def _positions(node: object | None) -> object | None:
    marks = getattr(node, "lc", None)
    return getattr(marks, "data", None)


def _child(node: object | None, part: str | int) -> object | None:
    if isinstance(node, Mapping):
        return cast("Mapping[object, object]", node).get(part)
    if isinstance(node, list) and isinstance(part, int):
        items = cast("list[object]", node)
        return items[part] if part < len(items) else None
    return None


def write_yaml_key(path: Path, key: str, value: JsonValue) -> None:
    """Set one dotted key in a YAML file, keeping its comments and order."""
    yaml = YAML()
    doc: object = None
    if path.is_file():
        doc = cast("object", yaml.load(path.read_text(encoding=FILE_ENCODING)))  # pyright: ignore[reportUnknownMemberType]
    root = doc if isinstance(doc, CommentedMap) else CommentedMap()
    node = root
    parts = split_key(key)
    for part in parts[:-1]:
        child = cast("object", node.get(part))  # pyright: ignore[reportUnknownMemberType]
        if not isinstance(child, CommentedMap):
            child = CommentedMap()
            node[part] = child
        node = child
    node[parts[-1]] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + _TMP_SUFFIX)
    with tmp.open("w", encoding=FILE_ENCODING) as handle:
        yaml.dump(root, handle)  # pyright: ignore[reportUnknownMemberType]
    os.replace(tmp, path)


def write_text(path: Path, text: str) -> None:
    """Write a text file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + _TMP_SUFFIX)
    tmp.write_text(text, encoding=FILE_ENCODING)
    os.replace(tmp, path)


def env_tree(env: Mapping[str, str]) -> tuple[Tree, list[ConfigIssue]]:
    """Build the env layer from ``HARNESS_<SECTION>__<KEY>`` variables."""
    tree: Tree = {}
    issues: list[ConfigIssue] = []
    for name in sorted(env):
        if not name.startswith(ENV_PREFIX) or name in ENV_RESERVED:
            continue
        parts = name.removeprefix(ENV_PREFIX).lower().split(ENV_NESTING_SEPARATOR)
        try:
            value = to_plain(cast("object", YAML(typ="safe").load(env[name])))  # pyright: ignore[reportUnknownMemberType]
        except YAMLError:
            issues.append(issue(MSG_ENV_VALUE.format(name=name), key=name))
            continue
        tree = set_path(tree, ".".join(parts), value)
    return tree, issues


def find_project_root(working_dir: Path, home_root: Path) -> Path | None:
    """Return the nearest ancestor of ``working_dir`` (itself included) holding ``.harness/``.

    The harness home itself never counts as a project.
    """
    start = working_dir.resolve()
    home = home_root.resolve()
    for candidate in (start, *start.parents):
        marker = candidate / PROJECT_DIRNAME
        if marker.is_dir() and marker.resolve() != home:
            return candidate
    return None
