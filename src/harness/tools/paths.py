"""Path normalisation and the checks the permission policy runs on paths (08 §5.3)."""

import os
import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

from harness.core.utils.paths import HarnessHome, is_within
from harness.tools.consts import (
    GLOB_ANY_DEPTH,
    PROTECTED_DIR_NAMES,
    PROTECTED_SYSTEM_DIRS,
    WORKING_DIR_PLACEHOLDER,
)

_CASE_INSENSITIVE = os.name == "nt"
_SEPARATOR = "/"


def resolve_path(raw: str, working_dir: str | Path) -> Path:
    """Return the absolute, symlink-free path ``raw`` points to.

    Relative paths start at the working directory. Resolving first means ``..`` and
    symlinks cannot slip past the checks that follow.
    """
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path(working_dir) / path
    return path.resolve()


def is_inside(path: Path, roots: Iterable[str | Path]) -> bool:
    """Return whether ``path`` is one of ``roots`` or below one."""
    return any(is_within(path, Path(root)) for root in roots)


def _fold(text: str) -> str:
    text = text.replace("\\", _SEPARATOR)
    return text.casefold() if _CASE_INSENSITIVE else text


@lru_cache(maxsize=512)
def _glob_regex(pattern: str) -> re.Pattern[str]:
    parts: list[str] = []
    index = 0
    while index < len(pattern):
        if pattern.startswith(GLOB_ANY_DEPTH, index):
            index += len(GLOB_ANY_DEPTH)
            if pattern.startswith(_SEPARATOR, index):
                # "**/" also matches no directory at all.
                parts.append(f"(?:.*{_SEPARATOR})?")
                index += 1
            else:
                parts.append(".*")
            continue
        char = pattern[index]
        if char == "*":
            parts.append(f"[^{_SEPARATOR}]*")
        elif char == "?":
            parts.append(f"[^{_SEPARATOR}]")
        else:
            parts.append(re.escape(char))
        index += 1
    return re.compile("".join(parts), re.DOTALL)


def path_matches(pattern: str, path: Path, working_dir: str | Path) -> bool:
    """Match a rule's ``path`` glob: ``*`` stays inside one directory, ``**`` crosses them."""
    expanded = pattern.replace(WORKING_DIR_PLACEHOLDER, Path(working_dir).as_posix())
    return _glob_regex(_fold(expanded)).fullmatch(_fold(path.as_posix())) is not None


def escape_glob(text: str) -> str:
    """Return ``text`` as a glob that matches only itself (for ``fnmatch`` and path rules)."""
    return "".join(f"[{char}]" if char in "*?[" else char for char in text)


def is_protected(path: Path, home: HarnessHome, extra_globs: Iterable[str] = ()) -> bool:
    """Return whether ``path`` may never be written or deleted by a tool (08 §4.1a).

    Protected: anything under a ``.git`` / ``.harness`` directory, the harness home
    (except agent workspaces), Windows system directories and the configured extras.
    """
    if is_within(path, home.workspaces_dir):
        inspected = path.relative_to(home.workspaces_dir.resolve()).parts
    elif is_within(path, home.root):
        return True
    else:
        inspected = path.parts
    if any(_fold(part) in PROTECTED_DIR_NAMES for part in inspected):
        return True
    if any(is_within(path, Path(system)) for system in PROTECTED_SYSTEM_DIRS):
        return True
    return any(path_matches(glob, path, path.anchor) for glob in extra_globs)
