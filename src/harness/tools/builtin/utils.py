"""Blocking file helpers shared by the built-in tools (run through ``asyncio.to_thread``)."""

import difflib
from pathlib import Path

from harness.tools.consts import (
    BINARY_PROBE_BYTES,
    DIFF_CONTEXT_LINES,
    DIFF_MAX_CHARS,
    PREVIEW_HEAD_LINES,
    TEXT_ENCODING,
)


def looks_binary(data: bytes) -> bool:
    """Return whether the first bytes contain a NUL, the usual sign of a binary file."""
    return b"\0" in data[:BINARY_PROBE_BYTES]


def read_text(path: Path) -> str:
    """Read a file as UTF-8, replacing undecodable bytes; newlines are kept as written."""
    return path.read_bytes().decode(TEXT_ENCODING, errors="replace")


def read_text_or_empty(path: Path) -> str:
    """Read a text file; a missing or binary file reads as empty."""
    if not path.is_file():
        return ""
    data = path.read_bytes()
    return "" if looks_binary(data) else data.decode(TEXT_ENCODING, errors="replace")


def write_text(path: Path, text: str) -> int:
    """Write ``text`` as UTF-8 without newline translation; return the size in bytes."""
    data = text.encode(TEXT_ENCODING)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return len(data)


def unified_diff(before: str, after: str, path: Path) -> str:
    """Return a bounded unified diff of two versions of one file."""
    lines = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{path.name}",
        tofile=f"b/{path.name}",
        n=DIFF_CONTEXT_LINES,
    )
    text = "".join(line if line.endswith("\n") else f"{line}\n" for line in lines)
    return text[:DIFF_MAX_CHARS]


def head(data: bytes) -> str | None:
    """Return the first lines of a text file, or ``None`` for a binary one."""
    if looks_binary(data):
        return None
    lines = data.decode(TEXT_ENCODING, errors="replace").splitlines()
    return "\n".join(lines[:PREVIEW_HEAD_LINES])
