"""The single-core lock: only one process may run the core at a time (02 §3)."""

import os
import sys
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Self

from harness.core.consts import CORE_LOCK_OFFSET, MSG_CORE_LOCKED
from harness.core.errors import CoreLockedError

_LOCK_BYTES = 1


def _lock(handle: BinaryIO) -> None:
    """Take a non-blocking exclusive OS lock; raise ``OSError`` when it is held."""
    if sys.platform == "win32":
        import msvcrt  # noqa: PLC0415 - platform specific

        # The locked byte sits far past the PID text so other processes can still read it.
        handle.seek(CORE_LOCK_OFFSET)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, _LOCK_BYTES)
    else:
        import fcntl  # noqa: PLC0415 - platform specific

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(handle: BinaryIO) -> None:
    if sys.platform == "win32":
        import msvcrt  # noqa: PLC0415 - platform specific

        handle.seek(CORE_LOCK_OFFSET)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, _LOCK_BYTES)
    else:
        import fcntl  # noqa: PLC0415 - platform specific

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class CoreLock:
    """An OS lock on ``core.lock`` plus the owner's PID.

    The operating system drops the lock when the owning process dies, so a stale
    lock file never blocks the next start.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._handle: BinaryIO | None = None

    @property
    def held(self) -> bool:
        """Whether this object currently owns the lock."""
        return self._handle is not None

    def acquire(self) -> None:
        """Take the lock and record this process's PID.

        Raises:
            CoreLockedError: Another process (or another ``CoreLock``) holds it.
        """
        if self._handle is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle = self._path.open("a+b")
        try:
            _lock(handle)
        except OSError as error:
            handle.close()
            raise CoreLockedError(MSG_CORE_LOCKED.format(path=self._path)) from error
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()).encode())
        handle.flush()
        self._handle = handle

    def release(self) -> None:
        """Give the lock back; safe to call twice."""
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            _unlock(handle)
        finally:
            handle.close()

    def owner_pid(self) -> int | None:
        """Return the PID written by the current owner, if it can be read."""
        try:
            text = self._path.read_text(encoding="ascii").strip()
        except (OSError, ValueError):
            return None
        return int(text) if text.isdigit() else None

    def __enter__(self) -> Self:
        """Acquire on entry."""
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release on exit."""
        self.release()
