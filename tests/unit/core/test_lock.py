"""The single-core lock (02 §3)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from harness.core.errors import CoreLockedError
from harness.core.lock import CoreLock

HOLD_AND_DIE = """
import os, sys
from pathlib import Path
from harness.core.lock import CoreLock
CoreLock(Path(sys.argv[1])).acquire()
sys.stdout.write(str(os.getpid()))
sys.stdout.flush()
os._exit(0)
"""


@pytest.fixture
def lock_path(tmp_path: Path) -> Path:
    return tmp_path / "home" / "core.lock"


def test_acquire_records_the_owner_pid(lock_path: Path) -> None:
    lock = CoreLock(lock_path)

    lock.acquire()
    pid = lock.owner_pid()
    lock.release()

    assert pid == os.getpid()


def test_second_holder_is_refused_until_release(lock_path: Path) -> None:
    first, second = CoreLock(lock_path), CoreLock(lock_path)
    first.acquire()

    with pytest.raises(CoreLockedError, match=lock_path.name.split(".")[0]):
        second.acquire()
    first.release()
    second.acquire()

    assert (first.held, second.held) == (False, True)
    second.release()


def test_context_manager_releases_on_error(lock_path: Path) -> None:
    lock = CoreLock(lock_path)

    with pytest.raises(RuntimeError), lock:
        raise RuntimeError

    with CoreLock(lock_path) as again:
        assert again.held
    assert not lock.held


def test_release_twice_is_harmless(lock_path: Path) -> None:
    lock = CoreLock(lock_path)
    lock.acquire()

    lock.release()
    lock.release()

    assert not lock.held


@pytest.mark.slow
def test_lock_of_a_dead_process_is_free(lock_path: Path) -> None:
    dead = subprocess.run(  # noqa: S603 - fixed interpreter and script
        [sys.executable, "-c", HOLD_AND_DIE, str(lock_path)],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    lock = CoreLock(lock_path)

    lock.acquire()
    owner = lock.owner_pid()
    lock.release()

    assert int(dead.stdout) != os.getpid()
    assert owner == os.getpid()
