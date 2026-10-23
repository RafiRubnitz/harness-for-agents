"""Logging setup: library loggers reach the sinks only from INFO up."""

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from loguru import logger

from harness.core.consts import LOG_FILENAME
from harness.core.logging import setup_logging

LIBRARY_LOGGER = "markdown_it.rules_block.table"
DEBUG_LINE = "entering table"
INFO_LINE = "library says hello"


@pytest.fixture
def stdlib_logging() -> Iterator[None]:
    """Put the root stdlib logger back the way the test found it."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    logger.remove()
    root.handlers[:] = handlers
    root.setLevel(level)


@pytest.mark.usefixtures("stdlib_logging")
def test_library_debug_records_stay_out_of_the_log_file(tmp_path: Path) -> None:
    setup_logging("INFO", tmp_path, console=False)
    library = logging.getLogger(LIBRARY_LOGGER)

    library.debug(DEBUG_LINE)
    library.info(INFO_LINE)
    logger.complete()

    assert not library.isEnabledFor(logging.DEBUG)
    written = (tmp_path / LOG_FILENAME).read_text(encoding="utf-8")
    assert INFO_LINE in written
    assert DEBUG_LINE not in written
