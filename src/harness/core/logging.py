"""Logging setup. The only module that configures loguru sinks."""

import inspect
import logging
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from harness.core.consts import (
    CONSOLE_FORMAT,
    LOG_FILE_LEVEL,
    LOG_FILENAME,
    LOG_RETENTION,
    LOG_ROTATION,
    REDACTED,
    SECRET_PATTERNS,
)

if TYPE_CHECKING:
    from loguru import Record

_SECRET_RE = re.compile("|".join(f"(?:{pattern})" for pattern in SECRET_PATTERNS))
_known_secrets: set[str] = set()


def register_secret(value: str) -> None:
    """Make ``value`` redacted in every later log record."""
    if value:
        _known_secrets.add(value)


def redact(text: str) -> str:
    """Return ``text`` with API keys and registered secrets replaced."""
    for secret in _known_secrets:
        text = text.replace(secret, REDACTED)
    return _SECRET_RE.sub(REDACTED, text)


def redact_record(record: "Record") -> None:
    """Loguru patcher: redact secrets in the message before any sink writes it."""
    record["message"] = redact(record["message"])


class InterceptHandler(logging.Handler):
    """Route standard-library logging (httpx, lancedb, anthropic) into loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        """Forward one stdlib record to loguru, keeping level and caller depth."""
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = inspect.currentframe(), 0
        while frame is not None and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(level: str, log_dir: Path, *, console: bool = True) -> None:
    """Configure console and JSONL file sinks with secret redaction."""
    logger.remove()
    if console:
        logger.add(sys.stderr, level=level, format=CONSOLE_FORMAT, enqueue=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / LOG_FILENAME,
        level=LOG_FILE_LEVEL,
        serialize=True,
        rotation=LOG_ROTATION,
        retention=LOG_RETENTION,
        enqueue=True,
        diagnose=False,
    )
    logger.configure(patcher=redact_record)
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
