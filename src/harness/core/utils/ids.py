"""Create and validate ids of the form ``prefix + ULID``."""

import re

from ulid import ULID

from harness.core.consts import MSG_BAD_ID, ULID_LENGTH

_ULID_RE = re.compile(rf"[0-9A-HJKMNP-TV-Z]{{{ULID_LENGTH}}}")


class UlidGenerator:
    """Production id generator: ``prefix`` followed by a fresh ULID."""

    def new(self, prefix: str) -> str:
        """Return a new id starting with ``prefix``."""
        return f"{prefix}{ULID()}"


def is_valid_id(value: str, prefix: str) -> bool:
    """Return whether ``value`` is ``prefix`` followed by a ULID."""
    return value.startswith(prefix) and _ULID_RE.fullmatch(value[len(prefix) :]) is not None


def require_id(value: str, prefix: str) -> str:
    """Return ``value`` if it is a valid id for ``prefix``; raise ``ValueError`` otherwise."""
    if not is_valid_id(value, prefix):
        raise ValueError(MSG_BAD_ID.format(value=value, prefix=prefix))
    return value
