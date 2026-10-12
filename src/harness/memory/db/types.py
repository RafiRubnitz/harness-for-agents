"""Column types and naming shared by the ORM models and the migrations."""

from datetime import UTC, datetime
from typing import Final

from sqlalchemy import String
from sqlalchemy.engine import Dialect
from sqlalchemy.types import TypeDecorator

from harness.memory.consts import MSG_NAIVE_DATETIME

NAMING_CONVENTION: Final = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class UTCDateTime(TypeDecorator[datetime]):
    """Stores aware UTC datetimes as fixed-format ISO-8601 text.

    SQLite has no datetime type and the stock ``DateTime`` drops the tzinfo; fixed-width
    text also sorts correctly.
    """

    impl = String
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> str | None:
        """Convert to UTC ISO text; reject naive values."""
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError(MSG_NAIVE_DATETIME)
        return value.astimezone(UTC).isoformat(timespec="microseconds")

    def process_result_value(self, value: str | None, dialect: Dialect) -> datetime | None:
        """Parse the stored ISO text back to an aware datetime."""
        return None if value is None else datetime.fromisoformat(value)
