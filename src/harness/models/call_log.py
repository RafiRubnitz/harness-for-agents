"""Model call sinks: the JSONL log of every request and response (decision D8)."""

import asyncio
import json
from pathlib import Path
from typing import Protocol

from harness.core.errors import GatewayError
from harness.core.logging import redact
from harness.core.records import ModelCallRecord
from harness.core.types import ModelRequest, ModelResponse
from harness.models.consts import FILE_ENCODING


class CallSink(Protocol):
    """Receives one entry per gateway call (after retries and fallback)."""

    async def record_call(
        self,
        record: ModelCallRecord,
        request: ModelRequest,
        response: ModelResponse | None,
        error: GatewayError | None,
    ) -> None:
        """Persist the call."""
        ...


class JsonlCallLog:
    """Appends the full request, response and usage of each call to a JSONL file."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()

    async def record_call(
        self,
        record: ModelCallRecord,
        request: ModelRequest,
        response: ModelResponse | None,
        error: GatewayError | None,
    ) -> None:
        """Append one line; secrets are redacted."""
        entry = {
            "call": record.model_dump(mode="json"),
            "request": request.model_dump(mode="json"),
            "response": response.model_dump(mode="json") if response else None,
            "error": (
                {"kind": error.kind.value, "code": error.code.value, "message": error.message}
                if error
                else None
            ),
        }
        line = redact(json.dumps(entry, ensure_ascii=False))
        async with self._lock:
            await asyncio.to_thread(self._append, line)

    def _append(self, line: str) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding=FILE_ENCODING) as handle:
            handle.write(line + "\n")
