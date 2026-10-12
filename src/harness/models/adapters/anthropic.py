"""The Anthropic adapter (thinking, prompt caching, full tool use)."""

from collections.abc import AsyncIterator, Mapping
from typing import Any, cast

import anthropic
import httpx2
from pydantic import JsonValue

from harness.core.consts import ErrorKind, ProviderId
from harness.core.errors import GatewayError
from harness.core.types import Capabilities, ModelRequest, ModelResponse, StreamEvent
from harness.core.utils.tool_names import build_wire_map
from harness.models.adapters.anthropic_wire import (
    StreamParser,
    content_block,
    parse_usage,
    stop_reason,
    to_provider,
)
from harness.models.adapters.base import AdapterSettings, SettingsProvider
from harness.models.catalog import load_catalog
from harness.models.consts import (
    ANTHROPIC_OVERFLOW_MARKERS,
    HTTP_STATUS_OVERLOADED,
    HTTP_STATUS_SERVER_ERROR_MIN,
    HTTP_STATUS_TIMEOUT,
    HTTP_STATUS_TOO_LARGE,
    RETRY_AFTER_HEADER,
)

_KIND_BY_ERROR_TYPE = {
    "rate_limit_error": ErrorKind.RATE_LIMIT,
    "overloaded_error": ErrorKind.OVERLOADED,
    "api_error": ErrorKind.OVERLOADED,
    "timeout_error": ErrorKind.TIMEOUT,
    "authentication_error": ErrorKind.AUTH,
    "permission_error": ErrorKind.AUTH,
}
_COUNT_KEYS = ("model", "messages", "system", "tools", "tool_choice", "thinking")
_UNKNOWN_MODEL = "unknown"


def _to_mapping(raw: object) -> Mapping[str, object]:
    if isinstance(raw, Mapping):
        return cast("Mapping[str, object]", raw)
    to_dict = getattr(raw, "to_dict", None)
    if callable(to_dict):
        return cast("Mapping[str, object]", to_dict())
    return {}


def _error_type(body: object) -> str | None:
    if not isinstance(body, Mapping):
        return None
    error = cast("Mapping[str, object]", body).get("error")
    if not isinstance(error, Mapping):
        return None
    kind = cast("Mapping[str, object]", error).get("type")
    return kind if isinstance(kind, str) else None


def _retry_after(response: httpx2.Response) -> float | None:
    value = response.headers.get(RETRY_AFTER_HEADER)
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _status_kind(error: anthropic.APIStatusError) -> ErrorKind:
    by_type = _KIND_BY_ERROR_TYPE.get(_error_type(error.body) or "")
    status = error.status_code
    if isinstance(error, anthropic.RateLimitError):
        return ErrorKind.RATE_LIMIT
    if isinstance(error, anthropic.AuthenticationError | anthropic.PermissionDeniedError):
        return ErrorKind.AUTH
    if by_type is not None:
        return by_type
    if status == HTTP_STATUS_OVERLOADED or status >= HTTP_STATUS_SERVER_ERROR_MIN:
        return ErrorKind.OVERLOADED
    if status == HTTP_STATUS_TIMEOUT:
        return ErrorKind.TIMEOUT
    message = str(error).lower()
    if status == HTTP_STATUS_TOO_LARGE or any(m in message for m in ANTHROPIC_OVERFLOW_MARKERS):
        return ErrorKind.CONTEXT_OVERFLOW
    return ErrorKind.INVALID_REQUEST


class AnthropicAdapter:
    """``ProviderAdapter`` for the Anthropic Messages API."""

    id = ProviderId.ANTHROPIC

    def __init__(
        self, settings: SettingsProvider, http_client: httpx2.AsyncClient | None = None
    ) -> None:
        self._settings = settings
        self._http_client = http_client
        self._client: anthropic.AsyncAnthropic | None = None
        self._client_settings: AdapterSettings | None = None

    def _get_client(self) -> anthropic.AsyncAnthropic:
        settings = self._settings()
        if self._client is None or settings != self._client_settings:
            self._client = anthropic.AsyncAnthropic(
                api_key=settings.api_key,
                base_url=settings.base_url,
                timeout=settings.timeout_s,
                max_retries=0,  # retries are the gateway's job
                http_client=self._http_client,
            )
            self._client_settings = settings
        return self._client

    def capabilities(self, model_id: str) -> Capabilities:
        """Return the catalog capabilities of a model (defaults when unknown)."""
        entry = load_catalog().find(self.id, model_id)
        return entry.capabilities if entry else Capabilities()

    def to_provider(self, req: ModelRequest) -> dict[str, JsonValue]:
        """Build the Messages API request body."""
        return to_provider(req)

    def from_provider(self, raw: object) -> ModelResponse:
        """Convert a complete Messages API response (SDK object or plain mapping)."""
        message = _to_mapping(raw)
        blocks = message.get("content")
        content = [
            block
            for item in (cast("list[object]", blocks) if isinstance(blocks, list) else [])
            if (block := content_block(_to_mapping(item), {})) is not None
        ]
        model = message.get("model")
        return ModelResponse(
            content=content,
            stop_reason=stop_reason(message.get("stop_reason")),
            usage=parse_usage(_to_mapping(message.get("usage"))),
            model_used=model if isinstance(model, str) else _UNKNOWN_MODEL,
        )

    async def stream(self, req: ModelRequest) -> AsyncIterator[StreamEvent]:
        """Stream one call.

        Raises:
            GatewayError: Any provider or network failure, classified.
        """
        body = cast("dict[str, Any]", self.to_provider(req))
        parser = StreamParser(req.model, build_wire_map([tool.name for tool in req.tools or []]))
        try:
            client = self._get_client()
            raw_stream = cast(
                "AsyncIterator[object]", await client.messages.create(**body, stream=True)
            )
            async for raw_event in raw_stream:
                for event in parser.feed(_to_mapping(raw_event)):
                    yield event
            if not parser.finished:
                parser.response()  # raises: the stream ended without message_stop
        except GatewayError:
            raise
        except (anthropic.AnthropicError, httpx2.HTTPError) as error:
            raise self._classify(error, req.model) from error

    def classify_error(self, e: Exception) -> GatewayError:
        """Classify an SDK / network exception into a ``GatewayError``."""
        return self._classify(e, _UNKNOWN_MODEL)

    def _classify(self, e: Exception, model: str) -> GatewayError:
        if isinstance(e, GatewayError):
            return e
        retry_after: float | None = None
        if isinstance(e, anthropic.APIStatusError):
            kind = _status_kind(e)
            retry_after = _retry_after(e.response)
        elif isinstance(e, anthropic.APIConnectionError | httpx2.TransportError | TimeoutError):
            kind = ErrorKind.TIMEOUT
        else:
            kind = ErrorKind.INVALID_REQUEST
        return GatewayError(
            str(e) or type(e).__name__,
            kind=kind,
            provider=self.id,
            model=model,
            retry_after_s=retry_after,
        )

    async def count_tokens(self, req: ModelRequest) -> int | None:
        """Count input tokens with the provider's counting endpoint.

        Raises:
            GatewayError: The counting call failed.
        """
        body = self.to_provider(req)
        params = cast("dict[str, Any]", {key: body[key] for key in _COUNT_KEYS if key in body})
        try:
            result = await self._get_client().messages.count_tokens(**params)
        except (anthropic.AnthropicError, httpx2.HTTPError) as error:
            raise self._classify(error, req.model) from error
        return result.input_tokens
