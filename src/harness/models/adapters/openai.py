"""The OpenAI adapter (Chat Completions) and its thin DeepSeek subclass."""

from collections.abc import AsyncIterator, Mapping
from typing import Any, ClassVar, cast

import httpx2
import openai
from pydantic import JsonValue

from harness.core.consts import ErrorKind, ProviderId
from harness.core.errors import GatewayError
from harness.core.types import Capabilities, ModelRequest, ModelResponse, StreamEvent
from harness.core.utils.tool_names import build_wire_map
from harness.models.adapters.base import AdapterSettings, SettingsProvider
from harness.models.adapters.openai_wire import (
    StreamParser,
    WireOptions,
    message_content,
    parse_usage,
    stop_reason,
    to_provider,
)
from harness.models.catalog import load_catalog
from harness.models.consts import (
    HTTP_STATUS_SERVER_ERROR_MIN,
    HTTP_STATUS_TIMEOUT,
    HTTP_STATUS_TOO_LARGE,
    OPENAI_LEGACY_MAX_TOKENS_KEY,
    OPENAI_OVERFLOW_CODE,
    OPENAI_OVERFLOW_MARKERS,
    OPENAI_QUOTA_CODE,
    RETRY_AFTER_HEADER,
)

_UNKNOWN_MODEL = "unknown"
_STREAM_OPTIONS = {"include_usage": True}


def _to_mapping(raw: object) -> Mapping[str, object]:
    if isinstance(raw, Mapping):
        return cast("Mapping[str, object]", raw)
    to_dict = getattr(raw, "to_dict", None)
    if callable(to_dict):
        return cast("Mapping[str, object]", to_dict())
    return {}


def _retry_after(response: httpx2.Response) -> float | None:
    value = response.headers.get(RETRY_AFTER_HEADER)
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _status_kind(error: openai.APIStatusError) -> ErrorKind:
    status = error.status_code
    code = error.code or ""
    message = str(error).lower()
    if isinstance(error, openai.AuthenticationError | openai.PermissionDeniedError):
        return ErrorKind.AUTH
    if isinstance(error, openai.RateLimitError):
        # an empty account answers 429 too, and waiting does not refill it
        return ErrorKind.AUTH if code == OPENAI_QUOTA_CODE else ErrorKind.RATE_LIMIT
    if status >= HTTP_STATUS_SERVER_ERROR_MIN:
        return ErrorKind.OVERLOADED
    if status == HTTP_STATUS_TIMEOUT:
        return ErrorKind.TIMEOUT
    overflow = code == OPENAI_OVERFLOW_CODE or any(m in message for m in OPENAI_OVERFLOW_MARKERS)
    if status == HTTP_STATUS_TOO_LARGE or overflow:
        return ErrorKind.CONTEXT_OVERFLOW
    return ErrorKind.INVALID_REQUEST


class OpenAIAdapter:
    """``ProviderAdapter`` for the OpenAI Chat Completions API."""

    id: ProviderId = ProviderId.OPENAI
    options: ClassVar[WireOptions] = WireOptions()

    def __init__(
        self, settings: SettingsProvider, http_client: httpx2.AsyncClient | None = None
    ) -> None:
        self._settings = settings
        self._http_client = http_client
        self._client: openai.AsyncOpenAI | None = None
        self._client_settings: AdapterSettings | None = None

    def _get_client(self) -> openai.AsyncOpenAI:
        settings = self._settings()
        if self._client is None or settings != self._client_settings:
            self._client = openai.AsyncOpenAI(
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
        """Build the Chat Completions request body."""
        return to_provider(req, self.options)

    def from_provider(self, raw: object) -> ModelResponse:
        """Convert a complete chat completion (SDK object or plain mapping)."""
        completion = _to_mapping(raw)
        choices = completion.get("choices")
        listed = cast("list[object]", choices) if isinstance(choices, list) else []
        choice = _to_mapping(listed[0] if listed else None)
        model = completion.get("model")
        return ModelResponse(
            content=message_content(_to_mapping(choice.get("message")), {}),
            stop_reason=stop_reason(choice.get("finish_reason")),
            usage=parse_usage(_to_mapping(completion.get("usage"))),
            model_used=model if isinstance(model, str) else _UNKNOWN_MODEL,
        )

    async def stream(self, req: ModelRequest) -> AsyncIterator[StreamEvent]:
        """Stream one call.

        Raises:
            GatewayError: Any provider or network failure, classified.
        """
        body = cast("dict[str, Any]", self.to_provider(req))
        body["stream_options"] = dict(_STREAM_OPTIONS)
        wire_map = build_wire_map([tool.name for tool in req.tools or []])
        parser = StreamParser(req.model, wire_map, self.id)
        try:
            client = self._get_client()
            raw_stream = cast(
                "AsyncIterator[object]",
                await client.chat.completions.create(**body, stream=True),
            )
            async for chunk in raw_stream:
                for event in parser.feed(_to_mapping(chunk)):
                    yield event
            yield parser.finish()  # raises when the stream ended without a finish reason
        except GatewayError:
            raise
        except (openai.OpenAIError, httpx2.HTTPError) as error:
            raise self._classify(error, req.model) from error

    def classify_error(self, e: Exception) -> GatewayError:
        """Classify an SDK / network exception into a ``GatewayError``."""
        return self._classify(e, _UNKNOWN_MODEL)

    def _classify(self, e: Exception, model: str) -> GatewayError:
        if isinstance(e, GatewayError):
            return e
        retry_after: float | None = None
        if isinstance(e, openai.APIStatusError):
            kind = _status_kind(e)
            retry_after = _retry_after(e.response)
        elif isinstance(e, openai.APIConnectionError | httpx2.TransportError | TimeoutError):
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
        """Chat Completions has no counting endpoint: the gateway estimates locally."""
        return None


class DeepSeekAdapter(OpenAIAdapter):
    """DeepSeek speaks the OpenAI format; only a few request fields differ.

    The reasoning model's ``reasoning_content`` is read by the shared stream parser and
    is never sent back, as the DeepSeek API requires.
    """

    id: ProviderId = ProviderId.DEEPSEEK
    options: ClassVar[WireOptions] = WireOptions(
        provider=ProviderId.DEEPSEEK,
        max_tokens_key=OPENAI_LEGACY_MAX_TOKENS_KEY,
        reasoning_effort=False,
        json_schema=False,
    )
