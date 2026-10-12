"""The model gateway: alias resolution, streaming, retry, fallback, cost, call log."""

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass

from loguru import logger

from harness.config import ConfigManager, LimitsConfig
from harness.core.consts import ErrorKind, ProviderId, StreamEventType
from harness.core.errors import ConfigError, GatewayError
from harness.core.records import ModelCallRecord
from harness.core.types import (
    Clock,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    StreamEvent,
)
from harness.core.utils.time import elapsed_ms
from harness.models.adapters.base import ProviderAdapter
from harness.models.budget import BudgetGuard
from harness.models.call_log import CallSink
from harness.models.catalog import build_profiles, entry_profile, load_catalog
from harness.models.consts import (
    MS_PER_SECOND,
    MSG_BREAKER_OPEN,
    MSG_IDLE_TIMEOUT,
    MSG_NO_ADAPTER,
    MSG_STREAM_ENDED_EARLY,
    MSG_UNKNOWN_MODEL,
)
from harness.models.reliability import CircuitBreaker
from harness.models.utils import backoff_s, cost_usd, estimate_tokens

_CONTENT_EVENTS = frozenset(
    {
        StreamEventType.TEXT_DELTA,
        StreamEventType.THINKING_DELTA,
        StreamEventType.TOOL_CALL_START,
        StreamEventType.TOOL_CALL_DELTA,
        StreamEventType.TOOL_CALL_END,
    }
)


@dataclass(slots=True)
class _CallState:
    """What one gateway call has done so far."""

    emitted: bool = False


class DefaultModelGateway:
    """``ModelGateway`` over a set of provider adapters."""

    def __init__(
        self,
        config: ConfigManager,
        adapters: Mapping[ProviderId, ProviderAdapter],
        clock: Clock,
        sinks: Sequence[CallSink] = (),
    ) -> None:
        self._config = config
        self._adapters = dict(adapters)
        self._clock = clock
        self._sinks = list(sinks)
        self._breakers: dict[ProviderId, CircuitBreaker] = {}
        self._budget: BudgetGuard | None = None

    def add_sink(self, sink: CallSink) -> None:
        """Add a call sink (the cost ledger is attached once the memory store is open)."""
        self._sinks.append(sink)

    def set_budget(self, budget: BudgetGuard) -> None:
        """Check this budget before every call (set once the cost ledger is open)."""
        self._budget = budget

    # --- models ----------------------------------------------------------------------

    def _profiles(self) -> dict[str, ModelProfile]:
        return build_profiles(self._config.config().models, load_catalog())

    def list_models(self) -> list[ModelProfile]:
        """Return the profile of every configured alias."""
        return list(self._profiles().values())

    def resolve(self, alias_or_id: str) -> ModelProfile:
        """Resolve an alias (or a full model id) to its profile.

        Raises:
            GatewayError: The name is neither an alias nor a known model id.
        """
        try:
            profiles = self._profiles()
        except ConfigError as error:
            raise self._invalid(str(error), alias_or_id) from error
        if alias_or_id in profiles:
            return profiles[alias_or_id]
        for profile in profiles.values():
            if profile.id == alias_or_id:
                return profile
        entry = load_catalog().find_by_id(alias_or_id)
        if entry is None:
            raise self._invalid(MSG_UNKNOWN_MODEL.format(name=alias_or_id), alias_or_id)
        return entry_profile(entry)

    @staticmethod
    def _invalid(message: str, model: str) -> GatewayError:
        return GatewayError(message, kind=ErrorKind.INVALID_REQUEST, provider="", model=model)

    def _chain(self, name: str) -> list[ModelProfile]:
        primary = self.resolve(name)
        chain = [primary]
        for fallback in primary.fallback:
            profile = self.resolve(fallback)
            if all(profile.id != known.id for known in chain):
                chain.append(profile)
        return chain

    def _adapter(self, profile: ModelProfile) -> ProviderAdapter:
        adapter = self._adapters.get(profile.provider)
        if adapter is None:
            message = MSG_NO_ADAPTER.format(provider=profile.provider.value)
            raise GatewayError(
                message,
                kind=ErrorKind.INVALID_REQUEST,
                provider=profile.provider,
                model=profile.id,
            )
        return adapter

    def _breaker(self, provider: ProviderId) -> CircuitBreaker:
        return self._breakers.setdefault(provider, CircuitBreaker())

    # --- calls -----------------------------------------------------------------------

    async def complete(self, req: ModelRequest) -> ModelResponse:
        """Run one call to completion (built on ``stream``).

        Raises:
            GatewayError: The call failed after retries and fallback.
        """
        response: ModelResponse | None = None
        async for event in self.stream(req):
            if event.type is StreamEventType.END:
                response = event.response
        if response is None:
            raise GatewayError(
                MSG_STREAM_ENDED_EARLY, kind=ErrorKind.OVERLOADED, provider="", model=req.model
            )
        return response

    async def stream(self, req: ModelRequest) -> AsyncIterator[StreamEvent]:
        """Stream one call with retry and fallback; failures raise ``GatewayError``.

        Retry and fallback happen only before any content was emitted, so a consumer
        never sees the same output twice.

        Raises:
            GatewayError: The call failed after retries and fallback, or the budget is used up.
        """
        if self._budget is not None:
            await self._budget.check(req)
        chain = self._chain(req.model)
        state = _CallState()
        failure: GatewayError | None = None
        for profile in chain:
            try:
                async for event in self._call(req, profile, state):
                    yield event
            except GatewayError as error:
                failure = error
                logger.warning("model call failed on {}: {}", profile.id, error.message)
                if not error.retryable or state.emitted:
                    break
            else:
                return
        if failure is None:
            failure = self._invalid(MSG_UNKNOWN_MODEL.format(name=req.model), req.model)
        await self._record(req, chain[0], None, failure, None)
        raise failure

    async def _call(
        self, req: ModelRequest, profile: ModelProfile, state: _CallState
    ) -> AsyncIterator[StreamEvent]:
        adapter = self._adapter(profile)
        breaker = self._breaker(profile.provider)
        limits = self._config.section(LimitsConfig)
        max_tokens = min(req.max_tokens or profile.max_output_tokens, profile.max_output_tokens)
        provider_req = req.model_copy(update={"model": profile.id, "max_tokens": max_tokens})
        timeout_s = limits.timeout_ms / MS_PER_SECOND
        for attempt in range(limits.retries + 1):
            if breaker.is_open(self._clock.now()):
                raise GatewayError(
                    MSG_BREAKER_OPEN.format(provider=profile.provider.value),
                    kind=ErrorKind.OVERLOADED,
                    provider=profile.provider,
                    model=profile.id,
                )
            started = self._clock.now()
            try:
                async for event in self._events(adapter, provider_req, timeout_s, profile):
                    if event.type in _CONTENT_EVENTS:
                        state.emitted = True
                    if event.type is StreamEventType.END and event.response is not None:
                        breaker.record_success()
                        latency = elapsed_ms(started, self._clock.now())
                        response = event.response.model_copy(
                            update={
                                "cost_usd": cost_usd(event.response.usage, profile.prices),
                                "model_used": profile.id,
                                "latency_ms": latency,
                            }
                        )
                        await self._record(req, profile, response, None, latency)
                        yield event.model_copy(update={"response": response})
                    else:
                        yield event
            except GatewayError as error:
                if error.retryable:
                    breaker.record_failure(self._clock.now())
                if state.emitted or not error.retryable or attempt == limits.retries:
                    raise
                await self._clock.sleep(backoff_s(attempt, error.retry_after_s))
            else:
                return

    async def _events(
        self,
        adapter: ProviderAdapter,
        req: ModelRequest,
        timeout_s: float,
        profile: ModelProfile,
    ) -> AsyncIterator[StreamEvent]:
        """Iterate an adapter stream with an idle timeout and classified errors."""
        iterator = aiter(adapter.stream(req))
        try:
            while True:
                try:
                    async with asyncio.timeout(timeout_s):
                        event = await anext(iterator)
                except StopAsyncIteration:
                    return
                except TimeoutError:
                    raise GatewayError(
                        MSG_IDLE_TIMEOUT.format(seconds=timeout_s),
                        kind=ErrorKind.TIMEOUT,
                        provider=profile.provider,
                        model=profile.id,
                    ) from None
                except GatewayError:
                    raise
                except Exception as error:
                    raise adapter.classify_error(error) from error
                yield event
        finally:
            close = getattr(iterator, "aclose", None)
            if close is not None:
                await close()

    async def _record(
        self,
        req: ModelRequest,
        profile: ModelProfile,
        response: ModelResponse | None,
        error: GatewayError | None,
        latency_ms: int | None,
    ) -> None:
        record = ModelCallRecord(
            id=req.metadata.request_id,
            agent_id=req.metadata.agent_id,
            run_id=req.metadata.run_id,
            step=req.metadata.step,
            purpose=req.metadata.purpose,
            model_alias=req.model,
            model_used=profile.id,
            provider=profile.provider,
            usage=response.usage if response else ModelCallRecord.model_fields["usage"].default,
            cost_usd=response.cost_usd if response else 0.0,
            latency_ms=latency_ms,
            stop_reason=response.stop_reason if response else None,
            created_at=self._clock.now(),
        )
        for sink in self._sinks:
            try:
                await sink.record_call(record, req, response, error)
            except Exception as sink_error:
                logger.error("model call sink failed: {}", sink_error)

    async def count_tokens(self, req: ModelRequest) -> int:
        """Count input tokens with the provider API, or estimate locally when it fails."""
        profile = self.resolve(req.model)
        provider_req = req.model_copy(update={"model": profile.id})
        try:
            counted = await self._adapter(profile).count_tokens(provider_req)
        except GatewayError as error:
            logger.debug("provider token count failed, estimating: {}", error.message)
            counted = None
        return counted if counted is not None else estimate_tokens(req)
