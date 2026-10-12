"""Model gateway (component 1): one interface to every model provider."""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol

from harness.config import ConfigManager, LimitsConfig
from harness.core.consts import ErrorKind, ProviderId
from harness.core.errors import ConfigError, GatewayError
from harness.core.types import Clock, ModelProfile, ModelRequest, ModelResponse, StreamEvent
from harness.models import consts
from harness.models.adapters.anthropic import AnthropicAdapter
from harness.models.adapters.base import AdapterSettings, ProviderAdapter, SettingsProvider
from harness.models.adapters.openai import DeepSeekAdapter, OpenAIAdapter
from harness.models.budget import BudgetGuard, BudgetNotifier, CostLedger
from harness.models.call_log import CallSink, JsonlCallLog
from harness.models.gateway import DefaultModelGateway
from harness.models.utils import cost_usd, estimate_tokens


class ModelGateway(Protocol):
    """How every other component talks to a model."""

    async def complete(self, req: ModelRequest) -> ModelResponse:
        """Run one call to completion."""
        ...

    def stream(self, req: ModelRequest) -> AsyncIterator[StreamEvent]:
        """Stream one call."""
        ...

    async def count_tokens(self, req: ModelRequest) -> int:
        """Count the input tokens of a request."""
        ...

    def list_models(self) -> list[ModelProfile]:
        """Return every configured model."""
        ...

    def resolve(self, alias_or_id: str) -> ModelProfile:
        """Resolve an alias or model id."""
        ...


def provider_settings(config: ConfigManager, provider: ProviderId) -> SettingsProvider:
    """Return a function that reads a provider's connection settings from the config."""

    def read() -> AdapterSettings:
        entry = config.config().providers.get(provider)
        if entry is None or entry.api_key is None:
            raise GatewayError(
                consts.MSG_NO_API_KEY.format(provider=provider.value),
                kind=ErrorKind.AUTH,
                provider=provider,
                model="",
            )
        try:
            api_key = config.expand(entry.api_key)
        except ConfigError as error:
            raise GatewayError(
                error.message, kind=ErrorKind.AUTH, provider=provider, model=""
            ) from error
        timeout_s = config.section(LimitsConfig).timeout_ms / consts.MS_PER_SECOND
        return AdapterSettings(api_key=api_key, base_url=entry.base_url, timeout_s=timeout_s)

    return read


def build_gateway(config: ConfigManager, clock: Clock, logs_dir: Path) -> DefaultModelGateway:
    """Build the gateway with every provider adapter of this version and the JSONL call log."""
    adapters: dict[ProviderId, ProviderAdapter] = {
        ProviderId.ANTHROPIC: AnthropicAdapter(provider_settings(config, ProviderId.ANTHROPIC)),
        ProviderId.OPENAI: OpenAIAdapter(provider_settings(config, ProviderId.OPENAI)),
        ProviderId.DEEPSEEK: DeepSeekAdapter(provider_settings(config, ProviderId.DEEPSEEK)),
    }
    call_log = JsonlCallLog(logs_dir / consts.CALL_LOG_FILENAME)
    return DefaultModelGateway(config, adapters, clock, [call_log])


__all__ = [
    "AdapterSettings",
    "AnthropicAdapter",
    "BudgetGuard",
    "BudgetNotifier",
    "CallSink",
    "CostLedger",
    "DeepSeekAdapter",
    "DefaultModelGateway",
    "JsonlCallLog",
    "ModelGateway",
    "OpenAIAdapter",
    "ProviderAdapter",
    "SettingsProvider",
    "build_gateway",
    "consts",
    "cost_usd",
    "estimate_tokens",
    "provider_settings",
]
