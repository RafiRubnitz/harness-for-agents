"""The model catalog and alias profiles."""

import pytest

from harness.config import HarnessConfig, ModelAliasConfig, ProfileOverride
from harness.core.consts import ProviderId
from harness.core.errors import ConfigError
from harness.core.types import ModelPrices, Usage
from harness.models import consts, cost_usd
from harness.models.catalog import build_profile, build_profiles, load_catalog

UNKNOWN_ID = "some-new-model"
WINDOW = 400_000
MAX_OUTPUT = 32_000
PRICES = ModelPrices(input=2.0, output=8.0)


def test_every_default_alias_has_a_catalog_profile() -> None:
    aliases = HarnessConfig().models

    profiles = build_profiles(aliases, load_catalog())

    assert set(profiles) == set(aliases)
    assert all(profile.context_window > profile.max_output_tokens for profile in profiles.values())


def test_model_without_catalog_entry_or_profile_is_a_config_error() -> None:
    alias = ModelAliasConfig(provider=ProviderId.OPENAI, id=UNKNOWN_ID)

    with pytest.raises(ConfigError):
        build_profile("big", alias, load_catalog())


def test_profile_override_supplies_an_unknown_model() -> None:
    override = ProfileOverride(context_window=WINDOW, max_output_tokens=MAX_OUTPUT, prices=PRICES)
    alias = ModelAliasConfig(provider=ProviderId.OPENAI, id=UNKNOWN_ID, profile=override)

    profile = build_profile("big", alias, load_catalog())

    assert (profile.context_window, profile.max_output_tokens) == (WINDOW, MAX_OUTPUT)
    assert profile.prices == PRICES


def test_unknown_fallback_is_a_config_error() -> None:
    aliases = HarnessConfig().models
    first = next(iter(aliases))
    aliases[first] = aliases[first].model_copy(update={"fallback": ["missing"]})

    with pytest.raises(ConfigError):
        build_profiles(aliases, load_catalog())


def test_cost_is_usage_times_price_per_million() -> None:
    usage = Usage(input=consts.TOKENS_PER_MILLION, output=consts.TOKENS_PER_MILLION)

    assert cost_usd(usage, PRICES) == PRICES.input + PRICES.output
