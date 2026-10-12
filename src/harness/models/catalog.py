"""The built-in model catalog and alias -> profile resolution."""

from functools import cache
from importlib import resources

from pydantic import BaseModel, ConfigDict
from ruamel.yaml import YAML

from harness.config import ModelAliasConfig
from harness.core.consts import ProviderId
from harness.core.errors import ConfigError
from harness.core.types import Capabilities, ModelPrices, ModelProfile
from harness.models.consts import (
    CATALOG_FILENAME,
    FILE_ENCODING,
    MSG_BAD_FALLBACK,
    MSG_NO_PROFILE,
)


class CatalogEntry(BaseModel):
    """One model of the built-in catalog."""

    model_config = ConfigDict(extra="forbid")
    provider: ProviderId
    id: str
    context_window: int
    max_output_tokens: int
    capabilities: Capabilities = Capabilities()
    prices: ModelPrices


class Catalog(BaseModel):
    """The whole ``catalog.yaml``."""

    model_config = ConfigDict(extra="forbid")
    models: list[CatalogEntry]

    def find(self, provider: ProviderId, model_id: str) -> CatalogEntry | None:
        """Return the entry for a provider + id."""
        for entry in self.models:
            if entry.provider == provider and entry.id == model_id:
                return entry
        return None

    def find_by_id(self, model_id: str) -> CatalogEntry | None:
        """Return the first entry with this id, whatever its provider."""
        for entry in self.models:
            if entry.id == model_id:
                return entry
        return None


@cache
def load_catalog() -> Catalog:
    """Load the catalog packaged with the harness."""
    text = resources.files(__package__).joinpath(CATALOG_FILENAME).read_text(FILE_ENCODING)
    return Catalog.model_validate(YAML(typ="safe").load(text))  # pyright: ignore[reportUnknownMemberType]


def entry_profile(entry: CatalogEntry) -> ModelProfile:
    """Build a profile for a model addressed by its full id (no alias)."""
    return ModelProfile(
        provider=entry.provider,
        id=entry.id,
        context_window=entry.context_window,
        max_output_tokens=entry.max_output_tokens,
        capabilities=entry.capabilities,
        prices=entry.prices,
    )


def build_profile(alias: str, config: ModelAliasConfig, catalog: Catalog) -> ModelProfile:
    """Combine the catalog entry of an alias with its ``profile`` override.

    Raises:
        ConfigError: The model has neither a catalog entry nor a complete override.
    """
    entry = catalog.find(config.provider, config.id)
    override = config.profile
    context_window = (override and override.context_window) or (entry and entry.context_window)
    max_output = (override and override.max_output_tokens) or (entry and entry.max_output_tokens)
    prices = (override and override.prices) or (entry and entry.prices)
    capabilities = (override and override.capabilities) or (entry and entry.capabilities)
    if not context_window or not max_output or prices is None:
        raise ConfigError(
            MSG_NO_PROFILE.format(alias=alias, provider=config.provider.value, model=config.id)
        )
    return ModelProfile(
        alias=alias,
        provider=config.provider,
        id=config.id,
        context_window=context_window,
        max_output_tokens=max_output,
        capabilities=capabilities or Capabilities(),
        prices=prices,
        fallback=list(config.fallback),
    )


def build_profiles(
    aliases: dict[str, ModelAliasConfig], catalog: Catalog
) -> dict[str, ModelProfile]:
    """Resolve every configured alias.

    Raises:
        ConfigError: An alias has no profile, or lists a fallback that is not an alias.
    """
    profiles = {alias: build_profile(alias, config, catalog) for alias, config in aliases.items()}
    for alias, profile in profiles.items():
        for fallback in profile.fallback:
            if fallback not in profiles:
                raise ConfigError(MSG_BAD_FALLBACK.format(alias=alias, fallback=fallback))
    return profiles
