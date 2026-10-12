"""The config manager: layered settings, secrets, templates, reload."""

import asyncio
import getpass
import os
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from loguru import logger
from pydantic import BaseModel, JsonValue

from harness.config.consts import (
    FILE_ENCODING,
    GLOBAL_ONLY_KEYS,
    MSG_AGENT_ID_REQUIRED,
    MSG_GLOBAL_ONLY,
    MSG_LAYER_NOT_WRITABLE,
    MSG_NO_PROJECT,
    MSG_NO_STORE,
    MSG_NOT_A_SECTION,
    MSG_SECRET_MISSING,
    MSG_UNKNOWN_KEY,
    RESTART_KEYS,
)
from harness.config.layers import (
    Layer,
    Tree,
    flatten,
    get_path,
    matches_prefix,
    resolve_sources,
    set_path,
    top_level,
)
from harness.config.loader import (
    LayerLoad,
    load_file_layer,
    load_templates,
    memory_layer,
    untrusted_warning,
    validate_layers,
)
from harness.config.schema import (
    SECTION_KEYS,
    AgentTemplate,
    ConfigIssue,
    HarnessConfig,
    ResolvedConfig,
)
from harness.config.secrets import KeyringSecretStore, SecretStore, expand, secret_names
from harness.config.utils import env_tree, find_project_root, issue, write_text, write_yaml_key
from harness.core.consts import (
    CONFIG_FILENAME,
    HOOKS_FILENAME,
    PROJECT_DIRNAME,
    ConfigLayer,
    ErrorCode,
)
from harness.core.errors import ConfigError, SecretMissingError
from harness.core.logging import register_secret
from harness.core.types import AgentId
from harness.core.utils.paths import HarnessHome

type ChangeCallback = Callable[[list[str]], Awaitable[None]]
type ErrorCallback = Callable[[list[ConfigIssue]], Awaitable[None]]


class AgentLayerStore(Protocol):
    """Where the agent config layer is persisted (implemented by the memory store)."""

    async def get_config_overrides(self, agent_id: AgentId) -> dict[str, JsonValue]:
        """Return the agent's stored overrides."""
        ...

    async def set_config_overrides(
        self, agent_id: AgentId, overrides: dict[str, JsonValue]
    ) -> None:
        """Persist the agent's overrides."""
        ...


@dataclass(slots=True)
class _Snapshot:
    """Everything read from disk and the environment in one load."""

    base: list[Layer]
    env: Layer
    projects: dict[Path, Layer] = field(default_factory=dict[Path, Layer])
    errors: list[ConfigIssue] = field(default_factory=list[ConfigIssue])
    warnings: list[ConfigIssue] = field(default_factory=list[ConfigIssue])

    def absorb(self, load: LayerLoad) -> Layer:
        """Collect a load's issues and return its layer."""
        self.errors.extend(load.errors)
        self.warnings.extend(load.warnings)
        return load.layer


class FileConfigManager:
    """``ConfigManager`` backed by YAML files, the keyring, and an agent layer store."""

    def __init__(
        self,
        home: HarnessHome,
        *,
        secrets: SecretStore | None = None,
        env: Mapping[str, str] | None = None,
        store: AgentLayerStore | None = None,
        username: str | None = None,
    ) -> None:
        self._home = home
        self._secrets: SecretStore = secrets or KeyringSecretStore()
        self._env: Mapping[str, str] = os.environ if env is None else env
        self._store = store
        self._username = username
        self._agents: dict[AgentId, Tree] = {}
        self._agent_projects: dict[AgentId, Path | None] = {}
        self._subscribers: list[tuple[list[str], ChangeCallback]] = []
        self._error_subscribers: list[ErrorCallback] = []
        self._cache: dict[AgentId | None, tuple[HarnessConfig, Tree]] = {}
        self._snapshot = self._read(set())
        self._raise_on_errors(self._check(self._snapshot))

    # --- loading ---------------------------------------------------------------------

    def _defaults(self) -> Layer:
        data = HarnessConfig().model_dump(mode="json", by_alias=True)
        user = self._username if self._username is not None else getpass.getuser()
        return Layer(ConfigLayer.DEFAULT, set_path(data, "user.id", user))

    def _read(self, project_roots: set[Path]) -> _Snapshot:
        env_data, env_issues = env_tree(self._env)
        env = memory_layer(ConfigLayer.ENV, env_data).layer
        snapshot = _Snapshot(base=[self._defaults()], env=env, errors=env_issues)
        global_load = load_file_layer(
            ConfigLayer.GLOBAL, self._home.config_file, self._home.hooks_file
        )
        snapshot.base.append(snapshot.absorb(global_load))
        trusted = self._trusted_roots(snapshot.base)
        for root in sorted(project_roots):
            is_trusted = root.resolve() in trusted
            load = load_file_layer(
                ConfigLayer.PROJECT,
                root / PROJECT_DIRNAME / CONFIG_FILENAME,
                root / PROJECT_DIRNAME / HOOKS_FILENAME,
                trusted=is_trusted,
            )
            snapshot.projects[root] = snapshot.absorb(load)
            if not is_trusted:
                snapshot.warnings.append(untrusted_warning(root))
        return snapshot

    @staticmethod
    def _trusted_roots(base: list[Layer]) -> set[Path]:
        config, _ = validate_layers(base)
        if config is None:
            return set()
        return {Path(entry).expanduser().resolve() for entry in config.projects.trusted}

    def _layers(self, snapshot: _Snapshot, agent_id: AgentId | None) -> list[Layer]:
        layers = list(snapshot.base)
        if agent_id is not None:
            root = self._agent_projects.get(agent_id)
            if root is not None and root in snapshot.projects:
                layers.append(snapshot.projects[root])
            overrides = self._agents.get(agent_id)
            if overrides:
                layers.append(memory_layer(ConfigLayer.AGENT, overrides).layer)
        layers.append(snapshot.env)
        return layers

    def _check(self, snapshot: _Snapshot) -> list[ConfigIssue]:
        """Return the blocking issues of a snapshot for the global scope and every agent."""
        errors = list(snapshot.errors)
        for agent_id in (None, *self._agents):
            _, issues = validate_layers(self._layers(snapshot, agent_id))
            errors.extend(found for found in issues if found not in errors)
        return errors

    @staticmethod
    def _raise_on_errors(errors: list[ConfigIssue]) -> None:
        if errors:
            first = errors[0]
            raise ConfigError(first.message, file=first.file, line=first.line)

    def _resolved(self, agent_id: AgentId | None) -> tuple[HarnessConfig, Tree]:
        scope = agent_id if agent_id in self._agents else None
        if scope not in self._cache:
            config, issues = validate_layers(self._layers(self._snapshot, scope))
            if config is None:
                self._raise_on_errors(issues)
                raise ConfigError(MSG_UNKNOWN_KEY.format(key=""))
            self._cache[scope] = (config, config.model_dump(mode="json", by_alias=True))
        return self._cache[scope]

    # --- reading ---------------------------------------------------------------------

    def config(self, agent_id: AgentId | None = None) -> HarnessConfig:
        """Return the whole resolved config for the global scope or one agent."""
        return self._resolved(agent_id)[0]

    def get(self, key: str, agent_id: AgentId | None = None) -> JsonValue:
        """Return one value by dotted key (``"models.main.id"``).

        Raises:
            ConfigError: The key does not exist.
        """
        try:
            return get_path(self._resolved(agent_id)[1], key)
        except KeyError:
            raise ConfigError(MSG_UNKNOWN_KEY.format(key=key)) from None

    def section[T: BaseModel](self, model: type[T], agent_id: AgentId | None = None) -> T:
        """Return one config section as its Pydantic model.

        Raises:
            ConfigError: ``model`` is not a known section type.
        """
        node: object = self._resolved(agent_id)[0]
        key = SECTION_KEYS.get(model)
        if key is None:
            raise ConfigError(MSG_NOT_A_SECTION.format(model=model.__name__))
        for part in key:
            node = getattr(node, part)
        if not isinstance(node, model):
            raise ConfigError(MSG_NOT_A_SECTION.format(model=model.__name__))
        return node

    def resolved(self, agent_id: AgentId | None = None) -> ResolvedConfig:
        """Return every value with the layer it came from."""
        scope = agent_id if agent_id in self._agents else None
        values = resolve_sources(self._layers(self._snapshot, scope))
        return ResolvedConfig(agent_id=scope, values=values)

    def validate(self) -> list[ConfigIssue]:
        """Check the files on disk now; returns errors, warnings, and missing secrets."""
        snapshot = self._read(self._project_roots())
        issues = [*self._check(snapshot), *snapshot.warnings]
        _, template_issues = load_templates(self._home.templates_dir)
        issues.extend(template_issues)
        seen: set[str] = set()
        for layer in (*snapshot.base, *snapshot.projects.values()):
            for key, name in secret_names(layer.data):
                if name not in seen and self._secrets.get(name) is None:
                    seen.add(name)
                    message = MSG_SECRET_MISSING.format(name=name)
                    code = ErrorCode.CONFIG_SECRET_MISSING
                    issues.append(issue(message, file=layer.file, key=key, code=code))
        return issues

    def warnings(self) -> list[ConfigIssue]:
        """Non-blocking findings of the active config (e.g. untrusted projects)."""
        return list(self._snapshot.warnings)

    # --- agents ----------------------------------------------------------------------

    def attach_store(self, store: AgentLayerStore) -> None:
        """Attach the store that persists agent layers (set once the memory store is open)."""
        self._store = store

    async def load_agent(self, agent_id: AgentId, working_dir: Path | None) -> None:
        """Load an agent's layer and project so that ``get`` can resolve them synchronously."""
        overrides: Tree = {}
        if self._store is not None:
            overrides = await self._store.get_config_overrides(agent_id)
        root = find_project_root(working_dir, self._home.root) if working_dir else None
        self._agent_projects[agent_id] = root
        self._agents[agent_id] = overrides
        self._snapshot = self._read(self._project_roots())
        self._cache.clear()
        _, issues = validate_layers(self._layers(self._snapshot, agent_id))
        if issues:
            logger.warning("agent {} config overrides ignored: {}", agent_id, issues[0].message)
            self._agents[agent_id] = {}

    def forget_agent(self, agent_id: AgentId) -> None:
        """Drop a deleted agent's cached layer."""
        self._agents.pop(agent_id, None)
        self._agent_projects.pop(agent_id, None)
        self._cache.pop(agent_id, None)

    def project_root(self, agent_id: AgentId) -> Path | None:
        """Return the project directory that applies to an agent, if any."""
        return self._agent_projects.get(agent_id)

    def is_trusted(self, root: Path) -> bool:
        """Return whether a project directory is in ``projects.trusted``."""
        return root.resolve() in self._trusted_roots(self._snapshot.base)

    def watch_dirs(self) -> list[Path]:
        """The directories whose files make up the config: home, templates, projects."""
        projects = [root / PROJECT_DIRNAME for root in self._project_roots()]
        return [self._home.root, self._home.templates_dir, *projects]

    def _project_roots(self) -> set[Path]:
        return {root for root in self._agent_projects.values() if root is not None}

    # --- writing ---------------------------------------------------------------------

    async def set(
        self,
        key: str,
        value: JsonValue,
        layer: ConfigLayer,
        agent_id: AgentId | None = None,
    ) -> None:
        """Validate and persist one value, then notify subscribers.

        Raises:
            ConfigError: The layer is not writable, the key is not allowed in it, or the
                resulting config is invalid. Nothing is written in that case.
        """
        if layer is not ConfigLayer.GLOBAL and top_level(key) in GLOBAL_ONLY_KEYS:
            raise ConfigError(MSG_GLOBAL_ONLY.format(key=key))
        if layer is ConfigLayer.AGENT:
            await self._set_agent(key, value, agent_id)
        elif layer in (ConfigLayer.GLOBAL, ConfigLayer.PROJECT):
            path = self._layer_file(layer, agent_id)
            self._validate_candidate(key, value, layer, agent_id)
            await asyncio.to_thread(write_yaml_key, path, key, value)
            self._snapshot = self._read(self._project_roots())
        else:
            raise ConfigError(MSG_LAYER_NOT_WRITABLE.format(layer=layer.value))
        self._cache.clear()
        await self._notify([key])

    def _layer_file(self, layer: ConfigLayer, agent_id: AgentId | None) -> Path:
        if layer is ConfigLayer.GLOBAL:
            return self._home.config_file
        root = self._agent_projects.get(agent_id) if agent_id is not None else None
        if root is None:
            raise ConfigError(MSG_NO_PROJECT.format(agent_id=agent_id))
        return root / PROJECT_DIRNAME / CONFIG_FILENAME

    def _validate_candidate(
        self, key: str, value: JsonValue, layer: ConfigLayer, agent_id: AgentId | None
    ) -> None:
        candidate = memory_layer(layer, set_path({}, key, value)).layer
        order = list(ConfigLayer)
        layers = [*self._layers(self._snapshot, agent_id), candidate]
        ordered = sorted(layers, key=lambda item: order.index(item.kind))
        _, issues = validate_layers(ordered)
        self._raise_on_errors(issues)

    async def _set_agent(self, key: str, value: JsonValue, agent_id: AgentId | None) -> None:
        if agent_id is None:
            raise ConfigError(MSG_AGENT_ID_REQUIRED)
        if self._store is None:
            raise ConfigError(MSG_NO_STORE)
        updated = set_path(self._agents.get(agent_id, {}), key, value)
        previous = self._agents.get(agent_id)
        self._agents[agent_id] = updated
        _, issues = validate_layers(self._layers(self._snapshot, agent_id))
        if issues:
            if previous is None:
                del self._agents[agent_id]
            else:
                self._agents[agent_id] = previous
            self._raise_on_errors(issues)
        await self._store.set_config_overrides(agent_id, updated)

    async def set_agent_layer(self, agent_id: AgentId, overrides: Tree) -> None:
        """Replace an agent's whole layer (template copy at creation, ``template.apply``).

        Raises:
            ConfigError: The overrides do not validate; the previous layer is kept.
        """
        if self._store is None:
            raise ConfigError(MSG_NO_STORE)
        previous = self._agents.get(agent_id, {})
        self._agents[agent_id] = overrides
        _, issues = validate_layers(self._layers(self._snapshot, agent_id))
        if issues:
            self._agents[agent_id] = previous
            self._raise_on_errors(issues)
        await self._store.set_config_overrides(agent_id, overrides)
        self._cache.clear()

    # --- subscriptions and reload ----------------------------------------------------

    def subscribe(self, keys: list[str], callback: ChangeCallback) -> None:
        """Call ``callback(changed_keys)`` when any key under ``keys`` changes."""
        self._subscribers.append((keys, callback))

    def subscribe_errors(self, callback: ErrorCallback) -> None:
        """Call ``callback(issues)`` when a reload is rejected."""
        self._error_subscribers.append(callback)

    async def _notify(self, changed: list[str]) -> None:
        for keys, callback in list(self._subscribers):
            relevant = [key for key in changed if matches_prefix(key, keys)]
            if relevant:
                await callback(relevant)

    async def reload(self) -> None:
        """Re-read every file. An invalid result keeps the previous config active."""
        snapshot = await asyncio.to_thread(self._read, self._project_roots())
        errors = self._check(snapshot)
        if errors:
            logger.warning("config reload rejected: {}", errors[0].message)
            for callback in list(self._error_subscribers):
                await callback(errors)
            return
        before = self._flat_all()
        self._snapshot = snapshot
        self._cache.clear()
        after = self._flat_all()
        changed = sorted(
            key for key in before.keys() | after.keys() if before.get(key) != after.get(key)
        )
        if changed:
            await self._notify(changed)

    def _flat_all(self) -> dict[str, JsonValue]:
        flat: dict[str, JsonValue] = {}
        for agent_id in (*self._agents, None):
            flat.update(flatten(self._resolved(agent_id)[1]))
        return flat

    # --- secrets, templates, profile -------------------------------------------------

    def secret(self, name: str) -> str:
        """Return a secret from the keyring and register it for log redaction.

        Raises:
            SecretMissingError: The secret is not set.
        """
        value = self._secrets.get(name)
        if value is None:
            raise SecretMissingError(MSG_SECRET_MISSING.format(name=name))
        register_secret(value)
        return value

    async def set_secret(self, name: str, value: str) -> None:
        """Store a secret in the keyring (never in YAML)."""
        register_secret(value)
        await asyncio.to_thread(self._secrets.set, name, value)

    def expand(self, value: str) -> str:
        """Replace ``${secret:x}`` / ``${env:X}`` references in a config string.

        Raises:
            ConfigError: A reference is malformed or names an unset variable.
            SecretMissingError: A referenced secret is not set.
        """
        expanded = expand(value, self._secrets, self._env)
        if expanded != value:
            register_secret(expanded)
        return expanded

    def templates(self) -> list[AgentTemplate]:
        """Return the valid agent templates in ``templates/``."""
        templates, issues = load_templates(self._home.templates_dir)
        for found in issues:
            logger.warning("template skipped: {}", found.message)
        return templates

    def user_profile(self) -> str:
        """Return the content of ``user_profile.md`` (empty when missing)."""
        path = self._home.user_profile_file
        return path.read_text(encoding=FILE_ENCODING) if path.is_file() else ""

    async def set_user_profile(self, text: str) -> None:
        """Replace ``user_profile.md``."""
        await asyncio.to_thread(write_text, self._home.user_profile_file, text)


def requires_restart(keys: list[str]) -> bool:
    """Return whether any changed key only takes effect after a restart."""
    return any(matches_prefix(key, RESTART_KEYS) for key in keys)
