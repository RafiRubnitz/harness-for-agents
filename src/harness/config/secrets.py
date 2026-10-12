"""Secrets: keyring access and ``${secret:x}`` / ``${env:X}`` references."""

import re
from collections.abc import Mapping
from typing import Protocol

import keyring
from pydantic import JsonValue

from harness.config.consts import (
    ENV_REF_PREFIX,
    KEYRING_SERVICE,
    MSG_BAD_REF,
    MSG_ENV_MISSING,
    MSG_SECRET_MISSING,
    REF_PATTERN,
    REF_SCAN_SKIP_KEYS,
    SECRET_MASK_FILL,
    SECRET_MASK_VISIBLE,
    SECRET_REF_PREFIX,
)
from harness.config.layers import Tree, join_key
from harness.core.errors import ConfigError, SecretMissingError

_REF = re.compile(REF_PATTERN)


class SecretStore(Protocol):
    """Where secret values live (the OS keyring in production)."""

    def get(self, name: str) -> str | None:
        """Return the secret, or ``None`` when it is not set."""
        ...

    def set(self, name: str, value: str) -> None:
        """Store the secret."""
        ...


class KeyringSecretStore:
    """Secrets in the OS keyring (Windows Credential Manager)."""

    def __init__(self, service: str = KEYRING_SERVICE) -> None:
        self._service = service

    def get(self, name: str) -> str | None:
        """Return the secret, or ``None`` when it is not set."""
        return keyring.get_password(self._service, name)

    def set(self, name: str, value: str) -> None:
        """Store the secret."""
        keyring.set_password(self._service, name, value)


def is_valid_ref(ref: str) -> bool:
    """Return whether the inside of ``${..}`` is a ``secret:`` or ``env:`` reference."""
    for prefix in (SECRET_REF_PREFIX, ENV_REF_PREFIX):
        if ref.startswith(prefix) and len(ref) > len(prefix):
            return True
    return False


def find_refs(data: JsonValue, prefix: str = "") -> list[tuple[str, str]]:
    """Return ``(dotted key, reference body)`` for every ``${..}`` in string values."""
    found: list[tuple[str, str]] = []
    if isinstance(data, str):
        found.extend((prefix, match.group(1)) for match in _REF.finditer(data))
    elif isinstance(data, dict):
        for name, value in data.items():
            found.extend(find_refs(value, join_key(prefix, name)))
    elif isinstance(data, list):
        for item in data:
            found.extend(find_refs(item, prefix))
    return found


def scan_refs(data: Tree) -> list[tuple[str, str]]:
    """Find references in a config tree, skipping sections that hold shell commands."""
    scanned = {name: value for name, value in data.items() if name not in REF_SCAN_SKIP_KEYS}
    return find_refs(scanned)


def bad_refs(data: Tree) -> list[tuple[str, str]]:
    """Return ``(key, message)`` for every reference without a valid prefix."""
    return [
        (key, MSG_BAD_REF.format(ref=ref, key=key))
        for key, ref in scan_refs(data)
        if not is_valid_ref(ref)
    ]


def secret_names(data: Tree) -> list[tuple[str, str]]:
    """Return ``(key, secret name)`` for every ``${secret:..}`` reference."""
    return [
        (key, ref.removeprefix(SECRET_REF_PREFIX))
        for key, ref in scan_refs(data)
        if ref.startswith(SECRET_REF_PREFIX) and is_valid_ref(ref)
    ]


def expand(value: str, store: SecretStore, env: Mapping[str, str]) -> str:
    """Replace every reference in ``value`` with its secret / environment value.

    Raises:
        ConfigError: A reference has no valid prefix or names an unset variable.
        SecretMissingError: A secret is not in the keyring.
    """

    def replace(match: re.Match[str]) -> str:
        ref = match.group(1)
        if not is_valid_ref(ref):
            raise ConfigError(MSG_BAD_REF.format(ref=ref, key=value))
        if ref.startswith(SECRET_REF_PREFIX):
            name = ref.removeprefix(SECRET_REF_PREFIX)
            secret = store.get(name)
            if secret is None:
                raise SecretMissingError(MSG_SECRET_MISSING.format(name=name))
            return secret
        name = ref.removeprefix(ENV_REF_PREFIX)
        if name not in env:
            raise ConfigError(MSG_ENV_MISSING.format(name=name))
        return env[name]

    return _REF.sub(replace, value)


def mask(value: str) -> str:
    """Return a display form of a secret (``sk-...a3f``)."""
    if len(value) <= SECRET_MASK_VISIBLE * 2:
        return SECRET_MASK_FILL
    return f"{value[:SECRET_MASK_VISIBLE]}{SECRET_MASK_FILL}{value[-SECRET_MASK_VISIBLE:]}"
