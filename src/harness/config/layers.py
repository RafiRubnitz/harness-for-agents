"""Config layers: deep merge, accumulating lists, and per-value source tracking."""

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import JsonValue

from harness.config.consts import (
    ACCUMULATING_KEYS,
    HOOK_SPECS_KEY,
    HOOKS_KEY,
    KEY_SEPARATOR,
    PERMISSIONS_KEY,
    RULE_LAYER_FIELD,
    RULES_KEY,
    SPEC_SCOPE_FIELD,
    SPEC_SOURCE_FIELD,
)
from harness.config.schema import ResolvedValue
from harness.core.consts import ConfigLayer, HookScope

type Tree = dict[str, JsonValue]

_SCOPE_BY_LAYER = {
    ConfigLayer.PROJECT: HookScope.PROJECT,
    ConfigLayer.AGENT: HookScope.AGENT,
}


@dataclass(frozen=True, slots=True)
class Layer:
    """One config layer: its (tagged) data and where it came from."""

    kind: ConfigLayer
    data: Tree
    file: str | None = None
    doc: object | None = None


def join_key(prefix: str, part: str) -> str:
    """Append one segment to a dotted key."""
    return f"{prefix}{KEY_SEPARATOR}{part}" if prefix else part


def split_key(key: str) -> list[str]:
    """Split a dotted key into its segments."""
    return key.split(KEY_SEPARATOR)


def normalise(data: Tree) -> Tree:
    """Rewrite the ``hooks: [..]`` list form into ``hooks: {specs: [..]}``."""
    hooks = data.get(HOOKS_KEY)
    if isinstance(hooks, list):
        return {**data, HOOKS_KEY: {HOOK_SPECS_KEY: hooks}}
    return data


def _tag_items(items: JsonValue, fields: dict[str, JsonValue]) -> JsonValue:
    if not isinstance(items, list):
        return items
    tagged: list[JsonValue] = []
    for item in items:
        tagged.append({**fields, **item} if isinstance(item, dict) else item)
    return tagged


def tag_tree(data: Tree, kind: ConfigLayer, file: str | None) -> Tree:
    """Normalise a raw layer and stamp its accumulating items with their origin."""
    tree: Tree = dict(normalise(data))
    hooks = tree.get(HOOKS_KEY)
    if isinstance(hooks, dict) and HOOK_SPECS_KEY in hooks:
        scope = _SCOPE_BY_LAYER.get(kind, HookScope.GLOBAL)
        fields: dict[str, JsonValue] = {SPEC_SCOPE_FIELD: scope.value, SPEC_SOURCE_FIELD: file}
        tagged_hooks: Tree = dict(hooks)
        tagged_hooks[HOOK_SPECS_KEY] = _tag_items(hooks[HOOK_SPECS_KEY], fields)
        tree[HOOKS_KEY] = tagged_hooks
    permissions = tree.get(PERMISSIONS_KEY)
    if isinstance(permissions, dict) and RULES_KEY in permissions:
        rules = permissions[RULES_KEY]
        if isinstance(rules, list):
            stamped: list[JsonValue] = [
                {**rule, RULE_LAYER_FIELD: kind.value} if isinstance(rule, dict) else rule
                for rule in rules
            ]
            tagged_permissions: Tree = dict(permissions)
            tagged_permissions[RULES_KEY] = stamped
            tree[PERMISSIONS_KEY] = tagged_permissions
    return tree


def deep_merge(base: Tree, over: Tree, prefix: str = "") -> Tree:
    """Merge ``over`` onto ``base``: dicts deeply, lists replaced, accumulating keys appended."""
    merged: Tree = dict(base)
    for name, value in over.items():
        key = join_key(prefix, name)
        current = merged.get(name)
        if isinstance(value, dict) and isinstance(current, dict):
            merged[name] = deep_merge(current, value, key)
        elif key in ACCUMULATING_KEYS and isinstance(value, list) and isinstance(current, list):
            merged[name] = [*current, *value]
        else:
            merged[name] = value
    return merged


def merge_layers(layers: Sequence[Layer]) -> Tree:
    """Merge layers from lowest to highest precedence."""
    merged: Tree = {}
    for layer in layers:
        merged = deep_merge(merged, layer.data)
    return merged


def flatten(data: Tree, prefix: str = "") -> dict[str, JsonValue]:
    """Flatten a tree into dotted leaf keys (empty dicts are dropped)."""
    flat: dict[str, JsonValue] = {}
    for name, value in data.items():
        key = join_key(prefix, name)
        if isinstance(value, dict):
            flat.update(flatten(value, key))
        else:
            flat[key] = value
    return flat


def resolve_sources(layers: Sequence[Layer]) -> dict[str, ResolvedValue]:
    """Return each dotted key with the layer that supplied it (per item for accumulating lists)."""
    values: dict[str, ResolvedValue] = {}
    counters: dict[str, int] = {}
    for layer in layers:
        for key, value in flatten(layer.data).items():
            if key in ACCUMULATING_KEYS and isinstance(value, list):
                for item in value:
                    index = counters.get(key, 0)
                    counters[key] = index + 1
                    values[f"{key}[{index}]"] = ResolvedValue(
                        value=item, layer=layer.kind, file=layer.file
                    )
            else:
                values[key] = ResolvedValue(value=value, layer=layer.kind, file=layer.file)
    return values


def has_path(data: Tree, parts: Sequence[str]) -> bool:
    """Return whether the tree defines the given path."""
    node: JsonValue = data
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def get_path(data: Tree, key: str) -> JsonValue:
    """Read a dotted key.

    Raises:
        KeyError: The key does not exist.
    """
    node: JsonValue = data
    for part in split_key(key):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(key)
        node = node[part]
    return node


def set_path(data: Tree, key: str, value: JsonValue) -> Tree:
    """Return a copy of the tree with a dotted key set."""
    head, _, rest = key.partition(KEY_SEPARATOR)
    updated: Tree = dict(data)
    if not rest:
        updated[head] = value
        return updated
    child = data.get(head)
    updated[head] = set_path(child if isinstance(child, dict) else {}, rest, value)
    return updated


def top_level(key: str) -> str:
    """Return the first segment of a dotted key."""
    return split_key(key)[0]


def matches_prefix(key: str, prefixes: Sequence[str] | frozenset[str]) -> bool:
    """Return whether ``key`` equals, is under, or is a parent of any of the prefixes."""
    for prefix in prefixes:
        if key == prefix or key.startswith(prefix + KEY_SEPARATOR):
            return True
        if prefix.startswith(key + KEY_SEPARATOR):
            return True
    return False
