"""Layer merging, accumulating lists, and source tracking."""

import pytest

from harness.config.consts import RULE_LAYER_FIELD, SPEC_SCOPE_FIELD
from harness.config.layers import (
    Layer,
    Tree,
    deep_merge,
    flatten,
    get_path,
    merge_layers,
    normalise,
    resolve_sources,
    set_path,
    tag_tree,
)
from harness.core.consts import ConfigLayer, Decision, HookEvent, HookScope

LOW_RETRIES = 1
HIGH_RETRIES = 7
RULE_GLOBAL: Tree = {"tool": "shell", "decision": Decision.DENY.value}
RULE_AGENT: Tree = {"tool": "shell", "decision": Decision.ALLOW.value}
SPEC: Tree = {"name": "h", "event": HookEvent.BEFORE_TOOL_CALL.value, "run": "check.py"}


def test_deep_merge_overrides_nested_keys_and_keeps_siblings() -> None:
    base: Tree = {"limits": {"retries": LOW_RETRIES, "timeout_ms": LOW_RETRIES}}
    over: Tree = {"limits": {"retries": HIGH_RETRIES}}

    merged = deep_merge(base, over)

    assert get_path(merged, "limits.retries") == HIGH_RETRIES
    assert get_path(merged, "limits.timeout_ms") == LOW_RETRIES


def test_deep_merge_replaces_plain_lists() -> None:
    base: Tree = {"tools": {"enabled": ["a", "b"]}}
    over: Tree = {"tools": {"enabled": ["c"]}}

    merged = deep_merge(base, over)

    assert get_path(merged, "tools.enabled") == get_path(over, "tools.enabled")


@pytest.mark.acceptance("06-5.1")
def test_permission_rules_accumulate_across_layers() -> None:
    layers = [
        Layer(
            ConfigLayer.GLOBAL,
            tag_tree({"permissions": {"rules": [RULE_GLOBAL]}}, ConfigLayer.GLOBAL, None),
        ),
        Layer(
            ConfigLayer.AGENT,
            tag_tree({"permissions": {"rules": [RULE_AGENT]}}, ConfigLayer.AGENT, None),
        ),
    ]

    rules = get_path(merge_layers(layers), "permissions.rules")

    assert rules == [
        {**RULE_GLOBAL, RULE_LAYER_FIELD: ConfigLayer.GLOBAL.value},
        {**RULE_AGENT, RULE_LAYER_FIELD: ConfigLayer.AGENT.value},
    ]


def test_hooks_list_form_is_normalised_and_scoped() -> None:
    tree = tag_tree({"hooks": [SPEC]}, ConfigLayer.PROJECT, "p.yaml")

    specs = get_path(tree, "hooks.specs")

    assert isinstance(specs, list)
    assert isinstance(specs[0], dict)
    assert specs[0][SPEC_SCOPE_FIELD] == HookScope.PROJECT.value


def test_normalise_leaves_mapping_form_untouched() -> None:
    data: Tree = {"hooks": {"timeout_ms": LOW_RETRIES}}

    assert normalise(data) == data


def test_resolve_sources_reports_highest_layer_and_each_accumulated_item() -> None:
    layers = [
        Layer(ConfigLayer.DEFAULT, {"limits": {"retries": LOW_RETRIES}}),
        Layer(
            ConfigLayer.GLOBAL,
            {"limits": {"retries": HIGH_RETRIES}, "permissions": {"rules": [RULE_GLOBAL]}},
            "g.yaml",
        ),
        Layer(ConfigLayer.AGENT, {"permissions": {"rules": [RULE_AGENT]}}),
    ]

    values = resolve_sources(layers)

    assert values["limits.retries"].layer is ConfigLayer.GLOBAL
    assert values["limits.retries"].value == HIGH_RETRIES
    assert values["permissions.rules[0]"].layer is ConfigLayer.GLOBAL
    assert values["permissions.rules[1]"].layer is ConfigLayer.AGENT


def test_set_path_does_not_mutate_the_original() -> None:
    original: Tree = {"a": {"b": LOW_RETRIES}}

    updated = set_path(original, "a.c", HIGH_RETRIES)

    assert flatten(original) == {"a.b": LOW_RETRIES}
    assert flatten(updated) == {"a.b": LOW_RETRIES, "a.c": HIGH_RETRIES}


def test_get_path_raises_for_missing_key() -> None:
    with pytest.raises(KeyError):
        get_path({"a": {}}, "a.b")
