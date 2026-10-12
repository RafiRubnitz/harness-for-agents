"""Loading config layers from disk and validating the merged result."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import JsonValue, ValidationError

from harness.config.consts import (
    GLOBAL_ONLY_KEYS,
    HOOK_SPECS_KEY,
    HOOKS_KEY,
    KEY_SEPARATOR,
    MSG_GLOBAL_ONLY_DROPPED,
    MSG_HOOKS_FILE_SHAPE,
    MSG_PROJECT_UNTRUSTED,
    MSG_TEMPLATE_INVALID,
    MSG_VALIDATION,
    PERMISSIONS_KEY,
    SPEC_SCOPE_FIELD,
    SPEC_SOURCE_FIELD,
    UNTRUSTED_PROJECT_KEYS,
    YAML_SUFFIX,
)
from harness.config.layers import Layer, Tree, deep_merge, has_path, merge_layers, tag_tree
from harness.config.schema import AgentTemplate, ConfigIssue, HarnessConfig
from harness.config.secrets import bad_refs
from harness.config.utils import issue, line_of, read_yaml
from harness.core.consts import ConfigLayer, ErrorCode
from harness.core.hook_types import HookSpec


@dataclass(slots=True)
class LayerLoad:
    """A loaded layer plus what went wrong while loading it."""

    layer: Layer
    errors: list[ConfigIssue] = field(default_factory=list[ConfigIssue])
    warnings: list[ConfigIssue] = field(default_factory=list[ConfigIssue])


def _key_of(loc: Sequence[str | int]) -> str:
    return KEY_SEPARATOR.join(str(part) for part in loc)


def _drop_keys(
    data: Tree, keys: frozenset[str], kind: ConfigLayer, file: str | None, template: str
) -> tuple[Tree, list[ConfigIssue]]:
    kept: Tree = {}
    warnings: list[ConfigIssue] = []
    for name, value in data.items():
        if name in keys:
            message = template.format(key=name, layer=kind.value, root=file)
            warnings.append(issue(message, file=file, key=name))
        else:
            kept[name] = value
    return kept, warnings


def _load_hooks_file(path: Path, kind: ConfigLayer) -> tuple[Tree, list[ConfigIssue]]:
    data, doc, errors = read_yaml(path)
    if errors or not data:
        return {}, errors
    specs = data.get(HOOKS_KEY)
    if set(data) != {HOOKS_KEY} or not isinstance(specs, list):
        return {}, [issue(MSG_HOOKS_FILE_SHAPE, file=str(path), line=1, key=HOOKS_KEY)]
    for index, spec in enumerate(specs):
        try:
            HookSpec.model_validate(spec)
        except ValidationError as exc:
            for error in exc.errors():
                loc = (HOOKS_KEY, index, *error["loc"])
                message = MSG_VALIDATION.format(key=_key_of(loc), detail=error["msg"])
                errors.append(
                    issue(message, file=str(path), line=line_of(doc, loc), key=_key_of(loc))
                )
    if errors:
        return {}, errors
    return tag_tree({HOOKS_KEY: {HOOK_SPECS_KEY: specs}}, kind, str(path)), []


def load_file_layer(
    kind: ConfigLayer,
    config_file: Path,
    hooks_file: Path | None,
    *,
    trusted: bool = True,
) -> LayerLoad:
    """Load one file-backed layer (``config.yaml`` plus optional ``hooks.yaml``)."""
    data, doc, errors = read_yaml(config_file)
    file = str(config_file)
    load = LayerLoad(Layer(kind, {}, file, doc), errors=list(errors))
    for key, message in bad_refs(data):
        parts = key.split(KEY_SEPARATOR)
        load.errors.append(issue(message, file=file, line=line_of(doc, parts), key=key))
    if kind is not ConfigLayer.GLOBAL:
        data, dropped = _drop_keys(data, GLOBAL_ONLY_KEYS, kind, file, MSG_GLOBAL_ONLY_DROPPED)
        load.warnings.extend(dropped)
    if not trusted:
        data, _ = _drop_keys(data, UNTRUSTED_PROJECT_KEYS, kind, file, MSG_PROJECT_UNTRUSTED)
    tree = tag_tree(data, kind, file)
    if hooks_file is not None and trusted:
        hooks_tree, hook_errors = _load_hooks_file(hooks_file, kind)
        load.errors.extend(hook_errors)
        tree = deep_merge(tree, hooks_tree)
    load.layer = Layer(kind, tree, file, doc)
    return load


def untrusted_warning(root: Path) -> ConfigIssue:
    """The warning sent when a project's code is not loaded."""
    return issue(
        MSG_PROJECT_UNTRUSTED.format(root=root),
        file=str(root),
        code=ErrorCode.PROJECT_UNTRUSTED,
    )


def memory_layer(kind: ConfigLayer, data: Tree) -> LayerLoad:
    """Build a layer that has no file (agent overrides, environment)."""
    warnings: list[ConfigIssue] = []
    if kind is ConfigLayer.AGENT:
        data, warnings = _drop_keys(data, GLOBAL_ONLY_KEYS, kind, None, MSG_GLOBAL_ONLY_DROPPED)
    return LayerLoad(Layer(kind, tag_tree(data, kind, None)), warnings=warnings)


def _origin(layers: Sequence[Layer], loc: Sequence[str | int]) -> Layer | None:
    parts = [part for part in loc if isinstance(part, str)]
    while parts:
        for layer in reversed(layers):
            if has_path(layer.data, parts):
                return layer
        parts.pop()
    return None


def validate_layers(layers: Sequence[Layer]) -> tuple[HarnessConfig | None, list[ConfigIssue]]:
    """Merge the layers and validate against the schema."""
    merged = merge_layers(layers)
    try:
        return HarnessConfig.model_validate(merged), []
    except ValidationError as exc:
        issues: list[ConfigIssue] = []
        for error in exc.errors():
            loc = error["loc"]
            origin = _origin(layers, loc)
            key = _key_of(loc)
            issues.append(
                issue(
                    MSG_VALIDATION.format(key=key, detail=error["msg"]),
                    file=origin.file if origin else None,
                    line=line_of(origin.doc, loc) if origin else None,
                    key=key,
                )
            )
        return None, issues


def load_templates(directory: Path) -> tuple[list[AgentTemplate], list[ConfigIssue]]:
    """Load every ``templates/*.yaml``; invalid files become issues and are skipped."""
    templates: list[AgentTemplate] = []
    issues: list[ConfigIssue] = []
    if not directory.is_dir():
        return templates, issues
    for path in sorted(directory.glob(f"*{YAML_SUFFIX}")):
        data, doc, errors = read_yaml(path)
        if errors:
            issues.extend(errors)
            continue
        try:
            templates.append(AgentTemplate.model_validate(data))
        except ValidationError as exc:
            for error in exc.errors():
                message = MSG_TEMPLATE_INVALID.format(file=path.name, detail=error["msg"])
                key = _key_of(error["loc"])
                issues.append(
                    issue(message, file=str(path), line=line_of(doc, error["loc"]), key=key)
                )
    return templates, issues


def template_overrides(template: AgentTemplate) -> Tree:
    """Return the agent-layer overrides a template is copied into at agent creation."""
    overrides: Tree = dict(template.config)
    if template.hooks:
        specs: list[JsonValue] = [
            spec.model_dump(
                mode="json",
                by_alias=True,
                exclude_none=True,
                exclude={SPEC_SCOPE_FIELD, SPEC_SOURCE_FIELD},
            )
            for spec in template.hooks
        ]
        overrides[HOOKS_KEY] = {HOOK_SPECS_KEY: specs}
    if template.permissions:
        overrides[PERMISSIONS_KEY] = dict(template.permissions)
    return overrides
