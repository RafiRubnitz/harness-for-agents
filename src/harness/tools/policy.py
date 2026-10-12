"""Permission rules: who wins when several rules match one tool call (08 §5.1)."""

import json
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path

from pydantic import JsonValue

from harness.config import Rule
from harness.core.consts import ConfigLayer, Decision
from harness.tools.paths import escape_glob, path_matches, resolve_path
from harness.tools.types import ToolTraits

_LAYER_RANK = {layer: rank for rank, layer in enumerate(ConfigLayer)}


@dataclass(frozen=True, slots=True)
class Subject:
    """What the rules look at for one call."""

    tool: str
    match_text: str
    path: Path | None


@dataclass(frozen=True, slots=True)
class Verdict:
    """The policy's answer and the rule that gave it (``None`` = no rule matched)."""

    decision: Decision
    rule: Rule | None


def subject_of(
    tool: str, traits: ToolTraits, args: dict[str, JsonValue], working_dir: str
) -> Subject:
    """Build the rule subject from a clean tool input."""
    raw_path = args.get(traits.path_param) if traits.path_param else None
    path = resolve_path(raw_path, working_dir) if isinstance(raw_path, str) and raw_path else None
    matched = args.get(traits.match_param) if traits.match_param else None
    text = matched if isinstance(matched, str) else json.dumps(args, sort_keys=True)
    return Subject(tool=tool, match_text=text, path=path)


def rule_matches(rule: Rule, subject: Subject, working_dir: str) -> bool:
    """Return whether every condition of ``rule`` holds for ``subject``."""
    if not fnmatchcase(subject.tool, rule.tool):
        return False
    if rule.match is not None and not fnmatchcase(subject.match_text, rule.match):
        return False
    if rule.path is not None:
        return subject.path is not None and path_matches(rule.path, subject.path, working_dir)
    return True


def evaluate(rules: list[Rule], subject: Subject, working_dir: str, default: Decision) -> Verdict:
    """Apply the rules of every layer.

    A matching ``deny`` from any layer always wins. Otherwise the most specific layer
    that has a match decides, and inside a layer the last matching rule. With no match
    the tool's own default applies.
    """
    matching = [rule for rule in rules if rule_matches(rule, subject, working_dir)]
    for rule in matching:
        if rule.decision is Decision.DENY:
            return Verdict(Decision.DENY, rule)
    if not matching:
        return Verdict(default, None)
    top = max(_LAYER_RANK.get(rule.layer or ConfigLayer.GLOBAL, 0) for rule in matching)
    winner = [
        rule for rule in matching if _LAYER_RANK.get(rule.layer or ConfigLayer.GLOBAL, 0) == top
    ][-1]
    return Verdict(winner.decision, winner)


def exact_rule(subject: Subject, traits: ToolTraits) -> Rule:
    """The allow rule written by "always approve for this agent": this exact call only."""
    if traits.path_param and subject.path is not None:
        return Rule(
            tool=subject.tool, path=escape_glob(subject.path.as_posix()), decision=Decision.ALLOW
        )
    return Rule(tool=subject.tool, match=escape_glob(subject.match_text), decision=Decision.ALLOW)
