"""Stages 0-2 of the permission pipeline: reason, policy, hooks (08 §5.2)."""

from dataclasses import dataclass, field

from pydantic import JsonValue

from harness.config import ConfigManager, ToolsConfig
from harness.core.consts import (
    AskedBy,
    Decision,
    DenySource,
    ErrorCode,
    HookEvent,
    ReasonRequiredFor,
)
from harness.core.hook_payloads import BeforeToolCallPayload
from harness.core.hook_types import HookInput
from harness.core.records import AgentState, PermissionDecision, ToolDef
from harness.core.types import Clock, InjectedFields
from harness.core.utils.paths import HarnessHome
from harness.hooks import HooksManager
from harness.tools.consts import (
    MSG_DENIED_HOOK,
    MSG_DENIED_POLICY,
    MSG_DENIED_PROTECTED,
    MSG_REASON_MISSING,
)
from harness.tools.paths import is_inside, is_protected
from harness.tools.policy import evaluate, subject_of
from harness.tools.types import Tool

_INPUT_KEY = "input"


@dataclass(frozen=True, slots=True)
class PolicyResult:
    """Stage 1: the policy's decision and why the user must be asked, if so."""

    decision: Decision
    denied_by: DenySource | None = None
    asked_by: list[AskedBy] = field(default_factory=list[AskedBy])
    must_ask: bool = False
    message: str | None = None


def reason_required(definition: ToolDef, required_for: ReasonRequiredFor) -> bool:
    """Return whether calls to this tool must carry a reason (08 §3a)."""
    if definition.reason_required is not None:
        return definition.reason_required
    if required_for is ReasonRequiredFor.ALL:
        return True
    if required_for is ReasonRequiredFor.NONE:
        return False
    return not definition.read_only


class PermissionPipeline:
    """Decides whether a tool call runs, is blocked, or needs the user."""

    def __init__(
        self, config: ConfigManager, hooks: HooksManager, home: HarnessHome, clock: Clock
    ) -> None:
        self._config = config
        self._hooks = hooks
        self._home = home
        self._clock = clock

    def policy(self, agent: AgentState, tool: Tool, args: dict[str, JsonValue]) -> PolicyResult:
        """Stage 1: protected paths, rules, working directory, forced confirmations."""
        definition, traits = tool.definition, tool.traits
        working_dir = str(agent.settings.working_dir)
        config = self._config.config(agent.id)
        subject = subject_of(definition.name, traits, args, working_dir)
        path = subject.path
        extra = config.tools.protected_paths_extra
        if path is not None and not definition.read_only and is_protected(path, self._home, extra):
            message = MSG_DENIED_PROTECTED.format(tool=definition.name, path=path)
            return PolicyResult(Decision.DENY, DenySource.PROTECTED_PATH, message=message)
        verdict = evaluate(
            config.permissions.rules, subject, working_dir, definition.default_permission
        )
        if verdict.decision is Decision.DENY:
            message = MSG_DENIED_POLICY.format(tool=definition.name)
            return PolicyResult(Decision.DENY, DenySource.POLICY, message=message)
        asked: list[AskedBy] = []
        roots = [working_dir, *agent.settings.allowed_paths]
        allowed_by_rule = verdict.rule is not None and verdict.decision is Decision.ALLOW
        if path is not None and not is_inside(path, roots) and not allowed_by_rule:
            asked.append(AskedBy.OUTSIDE_WORKING_DIR)
        if traits.confirm_param and args.get(traits.confirm_param) is True:
            asked.append(AskedBy.PERMANENT_DELETE)
        if asked:
            return PolicyResult(Decision.ASK, asked_by=asked, must_ask=True)
        if verdict.decision is Decision.ASK:
            return PolicyResult(Decision.ASK, asked_by=[AskedBy.POLICY])
        return PolicyResult(Decision.ALLOW)

    async def decide(
        self, agent: AgentState, tool: Tool, fields: InjectedFields, call_id: str
    ) -> PermissionDecision:
        """Run stages 0-2 for one call."""
        definition = tool.definition
        tools_config = self._config.section(ToolsConfig, agent.id)
        if reason_required(definition, tools_config.reason.required_for) and fields.reason is None:
            return PermissionDecision(
                decision=Decision.DENY,
                input=fields.input,
                error_code=ErrorCode.TOOL_REASON_MISSING,
                deny_message=MSG_REASON_MISSING.format(tool=definition.name),
            )
        args = fields.input
        result = self.policy(agent, tool, args)
        if result.decision is Decision.DENY:
            return _blocked(args, result)
        payload = BeforeToolCallPayload(
            tool=definition.name,
            call_id=call_id,
            input=args,
            reason=fields.reason,
            read_only=definition.read_only,
            risk=definition.risk,
            run_in_background=fields.run_in_background,
        )
        merged = await self._hooks.emit(
            HookEvent.BEFORE_TOOL_CALL,
            HookInput(
                event=HookEvent.BEFORE_TOOL_CALL,
                agent_id=agent.id,
                agent_name=agent.name,
                run_id=agent.run_id,
                step=agent.step,
                working_dir=agent.settings.working_dir,
                payload=payload.model_dump(mode="json"),
                timestamp=self._clock.now(),
            ),
        )
        updated = merged.payload.get(_INPUT_KEY)
        if isinstance(updated, dict) and updated != args:
            # A hook rewrote the input: the policy must see what will actually run.
            args = updated
            result = self.policy(agent, tool, args)
            if result.decision is Decision.DENY:
                return _blocked(args, result)
        decision = _after_hooks(definition.name, args, result, merged.decision, merged.reasons)
        return decision.model_copy(update={"hook_context": merged.additional_context})


def _blocked(args: dict[str, JsonValue], result: PolicyResult) -> PermissionDecision:
    return PermissionDecision(
        decision=Decision.DENY,
        input=args,
        denied_by=result.denied_by,
        deny_message=result.message,
        error_code=ErrorCode.PERMISSION_DENIED,
    )


def _after_hooks(
    tool: str,
    args: dict[str, JsonValue],
    result: PolicyResult,
    hook_decision: Decision | None,
    reasons: list[str],
) -> PermissionDecision:
    """Stage 2: a hook may tighten the policy, or relax ``ask`` to ``allow``."""
    if hook_decision is Decision.DENY:
        return PermissionDecision(
            decision=Decision.DENY,
            input=args,
            denied_by=DenySource.HOOK,
            hook_reasons=reasons,
            deny_message=MSG_DENIED_HOOK.format(tool=tool, reasons="; ".join(reasons)),
            error_code=ErrorCode.PERMISSION_DENIED,
        )
    asked = list(result.asked_by)
    decision = result.decision
    if hook_decision is Decision.ASK:
        decision = Decision.ASK
        asked.append(AskedBy.HOOK)
    elif hook_decision is Decision.ALLOW and not result.must_ask:
        decision, asked = Decision.ALLOW, []
    return PermissionDecision(decision=decision, input=args, asked_by=asked, hook_reasons=reasons)
