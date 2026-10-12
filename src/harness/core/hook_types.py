"""Hook data models shared by the hooks manager, config templates, and hook callers."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from harness.core.consts import Decision, ErrorCode, HookEvent, HookOnError, HookScope
from harness.core.types import AgentId, RunId


class HookInput(BaseModel):
    """What every hook receives, for every event."""

    model_config = ConfigDict(extra="forbid")
    event: HookEvent
    agent_id: AgentId | None
    agent_name: str
    run_id: RunId | None = None
    step: int | None = None
    working_dir: str | None = None
    payload: dict[str, JsonValue]
    timestamp: datetime


class HookOutput(BaseModel):
    """What every hook returns, for every event."""

    decision: Decision | None = None
    reason: str | None = None
    additional_context: str | None = None
    updated_payload: dict[str, JsonValue] | None = None
    suppress_output: bool = False


class HookSpec(BaseModel):
    """One hook definition (a row in ``hooks.yaml`` or in the agent config layer)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    name: str
    event: HookEvent
    match: dict[str, str] = {}
    run: str
    timeout_ms: int | None = None
    on_error: HookOnError | None = None
    is_async: bool = Field(default=False, alias="async")
    scope: HookScope = HookScope.GLOBAL
    source_file: str | None = None


class HookRunRecord(BaseModel):
    """Log entry of one hook invocation."""

    hook: str
    event: HookEvent
    agent_id: AgentId | None
    decision: Decision | None
    reason: str | None
    duration_ms: int
    error: ErrorCode | None = None
    suppressed: bool = False


class MergedOutput(BaseModel):
    """The combined result of every hook that ran for one event."""

    decision: Decision | None = None
    reasons: list[str] = []
    additional_context: str | None = None
    payload: dict[str, JsonValue] = {}
    fired: list[HookRunRecord] = []
