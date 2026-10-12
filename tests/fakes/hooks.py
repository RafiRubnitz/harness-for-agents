"""FakeHooksManager: scripted hook decisions per event, with a log of what was emitted."""

from collections import defaultdict

from pydantic import JsonValue

from harness.core.consts import Decision, HookEvent
from harness.core.hook_types import HookInput, HookRunRecord, HookSpec, MergedOutput
from harness.core.types import AgentId

FAKE_HOOK_NAME = "fake-hook"
FAKE_HOOK_DURATION_MS = 1


class FakeHooksManager:
    """Answers each event with the next scripted output (or passes it through)."""

    def __init__(self) -> None:
        self.scripted: defaultdict[HookEvent, list[MergedOutput]] = defaultdict(list)
        self.emitted: list[HookInput] = []
        self.specs: list[HookSpec] = []

    def script(
        self,
        event: HookEvent,
        decision: Decision | None = None,
        *,
        reason: str | None = None,
        additional_context: str | None = None,
        updated: dict[str, JsonValue] | None = None,
    ) -> None:
        """Queue one answer for the next emission of ``event``."""
        fired = HookRunRecord(
            hook=FAKE_HOOK_NAME,
            event=event,
            agent_id=None,
            decision=decision,
            reason=reason,
            duration_ms=FAKE_HOOK_DURATION_MS,
        )
        self.scripted[event].append(
            MergedOutput(
                decision=decision,
                reasons=[reason] if reason else [],
                additional_context=additional_context,
                payload=updated or {},
                fired=[fired],
            )
        )

    def events(self) -> list[HookEvent]:
        """The events emitted so far, in order."""
        return [inp.event for inp in self.emitted]

    async def emit(self, event: HookEvent, inp: HookInput) -> MergedOutput:
        """Return the next scripted output merged over the incoming payload."""
        self.emitted.append(inp)
        if not self.scripted[event]:
            return MergedOutput(payload=dict(inp.payload))
        out = self.scripted[event].pop(0)
        return out.model_copy(update={"payload": {**inp.payload, **out.payload}})

    def register(self, hook: HookSpec) -> None:
        """Remember the hook."""
        self.specs.append(hook)

    def unregister(self, name: str) -> None:
        """Forget the hook."""
        self.specs = [spec for spec in self.specs if spec.name != name]

    def list(
        self, event: HookEvent | None = None, agent_id: AgentId | None = None
    ) -> list[HookSpec]:
        """Return the registered hooks for an event."""
        return [spec for spec in self.specs if event in (None, spec.event)]

    async def reload(self) -> None:
        """Nothing to reload."""
