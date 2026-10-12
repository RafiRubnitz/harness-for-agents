"""Merging what several hooks answered for one event (05 §4)."""

from loguru import logger
from pydantic import JsonValue, ValidationError

from harness.core.consts import Decision, HookEvent
from harness.core.hook_payloads import MUTABLE_PAYLOAD_FIELDS, PAYLOAD_BY_HOOK_EVENT
from harness.core.hook_types import HookOutput, HookRunRecord, MergedOutput
from harness.hooks import consts


class Merge:
    """The running result of one event: the strictest decision and the chained payload."""

    def __init__(self, event: HookEvent, payload: dict[str, JsonValue]) -> None:
        self._event = event
        self.payload: dict[str, JsonValue] = dict(payload)
        self.decision: Decision | None = None
        self._reasons: list[str] = []
        self._contexts: list[str] = []
        self._fired: list[HookRunRecord] = []

    @property
    def denied(self) -> bool:
        """Whether a hook denied; the chain stops there."""
        return self.decision is Decision.DENY

    def change_payload(self, hook: str, updated: dict[str, JsonValue]) -> str | None:
        """Apply a hook's ``updated_payload``; return an error text when it is invalid.

        Fields the event does not let hooks change are dropped with a warning. The changed
        payload must still be a valid payload of the event, otherwise nothing changes.
        """
        allowed = MUTABLE_PAYLOAD_FIELDS.get(self._event, frozenset[str]())
        for key in sorted(set(updated) - allowed):
            logger.warning(consts.MSG_FIELD_DROPPED, hook, key, self._event.value)
        kept = {key: value for key, value in updated.items() if key in allowed}
        if not kept:
            return None
        candidate = {**self.payload, **kept}
        try:
            PAYLOAD_BY_HOOK_EVENT[self._event].model_validate(candidate)
        except ValidationError as error:
            return consts.MSG_INVALID_PAYLOAD.format(hook=hook, error=error)
        self.payload = candidate
        return None

    def add(self, record: HookRunRecord, output: HookOutput) -> None:
        """Fold one hook's answer in: decisions by severity, reasons and context accumulate."""
        if consts.DECISION_RANK[output.decision] > consts.DECISION_RANK[self.decision]:
            self.decision = output.decision
        if output.reason:
            self._reasons.append(output.reason)
        if output.additional_context:
            self._contexts.append(output.additional_context)
        self._fired.append(record)

    def result(self) -> MergedOutput:
        """The merged output handed back to the component that fired the event."""
        return MergedOutput(
            decision=self.decision,
            reasons=list(self._reasons),
            additional_context=consts.CONTEXT_JOINER.join(self._contexts) or None,
            payload=self.payload,
            fired=list(self._fired),
        )
