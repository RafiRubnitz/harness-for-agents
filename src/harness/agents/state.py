"""The agent state machine: the transition table of 04 §4."""

from typing import Final

from harness.agents.consts import MSG_INVALID_TRANSITION, MSG_NOT_ALLOWED_IN
from harness.core.consts import AgentStatus
from harness.core.errors import InvalidTransitionError

_S = AgentStatus

TRANSITIONS: Final[frozenset[tuple[AgentStatus, AgentStatus]]] = frozenset(
    {
        (_S.IDLE, _S.RUNNING),  # a message arrived
        (_S.RUNNING, _S.WAITING_APPROVAL),  # ask (policy, hook, run limit)
        (_S.WAITING_APPROVAL, _S.RUNNING),  # the request was answered
        (_S.RUNNING, _S.IDLE),  # run finished / stopped
        (_S.WAITING_APPROVAL, _S.IDLE),  # stopped while waiting
        (_S.RUNNING, _S.PAUSED),  # pause between steps / budget exceeded
        (_S.PAUSED, _S.RUNNING),  # resume
        (_S.RUNNING, _S.ERROR),  # unrecoverable model error
        (_S.ERROR, _S.RUNNING),  # retry / new message
        (_S.IDLE, _S.ARCHIVED),
        (_S.PAUSED, _S.ARCHIVED),
        (_S.ERROR, _S.ARCHIVED),
        (_S.ARCHIVED, _S.IDLE),  # unarchive
    }
)

ACTIVE: Final = frozenset({_S.RUNNING, _S.WAITING_APPROVAL})
ARCHIVABLE: Final = frozenset({_S.IDLE, _S.PAUSED, _S.ERROR})


def can_transition(current: AgentStatus, target: AgentStatus) -> bool:
    """Return whether ``current -> target`` is in the transition table."""
    return (current, target) in TRANSITIONS


def check_transition(current: AgentStatus, target: AgentStatus) -> None:
    """Raise ``InvalidTransitionError`` unless ``current -> target`` is in the table."""
    if not can_transition(current, target):
        raise InvalidTransitionError(
            MSG_INVALID_TRANSITION.format(current=current.value, target=target.value)
        )


def require_status(current: AgentStatus, allowed: frozenset[AgentStatus], action: str) -> None:
    """Raise ``InvalidTransitionError`` unless the agent is in one of ``allowed``."""
    if current not in allowed:
        raise InvalidTransitionError(MSG_NOT_ALLOWED_IN.format(action=action, status=current.value))
