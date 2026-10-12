"""AppState: events in, agents / conversations / stale regions out."""

import pytest

from harness.core import events as e
from harness.core.consts import AgentStatus, ErrorCode, MessageRole, PermissionResolution
from harness.core.types import TextBlock, ToolCallBlock
from harness.gui import consts
from harness.gui.consts import NoticeLevel, Region, ToolStatus
from harness.gui.forms import approval_lines
from harness.gui.items import AssistantItem, NoticeItem, ToolItem, UserItem, from_history
from harness.gui.render import item_lines, item_text
from tests.consts import AGENT_ID, OTHER_AGENT_ID
from tests.unit.gui.conftest import (
    CALL_ID,
    REASON,
    REQUEST_ID,
    THIRD_AGENT_ID,
    TOOL,
    Feed,
    agent_state,
    permission_request,
)
from tests.utils import make_message

CHUNKS = ["שלום ", "world", "!"]
QUESTION = "what is up?"
DURATION_MS = 1500


def _running(feed: Feed, agent_id: object = AGENT_ID) -> None:
    feed.send(
        e.AgentStatusChanged(status=AgentStatus.RUNNING, previous=AgentStatus.IDLE),
        agent_id,  # type: ignore[arg-type]
    )


def test_listing_selects_the_first_agent_and_hides_archived(feed: Feed) -> None:
    archived = agent_state(OTHER_AGENT_ID, status=AgentStatus.ARCHIVED, order=1)

    feed.app(e.AgentUpdated(agent=archived))

    state = feed.state
    assert state.selected == AGENT_ID
    assert [view.state.id for view in state.ordered()] == [AGENT_ID, THIRD_AGENT_ID]
    assert [view.state.id for view in state.ordered(archived=True)] == [OTHER_AGENT_ID]
    assert Region.SIDEBAR in state.take_changes()


def test_each_agent_shows_its_own_live_status(feed: Feed) -> None:
    statuses = {
        AGENT_ID: AgentStatus.RUNNING,
        OTHER_AGENT_ID: AgentStatus.WAITING_APPROVAL,
        THIRD_AGENT_ID: AgentStatus.ERROR,
    }

    for agent_id, status in statuses.items():
        feed.send(e.AgentStatusChanged(status=status, previous=AgentStatus.IDLE), agent_id)

    shown = {view.state.id: view.state.status for view in feed.state.ordered()}
    assert shown == statuses
    assert {view.state.id for view in feed.state.active()} == {AGENT_ID, OTHER_AGENT_ID}


@pytest.mark.acceptance("02-10.3")
def test_streamed_text_grows_one_item_and_marks_the_stream_stale(feed: Feed) -> None:
    conversation = feed.state.agents[AGENT_ID].conversation
    _running(feed)
    feed.state.take_changes()
    seen: list[str] = []

    for chunk in CHUNKS:
        feed.send(e.TextDelta(text=chunk))
        assert Region.STREAM in feed.state.take_changes()
        assert conversation.take_dirty() is not None
        seen.append(item_text(conversation.items[-1]))

    assert seen == ["".join(CHUNKS[: count + 1]) for count in range(len(CHUNKS))]
    assert [type(item) for item in conversation.items] == [AssistantItem]


def test_the_stored_message_replaces_the_streamed_one(feed: Feed) -> None:
    text = "".join(CHUNKS)
    _running(feed)
    for chunk in CHUNKS:
        feed.send(e.TextDelta(text=chunk))

    feed.send(e.MessageAppended(message=make_message("m1", text)))
    feed.send(e.AgentStatusChanged(status=AgentStatus.IDLE, previous=AgentStatus.RUNNING))

    items = feed.state.agents[AGENT_ID].conversation.items
    assert [(type(item), item_text(item)) for item in items] == [(AssistantItem, text)]
    assert not any(isinstance(item, AssistantItem) and item.streaming for item in items)


@pytest.mark.acceptance("02-10.2")
def test_hidden_agent_gets_a_badge_when_it_asks_and_others_keep_running(feed: Feed) -> None:
    _running(feed)
    _running(feed, OTHER_AGENT_ID)
    feed.state.take_changes()

    feed.send(e.PermissionRequest(request=permission_request()), OTHER_AGENT_ID)

    hidden = feed.state.agents[OTHER_AGENT_ID]
    assert hidden.badge == len(hidden.approvals) > 0
    assert feed.state.agents[AGENT_ID].badge == 0
    assert feed.state.agents[AGENT_ID].state.status is AgentStatus.RUNNING
    changed = feed.state.take_changes()
    assert Region.SIDEBAR in changed
    assert Region.STREAM not in changed


def test_answering_clears_the_badge_and_selecting_clears_unread(feed: Feed) -> None:
    feed.send(e.PermissionRequest(request=permission_request()), OTHER_AGENT_ID)
    feed.send(
        e.MessageAppended(message=make_message("m1", agent_id=THIRD_AGENT_ID)), THIRD_AGENT_ID
    )

    feed.send(
        e.PermissionResolved(request_id=REQUEST_ID, decision=PermissionResolution.APPROVE),
        OTHER_AGENT_ID,
    )
    unread_before = feed.state.agents[THIRD_AGENT_ID].badge
    feed.state.select(THIRD_AGENT_ID)

    assert feed.state.agents[OTHER_AGENT_ID].approvals == []
    assert unread_before > 0
    assert feed.state.agents[THIRD_AGENT_ID].badge == 0


@pytest.mark.acceptance("02-10.5")
def test_the_reason_is_in_the_approval_panel_and_in_the_tool_row(feed: Feed) -> None:
    feed.send(e.ToolCall(call_id=CALL_ID, tool=TOOL, input={"path": "note.txt"}, reason=REASON))
    feed.send(e.PermissionRequest(request=permission_request()))

    view = feed.state.agents[AGENT_ID]
    row = view.conversation.tool(CALL_ID)
    assert row is not None
    assert row.status is ToolStatus.WAITING
    panel = "\n".join(line.text for line in approval_lines(view.approvals[0]))
    opened = "\n".join(line.text for line in item_lines(row, expanded=True))
    assert REASON in panel
    assert REASON in opened


def test_tool_row_follows_the_call_to_its_result(feed: Feed) -> None:
    output = "saved 2 lines"
    feed.send(e.ToolCall(call_id=CALL_ID, tool=TOOL, input={"path": "note.txt"}, reason=REASON))
    feed.send(e.ToolProgress(call_id=CALL_ID, chunk="working"))

    feed.send(
        e.ToolResult(
            call_id=CALL_ID,
            tool=TOOL,
            reason=REASON,
            content=[TextBlock(text=output)],
            is_error=False,
            duration_ms=DURATION_MS,
        )
    )

    row = feed.state.agents[AGENT_ID].conversation.tool(CALL_ID)
    assert row is not None
    assert (row.status, row.duration_ms, row.output) == (ToolStatus.OK, DURATION_MS, output)
    assert len(feed.state.agents[AGENT_ID].conversation.items) == 1


def test_a_run_that_ends_fails_tools_left_running(feed: Feed) -> None:
    _running(feed)
    feed.send(e.ToolCall(call_id=CALL_ID, tool=TOOL, input={}, reason=None))

    feed.send(e.AgentStatusChanged(status=AgentStatus.IDLE, previous=AgentStatus.RUNNING))

    row = feed.state.agents[AGENT_ID].conversation.tool(CALL_ID)
    assert row is not None
    assert row.status is ToolStatus.FAILED


def test_errors_become_notices_or_the_banner(feed: Feed) -> None:
    message = "model unreachable"
    error = e.Error(
        code=ErrorCode.APP_INTERNAL, message=message, component="models", recoverable=True
    )

    feed.send(error)
    feed.app(error)

    last = feed.state.agents[AGENT_ID].conversation.items[-1]
    assert isinstance(last, NoticeItem)
    assert last.level is NoticeLevel.ERROR
    assert message in last.text
    assert feed.state.banner is not None
    assert message in feed.state.banner


def test_deleting_the_selected_agent_moves_the_selection(feed: Feed) -> None:
    feed.app(e.AgentDeleted(agent_id=AGENT_ID))

    assert AGENT_ID not in feed.state.agents
    assert feed.state.selected == OTHER_AGENT_ID


def test_config_change_marks_settings_stale(feed: Feed) -> None:
    feed.app(e.ConfigChanged(keys=["gui.theme"]))

    assert Region.SETTINGS in feed.state.take_changes()


def test_history_rebuilds_the_same_conversation_the_live_events_built(feed: Feed) -> None:
    call = ToolCallBlock(id=CALL_ID, name=TOOL, input={"path": "note.txt", "reason": REASON})
    messages = [
        make_message("m1", QUESTION, role=MessageRole.USER),
        make_message("m2", content=[TextBlock(text=CHUNKS[0]), call]),
    ]
    for message in messages:
        feed.send(e.MessageAppended(message=message))
    live = feed.state.agents[AGENT_ID].conversation

    rebuilt = from_history(messages, [])
    feed.send(e.HistoryLoaded(messages=messages, events=[], from_seq=0))

    assert [type(item) for item in rebuilt.items] == [UserItem, AssistantItem, ToolItem]
    assert [item_text(item) for item in rebuilt.items] == [item_text(item) for item in live.items]
    tool = rebuilt.tool(CALL_ID)
    assert tool is not None
    assert tool.reason == REASON
    assert "reason" not in tool.input
    assert feed.state.agents[AGENT_ID].loaded


def test_recovery_is_announced_in_the_conversation(feed: Feed) -> None:
    feed.send(
        e.AgentStatusChanged(
            status=AgentStatus.PAUSED, previous=AgentStatus.RUNNING, recovered=True
        )
    )

    last = feed.state.agents[AGENT_ID].conversation.items[-1]
    assert isinstance(last, NoticeItem)
    assert last.text == consts.MSG_RECOVERED
