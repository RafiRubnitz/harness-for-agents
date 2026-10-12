"""``MemoryStore`` contract: the SQLite store and the fake must behave the same."""

from datetime import timedelta

import pytest

from harness.core.consts import (
    AddressType,
    AgentEventType,
    AgentStatus,
    HookEvent,
    MessageRole,
    PermissionResolution,
    ProviderId,
    RequestPurpose,
    TaskStatus,
)
from harness.core.errors import AgentNotFoundError, MemoryStoreError
from harness.core.records import (
    AgentEventRecord,
    BackgroundTaskRecord,
    InboxItem,
    ModelCallRecord,
    PendingApproval,
    PermissionRequestPayload,
    RunRecord,
    SummaryRecord,
    ToolCallRecord,
)
from harness.core.types import (
    Address,
    MessageOrigin,
    PermissionRequestId,
    RunId,
    TaskId,
    TextBlock,
    ToolCallBlock,
    ToolCallRecordId,
    ToolResultBlock,
)
from harness.memory import Memory, MemoryStore
from tests.consts import AGENT_ID, HEBREW_PLAIN, OTHER_AGENT_ID, RUN_ID, USER_ID
from tests.fakes.clock import FakeClock
from tests.fakes.ids import FakeIds
from tests.fakes.memory import FakeMemoryStore
from tests.utils import make_agent, make_message

ORIGIN = MessageOrigin(source=Address(type=AddressType.USER, id=USER_ID))
CALL_ID = "toolu_1"
TOOL_CALL_ID = ToolCallRecordId("tc_0001")
REQUEST_ID = PermissionRequestId("perm_0001")
OVERRIDES = {"context": {"compact_threshold": 0.5}}
COSTS = (0.25, 0.5)
STEM = "קובץ"
PREFIXED_TEXT = "פתחתי את והקובץ הזה"
IDENTIFIER = "config_v2.yaml"


@pytest.fixture(params=["sqlite", "fake"])
def store(
    request: pytest.FixtureRequest, memory: Memory, clock: FakeClock, ids: FakeIds
) -> MemoryStore:
    """Both implementations, one after the other."""
    return memory.store if request.param == "sqlite" else FakeMemoryStore(clock, ids)


def _run(clock: FakeClock, run_id: RunId = RUN_ID) -> RunRecord:
    return RunRecord(id=run_id, agent_id=AGENT_ID, first_seq=1, started_at=clock.now())


def _inbox(item_id: str, text: str, clock: FakeClock) -> InboxItem:
    return InboxItem(
        id=item_id,
        agent_id=AGENT_ID,
        content=[TextBlock(text=text)],
        origin=ORIGIN,
        received_at=clock.now(),
    )


async def _tool_call(store: MemoryStore, clock: FakeClock) -> ToolCallRecord:
    call = ToolCallBlock(id=CALL_ID, name="read_file", input={"path": "a.txt"})
    request = await store.append_message(make_message("msg_call", content=[call]))
    record = ToolCallRecord(
        id=TOOL_CALL_ID,
        agent_id=AGENT_ID,
        call_id=CALL_ID,
        request_message_id=request.id,
        tool_name=call.name,
        input=call.input,
        created_at=clock.now(),
    )
    await store.save_tool_call(record)
    return record


async def test_agent_roundtrip_and_listing(store: MemoryStore) -> None:
    agent = make_agent()
    archived = make_agent(OTHER_AGENT_ID, status=AgentStatus.ARCHIVED)

    await store.save_agent(agent)
    await store.save_agent(archived)

    assert await store.get_agent(AGENT_ID) == agent
    assert await store.list_agents() == [agent]
    assert await store.list_agents(include_archived=True) == [agent, archived]


async def test_missing_agent_raises(store: MemoryStore) -> None:
    with pytest.raises(AgentNotFoundError):
        await store.get_agent(AGENT_ID)
    with pytest.raises(AgentNotFoundError):
        await store.append_message(make_message("msg_1"))
    with pytest.raises(AgentNotFoundError):
        await store.delete_agent(AGENT_ID)


async def test_saving_an_agent_does_not_undo_config_overrides(store: MemoryStore) -> None:
    agent = make_agent()
    await store.save_agent(agent)

    await store.set_config_overrides(AGENT_ID, OVERRIDES)
    await store.save_agent(agent.model_copy(update={"status": AgentStatus.RUNNING}))

    assert await store.get_config_overrides(AGENT_ID) == OVERRIDES
    assert (await store.get_agent(AGENT_ID)).status is AgentStatus.RUNNING


async def test_messages_get_consecutive_seq_per_agent(store: MemoryStore) -> None:
    for agent_id in (AGENT_ID, OTHER_AGENT_ID):
        await store.save_agent(make_agent(agent_id))

    first = await store.append_message(make_message("msg_1"))
    other = await store.append_message(make_message("msg_2", agent_id=OTHER_AGENT_ID))
    second = await store.append_message(make_message("msg_3"))

    assert [first.seq, second.seq] == [1, 2]
    assert other.seq == 1
    assert await store.get_messages(AGENT_ID) == [first, second]
    assert await store.get_messages(AGENT_ID, from_seq=second.seq) == [second]
    assert await store.get_messages(AGENT_ID, limit=1) == [first]
    assert await store.last_seq(AGENT_ID) == second.seq


@pytest.mark.acceptance("03-10.12")
async def test_inbox_is_injected_after_tool_results(store: MemoryStore, clock: FakeClock) -> None:
    await store.save_agent(make_agent())
    await store.save_run(_run(clock))
    call = ToolCallBlock(id=CALL_ID, name="read_file", input={})
    await store.append_message(make_message("msg_1", content=[call]))
    await store.enqueue_inbox(_inbox("inb_1", "while tools run", clock))
    result = ToolResultBlock(call_id=CALL_ID, content=[TextBlock(text="done")])
    tool_message = await store.append_message(
        make_message("msg_2", role=MessageRole.TOOL, content=[result])
    )

    pending = await store.pending_inbox(AGENT_ID)
    injected = await store.inject_inbox(AGENT_ID, RUN_ID)

    assert [item.id for item in pending] == ["inb_1"]
    assert len(injected) == len(pending)
    assert injected[0].seq == tool_message.seq + 1
    assert injected[0].role is MessageRole.USER
    assert injected[0].origin == ORIGIN
    assert injected[0].run_id == RUN_ID
    assert await store.pending_inbox(AGENT_ID) == []
    assert (await store.get_messages(AGENT_ID))[-1] == injected[0]


async def test_inbox_keeps_arrival_order(store: MemoryStore, clock: FakeClock) -> None:
    await store.save_agent(make_agent())
    await store.save_run(_run(clock))
    texts = ["first", "second", "third"]
    for index, text in enumerate(texts):
        await store.enqueue_inbox(_inbox(f"inb_{index}", text, clock))
        clock.advance(1)

    injected = await store.inject_inbox(AGENT_ID, RUN_ID)

    assert [m.content for m in injected] == [[TextBlock(text=text)] for text in texts]


async def test_open_run_until_it_ends(store: MemoryStore, clock: FakeClock) -> None:
    await store.save_agent(make_agent())
    run = _run(clock)

    await store.save_run(run)
    opened = await store.open_run(AGENT_ID)
    ended = run.model_copy(update={"ended_at": clock.now(), "stop_reason": "end", "step_count": 2})
    await store.save_run(ended)

    assert opened == run
    assert await store.open_run(AGENT_ID) is None


async def test_pending_approval_is_listed_until_closed(
    store: MemoryStore, clock: FakeClock
) -> None:
    await store.save_agent(make_agent())
    payload = PermissionRequestPayload(
        request_id=REQUEST_ID, subject=HookEvent.RUN_LIMIT_REACHED, summary="continue?"
    )
    pending = PendingApproval(
        request_id=REQUEST_ID,
        agent_id=AGENT_ID,
        run_id=None,
        request=payload,
        created_at=clock.now(),
    )

    await store.save_pending_approval(pending)
    listed = await store.pending_approvals(AGENT_ID)
    await store.close_pending_approval(REQUEST_ID, PermissionResolution.APPROVE)

    assert listed == [pending]
    assert await store.pending_approvals() == []


async def test_background_task_needs_its_tool_call(store: MemoryStore, clock: FakeClock) -> None:
    await store.save_agent(make_agent())
    task = BackgroundTaskRecord(
        id=TaskId("task_0001"),
        agent_id=AGENT_ID,
        call_id=CALL_ID,
        tool="run_command",
        status=TaskStatus.RUNNING,
        started_at=clock.now(),
    )

    with pytest.raises(MemoryStoreError):
        await store.save_task(task)
    await _tool_call(store, clock)
    await store.save_task(task)
    done = task.model_copy(update={"status": TaskStatus.DONE, "ended_at": clock.now()})
    await store.save_task(done)

    assert await store.list_tasks(AGENT_ID, status=TaskStatus.RUNNING) == []
    assert await store.list_tasks(AGENT_ID) == [done]
    assert await store.list_tasks(None, status=TaskStatus.DONE) == [done]


async def test_tool_call_record_is_updated_in_place(store: MemoryStore, clock: FakeClock) -> None:
    await store.save_agent(make_agent())
    record = await _tool_call(store, clock)
    finished = record.model_copy(update={"is_error": False, "finished_at": clock.now()})

    await store.save_tool_call(finished)

    assert await store.get_tool_calls(AGENT_ID) == [finished]


async def test_events_are_positioned_by_after_seq(store: MemoryStore, clock: FakeClock) -> None:
    await store.save_agent(make_agent())
    events = [
        AgentEventRecord(
            agent_id=AGENT_ID,
            run_id=None,
            after_seq=after_seq,
            type=AgentEventType.STATUS,
            payload={"status": AgentStatus.RUNNING.value},
            created_at=clock.now(),
        )
        for after_seq in (0, 3)
    ]

    for event in events:
        await store.append_event(event)

    assert await store.get_events(AGENT_ID) == events
    assert await store.get_events(AGENT_ID, from_seq=events[1].after_seq) == events[1:]


async def test_superseded_summaries_are_hidden_by_default(
    store: MemoryStore, clock: FakeClock
) -> None:
    await store.save_agent(make_agent())
    merged = SummaryRecord(
        id="sum_2", agent_id=AGENT_ID, from_seq=1, to_seq=20, text="all", created_at=clock.now()
    )
    old = SummaryRecord(
        id="sum_1", agent_id=AGENT_ID, from_seq=1, to_seq=10, text="part", created_at=clock.now()
    )

    await store.save_summary(old)
    await store.save_summary(merged)
    await store.save_summary(old.model_copy(update={"superseded_by": merged.id}))

    assert await store.get_summaries(AGENT_ID) == [merged]
    assert len(await store.get_summaries(AGENT_ID, active_only=False)) == len((old, merged))


async def test_cost_ledger_sums_per_agent_and_since(store: MemoryStore, clock: FakeClock) -> None:
    await store.save_agent(make_agent())
    start = clock.now()
    for index, cost in enumerate(COSTS):
        clock.advance(timedelta(hours=1).total_seconds())
        await store.record_model_call(
            ModelCallRecord(
                id=f"req_{index}",
                agent_id=AGENT_ID,
                purpose=RequestPurpose.STEP,
                model_alias="main",
                model_used="fake-model-1",
                provider=ProviderId.ANTHROPIC,
                cost_usd=cost,
                created_at=clock.now(),
            )
        )

    assert await store.cost_total(AGENT_ID) == pytest.approx(sum(COSTS))
    assert await store.cost_total(OTHER_AGENT_ID) == 0
    assert await store.cost_since(clock.now()) == pytest.approx(COSTS[-1])
    assert await store.cost_since(start) == pytest.approx(sum(COSTS))


async def test_cost_ledger_survives_agent_deletion(store: MemoryStore, clock: FakeClock) -> None:
    await store.save_agent(make_agent())
    await store.record_model_call(
        ModelCallRecord(
            id="req_1",
            agent_id=AGENT_ID,
            purpose=RequestPurpose.COMPACTION,
            model_alias="main",
            model_used="fake-model-1",
            provider=ProviderId.ANTHROPIC,
            cost_usd=COSTS[0],
            created_at=clock.now(),
        )
    )

    await store.delete_agent(AGENT_ID)

    assert await store.cost_since(clock.now()) == pytest.approx(COSTS[0])


@pytest.mark.acceptance("03-10.13")
async def test_delete_agent_removes_everything_it_owns(
    store: MemoryStore, clock: FakeClock
) -> None:
    for agent_id in (AGENT_ID, OTHER_AGENT_ID):
        await store.save_agent(make_agent(agent_id))
    await store.save_run(_run(clock))
    await _tool_call(store, clock)
    await store.enqueue_inbox(_inbox("inb_1", "queued", clock))
    kept = await store.append_message(make_message("msg_other", agent_id=OTHER_AGENT_ID))

    await store.delete_agent(AGENT_ID)
    await store.save_agent(make_agent())

    assert await store.get_messages(AGENT_ID) == []
    assert await store.pending_inbox(AGENT_ID) == []
    assert await store.open_run(AGENT_ID) is None
    assert await store.get_tool_calls(AGENT_ID) == []
    assert await store.last_seq(AGENT_ID) == 0
    assert await store.get_messages(OTHER_AGENT_ID) == [kept]


@pytest.mark.acceptance("03-10.8")
async def test_history_search_finds_a_prefixed_hebrew_word(store: MemoryStore) -> None:
    await store.save_agent(make_agent())
    hit = await store.append_message(make_message("msg_1", PREFIXED_TEXT))
    await store.append_message(make_message("msg_2", HEBREW_PLAIN))

    found = await store.search_history(AGENT_ID, STEM)

    assert found == [hit]


@pytest.mark.acceptance("03-10.9")
async def test_history_search_is_scoped_to_the_agent(store: MemoryStore) -> None:
    for agent_id in (AGENT_ID, OTHER_AGENT_ID):
        await store.save_agent(make_agent(agent_id))
    mine = await store.append_message(make_message("msg_1", f"edit {IDENTIFIER} now"))
    await store.append_message(make_message("msg_2", IDENTIFIER, agent_id=OTHER_AGENT_ID))

    assert await store.search_history(AGENT_ID, IDENTIFIER) == [mine]
    assert await store.search_history(AGENT_ID, "nothing-like-this") == []
    assert await store.search_history(AGENT_ID, '"*') == []


async def test_history_search_covers_tool_calls_but_not_thinking(store: MemoryStore) -> None:
    from harness.core.types import ThinkingBlock  # noqa: PLC0415 - only this test needs it

    await store.save_agent(make_agent())
    call = ToolCallBlock(id=CALL_ID, name="read_file", input={"path": IDENTIFIER}, reason="check")
    thinking = ThinkingBlock(text="secretthought")
    stored = await store.append_message(make_message("msg_1", content=[thinking, call]))

    assert await store.search_history(AGENT_ID, IDENTIFIER) == [stored]
    assert await store.search_history(AGENT_ID, thinking.text) == []


async def test_meta_roundtrip(store: MemoryStore) -> None:
    assert await store.get_meta("missing") is None

    await store.set_meta("key", "v1")
    await store.set_meta("key", "v2")

    assert await store.get_meta("key") == "v2"
