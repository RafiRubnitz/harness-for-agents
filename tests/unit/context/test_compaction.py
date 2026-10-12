"""Compaction: when it runs, what it keeps, and what it leaves behind (07 §5, §8)."""

import pytest

from harness.core.consts import (
    AgentEventType,
    CompactReason,
    Decision,
    ErrorKind,
    HookEvent,
    ProviderId,
    RequestPurpose,
)
from harness.core.errors import ContextError, GatewayError
from harness.core.types import Message, ModelRequest, ToolCallBlock, ToolResultBlock
from tests.consts import AGENT_ID, TEST_MODEL_ALIAS
from tests.fakes.adapter import text_response
from tests.unit.context.conftest import SMALL_WINDOW, SUMMARY_TEXT, ContextEnv, request_tokens

MANY = 20  # more summary answers than any single test uses
HEAVY = 40  # repeats that make one message a few hundred tokens
NEEDLE = "pomegranate"
HOOK_REASON = "not now"
HOOK_INSTRUCTIONS = "keep every file name"
MERGED_TEXT = "MERGED: one summary of everything"
THOUSANDS_OF_TURNS = 1_000
SUMMARIES_PCT_KEY = "context.budget.summaries_pct"
TINY_SHARE = 0.001


def compaction_calls(env: ContextEnv) -> list[ModelRequest]:
    return [r for r in env.gateway.requests if r.metadata.purpose is RequestPurpose.COMPACTION]


def sent_text(request: ModelRequest) -> str:
    return request.model_dump_json(exclude={"metadata"})


def unpaired(messages: list[Message]) -> set[str]:
    """Tool call ids that appear without their result, or results without their call."""
    calls = {b.id for m in messages for b in m.content if isinstance(b, ToolCallBlock)}
    answers = {b.call_id for m in messages for b in m.content if isinstance(b, ToolResultBlock)}
    return calls ^ answers


def failure(kind: ErrorKind = ErrorKind.OVERLOADED) -> GatewayError:
    return GatewayError("down", kind=kind, provider=ProviderId.ANTHROPIC, model="cheap")


async def heavy_history(env: ContextEnv, turns: int = 40) -> None:
    """A history several times larger than the small window."""
    env.shrink_window()
    env.script_summaries(MANY)
    await env.talk(turns, size=HEAVY)


# --- when it runs ---


async def test_a_history_inside_the_window_is_left_alone(env: ContextEnv) -> None:
    await env.talk(5)

    await env.manager.build(AGENT_ID)

    assert compaction_calls(env) == []
    assert await env.store.get_summaries(AGENT_ID) == []


async def test_passing_the_line_compacts_before_the_request_is_built(env: ContextEnv) -> None:
    await heavy_history(env)

    built = await env.manager.build(AGENT_ID)

    (summary,) = await env.store.get_summaries(AGENT_ID)
    assert summary.reason is CompactReason.THRESHOLD
    assert built.report.compacted_ranges == [(summary.from_seq, summary.to_seq)]
    assert request_tokens(built.request) < SMALL_WINDOW
    opening = built.request.messages[0].model_dump_json()
    assert "<conversation_summary" in opening
    assert SUMMARY_TEXT in opening


@pytest.mark.acceptance("07-12.1")
async def test_thousands_of_messages_still_fit_in_the_window(bulk_env: ContextEnv) -> None:
    env = bulk_env
    env.shrink_window()
    env.gateway.outcomes = [text_response(SUMMARY_TEXT) for _ in range(THOUSANDS_OF_TURNS)]
    await env.talk(THOUSANDS_OF_TURNS, size=5)

    first = await env.manager.build(AGENT_ID)
    await env.talk(THOUSANDS_OF_TURNS // 10, size=HEAVY)
    second = await env.manager.build(AGENT_ID)

    stored = await env.store.get_messages(AGENT_ID)
    assert len(stored) >= THOUSANDS_OF_TURNS * 2
    assert request_tokens(first.request) < SMALL_WINDOW
    assert request_tokens(second.request) < SMALL_WINDOW
    assert second.request.messages[-1].content == stored[-1].content


@pytest.mark.acceptance("07-12.7")
async def test_switching_to_a_smaller_model_compacts_instead_of_failing(
    env: ContextEnv,
) -> None:
    env.script_summaries(MANY)
    await env.talk(80, size=HEAVY)
    before = await env.manager.build(AGENT_ID)
    assert await env.store.get_summaries(AGENT_ID) == []
    assert request_tokens(before.request) > SMALL_WINDOW

    env.shrink_window()
    after = await env.manager.build(AGENT_ID)

    assert len(await env.store.get_summaries(AGENT_ID)) == 1
    assert after.report.window == SMALL_WINDOW
    assert request_tokens(after.request) < SMALL_WINDOW


async def test_the_exact_count_decides_near_the_line(env: ContextEnv) -> None:
    await heavy_history(env)
    env.gateway.counter = lambda _: 0  # the provider says the request is small

    await env.manager.build(AGENT_ID)

    assert await env.store.get_summaries(AGENT_ID) == []


async def test_the_estimate_decides_when_the_exact_count_fails(env: ContextEnv) -> None:
    await heavy_history(env)

    def broken(_: ModelRequest) -> int:
        raise failure()

    env.gateway.counter = broken

    await env.manager.build(AGENT_ID)

    assert len(await env.store.get_summaries(AGENT_ID)) == 1


async def test_a_failed_compaction_does_not_stop_the_build(env: ContextEnv) -> None:
    env.shrink_window()
    env.gateway.outcomes = [failure(), failure()]
    await env.talk(40, size=HEAVY)

    built = await env.manager.build(AGENT_ID)

    assert built.request.messages
    assert await env.store.get_summaries(AGENT_ID) == []


# --- what it keeps ---


async def test_the_newest_messages_stay_word_for_word(env: ContextEnv) -> None:
    await heavy_history(env)
    stored = await env.store.get_messages(AGENT_ID)

    summary = await env.manager.compact(AGENT_ID)
    built = await env.manager.build(AGENT_ID)

    kept = [m for m in stored if m.seq > summary.to_seq]
    assert summary.from_seq == stored[0].seq
    assert 0 < len(kept) < len(stored)
    assert built.request.messages[-1].content == kept[-1].content
    assert stored[0].model_dump_json() not in sent_text(built.request)


async def test_compaction_never_leaves_half_a_tool_call(env: ContextEnv) -> None:
    env.shrink_window()
    env.script_summaries(MANY)
    await env.user("start")
    for _ in range(30):
        await env.tool_step("row of data " * 60)

    built = await env.manager.build(AGENT_ID)

    assert len(await env.store.get_summaries(AGENT_ID)) == 1
    assert unpaired(built.request.messages) == set()
    assert built.request.messages[0].role == "user"


@pytest.mark.acceptance("07-12.3")
async def test_compacted_messages_can_still_be_searched(env: ContextEnv) -> None:
    await env.user(f"my favourite fruit is the {NEEDLE}")
    await heavy_history(env)

    summary = await env.manager.compact(AGENT_ID)
    found = await env.store.search_history(AGENT_ID, NEEDLE)

    assert [message.seq for message in found] == [summary.from_seq]
    assert len(await env.store.get_messages(AGENT_ID)) > summary.to_seq - summary.from_seq


async def test_a_second_compaction_continues_after_the_first(env: ContextEnv) -> None:
    await heavy_history(env)
    first = await env.manager.compact(AGENT_ID)
    await env.talk(40, size=HEAVY)

    second = await env.manager.compact(AGENT_ID)

    assert second.from_seq == first.to_seq + 1
    ranges = (await env.manager.inspect(AGENT_ID)).compacted_ranges
    assert ranges == [(first.from_seq, first.to_seq), (second.from_seq, second.to_seq)]


async def test_summaries_over_their_budget_are_folded_into_one(env: ContextEnv) -> None:
    await heavy_history(env)
    first = await env.manager.compact(AGENT_ID)
    await env.talk(40, size=HEAVY)
    await env.set(SUMMARIES_PCT_KEY, TINY_SHARE)
    env.gateway.outcomes = [text_response(SUMMARY_TEXT) for _ in compaction_calls(env)]
    env.gateway.outcomes.append(text_response(MERGED_TEXT))

    merged = await env.manager.compact(AGENT_ID)

    (active,) = await env.store.get_summaries(AGENT_ID)
    every = await env.store.get_summaries(AGENT_ID, active_only=False)
    assert active.id == merged.id
    assert (active.text, active.level) == (MERGED_TEXT, first.level + 1)
    assert active.from_seq == first.from_seq
    assert {s.superseded_by for s in every if s.id != merged.id} == {merged.id}


# --- manual and refused ---


async def test_a_manual_compaction_works_on_a_short_history(env: ContextEnv) -> None:
    env.script_summaries()
    await env.talk(2)
    stored = await env.store.get_messages(AGENT_ID)

    summary = await env.manager.compact(AGENT_ID, CompactReason.MANUAL)

    assert (summary.from_seq, summary.to_seq) == (stored[0].seq, stored[-1].seq)
    assert summary.reason is CompactReason.MANUAL
    assert (summary.text, summary.model_used) == (SUMMARY_TEXT, compaction_calls(env)[0].model)


async def test_there_must_be_something_to_compact(env: ContextEnv) -> None:
    await env.user("only one message")

    with pytest.raises(ContextError):
        await env.manager.compact(AGENT_ID)

    assert compaction_calls(env) == []


# --- hooks ---


async def test_a_hook_can_block_compaction(env: ContextEnv) -> None:
    await heavy_history(env)
    env.hooks.script(HookEvent.CONTEXT_COMPACT, Decision.DENY, reason=HOOK_REASON)

    with pytest.raises(ContextError) as raised:
        await env.manager.compact(AGENT_ID)

    assert HOOK_REASON in raised.value.message
    assert compaction_calls(env) == []
    assert await env.store.get_summaries(AGENT_ID) == []


async def test_a_hook_sees_the_range_and_can_add_summary_instructions(env: ContextEnv) -> None:
    await heavy_history(env)
    env.hooks.script(HookEvent.CONTEXT_COMPACT, updated={"instructions": HOOK_INSTRUCTIONS})

    summary = await env.manager.compact(AGENT_ID, CompactReason.MANUAL)

    (fired,) = [inp for inp in env.hooks.emitted if inp.event is HookEvent.CONTEXT_COMPACT]
    assert (fired.payload["from_seq"], fired.payload["to_seq"]) == (
        summary.from_seq,
        summary.to_seq,
    )
    assert fired.payload["reason"] == CompactReason.MANUAL.value
    assert all(HOOK_INSTRUCTIONS in sent_text(call) for call in compaction_calls(env))


# --- the summary model ---


async def test_the_summary_model_gets_the_old_messages_as_a_transcript(env: ContextEnv) -> None:
    await env.user(f"note the {NEEDLE}")
    await heavy_history(env)

    await env.manager.compact(AGENT_ID)

    calls = compaction_calls(env)
    summarizer = env.config.config().context.summarizer_model
    assert {call.model for call in calls} == {summarizer}
    assert NEEDLE in sent_text(calls[0])
    assert all(call.tools is None and call.max_tokens for call in calls)


async def test_a_long_transcript_is_summarised_in_rolling_chunks(env: ContextEnv) -> None:
    await heavy_history(env, turns=80)

    await env.manager.compact(AGENT_ID)

    calls = compaction_calls(env)
    assert len(calls) > 1
    assert SUMMARY_TEXT not in sent_text(calls[0])
    assert all(SUMMARY_TEXT in sent_text(call) for call in calls[1:])
    assert all(request_tokens(call) < SMALL_WINDOW for call in calls)


async def test_the_agents_own_model_steps_in_when_the_summary_model_fails(
    env: ContextEnv,
) -> None:
    env.script_summaries()
    await env.talk(2)
    env.gateway.outcomes.insert(0, failure(ErrorKind.AUTH))

    summary = await env.manager.compact(AGENT_ID, CompactReason.MANUAL)

    assert [call.model for call in compaction_calls(env)][-1] == TEST_MODEL_ALIAS
    assert summary.model_used == TEST_MODEL_ALIAS


async def test_no_model_answering_is_a_context_error(env: ContextEnv) -> None:
    await env.talk(2)
    env.gateway.outcomes = [failure(), failure()]

    with pytest.raises(ContextError):
        await env.manager.compact(AGENT_ID, CompactReason.MANUAL)

    assert await env.store.get_summaries(AGENT_ID) == []


async def test_an_empty_summary_is_refused(env: ContextEnv) -> None:
    await env.talk(2)
    env.gateway.outcomes = [text_response(""), text_response("  ")]

    with pytest.raises(ContextError):
        await env.manager.compact(AGENT_ID, CompactReason.MANUAL)


# --- telling the user ---


async def test_a_compaction_is_announced_and_kept_as_a_display_event(env: ContextEnv) -> None:
    await heavy_history(env)

    summary = await env.manager.compact(AGENT_ID)

    (shown,) = env.notices.summaries
    assert (shown.id, shown.from_seq, shown.to_seq) == (
        summary.id,
        summary.from_seq,
        summary.to_seq,
    )
    (event,) = [
        e for e in await env.store.get_events(AGENT_ID) if e.type is AgentEventType.COMPACTED
    ]
    assert event.payload["summary_id"] == summary.id
    assert (event.payload["from_seq"], event.payload["to_seq"]) == (
        summary.from_seq,
        summary.to_seq,
    )
