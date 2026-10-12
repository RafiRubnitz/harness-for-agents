"""Building a request: order, budgets, cache points, memories, tool results (07 §3-§8)."""

import pytest

from harness.context.consts import BASE_INSTRUCTIONS
from harness.core.consts import ErrorCode, MessageRole
from harness.core.records import AgentSettings
from harness.core.types import (
    Message,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
)
from harness.models.adapters.anthropic_wire import to_provider
from tests.consts import AGENT_ID, HEBREW_PLAIN
from tests.fakes.recall import memory_item
from tests.fakes.tools import text_output
from tests.unit.context.conftest import SMALL_WINDOW, TOOL_NAME, ContextEnv
from tests.utils import user_message

PROMPT = "You write release notes."
PROFILE = "The user prefers short answers."
QUESTION = "what changed in the last release?"
ANSWER = "three bug fixes"
HOOK_NOTE = "remember the house style"
FACT = "the release branch is called stable"
PINNED_FACT = "the user is called Dana"
RESULT_ID = "blob_0007"
SECTION_NAMES = [
    "system",
    "profile",
    "pinned",
    "tools",
    "summaries",
    "tail",
    "memories",
    "hook_context",
]
MAX_CACHE_POINTS = 4
SUMMARY_CALLS = 10  # a long transcript is summarised in several calls


def dump(message: Message) -> str:
    return message.model_dump_json()


def results(env_messages: list[Message]) -> list[ToolResultBlock]:
    return [b for m in env_messages for b in m.content if isinstance(b, ToolResultBlock)]


# --- order ---


async def test_system_holds_base_instructions_prompt_and_profile_in_that_order(
    env: ContextEnv,
) -> None:
    await env.agent(system_prompt=PROMPT)
    await env.config.set_user_profile(PROFILE)
    await env.user(QUESTION)

    system = str((await env.manager.build(AGENT_ID)).request.system)

    assert system.index(BASE_INSTRUCTIONS) < system.index(PROMPT) < system.index(PROFILE)


async def test_history_becomes_alternating_messages_with_tool_results_as_user(
    env: ContextEnv,
) -> None:
    await env.user(QUESTION)
    call_id = await env.tool_step("found it")
    await env.assistant(ANSWER)

    messages = (await env.manager.build(AGENT_ID)).request.messages

    assert [m.role for m in messages] == ["user", "assistant", "user", "assistant"]
    assert [block.call_id for block in results(messages)] == [call_id]


async def test_pending_messages_come_last(env: ContextEnv) -> None:
    await env.user(QUESTION)
    await env.assistant(ANSWER)

    built = await env.manager.build(AGENT_ID, [user_message(HEBREW_PLAIN)])

    assert HEBREW_PLAIN in dump(built.request.messages[-1])


async def test_hook_context_goes_into_the_last_user_message_not_the_system(
    env: ContextEnv,
) -> None:
    await env.agent(pending_hook_context=[HOOK_NOTE])
    await env.user("first")
    await env.assistant(ANSWER)
    await env.user(QUESTION)

    request = (await env.manager.build(AGENT_ID)).request

    assert HOOK_NOTE not in str(request.system)
    assert "<hook_context>" in dump(request.messages[-1])
    assert HOOK_NOTE in dump(request.messages[-1])
    assert HOOK_NOTE not in dump(request.messages[0])


async def test_thinking_of_earlier_turns_is_not_sent_again(env: ContextEnv) -> None:
    thought = ThinkingBlock(text="hmm", signature="sig")
    await env.user("first")
    await env.add(MessageRole.ASSISTANT, content=[thought, TextBlock(text=ANSWER)])
    await env.user(QUESTION)
    call = ToolCallBlock(id="c1", name=TOOL_NAME, input={})
    await env.add(MessageRole.ASSISTANT, content=[thought, call])
    await env.add(
        MessageRole.TOOL, content=[ToolResultBlock(call_id="c1", content=[TextBlock(text="ok")])]
    )

    messages = (await env.manager.build(AGENT_ID)).request.messages

    with_thinking = [
        index
        for index, message in enumerate(messages)
        if any(isinstance(block, ThinkingBlock) for block in message.content)
    ]
    assert with_thinking == [len(messages) - 2]


# --- the request's own settings ---


async def test_the_answer_room_is_bounded_by_the_model(env: ContextEnv) -> None:
    await env.user(QUESTION)
    model_max = env.gateway.profile.max_output_tokens

    default = (await env.manager.build(AGENT_ID)).request.max_tokens
    await env.agent(settings=AgentSettings(max_tokens_per_step=model_max // 2))
    chosen = (await env.manager.build(AGENT_ID)).request.max_tokens
    await env.agent(settings=AgentSettings(max_tokens_per_step=model_max * 2))
    too_much = (await env.manager.build(AGENT_ID)).request.max_tokens

    assert (default, chosen, too_much) == (model_max, model_max // 2, model_max)


# --- the report ---


@pytest.mark.acceptance("07-12.8")
async def test_the_report_says_how_many_tokens_each_part_takes(env: ContextEnv) -> None:
    env.tools.add(TOOL_NAME, text_output("ok"))
    await env.agent(system_prompt=PROMPT, pending_hook_context=[HOOK_NOTE])
    await env.config.set_user_profile(PROFILE)
    await env.user(QUESTION)
    requests_before = len(env.gateway.requests)

    report = await env.manager.inspect(AGENT_ID)

    by_name = {section.name: section for section in report.sections}
    assert list(by_name) == SECTION_NAMES
    used = ["system", "profile", "tools", "tail", "hook_context"]
    assert all(by_name[name].tokens > 0 for name in used)
    assert report.total_tokens == sum(section.tokens for section in report.sections)
    assert report.window == env.gateway.profile.context_window
    assert by_name["tail"].budget_tokens is None
    assert len(env.gateway.requests) == requests_before  # nothing was sent to the model
    assert report.full_text is None


async def test_the_full_report_carries_the_text_that_would_be_sent(env: ContextEnv) -> None:
    await env.user(QUESTION)

    report = await env.manager.inspect(AGENT_ID, full=True)

    assert report.full_text is not None
    assert QUESTION in report.full_text
    assert BASE_INSTRUCTIONS.splitlines()[0] in report.full_text


# --- prompt caching ---


@pytest.mark.acceptance("07-12.4")
async def test_the_second_step_repeats_the_first_step_as_a_cached_prefix(
    env: ContextEnv,
) -> None:
    """What a provider needs for a cache hit: the earlier request is an exact prefix.

    The first step's last message carries a cache point, and the second step sends the
    same system, tools and messages before anything new.
    """
    env.tools.add(TOOL_NAME, text_output("ok"))
    await env.user(QUESTION)
    first = (await env.manager.build(AGENT_ID)).request
    await env.tool_step("found it")

    second = (await env.manager.build(AGENT_ID)).request

    assert (second.system, second.tools) == (first.system, first.tools)
    assert second.messages[: len(first.messages)] == first.messages
    assert len(second.messages) > len(first.messages)
    wire = to_provider(first.model_copy(update={"model": env.gateway.profile.id}))
    assert "cache_control" in str(wire["messages"])
    assert "cache_control" in str(wire["system"])
    assert "cache_control" in str(wire["tools"])


async def test_there_are_never_more_cache_points_than_the_provider_allows(
    env: ContextEnv,
) -> None:
    env.tools.add(TOOL_NAME, text_output("ok"))
    env.shrink_window()
    env.script_summaries(SUMMARY_CALLS)
    await env.talk(40, size=40)

    built = await env.manager.build(AGENT_ID)

    hints = built.request.cache or []
    assert len(env.notices.summaries) == 1
    assert len(hints) == MAX_CACHE_POINTS
    assert len(built.report.cache_breakpoints) == MAX_CACHE_POINTS
    assert built.report.cache_breakpoints == sorted(built.report.cache_breakpoints)


async def test_models_that_cache_by_prefix_get_no_cache_points(env: ContextEnv) -> None:
    capabilities = env.gateway.profile.capabilities.model_copy(update={"prompt_cache": "auto"})
    env.gateway.profile = env.gateway.profile.model_copy(update={"capabilities": capabilities})
    await env.user(QUESTION)

    built = await env.manager.build(AGENT_ID)

    assert built.request.cache is None
    assert built.report.cache_breakpoints == []


# --- big and old tool results ---


@pytest.mark.acceptance("07-12.6")
async def test_a_huge_tool_result_is_cut_and_points_at_the_stored_copy(env: ContextEnv) -> None:
    limit = env.config.config().context.tool_results.truncate_tokens
    huge = "START " + "data " * 80_000 + " END"
    await env.user(QUESTION)
    await env.tool_step(huge, result_id=RESULT_ID)

    built = await env.manager.build(AGENT_ID)

    (sent,) = results(built.request.messages)
    text = "".join(block.text for block in sent.content if isinstance(block, TextBlock))
    assert len(huge) // 4 > limit * 10  # about 100k tokens went in
    assert len(text) // 4 < limit * 2
    assert text.startswith("START")
    assert text.endswith("END")
    assert RESULT_ID in text
    assert "read_tool_result" in text
    stored = (await env.store.get_messages(AGENT_ID))[-1]
    assert huge in stored.model_dump_json()  # the history keeps the whole result


async def test_tool_results_from_long_ago_are_replaced_by_a_note(env: ContextEnv) -> None:
    keep = env.config.config().context.tool_results.keep_recent_steps
    body = "row of data " * 200
    await env.user(QUESTION)
    for _ in range(keep * 2):
        await env.tool_step(body)

    built = await env.manager.build(AGENT_ID)

    texts = [block.model_dump_json() for block in results(built.request.messages)]
    assert [body in text for text in texts] == [False] * keep + [True] * keep


# --- semantic memory ---


@pytest.mark.acceptance("07-12.5")
async def test_a_relevant_fact_is_in_the_context_without_the_agent_searching(
    env: ContextEnv,
) -> None:
    item = env.recall.add(memory_item("mem_1", FACT, agent_id=AGENT_ID), hit=True)
    asked = await env.user(QUESTION, in_run=False)
    await env.open_run(asked.seq)

    built = await env.manager.build(AGENT_ID)

    opening = dump(built.request.messages[0])
    assert "<memory>" in opening
    assert FACT in opening
    assert item.observed_at.strftime("%Y-%m-%d") in opening
    assert built.report.retrieved_memory_ids == [item.id]
    assert env.recall.queries == [(QUESTION, env.config.config().context.auto_retrieve_k)]
    run = await env.store.open_run(AGENT_ID)
    assert run is not None
    assert run.retrieved_memory_ids == [item.id]


async def test_the_memory_block_stays_put_for_the_whole_run(env: ContextEnv) -> None:
    env.recall.add(memory_item("mem_1", FACT, agent_id=AGENT_ID), hit=True)
    asked = await env.user(QUESTION)
    await env.open_run(asked.seq)
    first = (await env.manager.build(AGENT_ID)).request
    await env.tool_step("found it")
    env.recall.add(memory_item("mem_2", "something newer"), hit=True)

    second = (await env.manager.build(AGENT_ID)).request

    assert second.messages[0] == first.messages[0]
    assert len(env.recall.queries) == 1  # searched once, at the start of the run


async def test_no_open_run_means_no_search(env: ContextEnv) -> None:
    env.recall.add(memory_item("mem_1", FACT, agent_id=AGENT_ID), hit=True)
    await env.user(QUESTION)

    built = await env.manager.build(AGENT_ID)

    assert env.recall.queries == []
    assert FACT not in dump(built.request.messages[0])


async def test_inspecting_never_searches(env: ContextEnv) -> None:
    env.recall.add(memory_item("mem_1", FACT, agent_id=AGENT_ID), hit=True)
    asked = await env.user(QUESTION)
    await env.open_run(asked.seq)

    await env.manager.inspect(AGENT_ID)

    assert env.recall.queries == []


async def test_expired_and_pinned_items_are_not_recalled_twice(env: ContextEnv) -> None:
    expired = memory_item("mem_old", "expired fact", valid_until=env.clock.now())
    env.recall.add(expired, hit=True)
    env.recall.add(memory_item("mem_pin", PINNED_FACT, pinned=True), hit=True)
    kept = env.recall.add(memory_item("mem_1", FACT, agent_id=AGENT_ID), hit=True)
    asked = await env.user(QUESTION)
    await env.open_run(asked.seq)

    built = await env.manager.build(AGENT_ID)

    assert built.report.retrieved_memory_ids == [kept.id]
    assert PINNED_FACT in str(built.request.system)
    assert PINNED_FACT not in dump(built.request.messages[0])


async def test_recalled_memories_stop_at_their_budget(env: ContextEnv) -> None:
    env.shrink_window()
    long_fact = "detail " * 200
    for number in range(8):
        env.recall.add(memory_item(f"mem_{number}", f"{number}: {long_fact}"), hit=True)
    asked = await env.user(QUESTION)
    await env.open_run(asked.seq)

    report = (await env.manager.build(AGENT_ID)).report

    memories = next(section for section in report.sections if section.name == "memories")
    assert memories.budget_tokens == int(SMALL_WINDOW * 0.05)
    assert 0 < memories.items < len(env.recall.hits)
    assert memories.tokens <= memories.budget_tokens + 10


async def test_the_oldest_pinned_items_go_when_the_system_part_is_full(env: ContextEnv) -> None:
    env.shrink_window()
    big = "pinned detail " * 150
    for number in range(6):
        item = memory_item(f"mem_{number}", f"pin {number}: {big}", pinned=True)
        env.recall.add(
            item.model_copy(update={"created_at": env.clock.now().replace(year=2020 + number)})
        )
    await env.user(QUESTION)

    first = await env.manager.build(AGENT_ID)
    await env.manager.build(AGENT_ID)

    system = str(first.request.system)
    pinned = next(section for section in first.report.sections if section.name == "pinned")
    assert 0 < pinned.items < len(env.recall.items)
    assert "pin 5:" in system  # the newest stays
    assert "pin 0:" not in system  # the oldest went
    assert [code for code, _ in env.notices.warnings] == [ErrorCode.CONTEXT_PINNED_TRIMMED]
