"""The real core behind ``CoreHandle``: commands in, events out (02 §3, §5; 12 §2)."""

import asyncio

import pytest

from harness.core.commands import (
    AgentCreate,
    AgentList,
    AgentLoadHistory,
    AgentSend,
    AgentStop,
    AppShutdown,
    ConfigSet,
    ConfigSetSecret,
    ContextInspect,
)
from harness.core.consts import (
    COMPONENT_COMMANDS,
    AgentStatus,
    ConfigLayer,
    ErrorCode,
    MessageRole,
)
from harness.core.errors import CoreLockedError
from harness.core.events import (
    AgentCreated,
    AgentListed,
    AgentStatusChanged,
    ConfigChanged,
    ContextReported,
    Error,
    HistoryLoaded,
    TextDelta,
)
from harness.core.handle import CoreHandle
from harness.core.lock import CoreLock
from harness.core.types import AgentId, TextBlock
from tests.consts import AGENT_NAME, TEST_API_KEY, TEST_MAX_STEPS
from tests.fakes.adapter import text_response
from tests.integration.conftest import CoreKit, next_payload

QUESTION = "what is up"
ANSWER = "all good"
FOLLOW_UP = "and tomorrow?"
SAID_BEFORE_CLOSING = (QUESTION, ANSWER, FOLLOW_UP)
SECRET_NAME = "anthropic"
WAIT_S = 8.0
MISSING_AGENT = AgentId("agt_missing")
MAX_STEPS_KEY = "agents.defaults.max_steps"
EMBEDDER_URL_KEY = "embedder.url"
EMBEDDER_URL = "http://192.168.1.20:8765"


async def create(core: CoreHandle) -> AgentId:
    events = core.events()
    await core.submit(AgentCreate(agent_name=AGENT_NAME))
    return (await next_payload(events, AgentCreated)).agent.id


async def converse(core: CoreHandle, agent_id: AgentId) -> list[str]:
    """Send the question and return the streamed text once the agent is idle again."""
    events = core.events()
    await core.submit(AgentSend(agent_id=agent_id, content=[TextBlock(text=QUESTION)]))
    streamed: list[str] = []
    async for event in events:
        payload = event.payload
        if isinstance(payload, TextDelta):
            streamed.append(payload.text)
        if isinstance(payload, AgentStatusChanged) and payload.status is AgentStatus.IDLE:
            break
    return streamed


async def test_send_streams_the_answer_and_persists_the_turn(
    core: CoreHandle, kit: CoreKit
) -> None:
    kit.gateway.outcomes = [text_response(ANSWER)]
    agent_id = await create(core)

    streamed = await converse(core, agent_id)
    events = core.events()
    await core.submit(AgentLoadHistory(agent_id=agent_id))
    history = await next_payload(events, HistoryLoaded)

    assert "".join(streamed) == ANSWER
    assert [m.role for m in history.messages] == [MessageRole.USER, MessageRole.ASSISTANT]
    assert history.messages[1].content == [TextBlock(text=ANSWER)]
    assert history.events


async def test_agents_survive_a_restart(kit: CoreKit) -> None:
    kit.gateway.outcomes = [text_response(ANSWER)]
    async with kit.launch() as first:
        agent_id = await create(first)
        await converse(first, agent_id)

    async with kit.launch() as second:
        events = second.events()
        await second.submit(AgentList())
        listed = (await next_payload(events, AgentListed)).agents
        await second.submit(AgentLoadHistory(agent_id=agent_id))
        history = await next_payload(events, HistoryLoaded)

    assert [(a.id, a.name, a.status) for a in listed] == [(agent_id, AGENT_NAME, AgentStatus.IDLE)]
    assert len(history.messages) == len([QUESTION, ANSWER])


@pytest.mark.acceptance("03-10.4")
async def test_closing_in_the_middle_of_a_run_keeps_every_message_so_far(kit: CoreKit) -> None:
    kit.gateway.outcomes = [text_response(ANSWER), text_response(ANSWER), text_response(ANSWER)]
    async with kit.launch() as first:
        agent_id = await create(first)
        await converse(first, agent_id)
        kit.gateway.hold = asyncio.Event()  # never set: the second answer is still streaming
        await first.submit(AgentSend(agent_id=agent_id, content=[TextBlock(text=FOLLOW_UP)]))
        async with asyncio.timeout(WAIT_S):
            await kit.gateway.streaming.wait()
    kit.gateway.hold = None

    async with kit.launch() as second:
        events = second.events()
        await second.submit(AgentLoadHistory(agent_id=agent_id))
        history = await next_payload(events, HistoryLoaded)

    kept = [(m.role, m.content) for m in history.messages[: len(SAID_BEFORE_CLOSING)]]
    assert kept == [
        (role, [TextBlock(text=text)])
        for role, text in zip(
            (MessageRole.USER, MessageRole.ASSISTANT, MessageRole.USER),
            SAID_BEFORE_CLOSING,
            strict=True,
        )
    ]


@pytest.mark.acceptance("06-11.5")
async def test_an_api_key_goes_to_the_keyring_and_into_no_file_or_log(kit: CoreKit) -> None:
    async with kit.launch() as core:
        events = core.events()
        await core.submit(ConfigSetSecret(secret_name=SECRET_NAME, value=TEST_API_KEY))
        await core.submit(AgentList())
        await next_payload(events, AgentListed)  # commands run in order: the secret is stored

    files = [path for path in kit.home.root.rglob("*") if path.is_file()]
    leaked = [path.name for path in files if TEST_API_KEY.encode() in path.read_bytes()]
    assert kit.secrets.values[SECRET_NAME] == TEST_API_KEY
    assert files  # the log and the database were written
    assert leaked == []


async def test_a_saved_api_key_is_shown_as_set_but_never_in_full(kit: CoreKit) -> None:
    async with kit.launch() as core:
        before = core.info().keys
        events = core.events()

        await core.submit(ConfigSetSecret(secret_name=SECRET_NAME, value=TEST_API_KEY))
        changed = await next_payload(events, ConfigChanged)
        after = core.info().keys

    assert before == {}
    assert list(after) == [SECRET_NAME]
    assert after[SECRET_NAME] != TEST_API_KEY
    assert TEST_API_KEY not in changed.model_dump_json()


async def test_each_model_names_its_provider(core: CoreHandle) -> None:
    info = core.info()

    assert set(info.model_providers) == set(info.models)
    assert set(info.model_providers.values()) == set(info.providers)


async def test_the_embeddings_server_url_is_set_and_cleared(core: CoreHandle) -> None:
    events = core.events()

    await core.submit(ConfigSet(key=EMBEDDER_URL_KEY, value=EMBEDDER_URL, layer=ConfigLayer.GLOBAL))
    changed = await next_payload(events, ConfigChanged)
    shown = core.info().embedder_url
    await core.submit(ConfigSet(key=EMBEDDER_URL_KEY, value=None, layer=ConfigLayer.GLOBAL))
    await next_payload(events, ConfigChanged)

    assert shown == EMBEDDER_URL
    assert changed.restart_required
    assert core.info().embedder_url == ""


async def test_failed_command_becomes_an_error_event(core: CoreHandle) -> None:
    events = core.events()

    await core.submit(AgentStop(agent_id=MISSING_AGENT))
    error = await next_payload(events, Error)

    assert (error.code, error.component) == (ErrorCode.AGENT_NOT_FOUND, COMPONENT_COMMANDS)


async def test_config_set_applies_and_reports_the_key(core: CoreHandle) -> None:
    events = core.events()

    await core.submit(ConfigSet(key=MAX_STEPS_KEY, value=TEST_MAX_STEPS, layer=ConfigLayer.GLOBAL))
    changed = await next_payload(events, ConfigChanged)
    await core.submit(AgentCreate(agent_name=AGENT_NAME))
    created = await next_payload(events, AgentCreated)

    assert changed.keys == [MAX_STEPS_KEY]
    assert created.agent.settings.max_steps == TEST_MAX_STEPS


async def test_context_inspect_reports_the_model_window(core: CoreHandle, kit: CoreKit) -> None:
    agent_id = await create(core)
    events = core.events()

    await core.submit(ContextInspect(agent_id=agent_id))
    reported = await next_payload(events, ContextReported)

    assert reported.report.window == kit.gateway.profile.context_window


async def test_shutdown_command_ends_the_event_stream(core: CoreHandle) -> None:
    events = core.events()

    await core.submit(AppShutdown())

    assert [event async for event in events] == []


async def test_second_core_is_refused_and_touches_nothing(kit: CoreKit) -> None:
    with CoreLock(kit.home.lock_file):
        with pytest.raises(CoreLockedError):
            async with kit.launch():
                pass

        assert not kit.home.db_file.exists()
