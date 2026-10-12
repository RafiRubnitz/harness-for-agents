"""Agent lifecycle: create, templates, update, archive, delete, shutdown (04 §4, §8)."""

from pathlib import Path

import pytest

from harness.core.consts import (
    AgentStatus,
    Decision,
    EventName,
    HookEvent,
    RunStopReason,
    TaskStatus,
)
from harness.core.errors import AgentError, AgentNotFoundError, InvalidTransitionError
from harness.core.events import AgentCreated, AgentDeleted, AgentUpdated
from harness.core.records import BackgroundTask
from harness.core.types import TaskId, ToolCallBlock
from tests.consts import AGENT_NAME
from tests.fakes.adapter import text_response, tool_response
from tests.unit.agents.conftest import AgentEnv

QUESTION = "hello there"
ANSWER = "hi"
TOOL = "lookup"
NEW_NAME = "renamed"
TEMPLATE_NAME = "researcher"
TEMPLATE_PROMPT = "You research things."
TEMPLATE_PROMPT_V2 = "You research things carefully."
TEMPLATE_STEPS = 7
TEMPLATE_TOOLS = ["files"]
TASK_ID = TaskId("task_0001")
TEMPLATE_YAML = """\
name: {name}
system_prompt: {prompt}
tools: {tools}
settings:
  max_steps: {steps}
permissions:
  default: allow
"""


def write_template(env: AgentEnv, prompt: str = TEMPLATE_PROMPT) -> None:
    """Put the test template into the harness home."""
    text = TEMPLATE_YAML.format(
        name=TEMPLATE_NAME, prompt=prompt, tools=TEMPLATE_TOOLS, steps=TEMPLATE_STEPS
    )
    path = env.deps.home.templates_dir / f"{TEMPLATE_NAME}.yaml"
    path.write_text(text, encoding="utf-8")


async def test_create_makes_an_idle_agent_with_a_workspace(env: AgentEnv) -> None:
    agent_id = await env.create()

    state = env.manager.get(agent_id)
    stored = await env.store.get_agent(agent_id)
    defaults = env.deps.config.config().agents.defaults
    assert state.status is AgentStatus.IDLE
    assert state.settings.working_dir == str(env.deps.home.workspace_for(agent_id))
    assert Path(str(state.settings.working_dir)).is_dir()
    assert state.settings.max_steps == defaults.max_steps
    assert (stored.name, stored.settings) == (AGENT_NAME, state.settings)
    assert [p.agent for p in env.payloads(AgentCreated)] == [state]


async def test_create_uses_and_creates_a_given_working_dir(env: AgentEnv, tmp_path: Path) -> None:
    wanted = tmp_path / "project" / "nested"

    agent_id = await env.create(working_dir=str(wanted))

    assert env.manager.get(agent_id).settings.working_dir == str(wanted)
    assert wanted.is_dir()


@pytest.mark.parametrize("event", [HookEvent.AGENT_CREATED, HookEvent.AGENT_DELETED])
async def test_hook_deny_cancels_create_and_delete(env: AgentEnv, event: HookEvent) -> None:
    kept = await env.create()
    env.hooks.script(event, Decision.DENY)

    with pytest.raises(AgentError):
        await (env.create() if event is HookEvent.AGENT_CREATED else env.manager.delete(kept))

    assert [a.id for a in env.manager.list()] == [kept]
    assert [a.id for a in await env.store.list_agents()] == [kept]


async def test_update_changes_fields_and_merges_settings(env: AgentEnv) -> None:
    agent_id = await env.create()
    before = env.manager.get(agent_id)

    await env.manager.update(
        agent_id, {"name": NEW_NAME, "settings": {"max_steps": TEMPLATE_STEPS}}
    )

    after = env.manager.get(agent_id)
    stored = await env.store.get_agent(agent_id)
    assert (after.name, after.settings.max_steps) == (NEW_NAME, TEMPLATE_STEPS)
    assert after.settings.working_dir == before.settings.working_dir
    assert (stored.name, stored.settings) == (after.name, after.settings)
    assert [p.agent for p in env.payloads(AgentUpdated)] == [after]


@pytest.mark.parametrize("changes", [{"status": "running"}, {"settings": {"max_steps": "many"}}])
async def test_update_rejects_bad_changes(env: AgentEnv, changes: dict[str, object]) -> None:
    agent_id = await env.create()
    before = env.manager.get(agent_id)

    with pytest.raises(AgentError):
        await env.manager.update(agent_id, changes)  # type: ignore[arg-type]

    assert env.manager.get(agent_id) == before


async def test_create_from_template_copies_the_template(env: AgentEnv) -> None:
    write_template(env)

    agent_id = await env.manager.create_from_template(TEMPLATE_NAME, overrides={"name": NEW_NAME})

    state = env.manager.get(agent_id)
    stored = await env.store.get_agent(agent_id)
    assert (state.name, state.template, state.tools) == (NEW_NAME, TEMPLATE_NAME, TEMPLATE_TOOLS)
    assert state.settings.max_steps == TEMPLATE_STEPS
    assert stored.system_prompt == TEMPLATE_PROMPT
    assert env.deps.config.config(agent_id).permissions.default is Decision.ALLOW
    assert env.deps.config.config().permissions.default is not Decision.ALLOW
    with pytest.raises(AgentError):
        await env.manager.create_from_template(NEW_NAME)


async def test_apply_template_copies_it_again(env: AgentEnv) -> None:
    write_template(env)
    agent_id = await env.manager.create_from_template(TEMPLATE_NAME)
    write_template(env, TEMPLATE_PROMPT_V2)

    await env.manager.apply_template(TEMPLATE_NAME, [agent_id])

    assert (await env.store.get_agent(agent_id)).system_prompt == TEMPLATE_PROMPT_V2


@pytest.mark.acceptance("06-11.7")
async def test_a_changed_template_reaches_only_the_agents_it_is_applied_to(env: AgentEnv) -> None:
    write_template(env)
    chosen = await env.manager.create_from_template(TEMPLATE_NAME)
    other = await env.manager.create_from_template(TEMPLATE_NAME, overrides={"name": NEW_NAME})

    write_template(env, TEMPLATE_PROMPT_V2)
    await env.deps.config.reload()
    untouched = [
        (await env.store.get_agent(agent_id)).system_prompt for agent_id in (chosen, other)
    ]
    await env.manager.apply_template(TEMPLATE_NAME, [chosen])

    assert untouched == [TEMPLATE_PROMPT, TEMPLATE_PROMPT]
    assert (await env.store.get_agent(chosen)).system_prompt == TEMPLATE_PROMPT_V2
    assert (await env.store.get_agent(other)).system_prompt == TEMPLATE_PROMPT


async def test_archive_hides_the_agent_until_unarchived(env: AgentEnv) -> None:
    agent_id = await env.create()

    await env.manager.archive(agent_id)
    listed = [a.id for a in env.manager.list()]
    listed_with_archived = [a.id for a in env.manager.list(include_archived=True)]
    with pytest.raises(InvalidTransitionError):
        await env.say(agent_id, QUESTION)
    await env.manager.unarchive(agent_id)

    assert listed == []
    assert listed_with_archived == [agent_id]
    assert (await env.store.get_agent(agent_id)).status is AgentStatus.IDLE


async def test_archive_closes_a_paused_run(env: AgentEnv) -> None:
    started, release = env.tools.add_blocking(TOOL)
    env.gateway.outcomes = [tool_response(ToolCallBlock(id="call_1", name=TOOL, input={}))]
    agent_id = await env.create()
    await env.say(agent_id, QUESTION)
    await started.wait()
    await env.manager.pause(agent_id)
    release.set()
    await env.until_status(agent_id, AgentStatus.PAUSED)

    await env.manager.archive(agent_id)

    assert env.stop_reasons() == [RunStopReason.ARCHIVED.value]
    assert await env.store.open_run(agent_id) is None
    assert env.manager.get(agent_id).status is AgentStatus.ARCHIVED


async def test_archive_is_refused_while_running(env: AgentEnv) -> None:
    started, _ = env.tools.add_blocking(TOOL)
    env.gateway.outcomes = [tool_response(ToolCallBlock(id="call_1", name=TOOL, input={}))]
    agent_id = await env.create()
    await env.say(agent_id, QUESTION)
    await started.wait()

    with pytest.raises(InvalidTransitionError):
        await env.manager.archive(agent_id)


async def test_delete_stops_the_run_and_removes_everything(env: AgentEnv) -> None:
    started, _ = env.tools.add_blocking(TOOL)
    env.gateway.outcomes = [tool_response(ToolCallBlock(id="call_1", name=TOOL, input={}))]
    agent_id = await env.create()
    other = await env.create()
    env.tools.tasks[agent_id] = [
        BackgroundTask(
            id=TASK_ID,
            call_id="call_0",
            tool=TOOL,
            status=TaskStatus.RUNNING,
            started_at=env.deps.clock.now(),
            duration_ms=None,
        )
    ]
    await env.say(agent_id, QUESTION)
    await started.wait()

    await env.manager.delete(agent_id)

    assert [a.id for a in env.manager.list()] == [other]
    assert [p.agent_id for p in env.payloads(AgentDeleted)] == [agent_id]
    assert env.stop_reasons() == [RunStopReason.DELETED.value]
    assert env.tools.cancelled_tasks == [TASK_ID]
    assert env.names()[-1] is EventName.AGENT_DELETED
    with pytest.raises(AgentNotFoundError):
        env.manager.get(agent_id)
    with pytest.raises(AgentNotFoundError):
        await env.store.get_agent(agent_id)


async def test_shutdown_with_stop_closes_runs_and_ends_the_stream(env: AgentEnv) -> None:
    started, _ = env.tools.add_blocking(TOOL)
    env.gateway.outcomes = [tool_response(ToolCallBlock(id="call_1", name=TOOL, input={}))]
    agent_id = await env.create()
    await env.say(agent_id, QUESTION)
    await started.wait()

    await env.manager.shutdown(stop_agents=True)
    names = [event.name async for event in env.events]

    assert (await env.store.get_agent(agent_id)).status is AgentStatus.IDLE
    assert await env.store.open_run(agent_id) is None
    assert names.count(EventName.AGENT_CREATED) == 1
    assert env.stop_reasons() == [RunStopReason.STOP.value]


async def test_idle_agent_answers_again_after_a_run(env: AgentEnv) -> None:
    env.gateway.outcomes = [text_response(ANSWER), text_response(ANSWER)]
    agent_id = await env.create()

    await env.converse(agent_id, QUESTION)
    await env.converse(agent_id, QUESTION)

    assert env.stop_reasons() == [RunStopReason.END.value] * 2
    assert env.manager.get(agent_id).cost_usd_total == pytest.approx(
        sum(m.cost_usd or 0 for m in await env.store.get_messages(agent_id))
    )
