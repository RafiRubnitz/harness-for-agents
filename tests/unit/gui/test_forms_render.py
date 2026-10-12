"""Forms to commands, state to bar text, and items to display lines."""

import pytest

from harness.core.consts import AgentStatus, ConfigLayer, HookEvent
from harness.core.events import UsageUpdate
from harness.core.handle import FrontendInfo
from harness.core.records import ApprovalPreview, PermissionRequestPayload
from harness.core.types import Usage
from harness.gui import consts
from harness.gui.consts import NoticeLevel, Tag, ToolStatus
from harness.gui.forms import (
    AgentForm,
    FormError,
    approval_lines,
    approval_title,
    build_create,
    build_embedder_url,
    build_update,
    header_line,
    key_status,
    model_providers,
    models_of,
    provider_of,
    split_lines,
    status_line,
)
from harness.gui.items import NoticeItem, ThinkingItem, ToolItem, UserItem
from harness.gui.render import foldable, item_lines, item_text, tool_summary
from harness.gui.state import AgentView
from tests.consts import AGENT_ID, TEST_MODEL_ALIAS
from tests.unit.gui.conftest import REASON, REQUEST_ID, TOOL, agent_state, permission_request

NAME = "helper"
PROMPT = "be brief"
TEMPLATE = "researcher"
WORKING_DIR = "C:/work"
PATHS = ["C:/a", "C:/b"]
COST = 0.25
TOKENS = 1500
WINDOW = 100_000
MODEL_PROVIDERS = {"main": "anthropic", "fast": "anthropic", "deepseek": "deepseek"}
MASKED_KEY = "sk-...a3f"
EMBEDDER_URL = "http://192.168.1.20:8765"


def _form(**changes: object) -> AgentForm:
    fields: dict[str, object] = {"name": NAME, "model": TEST_MODEL_ALIAS, "tools": [TOOL]}
    return AgentForm(**{**fields, **changes})  # type: ignore[arg-type]


def _info(**changes: object) -> FrontendInfo:
    fields: dict[str, object] = {
        "theme": "dark",
        "font_family": "Consolas",
        "font_size": 12,
        "window_width": 800,
        "window_height": 600,
        "shortcuts": {},
        "themes_dir": "",
        "models": list(MODEL_PROVIDERS),
        "templates": [],
        "tools": [],
        "tool_groups": [],
        "providers": ["anthropic", "openai", "deepseek"],
        "user_profile": "",
        "model_providers": MODEL_PROVIDERS,
    }
    return FrontendInfo.model_validate({**fields, **changes})


def test_models_are_offered_by_provider() -> None:
    info = _info()

    assert model_providers(info) == ["anthropic", "deepseek"]
    assert models_of(info, "anthropic") == ["main", "fast"]
    assert models_of(info, "deepseek") == ["deepseek"]
    assert models_of(info, "openai") == []
    assert provider_of(info, "deepseek") == "deepseek"
    assert provider_of(info, "some-raw-model-id") == "anthropic"


def test_a_provider_key_is_reported_as_set_with_its_masked_form() -> None:
    info = _info(keys={"anthropic": MASKED_KEY})

    assert MASKED_KEY in key_status(info, "anthropic")
    assert key_status(info, "openai") == consts.MSG_KEY_NOT_SET


def test_the_embeddings_url_form_builds_a_global_setting() -> None:
    command = build_embedder_url(f"  {EMBEDDER_URL} ")

    assert (command.key, command.value, command.layer) == (
        consts.KEY_EMBEDDER_URL,
        EMBEDDER_URL,
        ConfigLayer.GLOBAL,
    )
    assert build_embedder_url("  ").value is None
    with pytest.raises(FormError):
        build_embedder_url("localhost:8765")


def test_split_lines_drops_blanks() -> None:
    assert split_lines(f"  {PATHS[0]}\n\n{PATHS[1]}  \n") == PATHS


def test_create_without_a_template_sends_everything() -> None:
    form = _form(system_prompt=PROMPT, working_dir=WORKING_DIR, allowed_paths=PATHS)

    command = build_create(form)

    assert (command.agent_name, command.model, command.system_prompt) == (
        NAME,
        TEST_MODEL_ALIAS,
        PROMPT,
    )
    assert command.tools == [TOOL]
    assert command.settings is not None
    assert (command.settings.working_dir, command.settings.allowed_paths) == (WORKING_DIR, PATHS)


def test_create_from_a_template_sends_only_what_was_filled_in() -> None:
    command = build_create(_form(template=TEMPLATE, tools=[]))

    assert command.template == TEMPLATE
    assert command.model_fields_set == {"agent_name", "template"}


@pytest.mark.parametrize("build", [build_create, lambda form: build_update(agent_state(), form)])
def test_a_name_is_required(build: object) -> None:
    with pytest.raises(FormError):
        build(_form(name="   "))  # type: ignore[operator]


def test_update_sends_only_the_changes() -> None:
    agent = agent_state(name=NAME, tools=[TOOL])

    unchanged = build_update(agent, _form())
    renamed = build_update(agent, _form(name="other", working_dir=WORKING_DIR))

    assert unchanged is None
    assert renamed is not None
    assert set(renamed.changes) == {"name", "settings"}


def test_status_line_shows_status_tokens_and_cost() -> None:
    view = AgentView(agent_state(status=AgentStatus.RUNNING))
    view.usage = UsageUpdate(
        usage=Usage(input=1, output=1),
        cost_usd_total=COST,
        context_tokens=TOKENS,
        context_window=WINDOW,
        context_pct=TOKENS / WINDOW,
        model=TEST_MODEL_ALIAS,
    )

    line = status_line(view)

    assert AgentStatus.RUNNING.value in line
    assert consts.STATUS_COST.format(cost=COST) in line
    assert consts.STATUS_K.format(value=TOKENS / consts.TOKENS_PER_K) in line
    assert status_line(None) == consts.MSG_NO_AGENT


def test_header_line_names_the_agent() -> None:
    view = AgentView(agent_state(AGENT_ID, name=NAME))

    assert NAME in header_line(view)
    assert header_line(None) == consts.TITLE


def test_approval_panel_shows_diff_colours_and_the_permanent_warning() -> None:
    diff = "--- a\n+++ b\n-old\n+new\n same"
    request = PermissionRequestPayload(
        request_id=REQUEST_ID,
        subject=HookEvent.BEFORE_TOOL_CALL,
        tool=TOOL,
        reason=REASON,
        preview=ApprovalPreview(diff=diff, path="note.txt", permanent=True),
    )

    lines = approval_lines(request)

    by_text = {line.text: line for line in lines}
    assert REASON in lines[0].text
    assert Tag.SUCCESS in by_text["+new"].spans[0].tags
    assert Tag.ERROR in by_text["-old"].spans[0].tags
    assert consts.MSG_APPROVAL_PERMANENT in by_text
    assert not by_text["+new"].bidi


def test_approval_title_counts_the_waiting_requests() -> None:
    view = AgentView(agent_state(name=NAME))
    view.approvals = [permission_request(), permission_request()]

    title = approval_title(view)

    assert NAME in title
    assert str(len(view.approvals)) in title


def test_tool_row_folds_and_unfolds() -> None:
    output = "line one\nline two"
    item = ToolItem("c1", TOOL, {"path": "note.txt", "count": 3, "nested": {"a": 1}}, REASON)
    item.status, item.output = ToolStatus.OK, output

    folded = item_lines(item)
    opened = item_lines(item, expanded=True)

    assert foldable(item)
    assert len([line for line in folded if line.text]) == 1
    assert folded[0].text.startswith(consts.TOOL_COLLAPSED)
    assert opened[0].text.startswith(consts.TOOL_EXPANDED)
    assert tool_summary(item) == "note.txt 3"
    assert output.splitlines()[-1] in [line.text for line in opened]


def test_failed_tool_shows_its_first_error_line_while_folded() -> None:
    item = ToolItem("c1", TOOL, {})
    item.status, item.output = ToolStatus.FAILED, "boom\ntraceback"

    folded = item_lines(item)

    assert "boom" in folded[1].text
    assert "traceback" not in folded[1].text


def test_user_thinking_and_notice_lines() -> None:
    user = item_lines(UserItem("hello", "[agent] "))
    thinking = ThinkingItem("step one\nstep two")
    notice = item_lines(NoticeItem("careful", NoticeLevel.WARNING))

    assert user[0].text.startswith(consts.PROMPT_SYMBOL)
    assert "hello" in user[0].text
    assert foldable(thinking)
    assert len(item_lines(thinking)) < len(item_lines(thinking, expanded=True))
    assert item_text(thinking) == thinking.text
    assert any("careful" in line.text for line in notice)
