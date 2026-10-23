"""Pure helpers behind the dialogs and bars: form data to commands, state to text."""

from dataclasses import dataclass, field

from pydantic import JsonValue

from harness.core.commands import AgentCreate, AgentUpdate, ConfigSet
from harness.core.consts import ConfigLayer
from harness.core.handle import FrontendInfo
from harness.core.records import AgentSettings, AgentState, PermissionRequestPayload
from harness.gui import consts
from harness.gui.consts import Tag
from harness.gui.items import clip, shown_input
from harness.gui.markdown import Line, plain
from harness.gui.state import AgentView


class FormError(ValueError):
    """The form cannot be submitted as filled in."""


@dataclass
class AgentForm:
    """What the create / edit dialog collects."""

    name: str
    model: str
    system_prompt: str = ""
    description: str = ""
    tools: list[str] = field(default_factory=list[str])
    template: str | None = None
    working_dir: str = ""
    allowed_paths: list[str] = field(default_factory=list[str])


def split_lines(text: str) -> list[str]:
    """Non-empty trimmed lines of a multi-line field."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def _settings(form: AgentForm, base: AgentSettings) -> AgentSettings:
    return base.model_copy(
        update={
            "working_dir": form.working_dir.strip() or None,
            "allowed_paths": form.allowed_paths,
        }
    )


def build_create(form: AgentForm) -> AgentCreate:
    """The command that creates the agent described by the form.

    With a template, only the fields the user filled in are sent, so the template's own
    model, prompt, and tools stay in force.
    """
    name = form.name.strip()
    if not name:
        raise FormError(consts.MSG_NAME_REQUIRED)
    fields: dict[str, JsonValue] = {"agent_name": name}
    if form.description.strip():
        fields["description"] = form.description.strip()
    settings = _settings(form, AgentSettings())
    if settings != AgentSettings():
        fields["settings"] = settings.model_dump(mode="json")
    if form.template is None:
        fields["model"] = form.model
        fields["system_prompt"] = form.system_prompt
        fields["tools"] = list(form.tools)
        return AgentCreate.model_validate(fields)
    fields["template"] = form.template
    if form.system_prompt.strip():
        fields["system_prompt"] = form.system_prompt
    if form.tools:
        fields["tools"] = list(form.tools)
    return AgentCreate.model_validate(fields)


def build_update(agent: AgentState, form: AgentForm) -> AgentUpdate | None:
    """The command that applies the edited form; ``None`` when nothing changed.

    An empty system prompt means "keep the current one": the dialog cannot show it.
    """
    name = form.name.strip()
    if not name:
        raise FormError(consts.MSG_NAME_REQUIRED)
    changes: dict[str, JsonValue] = {}
    if name != agent.name:
        changes["name"] = name
    if form.description.strip() != agent.description:
        changes["description"] = form.description.strip()
    if form.model != agent.model:
        changes["model"] = form.model
    if form.system_prompt.strip():
        changes["system_prompt"] = form.system_prompt
    if sorted(form.tools) != sorted(agent.tools):
        changes["tools"] = list(form.tools)
    settings = _settings(form, agent.settings)
    if settings != agent.settings:
        changes["settings"] = settings.model_dump(mode="json")
    return AgentUpdate(agent_id=agent.id, changes=changes) if changes else None


def model_providers(info: FrontendInfo) -> list[str]:
    """The providers that serve at least one configured model, in the config's order."""
    return list(dict.fromkeys(info.model_providers.values()))


def models_of(info: FrontendInfo, provider: str) -> list[str]:
    """The model aliases one provider serves."""
    return [alias for alias, served_by in info.model_providers.items() if served_by == provider]


def provider_of(info: FrontendInfo, model: str) -> str:
    """The provider of a model alias; the first provider when the alias is not configured."""
    known = info.model_providers.get(model)
    if known is not None:
        return known
    providers = model_providers(info)
    return providers[0] if providers else ""


def key_status(info: FrontendInfo, provider: str) -> str:
    """Whether a provider already has an API key, with its masked form."""
    masked = info.keys.get(provider)
    if masked is None:
        return consts.MSG_KEY_NOT_SET
    return consts.MSG_KEY_SET.format(masked=masked)


def build_embedder_url(text: str) -> ConfigSet:
    """The command that stores the embeddings server URL; empty means the built-in server."""
    url = text.strip()
    if url and not url.startswith(consts.URL_SCHEMES):
        raise FormError(consts.MSG_URL_INVALID)
    return ConfigSet(key=consts.KEY_EMBEDDER_URL, value=url or None, layer=ConfigLayer.GLOBAL)


def _thousands(tokens: int) -> str:
    if tokens < consts.TOKENS_PER_K:
        return str(tokens)
    return consts.STATUS_K.format(value=tokens / consts.TOKENS_PER_K)


def status_line(view: AgentView | None) -> str:
    """The status bar: status, step, context use, cost, background tasks (14 §2)."""
    if view is None:
        return consts.MSG_NO_AGENT
    state, usage = view.state, view.usage
    parts = [state.status.value, consts.STATUS_MODEL.format(model=state.model)]
    if state.step is not None:
        limit = state.settings.max_steps
        parts.append(
            consts.STATUS_STEP.format(step=state.step, max_steps=limit)
            if limit is not None
            else consts.STATUS_STEP_UNBOUNDED.format(step=state.step)
        )
    if usage is not None:
        parts.append(
            consts.STATUS_TOKENS.format(
                used=_thousands(usage.context_tokens), window=_thousands(usage.context_window)
            )
        )
    cost = usage.cost_usd_total if usage is not None else state.cost_usd_total
    parts.append(consts.STATUS_COST.format(cost=cost))
    if view.running_tasks:
        parts.append(consts.STATUS_TASKS.format(count=view.running_tasks))
    return consts.SEPARATOR.join(parts)


def header_line(view: AgentView | None) -> str:
    """The title above the conversation: name, model, working directory."""
    if view is None:
        return consts.TITLE
    state = view.state
    parts = [state.name, state.model]
    if state.settings.working_dir:
        parts.append(state.settings.working_dir)
    return consts.SEPARATOR.join(parts)


def _diff_lines(diff: str) -> list[Line]:
    rows = diff.splitlines()
    shown = rows[: consts.APPROVAL_MAX_DIFF_LINES]
    lines: list[Line] = []
    for row in shown:
        tag = (
            Tag.SUCCESS if row.startswith("+") else Tag.ERROR if row.startswith("-") else Tag.MUTED
        )
        lines += plain(row, Tag.CODE_BLOCK, tag, bidi=False)
    if len(rows) > len(shown):
        hidden = consts.MSG_APPROVAL_DIFF_CUT.format(lines=len(rows) - len(shown))
        lines += plain(hidden, Tag.MUTED)
    return lines


def approval_lines(request: PermissionRequestPayload) -> list[Line]:
    """The body of the approval panel: the model's reason first, then what will happen."""
    lines: list[Line] = []
    if request.reason:
        lines += plain(consts.MSG_APPROVAL_REASON.format(reason=request.reason), Tag.BOLD)
    for reason in request.hook_reasons:
        lines += plain(consts.MSG_APPROVAL_HOOK.format(reason=reason), Tag.WARNING)
    if request.summary:
        lines += plain(consts.MSG_APPROVAL_SUMMARY.format(summary=request.summary))
    if request.tool:
        risk = request.risk.value if request.risk is not None else ""
        lines += plain(consts.MSG_APPROVAL_TOOL.format(tool=request.tool, risk=risk), Tag.MONO)
    preview = request.preview
    if preview is not None and preview.permanent:
        lines += plain(consts.MSG_APPROVAL_PERMANENT, Tag.ERROR, Tag.BOLD)
    if preview is not None and preview.path:
        lines += plain(consts.MSG_APPROVAL_PATH.format(path=preview.path), Tag.MONO, bidi=False)
    if preview is not None and preview.size is not None:
        lines += plain(consts.MSG_APPROVAL_SIZE.format(size=preview.size), Tag.MONO)
    if preview is not None and preview.command:
        lines += plain(preview.command, Tag.CODE_BLOCK, bidi=False)
    elif preview is not None and preview.diff:
        lines += _diff_lines(preview.diff)
    elif request.input:
        shown = clip(str(shown_input(request.input)), consts.NOTICE_MAX_CHARS)
        lines += plain(consts.MSG_APPROVAL_INPUT.format(input=shown), Tag.MONO, bidi=False)
    return lines


def approval_title(view: AgentView) -> str:
    """The heading of the panel, with a counter when several requests wait (02 §6)."""
    return consts.MSG_APPROVAL_TITLE.format(
        agent=view.state.name, index=1, total=len(view.approvals)
    )
