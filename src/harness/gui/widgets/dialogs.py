# pyright: basic
"""Dialogs: create / edit an agent, and settings (a face over the config manager)."""

import tkinter as tk
from collections.abc import Callable
from pathlib import Path

import customtkinter as ctk

from harness.core.commands import Command, ConfigSet, ConfigSetSecret, ProfileUpdate
from harness.core.consts import ConfigLayer
from harness.core.handle import FrontendInfo
from harness.core.records import AgentState
from harness.gui import consts
from harness.gui.forms import (
    AgentForm,
    FormError,
    build_embedder_url,
    key_status,
    model_providers,
    models_of,
    provider_of,
    split_lines,
)
from harness.gui.themes import Theme, available
from harness.gui.widgets.panels import button

_PAD = 12
_KEY_THEME = "gui.theme"
_MASK_CHAR = "*"
_TOOL_SEPARATOR = ","


def _label(master: tk.Misc, theme: Theme, text: str) -> ctk.CTkLabel:
    return ctk.CTkLabel(
        master,
        text=text,
        anchor="w",
        text_color=theme.color("muted"),
        font=(theme.font_mono, theme.font_size - 1),
    )


def _entry(master: tk.Misc, theme: Theme, **options: object) -> ctk.CTkEntry:
    return ctk.CTkEntry(
        master,
        fg_color=theme.color("surface"),
        text_color=theme.color("foreground"),
        border_color=theme.color("border"),
        font=(theme.font_text, theme.font_size),
        **options,  # type: ignore[arg-type]
    )


def _textbox(master: tk.Misc, theme: Theme, height: int) -> ctk.CTkTextbox:
    return ctk.CTkTextbox(
        master,
        height=height,
        fg_color=theme.color("surface"),
        text_color=theme.color("foreground"),
        border_color=theme.color("border"),
        border_width=1,
        font=(theme.font_text, theme.font_size),
    )


def _menu(
    master: tk.Misc, theme: Theme, values: list[str], variable: tk.StringVar
) -> ctk.CTkOptionMenu:
    return ctk.CTkOptionMenu(
        master,
        values=values,
        variable=variable,
        fg_color=theme.color("surface"),
        button_color=theme.color("border"),
        button_hover_color=theme.color("selection"),
        text_color=theme.color("foreground"),
        font=(theme.font_mono, theme.font_size),
    )


class _Dialog(ctk.CTkToplevel):
    """A modal window in the theme's colours."""

    def __init__(self, master: tk.Misc, theme: Theme, title: str, width: int, height: int) -> None:
        super().__init__(master, fg_color=theme.color("background"))
        self.title(title)
        self.geometry(f"{width}x{height}")
        self.transient(master)  # type: ignore[arg-type]
        self.grid_columnconfigure(0, weight=1)
        self.bind("<Escape>", lambda _event: self.destroy())
        self.after(100, self._focus)

    def _focus(self) -> None:
        self.lift()
        self.focus_force()
        self.grab_set()


class AgentDialog(_Dialog):
    """Create an agent, or edit one (02 §6)."""

    def __init__(
        self,
        master: tk.Misc,
        theme: Theme,
        info: FrontendInfo,
        on_submit: Callable[[AgentForm], None],
        agent: AgentState | None = None,
    ) -> None:
        title = consts.LABEL_NEW_AGENT if agent is None else f"{consts.LABEL_EDIT} {agent.name}"
        super().__init__(master, theme, title, consts.DIALOG_WIDTH, consts.DIALOG_HEIGHT)
        self._on_submit, self._agent, self._theme = on_submit, agent, theme
        settings = agent.settings if agent is not None else None
        self._name = _entry(self, theme)
        self._description = _entry(self, theme)
        self._info = info
        self._model = tk.StringVar(value=agent.model if agent else info.models[0])
        self._provider = tk.StringVar(value=provider_of(info, self._model.get()))
        self._model_menu = _menu(self, theme, self._models(), self._model)
        provider_menu = _menu(self, theme, model_providers(info), self._provider)
        provider_menu.configure(command=self._provider_chosen)
        self._template = tk.StringVar(value=consts.LABEL_NO_TEMPLATE)
        self._prompt = _textbox(self, theme, 110)
        self._working_dir = _entry(self, theme)
        self._paths = _textbox(self, theme, 60)
        self._extra_tools = _entry(self, theme)
        self._groups = {group: tk.BooleanVar(value=agent is None) for group in info.tool_groups}
        self._error = _label(self, theme, "")
        self._error.configure(text_color=theme.color("error"))

        rows: list[tuple[str, tk.Widget]] = [
            (consts.LABEL_NAME, self._name),
            (consts.LABEL_DESCRIPTION, self._description),
        ]
        if agent is None:
            templates = [consts.LABEL_NO_TEMPLATE, *info.templates]
            rows.append((consts.LABEL_TEMPLATE, _menu(self, theme, templates, self._template)))
        rows += [
            (consts.LABEL_PROVIDER, provider_menu),
            (consts.LABEL_MODEL, self._model_menu),
            (consts.LABEL_SYSTEM_PROMPT, self._prompt),
            (consts.LABEL_TOOLS, self._tools_row()),
            (consts.LABEL_WORKING_DIR, self._working_dir),
            (consts.LABEL_ALLOWED_PATHS, self._paths),
        ]
        for index, (text, widget) in enumerate(rows):
            _label(self, theme, text).grid(
                row=2 * index, column=0, sticky="ew", padx=_PAD, pady=(8, 0)
            )
            widget.grid(row=2 * index + 1, column=0, sticky="ew", padx=_PAD)
        last = 2 * len(rows)
        self._error.grid(row=last, column=0, sticky="ew", padx=_PAD, pady=(6, 0))
        buttons = ctk.CTkFrame(self, fg_color="transparent")
        buttons.grid(row=last + 1, column=0, sticky="e", padx=_PAD, pady=_PAD)
        confirm = consts.LABEL_CREATE if agent is None else consts.LABEL_SAVE
        button(buttons, theme, confirm, self._submit).grid(row=0, column=0, padx=4)
        button(buttons, theme, consts.LABEL_CANCEL, self.destroy).grid(row=0, column=1, padx=4)

        if agent is not None and settings is not None:
            self._name.insert(0, agent.name)
            self._description.insert(0, agent.description)
            self._working_dir.insert(0, settings.working_dir or "")
            self._paths.insert("1.0", "\n".join(settings.allowed_paths))
            for group, variable in self._groups.items():
                variable.set(group in agent.tools)
            others = [name for name in agent.tools if name not in self._groups]
            self._extra_tools.insert(0, f"{_TOOL_SEPARATOR} ".join(others))
        self._name.focus_set()

    def _models(self) -> list[str]:
        """The models of the chosen provider (every model when none is configured for it)."""
        return models_of(self._info, self._provider.get()) or list(self._info.models)

    def _provider_chosen(self, _provider: str) -> None:
        models = self._models()
        self._model_menu.configure(values=models)
        self._model.set(models[0])

    def _tools_row(self) -> ctk.CTkFrame:
        theme = self._theme
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.grid_columnconfigure(len(self._groups), weight=1)
        for column, (group, variable) in enumerate(self._groups.items()):
            ctk.CTkCheckBox(
                frame,
                text=group,
                variable=variable,
                checkbox_width=16,
                checkbox_height=16,
                text_color=theme.color("foreground"),
                font=(theme.font_mono, theme.font_size),
            ).grid(row=0, column=column, padx=(0, 10))
        self._extra_tools = _entry(frame, theme, placeholder_text="other tools, comma separated")
        self._extra_tools.grid(row=0, column=len(self._groups), sticky="ew")
        return frame

    def _form(self) -> AgentForm:
        chosen = [group for group, variable in self._groups.items() if variable.get()]
        extra = [n.strip() for n in self._extra_tools.get().split(_TOOL_SEPARATOR) if n.strip()]
        template = self._template.get()
        return AgentForm(
            name=self._name.get(),
            description=self._description.get(),
            model=self._model.get(),
            system_prompt=self._prompt.get("1.0", "end-1c"),
            tools=[*chosen, *extra],
            template=None if template == consts.LABEL_NO_TEMPLATE else template,
            working_dir=self._working_dir.get(),
            allowed_paths=split_lines(self._paths.get("1.0", "end-1c")),
        )

    def _submit(self) -> None:
        try:
            self._on_submit(self._form())
        except FormError as error:
            self._error.configure(text=str(error))
            return
        self.destroy()


class SettingsDialog(_Dialog):
    """Appearance, provider keys, and the user profile; every save is a command (02 §6)."""

    def __init__(
        self,
        master: tk.Misc,
        theme: Theme,
        info: FrontendInfo,
        submit: Callable[[Command], None],
    ) -> None:
        super().__init__(
            master, theme, consts.LABEL_SETTINGS, consts.SETTINGS_WIDTH, consts.SETTINGS_HEIGHT
        )
        self._submit, self._theme = submit, theme
        self.grid_rowconfigure(0, weight=1)
        tabs = ctk.CTkTabview(
            self,
            fg_color=theme.color("background"),
            segmented_button_fg_color=theme.color("surface"),
            segmented_button_selected_color=theme.color("selection"),
            segmented_button_unselected_color=theme.color("surface"),
            text_color=theme.color("foreground"),
        )
        tabs.grid(row=0, column=0, sticky="nsew", padx=_PAD, pady=_PAD)
        self._look(tabs.add(consts.LABEL_TAB_LOOK), info)
        self._keys(tabs.add(consts.LABEL_TAB_KEYS), info)
        self._memory(tabs.add(consts.LABEL_TAB_MEMORY), info)
        self._profile(tabs.add(consts.LABEL_TAB_PROFILE), info)

    def _look(self, tab: tk.Misc, info: FrontendInfo) -> None:
        theme = self._theme
        tab.grid_columnconfigure(0, weight=1)
        chosen = tk.StringVar(value=info.theme)
        _label(tab, theme, consts.LABEL_THEME).grid(row=0, column=0, sticky="ew", pady=(8, 0))
        names = available(Path(info.themes_dir))
        menu = _menu(tab, theme, names, chosen)
        menu.configure(
            command=lambda name: self._submit(
                ConfigSet(key=_KEY_THEME, value=name, layer=ConfigLayer.GLOBAL)
            )
        )
        menu.grid(row=1, column=0, sticky="w")

    def _keys(self, tab: tk.Misc, info: FrontendInfo) -> None:
        theme = self._theme
        tab.grid_columnconfigure(0, weight=1)
        note = _label(tab, theme, "")
        for index, provider in enumerate(info.providers):
            text = consts.LABEL_API_KEY.format(provider=provider, status=key_status(info, provider))
            title = _label(tab, theme, text)
            if provider in info.keys:
                title.configure(text_color=theme.color("success"))
            title.grid(row=2 * index, column=0, sticky="ew", pady=(8, 0))
            hint = consts.MSG_KEY_REPLACE if provider in info.keys else ""
            entry = _entry(tab, theme, show=_MASK_CHAR, placeholder_text=hint)
            entry.grid(row=2 * index + 1, column=0, sticky="ew")

            def save(
                name: str = provider, field: ctk.CTkEntry = entry, shown: ctk.CTkLabel = title
            ) -> None:
                value = field.get().strip()
                if value:
                    self._submit(ConfigSetSecret(secret_name=name, value=value))
                    field.delete(0, "end")
                    note.configure(text=consts.MSG_KEY_SAVED)
                    shown.configure(
                        text=consts.LABEL_API_KEY.format(
                            provider=name, status=consts.MSG_KEY_JUST_SAVED
                        ),
                        text_color=theme.color("success"),
                    )

            button(tab, theme, consts.LABEL_SAVE, save).grid(row=2 * index + 1, column=1, padx=6)
        note.grid(row=2 * len(info.providers), column=0, sticky="ew", pady=8)

    def _memory(self, tab: tk.Misc, info: FrontendInfo) -> None:
        theme = self._theme
        tab.grid_columnconfigure(0, weight=1)
        _label(tab, theme, consts.LABEL_EMBEDDER_URL).grid(
            row=0, column=0, sticky="ew", pady=(8, 0)
        )
        entry = _entry(tab, theme)
        entry.insert(0, info.embedder_url)
        entry.grid(row=1, column=0, sticky="ew")
        note = _label(tab, theme, "")
        note.grid(row=2, column=0, sticky="ew", pady=8)

        def save() -> None:
            try:
                command = build_embedder_url(entry.get())
            except FormError as error:
                note.configure(text=str(error), text_color=theme.color("error"))
                return
            self._submit(command)
            note.configure(text=consts.MSG_EMBEDDER_SAVED, text_color=theme.color("muted"))

        button(tab, theme, consts.LABEL_SAVE, save).grid(row=1, column=1, padx=6)

    def _profile(self, tab: tk.Misc, info: FrontendInfo) -> None:
        theme = self._theme
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)
        _label(tab, theme, consts.LABEL_PROFILE).grid(row=0, column=0, sticky="ew", pady=(8, 0))
        box = _textbox(tab, theme, 300)
        box.insert("1.0", info.user_profile)
        box.grid(row=1, column=0, sticky="nsew")
        save = button(
            tab,
            theme,
            consts.LABEL_SAVE,
            lambda: self._submit(ProfileUpdate(content=box.get("1.0", "end-1c"))),
        )
        save.grid(row=2, column=0, sticky="e", pady=8)
