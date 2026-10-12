# pyright: basic
"""The main window: polls the core's events and turns clicks into commands (02 §3)."""

from pathlib import Path
from tkinter import messagebox

import customtkinter as ctk
from loguru import logger

from harness.core import commands as c
from harness.core.consts import AgentStatus, PermissionAnswer, PermissionScope
from harness.core.errors import SlashError
from harness.core.records import PermissionRequestPayload
from harness.core.types import AgentId
from harness.core.utils.slash import parse_input
from harness.gui import consts
from harness.gui.bridge import BridgeError, CoreBridge
from harness.gui.consts import Region
from harness.gui.forms import AgentForm, build_create, build_update, header_line, status_line
from harness.gui.state import AgentView, AppState
from harness.gui.themes import Theme, load_theme
from harness.gui.widgets.dialogs import AgentDialog, SettingsDialog
from harness.gui.widgets.panels import ApprovalPanel, InputBox, Sidebar, button
from harness.gui.widgets.stream import StreamView

_APPROVAL_MARGIN = 10


class MainWindow(ctk.CTk):
    """Agents on the left, the selected conversation in the middle, status at the bottom."""

    def __init__(self, bridge: CoreBridge) -> None:
        super().__init__()
        self._bridge = bridge
        self._state = AppState()
        self._show_archived = False
        self._closing = False
        info = bridge.info()
        self._theme_key: tuple[str, str, int] | None = None
        self._theme = self._load_theme()
        self.title(consts.TITLE)
        self.geometry(f"{info.window_width}x{info.window_height}")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._build()
        self._apply_theme()
        self._bind_shortcuts()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self._submit(c.AgentList(include_archived=True))
        self.after(consts.POLL_INTERVAL_MS, self._tick)

    # --- construction ---

    def _load_theme(self) -> Theme:
        info = self._bridge.info()
        self._theme_key = (info.theme, info.font_family, info.font_size)
        theme, problem = load_theme(
            info.theme, Path(info.themes_dir), info.font_family, info.font_size
        )
        if problem is not None:
            self._state.set_banner(problem)
        return theme

    def _build(self) -> None:
        theme = self._theme
        self._sidebar = Sidebar(
            self,
            theme,
            on_select=self._select,
            on_new=self._new_agent,
            on_settings=self._open_settings,
            on_archived=self._toggle_archived,
        )
        self._sidebar.grid(row=0, column=0, rowspan=3, sticky="nsew")
        self._header = ctk.CTkFrame(self, corner_radius=0, height=36)
        self._header.grid(row=0, column=1, sticky="ew")
        self._header.grid_columnconfigure(0, weight=1)
        self._title = ctk.CTkLabel(self._header, anchor="w")
        self._title.grid(row=0, column=0, sticky="ew", padx=14, pady=6)
        actions = (
            (consts.LABEL_EDIT, self._edit_agent),
            (consts.LABEL_ARCHIVE, self._archive),
            (consts.LABEL_DELETE, self._delete),
        )
        self._actions = []
        for column, (label, command) in enumerate(actions, start=1):
            widget = button(self._header, theme, label, command)
            widget.configure(width=70)
            widget.grid(row=0, column=column, padx=3, pady=4)
            self._actions.append(widget)
        self._stream = StreamView(self, theme)
        self._stream.grid(row=1, column=1, sticky="nsew")
        self._approval = ApprovalPanel(self._stream, theme, self._answer)
        self._input = InputBox(
            self, theme, on_send=self._send, on_stop=self._stop, on_extra=self._extra
        )
        self._input.grid(row=2, column=1, sticky="ew", padx=10, pady=(6, 8))
        self._status = ctk.CTkLabel(self, anchor="w", height=24)
        self._status.grid(row=3, column=0, columnspan=2, sticky="ew")

    def _apply_theme(self) -> None:
        theme = self._theme
        color = theme.color
        ctk.set_appearance_mode("dark" if theme.dark else "light")
        self.configure(fg_color=color("background"))
        self.attributes("-alpha", theme.opacity)
        self._header.configure(fg_color=color("background"))
        self._title.configure(text_color=color("muted"), font=(theme.font_text, theme.font_size))
        for widget in self._actions:
            widget.configure(
                fg_color=color("background"),
                hover_color=color("selection"),
                text_color=color("muted"),
                border_color=color("border"),
                font=(theme.font_mono, theme.font_size - 1),
            )
        self._status.configure(
            fg_color=color("sidebar_bg"), font=(theme.font_mono, theme.font_size - 1)
        )
        for part in (self._sidebar, self._stream, self._approval, self._input):
            part.apply_theme(theme)
        self._refresh(set(Region))

    def _bind_shortcuts(self) -> None:
        shortcuts = {**consts.SHORTCUT_DEFAULTS, **self._bridge.info().shortcuts}
        actions = {
            consts.SHORTCUT_SEND: self._input.send,
            consts.SHORTCUT_STOP: self._stop,
            consts.SHORTCUT_NEW_AGENT: self._new_agent,
            consts.SHORTCUT_NEXT_AGENT: self._next_agent,
        }
        for name, action in actions.items():
            try:
                self.bind(shortcuts[name], lambda _event, run=action: (run(), "break")[1])
            except Exception:  # a bad key name in the config must not stop the window
                logger.warning("invalid shortcut {} = {}", name, shortcuts[name])

    # --- the loop ---

    def _tick(self) -> None:
        if self._closing:
            return
        try:
            for event in self._bridge.poll():
                self._state.apply(event)
            if not self._bridge.running and self._bridge.failure is not None:
                self._state.set_banner(str(self._bridge.failure))
            self._refresh(self._state.take_changes())
        except Exception:  # one bad frame must not kill the window
            logger.exception("GUI tick failed")
        self.after(consts.POLL_INTERVAL_MS, self._tick)

    def _refresh(self, changed: set[Region]) -> None:
        state = self._state
        view = state.current()
        if Region.SETTINGS in changed:
            info = self._bridge.info()
            if (info.theme, info.font_family, info.font_size) != self._theme_key:
                self._theme = self._load_theme()
                self._apply_theme()
                return
        if Region.SIDEBAR in changed:
            self._sidebar.refresh(state.ordered(archived=self._show_archived), state.selected)
        if Region.STREAM in changed:
            self._stream.show(view.conversation if view is not None else None)
        if Region.APPROVAL in changed or Region.STREAM in changed:
            self._place_approval(view)
        if changed & {Region.STATUS, Region.SIDEBAR, Region.STREAM}:
            self._refresh_bars(view)

    def _place_approval(self, view: AgentView | None) -> None:
        if self._approval.show(view):
            self._approval.place(relx=0.5, rely=1.0, anchor="s", relwidth=0.96, y=-_APPROVAL_MARGIN)
            self._approval.lift()
        else:
            self._approval.place_forget()

    def _refresh_bars(self, view: AgentView | None) -> None:
        color = self._theme.color
        banner = self._state.banner
        self._status.configure(
            text=f"  {banner or status_line(view)}",
            text_color=color("error") if banner else color("muted"),
        )
        self._title.configure(text=header_line(view))
        status = view.state.status if view is not None else None
        self._input.set_state(status)
        archived = status is AgentStatus.ARCHIVED
        self._actions[1].configure(
            text=consts.LABEL_UNARCHIVE if archived else consts.LABEL_ARCHIVE
        )
        for widget in self._actions:
            widget.configure(state="normal" if view is not None else "disabled")

    # --- commands ---

    def _submit(self, command: c.Command) -> None:
        try:
            self._bridge.submit(command)
        except BridgeError as error:
            self._state.set_banner(str(error))

    def _current_id(self) -> AgentId | None:
        view = self._state.current()
        return view.state.id if view is not None else None

    def _select(self, agent_id: AgentId) -> None:
        self._state.select(agent_id)
        view = self._state.current()
        if view is not None and not view.loaded:
            self._submit(c.AgentLoadHistory(agent_id=agent_id))

    def _next_agent(self) -> None:
        views = self._state.ordered(archived=self._show_archived)
        if not views:
            return
        ids = [view.state.id for view in views]
        current = self._state.selected
        position = ids.index(current) + 1 if current is not None and current in ids else 0
        self._select(ids[position % len(ids)])

    def _toggle_archived(self, shown: bool) -> None:
        self._show_archived = shown
        self._sidebar.refresh(self._state.ordered(archived=shown), self._state.selected)

    def _send(self, text: str) -> None:
        agent_id = self._current_id()
        if agent_id is None:
            self._state.set_banner(consts.MSG_NO_AGENT)
            return
        self._state.set_banner(None)
        try:
            self._submit(parse_input(text, agent_id))
        except SlashError as error:
            self._state.set_banner(error.message)

    def _stop(self) -> None:
        agent_id = self._current_id()
        if agent_id is not None:
            self._submit(c.AgentStop(agent_id=agent_id))

    def _extra(self) -> None:
        view = self._state.current()
        if view is None:
            return
        agent_id, status = view.state.id, view.state.status
        if status is AgentStatus.PAUSED:
            self._submit(c.AgentResume(agent_id=agent_id))
        elif status is AgentStatus.ERROR:
            self._submit(c.AgentRetry(agent_id=agent_id))
        else:
            self._submit(c.AgentPause(agent_id=agent_id))

    def _answer(
        self, request: PermissionRequestPayload, answer: PermissionAnswer, scope: PermissionScope
    ) -> None:
        self._submit(
            c.PermissionRespond(request_id=request.request_id, decision=answer, scope=scope)
        )

    def _new_agent(self) -> None:
        AgentDialog(
            self,
            self._theme,
            self._bridge.info(),
            lambda form: self._submit(build_create(form)),
        )

    def _edit_agent(self) -> None:
        view = self._state.current()
        if view is None:
            return

        def save(form: AgentForm) -> None:
            command = build_update(view.state, form)
            if command is not None:
                self._submit(command)

        AgentDialog(self, self._theme, self._bridge.info(), save, agent=view.state)

    def _archive(self) -> None:
        view = self._state.current()
        if view is None:
            return
        if view.state.status is AgentStatus.ARCHIVED:
            self._submit(c.AgentUnarchive(agent_id=view.state.id))
        else:
            self._submit(c.AgentArchive(agent_id=view.state.id))

    def _delete(self) -> None:
        view = self._state.current()
        if view is None:
            return
        question = consts.MSG_DELETE_QUESTION.format(name=view.state.name)
        if messagebox.askyesno(consts.LABEL_DELETE, question, parent=self):
            self._submit(c.AgentDelete(agent_id=view.state.id))

    def _open_settings(self) -> None:
        SettingsDialog(self, self._theme, self._bridge.info(), self._submit)

    # --- closing ---

    def close(self) -> None:
        """Ask what to do with running agents, then shut the core down (02 §8 G6)."""
        active = self._state.active()
        stop_agents = False
        if active:
            question = "\n\n".join(
                (
                    consts.MSG_CLOSE_QUESTION.format(count=len(active)),
                    f"Yes: {consts.LABEL_STOP_AGENTS}",
                    f"No: {consts.LABEL_LEAVE_AGENTS}",
                )
            )
            answer = messagebox.askyesnocancel(consts.MSG_CLOSE_TITLE, question, parent=self)
            if answer is None:
                return
            stop_agents = answer
        self._closing = True
        self.withdraw()
        self._bridge.stop(stop_agents=stop_agents)
        self.destroy()
