# pyright: basic
"""The fixed parts around the stream: agent list, approval card, input box."""

import tkinter as tk
from collections.abc import Callable
from tkinter import font as tkfont

import customtkinter as ctk

from harness.core.consts import AgentStatus, PermissionAnswer, PermissionScope
from harness.core.records import PermissionRequestPayload
from harness.core.types import AgentId
from harness.gui import bidi, consts
from harness.gui.forms import approval_lines, approval_title
from harness.gui.render import line_rows
from harness.gui.state import AgentView
from harness.gui.themes import Theme
from harness.gui.widgets.stream import StyledText

_ROW_HEIGHT = 32
_BADGE = " {count}"
_APPROVAL_ROWS = 12
_APPROVAL_WIDTH = 0.9
_RTL_TAG = "rtl"


def button(master: tk.Misc, theme: Theme, text: str, command: Callable[[], None]) -> ctk.CTkButton:
    """A flat button in the theme's colours."""
    return ctk.CTkButton(
        master,
        text=text,
        command=command,
        height=28,
        corner_radius=4,
        font=(theme.font_mono, theme.font_size),
        fg_color=theme.color("surface"),
        hover_color=theme.color("selection"),
        text_color=theme.color("foreground"),
        border_width=1,
        border_color=theme.color("border"),
    )


class Sidebar(ctk.CTkFrame):
    """The agent list: always on the left (14 §2)."""

    def __init__(
        self,
        master: tk.Misc,
        theme: Theme,
        *,
        on_select: Callable[[AgentId], None],
        on_new: Callable[[], None],
        on_settings: Callable[[], None],
        on_archived: Callable[[bool], None],
    ) -> None:
        super().__init__(master, width=consts.SIDEBAR_WIDTH, corner_radius=0)
        self.grid_propagate(False)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self._theme = theme
        self._on_select = on_select
        self._signature: object = None
        self._archived = tk.BooleanVar(value=False)
        self._title = ctk.CTkLabel(self, text=consts.LABEL_AGENTS, anchor="w")
        self._title.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 4))
        self._list = ctk.CTkScrollableFrame(self, corner_radius=0)
        self._list.grid(row=1, column=0, sticky="nsew")
        self._list.grid_columnconfigure(0, weight=1)
        self._new = button(self, theme, consts.LABEL_NEW_AGENT, on_new)
        self._new.grid(row=2, column=0, sticky="ew", padx=10, pady=(6, 2))
        self._toggle = ctk.CTkCheckBox(
            self,
            text=consts.LABEL_ARCHIVED,
            variable=self._archived,
            command=lambda: on_archived(self._archived.get()),
            checkbox_width=16,
            checkbox_height=16,
        )
        self._toggle.grid(row=3, column=0, sticky="w", padx=12, pady=2)
        self._settings = button(self, theme, consts.LABEL_SETTINGS, on_settings)
        self._settings.grid(row=4, column=0, sticky="ew", padx=10, pady=(2, 10))
        self.apply_theme(theme)

    def apply_theme(self, theme: Theme) -> None:
        """Switch colours and fonts."""
        self._theme, self._signature = theme, None
        color = theme.color
        mono = (theme.font_mono, theme.font_size)
        self.configure(fg_color=color("sidebar_bg"))
        self._list.configure(fg_color=color("sidebar_bg"))
        self._title.configure(text_color=color("muted"), font=mono)
        self._toggle.configure(text_color=color("sidebar_fg"), font=mono)
        for widget in (self._new, self._settings):
            widget.configure(
                fg_color=color("surface"),
                hover_color=color("selection"),
                text_color=color("foreground"),
                border_color=color("border"),
                font=mono,
            )

    def refresh(self, views: list[AgentView], selected: AgentId | None) -> None:
        """Redraw the rows when anything they show has changed."""
        signature = [
            (
                view.state.id,
                view.state.name,
                view.state.status,
                view.badge,
                view.state.id == selected,
            )
            for view in views
        ]
        if signature == self._signature:
            return
        self._signature = signature
        for child in self._list.winfo_children():
            child.destroy()
        for row, view in enumerate(views):
            self._row(view, selected=view.state.id == selected).grid(
                row=row, column=0, sticky="ew", padx=4, pady=1
            )

    def _row(self, view: AgentView, *, selected: bool) -> ctk.CTkFrame:
        theme, state = self._theme, view.state
        color = theme.color
        background = color("sidebar_selected") if selected else color("sidebar_bg")
        frame = ctk.CTkFrame(self._list, height=_ROW_HEIGHT, corner_radius=4, fg_color=background)
        frame.grid_columnconfigure(1, weight=1)
        dot = ctk.CTkLabel(
            frame,
            text=consts.STATUS_DOT,
            width=18,
            text_color=color(consts.STATUS_COLOR_TOKENS[state.status]),
        )
        dot.grid(row=0, column=0, padx=(8, 4))
        name = ctk.CTkLabel(
            frame,
            text=state.name,
            anchor="w",
            font=(theme.font_text, theme.font_size),
            text_color=color("sidebar_fg"),
        )
        name.grid(row=0, column=1, sticky="ew")
        badge_text = _BADGE.format(count=view.badge) if view.badge else ""
        waiting = state.status is AgentStatus.WAITING_APPROVAL
        badge = ctk.CTkLabel(
            frame,
            text=badge_text,
            width=24,
            font=(theme.font_mono, theme.font_size),
            text_color=color("warning") if waiting else color("accent"),
        )
        badge.grid(row=0, column=2, padx=(4, 8))
        for widget in (frame, dot, name, badge):
            widget.bind("<Button-1>", lambda _event, agent_id=state.id: self._on_select(agent_id))
        return frame


class ApprovalPanel(ctk.CTkFrame):
    """The approval card shown over the stream, above the input box (14 §3)."""

    def __init__(
        self,
        master: tk.Misc,
        theme: Theme,
        on_answer: Callable[[PermissionRequestPayload, PermissionAnswer, PermissionScope], None],
    ) -> None:
        super().__init__(master, corner_radius=6, border_width=1)
        self.grid_columnconfigure(0, weight=1)
        self._on_answer = on_answer
        self._request: PermissionRequestPayload | None = None
        self._title = ctk.CTkLabel(self, anchor="w")
        self._title.grid(row=0, column=0, sticky="ew", padx=12, pady=(8, 2))
        self._body = StyledText(self, theme, height=_APPROVAL_ROWS)
        self._body.grid(row=1, column=0, sticky="ew", padx=8)
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.grid(row=2, column=0, sticky="e", padx=8, pady=8)
        answers = (
            (consts.LABEL_APPROVE_ONCE, PermissionAnswer.APPROVE, PermissionScope.ONCE),
            (consts.LABEL_APPROVE_ALWAYS, PermissionAnswer.APPROVE, PermissionScope.AGENT),
            (consts.LABEL_DENY, PermissionAnswer.DENY, PermissionScope.ONCE),
        )
        self._buttons = []
        for column, (label, answer, scope) in enumerate(answers):
            widget = button(row, theme, label, lambda a=answer, s=scope: self._answer(a, s))
            widget.grid(row=0, column=column, padx=4)
            self._buttons.append(widget)
        self.apply_theme(theme)

    def apply_theme(self, theme: Theme) -> None:
        """Switch colours and fonts."""
        color = theme.color
        self.configure(fg_color=color("surface"), border_color=color("warning"))
        self._title.configure(
            text_color=color("warning"), font=(theme.font_text, theme.font_size, "bold")
        )
        self._body.apply_theme(theme)
        self._body.configure(background=color("surface"))

    def _answer(self, answer: PermissionAnswer, scope: PermissionScope) -> None:
        if self._request is not None:
            self._on_answer(self._request, answer, scope)

    def show(self, view: AgentView | None) -> bool:
        """Fill the card with the agent's first open request; ``False`` when there is none."""
        request = view.approvals[0] if view is not None and view.approvals else None
        if request is None or view is None:
            self._request = None
            return False
        self._title.configure(text=approval_title(view))
        if request is not self._request:
            self._request = request
            layout = self._body.layout(round(self.master.winfo_width() * _APPROVAL_WIDTH))
            rows = [row for line in approval_lines(request) for row in line_rows(line, layout)]
            self._body.configure(height=max(1, min(len(rows), _APPROVAL_ROWS)))
            self._body.replace_all(rows)
        return True


class InputBox(ctk.CTkFrame):
    """Where the user types to the agent; aligns itself to the language typed (02 §7)."""

    def __init__(
        self,
        master: tk.Misc,
        theme: Theme,
        *,
        on_send: Callable[[str], None],
        on_stop: Callable[[], None],
        on_extra: Callable[[], None],
    ) -> None:
        super().__init__(master, corner_radius=6, border_width=1)
        self.grid_columnconfigure(0, weight=1)
        self._on_send = on_send
        self._font = tkfont.Font(family=theme.font_text, size=theme.font_size)
        self.text = tk.Text(
            self,
            height=4,
            wrap="word",
            borderwidth=0,
            highlightthickness=0,
            undo=True,
            padx=8,
            pady=6,
        )
        self.text.grid(row=0, column=0, rowspan=3, sticky="nsew", padx=(4, 0), pady=4)
        self.text.tag_configure(_RTL_TAG, justify="right")
        self.text.bind("<KeyRelease>", self._align)
        self._send = button(self, theme, consts.LABEL_SEND, self.send)
        self._send.grid(row=0, column=1, padx=6, pady=(6, 2))
        self._stop = button(self, theme, consts.LABEL_STOP, on_stop)
        self._stop.grid(row=1, column=1, padx=6, pady=2)
        self._extra = button(self, theme, consts.LABEL_PAUSE, on_extra)
        self._extra.grid(row=2, column=1, padx=6, pady=(2, 6))
        self.apply_theme(theme)

    def apply_theme(self, theme: Theme) -> None:
        """Switch colours and fonts."""
        color = theme.color
        self._font.configure(family=theme.font_text, size=theme.font_size)
        self.configure(fg_color=color("surface"), border_color=color("border"))
        self.text.configure(
            font=self._font,
            background=color("surface"),
            foreground=color("foreground"),
            insertbackground=color("cursor"),
            selectbackground=color("selection"),
        )
        for widget in (self._send, self._stop, self._extra):
            widget.configure(
                fg_color=color("background"),
                hover_color=color("selection"),
                text_color=color("foreground"),
                border_color=color("border"),
                font=(theme.font_mono, theme.font_size),
            )

    def _align(self, _event: object = None) -> None:
        self.text.tag_remove(_RTL_TAG, "1.0", "end")
        last = int(self.text.index("end-1c").split(".")[0])
        for number in range(1, last + 1):
            if bidi.is_rtl(self.text.get(f"{number}.0", f"{number}.end")):
                self.text.tag_add(_RTL_TAG, f"{number}.0", f"{number}.end+1c")

    def send(self) -> None:
        """Hand the typed text over (as typed, never reordered) and clear the box."""
        typed = self.text.get("1.0", "end-1c")
        if typed.strip():
            self.text.delete("1.0", "end")
            self._on_send(typed)

    def set_state(self, status: AgentStatus | None) -> None:
        """Enable the buttons that make sense in this status."""
        active = status in consts.ACTIVE_STATUSES
        self._stop.configure(
            state="normal" if active or status is AgentStatus.PAUSED else "disabled"
        )
        self._send.configure(
            state="disabled" if status in (None, AgentStatus.ARCHIVED) else "normal"
        )
        label = {
            AgentStatus.PAUSED: consts.LABEL_RESUME,
            AgentStatus.ERROR: consts.LABEL_RETRY,
        }.get(status, consts.LABEL_PAUSE)  # type: ignore[arg-type]
        usable = status in (AgentStatus.RUNNING, AgentStatus.PAUSED, AgentStatus.ERROR)
        self._extra.configure(text=label, state="normal" if usable else "disabled")
