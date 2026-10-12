# pyright: basic
"""The conversation stream: one read-only text widget, items folded and styled by tags."""

import tkinter as tk
from collections.abc import Sequence
from tkinter import font as tkfont

import customtkinter as ctk

from harness.gui import consts
from harness.gui.consts import Tag
from harness.gui.items import Conversation, Item
from harness.gui.render import Layout, Row, Style, foldable, item_rows, item_text
from harness.gui.themes import Theme

_MARK = "item_{index}"
_END = "end-1c"
_HEADING_SCALE = {Tag.HEADING_1: 1.5, Tag.HEADING_2: 1.3, Tag.HEADING_3: 1.15}
_BOTTOM = 0.999
# inline code in a right-aligned row: a background there is painted across the whole row
_CODE_RTL = "code_rtl"
_MONO_TAGS = frozenset({Tag.CODE, Tag.CODE_BLOCK, Tag.MONO, Tag.TOOL})
_COLOR_TAGS = {
    Tag.PROMPT: "prompt",
    Tag.MUTED: "muted",
    Tag.ERROR: "error",
    Tag.WARNING: "warning",
    Tag.SUCCESS: "success",
    Tag.ACCENT: "accent",
    Tag.LINK: "accent",
}


class StyledText(tk.Text):
    """A read-only text widget that draws rows of styled pieces with correct RTL order."""

    def __init__(self, master: tk.Misc, theme: Theme, **options: object) -> None:
        super().__init__(
            master,
            wrap="word",
            borderwidth=0,
            highlightthickness=0,
            padx=consts.STREAM_PADDING,
            pady=consts.STREAM_PADDING,
            cursor="arrow",
            **options,  # type: ignore[arg-type]
        )
        self._fonts: dict[tuple[bool, bool, bool, float], tkfont.Font] = {}
        self._theme = theme
        self.apply_theme(theme)
        self.configure(state="disabled")

    # --- fonts and colours ---

    def _font(self, style: Sequence[Tag]) -> tkfont.Font:
        mono = any(tag in _MONO_TAGS for tag in style)
        bold = Tag.BOLD in style or any(tag in _HEADING_SCALE for tag in style)
        italic = Tag.ITALIC in style
        scale = max((_HEADING_SCALE.get(tag, 1.0) for tag in style), default=1.0)
        key = (mono, bold, italic, scale)
        if key not in self._fonts:
            theme = self._theme
            self._fonts[key] = tkfont.Font(
                family=theme.font_mono if mono else theme.font_text,
                size=round(theme.font_size * scale),
                weight="bold" if bold else "normal",
                slant="italic" if italic else "roman",
            )
        return self._fonts[key]

    def apply_theme(self, theme: Theme) -> None:
        """Recolour and refont every tag."""
        self._theme = theme
        self._fonts.clear()
        color = theme.color
        self.configure(
            background=color("background"),
            foreground=color("foreground"),
            insertbackground=color("cursor"),
            selectbackground=color("selection"),
            font=self._font(()),
        )
        for tag, token in _COLOR_TAGS.items():
            self.tag_configure(tag.value, foreground=color(token))
        self.tag_configure(Tag.CODE.value, background=color("surface"))
        self.tag_configure(Tag.CODE_BLOCK.value, background=color("surface"))
        self.tag_configure(_CODE_RTL, foreground=color("accent"))
        self.tag_configure(Tag.STRIKE.value, overstrike=True)
        self.tag_configure(Tag.LINK.value, underline=True)
        self.tag_configure(Tag.QUOTE.value, foreground=color("muted"))
        self.tag_configure(Tag.RTL.value, justify="right")
        self.tag_configure(Tag.LTR.value, justify="left")
        self.tag_raise("sel")

    def measure(self, text: str, style: Style) -> int:
        """The pixel width of ``text`` drawn in ``style``."""
        return self._font(style).measure(text)

    def layout(self, width: int | None = None) -> Layout:
        """The room a row has, for wrapping RTL lines (``width``: when not drawn yet)."""
        width = (width or self.winfo_width()) - 2 * consts.STREAM_PADDING - 4
        return Layout(max(width, consts.MIN_WRAP_WIDTH_PX), self.measure)

    # --- drawing ---

    def _font_tag(self, style: Style) -> str:
        font = self._font(style)
        name = f"font_{font.name}"
        self.tag_configure(name, font=font)
        return name

    def insert_rows(self, rows: Sequence[Row], extra: Sequence[str] = ()) -> None:
        """Append rows at the end; the caller has made the widget writable."""
        for row in rows:
            line_tags = [tag.value for tag in row.tags if tag in (Tag.RTL, Tag.LTR)]
            rtl_row = Tag.RTL in row.tags
            for piece in row.pieces:
                tags = [
                    _CODE_RTL if rtl_row and tag is Tag.CODE else tag.value for tag in piece.style
                ]
                tags.append(self._font_tag(piece.style))
                if piece.rtl:
                    tags.append(Tag.RTL_RUN.value)
                self.insert("end", piece.text, (*tags, *line_tags, *extra))
            block = [tag.value for tag in row.tags if tag is Tag.CODE_BLOCK]
            self.insert("end", "\n", (*line_tags, *block, *extra))

    def replace_all(self, rows: Sequence[Row]) -> None:
        """Show exactly these rows."""
        self.configure(state="normal")
        self.delete("1.0", "end")
        self.insert_rows(rows)
        self.configure(state="disabled")


class StreamView(ctk.CTkFrame):
    """The scrolling conversation of the selected agent."""

    def __init__(self, master: tk.Misc, theme: Theme) -> None:
        super().__init__(master, corner_radius=0, fg_color=theme.color("background"))
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.text = StyledText(self, theme)
        self.text.grid(row=0, column=0, sticky="nsew")
        self._scroll = ctk.CTkScrollbar(self, command=self.text.yview)
        self._scroll.grid(row=0, column=1, sticky="ns")
        self.text.configure(yscrollcommand=self._scroll.set)
        self._conversation: Conversation | None = None
        self._expanded: set[int] = set()
        self._drawn = 0
        self._width = 0
        self._resize_job: str | None = None
        self._menu = tk.Menu(self.text, tearoff=False)
        self._menu.add_command(label=consts.LABEL_COPY, command=self._copy_clicked)
        self._menu.add_command(label=consts.LABEL_COPY_ALL, command=self._copy_all)
        self._clicked: int | None = None
        self.text.bind("<ButtonRelease-1>", self._on_click)
        self.text.bind("<Button-3>", self._on_menu)
        self.text.bind("<Control-c>", self._on_copy)
        self.text.bind("<Configure>", self._on_resize)

    def apply_theme(self, theme: Theme) -> None:
        """Switch colours and fonts, then redraw."""
        self.configure(fg_color=theme.color("background"))
        self.text.apply_theme(theme)
        self.redraw()

    # --- content ---

    def show(self, conversation: Conversation | None) -> None:
        """Draw what changed in ``conversation`` (everything, when it is another one)."""
        if conversation is not self._conversation:
            self._conversation, self._expanded, self._drawn = conversation, set(), 0
            if conversation is not None:
                conversation.touch(0)
        if conversation is None:
            self.text.replace_all([])
            return
        dirty = conversation.take_dirty()
        if dirty is not None:
            self._draw_from(min(dirty, self._drawn))

    def redraw(self) -> None:
        """Draw the whole conversation again (new width or theme)."""
        if self._conversation is not None:
            self._conversation.take_dirty()
            self._draw_from(0)

    def _draw_from(self, first: int) -> None:
        conversation = self._conversation
        if conversation is None:
            return
        text = self.text
        at_bottom = text.yview()[1] >= _BOTTOM
        layout = text.layout()
        text.configure(state="normal")
        if _MARK.format(index=first) not in text.mark_names():
            first = 0
        # up to the last real character: deleting through "end" also eats the newline before
        text.delete(_MARK.format(index=first) if first else "1.0", _END)
        for index in range(first, len(conversation.items)):
            item = conversation.items[index]
            mark = _MARK.format(index=index)
            text.mark_set(mark, _END)
            text.mark_gravity(mark, "left")
            rows = item_rows(item, expanded=id(item) in self._expanded, layout=layout)
            text.insert_rows(rows)
        text.configure(state="disabled")
        self._drawn = len(conversation.items)
        if at_bottom:
            text.see("end")

    # --- interaction ---

    def _item_at(self, x: int, y: int) -> int | None:
        conversation = self._conversation
        if conversation is None or not conversation.items:
            return None
        return self._item_index(self.text.index(f"@{x},{y}"))

    def _item_index(self, position: str) -> int | None:
        conversation = self._conversation
        if conversation is None:
            return None
        found = None
        for index in range(len(conversation.items)):
            mark = _MARK.format(index=index)
            if mark in self.text.mark_names() and self.text.compare(mark, "<=", position):
                found = index
        return found

    def _item(self, index: int | None) -> Item | None:
        if self._conversation is None or index is None:
            return None
        return self._conversation.items[index]

    def _on_click(self, event: "tk.Event[tk.Text]") -> None:
        if self.text.tag_ranges("sel"):
            return
        index = self._item_at(event.x, event.y)
        item = self._item(index)
        if item is None or index is None or not foldable(item):
            return
        self._expanded ^= {id(item)}
        self._draw_from(index)

    def _on_menu(self, event: "tk.Event[tk.Text]") -> None:
        self._clicked = self._item_at(event.x, event.y)
        self._menu.tk_popup(event.x_root, event.y_root)

    def _to_clipboard(self, text: str) -> None:
        self.clipboard_clear()
        self.clipboard_append(text)

    def _copy_clicked(self) -> None:
        item = self._item(self._clicked)
        if item is not None:
            self._to_clipboard(item_text(item))

    def _copy_all(self) -> None:
        if self._conversation is not None:
            self._to_clipboard("\n\n".join(item_text(i) for i in self._conversation.items))

    def _on_copy(self, _event: "tk.Event[tk.Text]") -> str:
        """Copy the original text of the selected items, not the reordered display text."""
        ranges = self.text.tag_ranges("sel")
        if ranges and self._conversation is not None:
            first = self._item_index(str(ranges[0]))
            last = self._item_index(str(ranges[1]))
            if first is not None and last is not None:
                items = self._conversation.items[first : last + 1]
                self._to_clipboard("\n\n".join(item_text(item) for item in items))
        return "break"

    def _on_resize(self, event: "tk.Event[tk.Text]") -> None:
        if event.width == self._width:
            return
        self._width = event.width
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(consts.RESIZE_DEBOUNCE_MS, self.redraw)
