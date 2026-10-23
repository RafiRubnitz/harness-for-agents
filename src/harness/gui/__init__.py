"""Desktop GUI (CustomTkinter). Talks to the core only through ``harness.core``."""

import gc

from harness.core.errors import CoreLockedError
from harness.core.handle import CoreLauncher
from harness.gui import consts
from harness.gui.bridge import BridgeError, CoreBridge
from harness.gui.state import AgentView, AppState
from harness.gui.themes import Theme, load_theme

__all__ = [
    "AgentView",
    "AppState",
    "BridgeError",
    "CoreBridge",
    "Theme",
    "load_theme",
    "run_gui",
]


def _alert(title: str, message: str) -> None:
    """Show a message without a main window (Tk is imported only when a GUI is wanted)."""
    import tkinter as tk  # noqa: PLC0415
    from tkinter import messagebox  # noqa: PLC0415

    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(title, message, parent=root)
    root.destroy()


def run_gui(launcher: CoreLauncher) -> int:
    """Open the window over a core running on a background thread; returns an exit code."""
    from harness.gui.widgets.window import MainWindow  # noqa: PLC0415 - Tk loads on demand

    bridge = CoreBridge(launcher)
    try:
        bridge.start()
    except CoreLockedError as error:
        _alert(consts.TITLE, consts.MSG_LOCKED.format(detail=error.message))
        return consts.EXIT_CORE_LOCKED
    except Exception as error:
        _alert(consts.TITLE, consts.MSG_START_FAILED.format(detail=error))
        return consts.EXIT_CORE_LOCKED
    window = MainWindow(bridge)
    # A Tk object freed by the collector on another thread calls into Tcl and waits for this
    # loop; if this thread is waiting on that one (the log queue), both hang. The window
    # collects on its own tick instead.
    gc.disable()
    try:
        window.mainloop()  # pyright: ignore[reportUnknownMemberType]
    finally:
        gc.enable()
        bridge.stop()
    return consts.EXIT_OK
